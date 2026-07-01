"""
routing.router — 查询分类与记忆路由核心实现

本模块提供:

1. ``QueryRouter`` — 一个 ``typing.Protocol``, 定义 ``classify`` 协议。
2. ``LLMQueryRouter`` — 基于大模型 (OpenAI 兼容 API) 的具体实现, 支持:
     - Memora 长周期场景调优的 few-shot prompt
     - 鲁棒的 JSON 解析 (容忍 Markdown 代码块 / 多余文本)
     - 超时 / 异常时优雅降级到 ``fallback_type``
     - 同时实现 ``interfaces.QueryRouter`` ABC 的 ``classify_query`` /
       ``route`` / ``check_relevance`` 以兼容既有调用方
3. ``RuleBasedRouter`` — 纯规则兜底实现 (离线可用, 用于测试与降级)。

设计原则:
    - 单一职责: 分类逻辑 / JSON 解析 / 降级策略 各自独立可测。
    - 不吞掉关键异常: 所有异常都经过 logger 记录后再降级。
    - 离线可测: 通过注入 mock client 即可完全离线测试, 不依赖网络。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable

import yaml

from statebudgetmem.core import MemoryPiece, QueryRouter as QueryRouterABC, ViewType
from statebudgetmem.schemas import QueryType
from .models import QueryRecord
from .prompts import build_messages

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 协议定义
# ════════════════════════════════════════════════════════════════

@runtime_checkable
class QueryRouter(Protocol):
    """
    查询路由协议。

    任何实现了 ``classify(self, query: QueryRecord) -> QueryType`` 的对象
    都满足本协议, 可用于依赖注入。
    """

    def classify(self, query: QueryRecord) -> QueryType:  # pragma: no cover
        ...


# ════════════════════════════════════════════════════════════════
# 工具函数: 鲁棒的 JSON 解析
# ════════════════════════════════════════════════════════════════

# 匹配 ```json ... ``` 或 ``` ... ``` 代码块
_CODE_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
# 匹配裸 JSON 对象 {...}
_BARE_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

# query_type 合法值 → 枚举
_TYPE_MAP: dict[str, QueryType] = {
    "current": QueryType.CURRENT,
    "historical": QueryType.HISTORICAL,
    "change": QueryType.CHANGE,
    "general": QueryType.GENERAL,
}


def parse_query_type_from_response(raw: str) -> Optional[QueryType]:
    """
    从 LLM 的原始返回文本中极其稳健地解析出 ``QueryType``。

    解析顺序:
        1. 优先提取方括号中的关键词 (如 [CURRENT]、[change]) —— 对小模型最稳健
        2. 尝试直接 ``json.loads`` 
        3. 尝试提取 ```json ... ``` 代码块
        4. 尝试提取裸 ``{...}`` 对象
        5. 正则搜索 "query_type": "xxx"
        6. 逆向全文扫描（最后兜底，即使小模型胡言乱语，也能把句尾结论抓出来）
    """
    if not raw or not raw.strip():
        return None

    text = raw.strip()

    # ── 1. 优先提取方括号中的关键词 (如 [CURRENT], [change]) ─────────────────
    m_bracket = re.search(r"\[\s*(CURRENT|HISTORICAL|CHANGE|GENERAL)\s*\]", text, re.IGNORECASE)
    if m_bracket:
        qt = _TYPE_MAP.get(m_bracket.group(1).lower())
        if qt is not None:
            return qt

    # ── 2. 直接 json.loads ─────────────────────────────────────
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            qt = _extract_type_from_dict(obj)
            if qt is not None:
                return qt
    except (json.JSONDecodeError, ValueError):
        pass

    # ── 3. 提取 ```json ... ``` 代码块 ─────────────────────────
    m = _CODE_BLOCK_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                qt = _extract_type_from_dict(obj)
                if qt is not None:
                    return qt
        except (json.JSONDecodeError, ValueError):
            pass

    # ── 4. 提取裸 {...} 对象 ───────────────────────────────────
    for m in _BARE_JSON_RE.finditer(text):
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                qt = _extract_type_from_dict(obj)
                if qt is not None:
                    return qt
        except (json.JSONDecodeError, ValueError):
            continue

    # ── 5. 正则搜索 "query_type": "xxx" ───────────────────────
    m = re.search(r'"query_type"\s*:\s*"([a-zA-Z]+)"', text)
    if m:
        qt = _TYPE_MAP.get(m.group(1).lower())
        if qt is not None:
            return qt

    # ── 6. 逆向全文扫描（终极抢救通道） ───────────────────────────
    # 小模型经常输出大段废话，并在句尾输出结论。我们从后往前扫描最先出现的关键字。
    low = text.lower()
    found_types = []
    for key, qt in _TYPE_MAP.items():
        pos = low.rfind(key)  # 从后往前找
        if pos != -1:
            found_types.append((pos, qt))
            
    if found_types:
        # 按出现位置从后往前排序，选择最后出现的关键字作为最终结论
        found_types.sort(key=lambda x: x[0], reverse=True)
        return found_types[0][1]

    return None


def _extract_type_from_dict(obj: dict[str, Any]) -> Optional[QueryType]:
    """从字典中提取 query_type, 兼容多种键名写法。"""
    for key in ("query_type", "type", "category", "label", "classification"):
        if key in obj:
            val = obj[key]
            if isinstance(val, str):
                qt = _TYPE_MAP.get(val.strip().lower())
                if qt is not None:
                    return qt
            # 枚举值直接传入的情况
            if isinstance(val, QueryType):
                return val
    return None


# ════════════════════════════════════════════════════════════════
# 默认配置加载
# ════════════════════════════════════════════════════════════════

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    加载 YAML 配置文件。

    Args:
        config_path: 配置文件路径; None 则使用包内默认 config.yaml。

    Returns:
        配置字典; 文件不存在时返回空 dict (不抛异常, 由调用方用默认值)。
    """
    path = config_path or _DEFAULT_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        logger.debug("配置文件不存在, 使用默认值: %s", path)
        return {}
    except yaml.YAMLError as e:
        logger.warning("配置文件解析失败, 使用默认值: %s, 错误: %s", path, e)
        return {}


# ════════════════════════════════════════════════════════════════
# LLMQueryRouter — 核心实现
# ════════════════════════════════════════════════════════════════

class LLMQueryRouter(QueryRouterABC):
    """
    基于大模型的查询分类与记忆路由器。

    特性:
        - 使用 OpenAI 兼容 API (支持 DeepSeek / Qwen / GLM / OpenAI 等)。
        - Prompt 针对 Memora 长周期状态演进场景调优。
        - 鲁棒 JSON 解析, 容忍 Markdown / 多余文本。
        - 超时 / 异常时降级到 ``fallback_type``。
        - 同时实现 ``interfaces.QueryRouter`` ABC 接口, 兼容既有调用方。

    用法::

        router = LLMQueryRouter(api_key="sk-xxx", base_url="...", model="deepseek-chat")
        qtype = router.classify(QueryRecord(text="我现在还喜欢吃辣吗?"))
        # → QueryType.CHANGE

    离线测试用法 (注入 mock client)::

        router = LLMQueryRouter(client=mock_client)
        # mock_client.chat.completions.create 会被调用, 不触网。
    """

    def __init__(
        self,
        *,
        # ── LLM 客户端相关 ──
        client: Any = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        # ── 路由策略 ──
        fallback_type: Optional[QueryType] = None,
        empty_query_fallback: Optional[bool] = None,
        log_raw_response: Optional[bool] = None,
        # ── Prompt ──
        enable_few_shot: Optional[bool] = None,
        few_shot_count: Optional[int] = None,
        # ── 配置 ──
        config_path: Optional[str] = None,
    ) -> None:
        """
        构造路由器。参数优先级: 显式参数 > config.yaml > 内置默认值。

        Args:
            client            : 已构造的 OpenAI 客户端 (测试时注入 mock)。
                                若提供, 则忽略 api_key/base_url。
            api_key           : API Key; None 则读环境变量 OPENAI_API_KEY。
            base_url          : API Base URL; None 则读 OPENAI_BASE_URL。
            model             : 模型名。
            temperature       : 采样温度 (分类任务建议 0.0)。
            max_tokens        : 最大输出 token 数。
            timeout           : 单次请求超时 (秒)。
            max_retries       : 失败重试次数 (不含首次)。
            fallback_type     : 降级类型; None 则用 config 或 QueryType.GENERAL。
            empty_query_fallback: 空查询是否直接降级 (跳过 LLM)。
            log_raw_response  : 是否打印 LLM 原始返回 (调试用)。
            enable_few_shot   : 是否启用 few-shot 示例。
            few_shot_count    : few-shot 示例数量。
            config_path       : YAML 配置文件路径。
        """
        cfg = load_config(config_path)
        llm_cfg = cfg.get("llm", {}) or {}
        rt_cfg = cfg.get("routing", {}) or {}
        pr_cfg = cfg.get("prompt", {}) or {}

        # ── 解析各参数 (显式 > config > 默认) ──────────────────
        self.api_key = api_key if api_key is not None else (
            llm_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY") or ""
        )
        self.base_url = base_url if base_url is not None else (
            llm_cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL") or None
        )
        self.model = model if model is not None else llm_cfg.get("model", "gpt-4o-mini")
        self.temperature = float(
            temperature if temperature is not None else llm_cfg.get("temperature", 0.0)
        )
        self.max_tokens = int(
            max_tokens if max_tokens is not None else llm_cfg.get("max_tokens", 256)
        )
        self.timeout = float(
            timeout if timeout is not None else llm_cfg.get("timeout", 15)
        )
        self.max_retries = int(
            max_retries if max_retries is not None else llm_cfg.get("max_retries", 1)
        )

        # fallback_type 支持 str 或 QueryType
        fb_raw = fallback_type if fallback_type is not None else rt_cfg.get(
            "fallback_type", "general"
        )
        self.fallback_type = self._coerce_query_type(fb_raw, QueryType.GENERAL)

        self.empty_query_fallback = bool(
            empty_query_fallback if empty_query_fallback is not None
            else rt_cfg.get("empty_query_fallback", True)
        )
        self.log_raw_response = bool(
            log_raw_response if log_raw_response is not None
            else rt_cfg.get("log_raw_response", False)
        )

        self.enable_few_shot = bool(
            enable_few_shot if enable_few_shot is not None
            else pr_cfg.get("enable_few_shot", True)
        )
        self.few_shot_count = int(
            few_shot_count if few_shot_count is not None
            else pr_cfg.get("few_shot_count", 8)
        )

        # ── 构造客户端 ──────────────────────────────────────────
        self._client = client
        if self._client is None:
            self._client = self._build_client()

        # 统计
        self._call_count = 0
        self._fallback_count = 0

    # ── 客户端构造 ──────────────────────────────────────────────
    def _build_client(self) -> Any:
        """
        构造 OpenAI 兼容客户端。

        若 openai 未安装或 api_key 为空, 返回 None (后续 classify 会降级)。
        """
        if not self.api_key:
            logger.warning(
                "LLMQueryRouter 未配置 api_key (且环境变量 OPENAI_API_KEY 为空), "
                "所有分类将降级为 %s", self.fallback_type.value
            )
            return None
        try:
            import openai  # 延迟导入, 离线测试不需要
        except ImportError as e:
            logger.error("openai 包未安装: %s; 分类将降级", e)
            return None

        kwargs: dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        try:
            return openai.OpenAI(**kwargs)
        except Exception as e:  # noqa: BLE001
            logger.error("构造 OpenAI 客户端失败: %s; 分类将降级", e)
            return None

    # ── 核心方法: classify ─────────────────────────────────────
    def classify(self, query: QueryRecord) -> QueryType:
        """
        对查询进行时态分类。

        Args:
            query: QueryRecord, 包含 text / reference_time / context。

        Returns:
            QueryType 之一。异常 / 超时 / 解析失败时返回 fallback_type。
        """
        # ── 空查询快速降级 ──
        if self.empty_query_fallback and query.is_empty():
            logger.info("空查询, 直接降级为 %s", self.fallback_type.value)
            self._fallback_count += 1
            return self.fallback_type

        # ── 客户端不可用 ──
        if self._client is None:
            logger.warning("LLM 客户端不可用, 降级为 %s", self.fallback_type.value)
            self._fallback_count += 1
            return self.fallback_type

        # ── 构造 messages ──
        prompt_dict = query.to_prompt_dict()
        messages = build_messages(
            query_text=prompt_dict["text"],
            reference_time_str=prompt_dict["reference_time"],
            context=prompt_dict["context"],
            enable_few_shot=self.enable_few_shot,
            few_shot_count=self.few_shot_count,
        )

        # ── 调用 LLM (带重试) ──
        raw = self._call_llm_with_retry(messages)
        if raw is None:
            self._fallback_count += 1
            return self.fallback_type

        # ── 解析 ──
        qt = parse_query_type_from_response(raw)
        if qt is None:
            logger.warning(
                "无法从 LLM 返回中解析 query_type, 降级为 %s。原始返回: %s",
                self.fallback_type.value,
                raw[:200],
            )
            self._fallback_count += 1
            return self.fallback_type

        return qt

    def _call_llm_with_retry(self, messages: list[dict[str, str]]) -> Optional[str]:
        """
        调用 LLM, 失败时按 max_retries 重试。

        返回 LLM 文本内容; 全部失败返回 None。
        所有异常都会被 logger 记录 (不吞掉)。
        """
        attempts = self.max_retries + 1
        last_exc: Optional[BaseException] = None

        for i in range(attempts):
            try:
                self._call_count += 1
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = self._extract_content(resp)
                if self.log_raw_response:
                    logger.info("[LLM raw] %s", raw)
                return raw
            except Exception as e:  # noqa: BLE001
                last_exc = e
                logger.warning(
                    "LLM 调用失败 (第 %d/%d 次): %s: %s",
                    i + 1, attempts, type(e).__name__, e,
                )
                # 短暂退避后重试
                if i < attempts - 1:
                    time.sleep(0.3 * (i + 1))

        if last_exc is not None:
            logger.error(
                "LLM 调用 %d 次均失败, 降级。最后错误: %s: %s",
                attempts, type(last_exc).__name__, last_exc,
            )
        return None

    @staticmethod
    def _extract_content(resp: Any) -> str:
        """
        从 OpenAI 兼容的响应对象中提取文本内容。

        兼容多种结构:
            - resp.choices[0].message.content  (标准)
            - resp["choices"][0]["message"]["content"]  (dict)
            - resp.choices[0].text  (旧版 completions)
        """
        # 标准对象访问
        try:
            choices = resp.choices
            if choices and len(choices) > 0:
                msg = choices[0].message
                content = getattr(msg, "content", None)
                if content is not None:
                    return str(content)
                # dict 风格
                if isinstance(msg, dict):
                    return str(msg.get("content", ""))
        except AttributeError:
            pass

        # dict 风格访问
        try:
            if isinstance(resp, dict):
                choices = resp.get("choices") or []
                if choices:
                    msg = choices[0].get("message") or {}
                    content = msg.get("content")
                    if content is not None:
                        return str(content)
        except Exception:  # noqa: BLE001
            pass

        # 兜底: 尝试转 str
        return str(resp)

    # ── 兼容 interfaces.QueryRouter ABC ────────────────────────
    def classify_query(
        self, query: str, context: Optional[str] = None
    ) -> QueryType:
        """
        实现 ``interfaces.QueryRouter.classify_query``。

        把 (query_str, context) 包装成 QueryRecord 后调用 classify。
        reference_time 默认为当前时刻。
        """
        record = QueryRecord(text=query, context=context)
        return self.classify(record)

    def route(
        self, query: str, query_type: Optional[QueryType] = None
    ) -> ViewType:
        """
        实现 ``interfaces.QueryRouter.route``。

        把 QueryType 映射到 ViewType:
            CURRENT     → CURRENT  (只看当前有效记忆)
            HISTORICAL  → HISTORY  (看历史记忆)
            CHANGE      → BOTH     (需要对比过去与现在)
            GENERAL     → NONE     (通用查询不读取个人记忆)
        """
        if query_type is None:
            query_type = self.classify_query(query)
        mapping = {
            QueryType.CURRENT: ViewType.CURRENT,
            QueryType.HISTORICAL: ViewType.HISTORY,
            QueryType.CHANGE: ViewType.BOTH,
            QueryType.GENERAL: ViewType.NONE,
        }
        return mapping.get(query_type, ViewType.NONE)

    def check_relevance(self, memory: MemoryPiece, query: str) -> float:
        """
        实现 ``interfaces.QueryRouter.check_relevance``。

        简单实现: 基于关键词重叠的 0~1 相关性分数。
        生产环境可替换为 embedding 相似度。
        """
        if not query or not getattr(memory, "content", ""):
            return 0.0
        q_tokens = set(_tokenize(query))
        m_tokens = set(_tokenize(memory.content))
        if not q_tokens or not m_tokens:
            return 0.0
        overlap = len(q_tokens & m_tokens)
        return min(1.0, overlap / max(len(q_tokens), 1))

    # ── 统计 ────────────────────────────────────────────────────
    def stats(self) -> dict[str, int]:
        """Return compact call statistics."""
        return {
            "call_count": self._call_count,
            "fallback_count": self._fallback_count,
        }

    def get_stats(self) -> dict[str, int]:
        """Backward-compatible verbose statistics used by earlier demos."""
        return {
            "total_calls": self._call_count,
            "successful": max(0, self._call_count - self._fallback_count),
            "fallbacks": self._fallback_count,
        }

    # ── 工具 ────────────────────────────────────────────────────
    @staticmethod
    def _coerce_query_type(
        value: Any, default: QueryType = QueryType.GENERAL
    ) -> QueryType:
        """把 str / QueryType 统一成 QueryType。"""
        if isinstance(value, QueryType):
            return value
        if isinstance(value, str):
            return _TYPE_MAP.get(value.strip().lower(), default)
        return default


# ════════════════════════════════════════════════════════════════
# RuleBasedRouter — 纯规则兜底 (离线可用)
# ════════════════════════════════════════════════════════════════

# 通用知识关键词 (优先级最高: 天气/数学/常识等不依赖个人记忆)
_GENERAL_KEYWORDS: list[str] = [
    "天气", "气温", "温度多少", "下雨吗", "下雪吗",
    "数学", "计算", "算一下", "算算", "乘以", "除以", "加", "减",
    "首都", "省会", "公式", "定义", "是什么意思", "什么意思",
    "怎么回事", "帮我查", "百科", "维基",
    "光速", "圆周率", "勾股定理", "牛顿", "爱因斯坦",
]

# 时态关键词表 (按优先级 CHANGE > HISTORICAL > CURRENT 排序)
# 注意: GENERAL 已单独提前检查, 故不在此表中。
_TEMPORAL_KEYWORDS: list[tuple[QueryType, list[str]]] = [
    # CHANGE 优先级最高 (因为"现在还…吗"等隐含对比)
    (QueryType.CHANGE, [
        "变化", "变了", "改了", "换成", "对比", "相比", "演变", "转变",
        "什么时候开始", "从...到", "还...吗", "还喜欢", "还在", "还用",
        "之前和现在", "以前和现在", "上半年", "下半年", "改变", "差异",
        "不同了", "不一样了", "切换", "迁移", "有什么改变", "有什么变化",
    ]),
    # HISTORICAL
    (QueryType.HISTORICAL, [
        "以前", "之前", "那时候", "上周", "上个月", "去年", "曾经",
        "当时", "过去", "前年", "大学时候", "小时候", "从前", "早前",
        "前段时间", "先前", "昔日", "三个月前", "半年前",
    ]),
    # CURRENT
    (QueryType.CURRENT, [
        "现在", "目前", "当前", "最近", "现在还", "现在是不是",
        "我现在的", "我目前的", "我最近的", "今天", "此刻", "眼下",
        "当今", "现阶段", "现在适合", "现在在",
    ]),
]


class RuleBasedRouter(QueryRouterABC):
    """
    纯规则路由器 (离线可用, 不依赖 LLM)。

    用途:
        1. 作为 LLMQueryRouter 的离线兜底。
        2. 单元测试中的对照基线。
        3. 无网络 / 无 API Key 环境下的可用实现。

    分类策略 (按优先级):
        1. GENERAL  : 先检查通用知识关键词 (天气/数学/常识等)。
        2. CHANGE   : 再检查状态变化/对比关键词。
        3. HISTORICAL: 检查过去时态关键词。
        4. CURRENT  : 检查当前时态关键词。
        5. 无匹配   : 返回 fallback_type (默认 GENERAL)。
    """

    def __init__(self, fallback_type: QueryType = QueryType.GENERAL) -> None:
        self.fallback_type = fallback_type

    def classify(self, query: QueryRecord) -> QueryType:
        if query.is_empty():
            return self.fallback_type
        text = query.text.lower()

        # 1. 先检查 GENERAL (通用知识不依赖个人记忆, 优先级最高)
        for kw in _GENERAL_KEYWORDS:
            if kw in text:
                return QueryType.GENERAL

        # 2-4. 按优先级检查时态关键词
        for qtype, keywords in _TEMPORAL_KEYWORDS:
            for kw in keywords:
                if kw in text:
                    return qtype

        # 5. 无匹配 → fallback
        return self.fallback_type

    # 兼容 ABC
    def classify_query(
        self, query: str, context: Optional[str] = None
    ) -> QueryType:
        return self.classify(QueryRecord(text=query, context=context))

    def route(
        self, query: str, query_type: Optional[QueryType] = None
    ) -> ViewType:
        if query_type is None:
            query_type = self.classify_query(query)
        mapping = {
            QueryType.CURRENT: ViewType.CURRENT,
            QueryType.HISTORICAL: ViewType.HISTORY,
            QueryType.CHANGE: ViewType.BOTH,
            QueryType.GENERAL: ViewType.NONE,
        }
        return mapping.get(query_type, ViewType.NONE)

    def check_relevance(self, memory: MemoryPiece, query: str) -> float:
        if not query or not getattr(memory, "content", ""):
            return 0.0
        q_tokens = set(_tokenize(query))
        m_tokens = set(_tokenize(memory.content))
        if not q_tokens or not m_tokens:
            return 0.0
        overlap = len(q_tokens & m_tokens)
        return min(1.0, overlap / max(len(q_tokens), 1))


# ════════════════════════════════════════════════════════════════
# 辅助函数
# ════════════════════════════════════════════════════════════════

def _tokenize(text: str) -> list[str]:
    """
    简单分词: 中文按字, 英文按词。

    用于 check_relevance 的关键词重叠计算。
    """
    if not text:
        return []
    tokens: list[str] = []
    buf: list[str] = []
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            if buf:
                tokens.append("".join(buf).lower())
                buf = []
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf).lower())
                buf = []
    if buf:
        tokens.append("".join(buf).lower())
    return tokens


__all__ = [
    "QueryRouter",
    "LLMQueryRouter",
    "RuleBasedRouter",
    "parse_query_type_from_response",
    "load_config",
]
