from __future__ import annotations

import json
import os
from typing import Any

from statebudgetmem.preprocessing.compat import MemoryType, UpdateOperation
from statebudgetmem.preprocessing.models import ParsedMemory, RawMessage, parse_timestamp
from statebudgetmem.preprocessing.normalizer import canonical_attribute, clean_value


class ApiParser:
    """外部 API 版预处理器。

    默认使用 OpenAI-compatible Chat Completions 接口。
    可用于 DeepSeek / OpenAI / 其他兼容服务。

    环境变量：
    - SBM_API_KEY 或 DEEPSEEK_API_KEY 或 OPENAI_API_KEY
    - SBM_API_BASE_URL，可选
    - SBM_API_MODEL，可选
    """

    def __init__(self, model: str = "deepseek-chat") -> None:
        self.model = os.getenv("SBM_API_MODEL", model)

    def parse(self, raw: RawMessage) -> list[ParsedMemory]:
        data = self._call_api(raw)
        timestamp = parse_timestamp(raw.timestamp)

        memories: list[ParsedMemory] = []
        for item in data.get("memories", []):
            attribute = canonical_attribute(str(item.get("attribute", "fact")))
            value = clean_value(item.get("value"))
            previous_value = item.get("previous_value")
            operation = _parse_operation(item.get("operation", "add"))
            memory_type = _parse_memory_type(item.get("memory_type"), attribute)

            evidence = str(item.get("evidence_span") or raw.content)
            confidence = float(item.get("confidence", 0.7))
            tags = item.get("tags") or []
            if isinstance(tags, str):
                tags = [tags]

            memories.append(
                ParsedMemory(
                    content=str(item.get("content") or _build_content(attribute, value)),
                    timestamp=timestamp,
                    memory_type=memory_type,
                    operation=operation,
                    attribute=attribute,
                    value=value,
                    previous_value=clean_value(previous_value) if previous_value else None,
                    evidence_span=evidence,
                    tags=list(tags),
                    confidence=max(0.0, min(1.0, confidence)),
                    source=raw.source,
                    needs_review=bool(item.get("needs_review", False)),
                )
            )

        return memories

    def _call_api(self, raw: RawMessage) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("缺少 openai 依赖，请先安装：pip install openai") from exc

        api_key = (
            os.getenv("SBM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError("未检测到 API key，请设置 SBM_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY")

        base_url = os.getenv("SBM_API_BASE_URL")
        client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=self.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是 StateBudgetMem 的 preprocessing 模块。"
                        "请把用户自然语言记忆抽取成 JSON，不要输出 JSON 以外的文字。"
                    ),
                },
                {
                    "role": "user",
                    "content": _build_prompt(raw),
                },
            ],
        )

        content = response.choices[0].message.content or "{}"
        return json.loads(content)


def _build_prompt(raw: RawMessage) -> str:
    return f"""
请从下面文本中抽取结构化记忆，输出 JSON：

{{
  "memories": [
    {{
      "content": "自然语言形式的结构化记忆，例如：用户当前居住地是北京",
      "attribute": "标准化属性名，例如 home_location / preference / allergy / breakfast",
      "value": "当前值",
      "previous_value": "旧值；没有则为 null",
      "operation": "add / update / delete / noop / merge / supersede / temp_invalidate",
      "memory_type": "fact / event / preference / dialog",
      "evidence_span": "原文证据片段",
      "tags": ["标签"],
      "confidence": 0.0到1.0,
      "needs_review": false
    }}
  ]
}}

规则：
1. “以前 A，现在 B” 应抽取 B，previous_value 填 A，operation 用 supersede。
2. “暂时/这周/这几天” 可用 temp_invalidate。
3. “别记/忘掉/删除” 用 delete。
4. 没有明确事实时 memories 返回空列表。
5. 不要编造原文没有的信息。

原始消息：
role: {raw.role}
timestamp: {raw.timestamp}
content: {raw.content}
"""


def _parse_operation(value: Any) -> UpdateOperation:
    raw = str(value or "add").strip().lower()
    mapping = {
        "add": UpdateOperation.ADD,
        "update": UpdateOperation.UPDATE,
        "delete": UpdateOperation.DELETE,
        "noop": UpdateOperation.NOOP,
        "merge": UpdateOperation.MERGE,
        "supersede": UpdateOperation.SUPERSEDE,
        "temp_invalidate": UpdateOperation.TEMP_INVALIDATE,
        "temporary_invalidate": UpdateOperation.TEMP_INVALIDATE,
        "restore": UpdateOperation.RESTORE,
    }
    return mapping.get(raw, UpdateOperation.ADD)


def _parse_memory_type(value: Any, attribute: str) -> MemoryType:
    raw = str(value or "").strip().lower()

    if raw == "preference" or attribute == "preference":
        return MemoryType.PREFERENCE

    if raw == "event":
        return MemoryType.EVENT

    if raw == "dialog":
        return MemoryType.DIALOG

    if raw == "summary":
        return MemoryType.SUMMARY

    if raw == "portrait":
        return MemoryType.PORTRAIT

    return MemoryType.FACT


def _build_content(attribute: str, value: str) -> str:
    return f"用户的{attribute}是{value}"
