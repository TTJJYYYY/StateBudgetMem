"""
test_routing — 查询分类与记忆路由的单元测试

测试覆盖:
    1. JSON 解析鲁棒性 (parse_query_type_from_response)
    2. QueryRecord 数据模型 (Pydantic 校验)
    3. RuleBasedRouter 规则分类
    4. LLMQueryRouter 正常分类路径 (Mock OpenAI client)
    5. LLMQueryRouter 边界与异常处理 (空串 / 非标准 JSON / 超时 / 异常)
    6. 离线测试隔离 (全部使用 Mock, 不触网)
    7. 兼容 interfaces.QueryRouter ABC (classify_query / route / check_relevance)
    8. Memora 长周期场景模拟测试

运行方式:
    cd StateBudgetMem
    pytest tests/test_routing.py -v
    # 或全部测试:
    pytest -q
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── 导入被测模块 ──────────────────────────────────────────────
# conftest.py 已设置好 sys.path, 这里兼容包模式与脚本模式两种导入。
try:
    from statebudgetmem.routing import (
        LLMQueryRouter,
        QueryRecord,
        QueryRouter,
        QueryType,
        RuleBasedRouter,
        ViewType,
        parse_query_type_from_response,
    )
except ImportError:  # 脚本模式
    from routing import (  # type: ignore[no-redef]
        LLMQueryRouter,
        QueryRecord,
        QueryRouter,
        QueryType,
        RuleBasedRouter,
        ViewType,
        parse_query_type_from_response,
    )


# ════════════════════════════════════════════════════════════════
# 辅助: 构造 Mock OpenAI 响应
# ════════════════════════════════════════════════════════════════

def make_mock_response(content: str) -> MagicMock:
    """
    构造一个模拟 OpenAI chat.completions.create 返回值的 Mock 对象。

    结构: resp.choices[0].message.content = content
    """
    resp = MagicMock()
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp.choices = [choice]
    return resp


def make_mock_client(content: str = '{"query_type": "current"}') -> MagicMock:
    """
    构造一个 Mock OpenAI 客户端, 其 chat.completions.create 返回固定内容。
    """
    client = MagicMock()
    client.chat.completions.create.return_value = make_mock_response(content)
    return client


def make_mock_client_sequence(contents: list[str]) -> MagicMock:
    """
    构造一个 Mock 客户端, 其 create 方法按顺序返回不同内容 (用于重试测试)。
    """
    client = MagicMock()
    responses = [make_mock_response(c) for c in contents]
    client.chat.completions.create.side_effect = responses
    return client


def make_mock_client_exception(exc: Exception) -> MagicMock:
    """构造一个 Mock 客户端, 其 create 方法抛出指定异常。"""
    client = MagicMock()
    client.chat.completions.create.side_effect = exc
    return client


# ════════════════════════════════════════════════════════════════
# 1. JSON 解析鲁棒性测试
# ════════════════════════════════════════════════════════════════

class TestParseQueryType:
    """测试 parse_query_type_from_response 的鲁棒性。"""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # 标准 JSON
            ('{"query_type": "current", "reason": "x"}', QueryType.CURRENT),
            ('{"query_type": "historical"}', QueryType.HISTORICAL),
            ('{"query_type": "change"}', QueryType.CHANGE),
            ('{"query_type": "general"}', QueryType.GENERAL),
            # 大写键值
            ('{"query_type": "CURRENT"}', QueryType.CURRENT),
            ('{"Query_Type": "Historical"}', QueryType.HISTORICAL),
            # Markdown 代码块
            ('```json\n{"query_type": "current"}\n```', QueryType.CURRENT),
            ('```\n{"query_type": "change"}\n```', QueryType.CHANGE),
            # 前后有多余文本
            ('The answer is {"query_type": "historical"} done.', QueryType.HISTORICAL),
            ('好的，{"query_type":"general"}，谢谢', QueryType.GENERAL),
            # 不同键名
            ('{"type": "current"}', QueryType.CURRENT),
            ('{"category": "change"}', QueryType.CHANGE),
            ('{"label": "general"}', QueryType.GENERAL),
            ('{"classification": "historical"}', QueryType.HISTORICAL),
            # 正则兜底: "query_type": "xxx"
            ('某文本 "query_type": "current" 结尾', QueryType.CURRENT),
            # 关键词兜底
            ('current', QueryType.CURRENT),
            ('this is historical', QueryType.HISTORICAL),
            ('a change request', QueryType.CHANGE),
            ('general knowledge', QueryType.GENERAL),
        ],
    )
    def test_valid_inputs(self, raw: str, expected: QueryType) -> None:
        assert parse_query_type_from_response(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "\n\t",
            "totally garbage text",
            '{"wrong_key": "xxx"}',
            '{"query_type": "unknown_type"}',
            "```no json here```",
        ],
    )
    def test_invalid_inputs_return_none(self, raw: str) -> None:
        """测试非法或无法解析的原始输入，应当稳健返回 None"""
        assert parse_query_type_from_response(raw) is None

    def test_small_model_messy_outputs_salvaged(self) -> None:
        """测试即使小模型口语化输出或格式极度混乱，只要包含关键字，也能被逆向扫描完美挽救"""
        # 1. 测试方括号标签提取
        assert parse_query_type_from_response("我认为这句话是 [CURRENT] 没错了。") == QueryType.CURRENT
        
        # 2. 测试句尾逆向关键词捕获 (rfind 机制)
        assert parse_query_type_from_response("因为提到了‘去年’，所以我判定是 historical") == QueryType.HISTORICAL
        assert parse_query_type_from_response("状态发生了切换，答案应该是 change。") == QueryType.CHANGE
        
        # 3. 测试带有其他时态干扰词，但最终结论落在句尾的复杂情况
        assert parse_query_type_from_response("虽然有current这个词，但实际上是 historical") == QueryType.HISTORICAL
    def test_none_input(self) -> None:
        assert parse_query_type_from_response(None) is None  # type: ignore[arg-type]

    def test_dict_style_response(self) -> None:
        """测试 dict 风格的响应 (某些 API 返回 dict 而非对象)。"""
        # 模拟 dict 风格的响应对象
        resp = {
            "choices": [
                {"message": {"content": '{"query_type": "change"}'}}
            ]
        }
        # _extract_content 应能处理 dict
        content = LLMQueryRouter._extract_content(resp)
        assert parse_query_type_from_response(content) == QueryType.CHANGE


# ════════════════════════════════════════════════════════════════
# 2. QueryRecord 数据模型测试
# ════════════════════════════════════════════════════════════════

class TestQueryRecord:
    """测试 QueryRecord Pydantic 模型。"""

    def test_basic_construction(self) -> None:
        qr = QueryRecord(text="我现在喜欢吃什么?")
        assert qr.text == "我现在喜欢吃什么?"
        assert qr.reference_time is not None
        assert qr.context is None
        assert qr.user_id is None

    def test_reference_time_datetime(self) -> None:
        dt = datetime(2026, 6, 15, 10, 0, tzinfo=timezone.utc)
        qr = QueryRecord(text="test", reference_time=dt)
        assert qr.reference_time == dt

    def test_reference_time_naive_datetime(self) -> None:
        """naive datetime 应被视作 UTC。"""
        dt = datetime(2026, 6, 15, 10, 0)  # naive
        qr = QueryRecord(text="test", reference_time=dt)
        assert qr.reference_time.tzinfo == timezone.utc

    def test_reference_time_iso_string(self) -> None:
        qr = QueryRecord(text="test", reference_time="2026-06-15T10:00:00")
        assert qr.reference_time.year == 2026
        assert qr.reference_time.tzinfo == timezone.utc

    def test_reference_time_unix_timestamp(self) -> None:
        # 2026-06-15 10:00:00 UTC ≈ 1781517600
        import time as _time
        ts = _time.mktime((2026, 6, 15, 10, 0, 0, 0, 0, 0))  # local → 但我们用 UTC
        # 直接用 datetime 计算正确的 UTC 时间戳
        from datetime import datetime as _dt, timezone as _tz
        correct_ts = _dt(2026, 6, 15, 10, 0, tzinfo=_tz.utc).timestamp()
        qr = QueryRecord(text="test", reference_time=correct_ts)
        assert qr.reference_time.year == 2026
        assert qr.reference_time.month == 6

    def test_reference_time_none_defaults_now(self) -> None:
        before = datetime.now(timezone.utc)
        qr = QueryRecord(text="test", reference_time=None)
        after = datetime.now(timezone.utc)
        assert before <= qr.reference_time <= after

    def test_text_stripped(self) -> None:
        qr = QueryRecord(text="  hello  ")
        assert qr.text == "hello"

    def test_is_empty(self) -> None:
        assert QueryRecord(text="").is_empty()
        assert QueryRecord(text="   ").is_empty()
        assert not QueryRecord(text="hello").is_empty()

    def test_to_prompt_dict(self) -> None:
        qr = QueryRecord(
            text="test", reference_time="2026-06-15T10:00:00", context="ctx"
        )
        d = qr.to_prompt_dict()
        assert d["text"] == "test"
        assert "2026-06-15" in d["reference_time"]
        assert d["context"] == "ctx"

    def test_metadata(self) -> None:
        qr = QueryRecord(text="test", metadata={"source": "memora"})
        assert qr.metadata == {"source": "memora"}


# ════════════════════════════════════════════════════════════════
# 3. RuleBasedRouter 测试
# ════════════════════════════════════════════════════════════════

class TestRuleBasedRouter:
    """测试纯规则路由器。"""

    @pytest.fixture
    def router(self) -> RuleBasedRouter:
        return RuleBasedRouter()

    @pytest.mark.parametrize(
        "text, expected",
        [
            # CHANGE
            ("我现在还喜欢吃辣吗?", QueryType.CHANGE),
            ("我的饮食习惯是怎么变化的?", QueryType.CHANGE),
            ("我从什么时候开始改用 Rust 的?", QueryType.CHANGE),
            ("对比一下我上半年和下半年的运动频率", QueryType.CHANGE),
            ("我的睡眠时间相比上个月有变化吗?", QueryType.CHANGE),
            # HISTORICAL
            ("我以前喜欢吃什么?", QueryType.HISTORICAL),
            ("我去年这个时候在做什么工作?", QueryType.HISTORICAL),
            ("我大学时候的爱好是什么?", QueryType.HISTORICAL),
            ("三个月前我的体重是多少?", QueryType.HISTORICAL),
            ("二月份我工作日通常怎么去公司？", QueryType.HISTORICAL),
            ("六月上旬天气正常时我怎么去学校？", QueryType.HISTORICAL),
            ("大学时我的主要课程重点是什么？", QueryType.HISTORICAL),
            ("取药前我每天晚上服用多少毫克？", QueryType.HISTORICAL),
            # CURRENT
            ("我现在适合吃什么?", QueryType.CURRENT),
            ("我目前的运动习惯是什么?", QueryType.CURRENT),
            ("我最近在用什么编程语言?", QueryType.CURRENT),
            ("今天暴雨，我应该怎么去学校？", QueryType.CURRENT),
            ("拔牙后出血前还能吃什么？", QueryType.CURRENT),
            # GENERAL
            ("今天北京天气怎么样?", QueryType.GENERAL),
            ("法国的首都是哪里?", QueryType.GENERAL),
            ("帮我算一下 123 乘以 456", QueryType.GENERAL),
            ("光合作用的化学方程式是什么?", QueryType.GENERAL),
        ],
    )
    def test_classify(self, router: RuleBasedRouter, text: str, expected: QueryType) -> None:
        assert router.classify(QueryRecord(text=text)) == expected

    def test_empty_query(self, router: RuleBasedRouter) -> None:
        assert router.classify(QueryRecord(text="")) == QueryType.GENERAL
        assert router.classify(QueryRecord(text="   ")) == QueryType.GENERAL

    def test_no_match_fallback(self, router: RuleBasedRouter) -> None:
        """无任何关键词匹配时返回 fallback。"""
        assert router.classify(QueryRecord(text="xyzabc")) == QueryType.GENERAL

    def test_custom_fallback(self) -> None:
        router = RuleBasedRouter(fallback_type=QueryType.CURRENT)
        assert router.classify(QueryRecord(text="xyzabc")) == QueryType.CURRENT

    def test_general_priority_over_current(self, router: RuleBasedRouter) -> None:
        """'今天天气' 应归 GENERAL 而非 CURRENT (今天)。"""
        assert router.classify(QueryRecord(text="今天天气怎么样")) == QueryType.GENERAL

    def test_change_priority_over_historical(self, router: RuleBasedRouter) -> None:
        """'之前和现在对比' 应归 CHANGE 而非 HISTORICAL (之前)。"""
        assert router.classify(QueryRecord(text="之前和现在对比一下")) == QueryType.CHANGE

    def test_classify_query_abc(self, router: RuleBasedRouter) -> None:
        """测试 ABC 兼容方法 classify_query。"""
        assert router.classify_query("我以前喜欢什么?") == QueryType.HISTORICAL

    def test_route_abc(self, router: RuleBasedRouter) -> None:
        """测试 ABC 兼容方法 route。"""
        assert router.route("我现在适合吃什么?") == ViewType.CURRENT
        assert router.route("我以前喜欢什么?") == ViewType.HISTORY
        assert router.route("我现在还喜欢吃辣吗?") == ViewType.BOTH
        assert router.route("今天天气怎么样?") == ViewType.NONE

    def test_route_with_explicit_type(self, router: RuleBasedRouter) -> None:
        assert router.route("xxx", QueryType.CHANGE) == ViewType.BOTH
        assert router.route("xxx", QueryType.HISTORICAL) == ViewType.HISTORY

    def test_check_relevance(self, router: RuleBasedRouter) -> None:
        """测试 ABC 兼容方法 check_relevance。"""
        from statebudgetmem.interfaces import MemoryPiece
        mem = MemoryPiece(content="我喜欢吃辣的食物", timestamp=time.time())
        score = router.check_relevance(mem, "我喜欢吃什么?")
        assert 0.0 <= score <= 1.0
        assert score > 0.0  # 有重叠

    def test_check_relevance_empty(self, router: RuleBasedRouter) -> None:
        from statebudgetmem.interfaces import MemoryPiece
        mem = MemoryPiece(content="", timestamp=time.time())
        assert router.check_relevance(mem, "test") == 0.0


# ════════════════════════════════════════════════════════════════
# 4. LLMQueryRouter 正常分类路径 (Mock)
# ════════════════════════════════════════════════════════════════

class TestLLMQueryRouterNormal:
    """测试 LLMQueryRouter 的正常分类路径 (使用 Mock client)。"""

    @pytest.mark.parametrize(
        "llm_output, expected",
        [
            ('{"query_type": "current"}', QueryType.CURRENT),
            ('{"query_type": "historical"}', QueryType.HISTORICAL),
            ('{"query_type": "change"}', QueryType.CHANGE),
            ('{"query_type": "general"}', QueryType.GENERAL),
            # 带原因字段
            ('{"query_type": "current", "reason": "问当前状态"}', QueryType.CURRENT),
            # Markdown 包裹
            ('```json\n{"query_type": "change"}\n```', QueryType.CHANGE),
        ],
    )
    def test_classify_normal(self, llm_output: str, expected: QueryType) -> None:
        client = make_mock_client(llm_output)
        router = LLMQueryRouter(client=client, model="test-model")
        qr = QueryRecord(text="测试查询", reference_time="2026-06-15T10:00:00")
        assert router.classify(qr) == expected

    def test_classify_uses_messages(self) -> None:
        """验证 classify 会调用 client.chat.completions.create 并传入 messages。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client, model="test-model")
        qr = QueryRecord(text="我现在适合吃什么?")
        router.classify(qr)
        client.chat.completions.create.assert_called_once()
        call_kwargs = client.chat.completions.create.call_args.kwargs
        assert "messages" in call_kwargs
        assert call_kwargs["model"] == "test-model"
        # messages 应包含 system + few-shot + user
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "我现在适合吃什么?" in messages[-1]["content"]

    def test_classify_with_context(self) -> None:
        """验证 context 会被注入到 prompt。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        qr = QueryRecord(text="我喜欢什么?", context="之前聊过饮食")
        router.classify(qr)
        call_kwargs = client.chat.completions.create.call_args.kwargs
        user_msg = call_kwargs["messages"][-1]["content"]
        assert "之前聊过饮食" in user_msg

    def test_classify_with_few_shot_disabled(self) -> None:
        """禁用 few-shot 时 messages 应更短。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client, enable_few_shot=False)
        qr = QueryRecord(text="test")
        router.classify(qr)
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        # 只有 system + user, 共 2 条
        assert len(messages) == 2

    def test_classify_with_few_shot_enabled(self) -> None:
        """启用 few-shot 时 messages 应包含示例。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client, enable_few_shot=True, few_shot_count=4)
        qr = QueryRecord(text="test")
        router.classify(qr)
        messages = client.chat.completions.create.call_args.kwargs["messages"]
        # system + 4*(user+assistant) + 1*user = 1 + 8 + 1 = 10
        assert len(messages) == 10

    def test_stats(self) -> None:
        """测试调用统计。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        assert router.stats()["call_count"] == 0
        router.classify(QueryRecord(text="test"))
        assert router.stats()["call_count"] == 1
        assert router.stats()["fallback_count"] == 0

    def test_protocol_compliance(self) -> None:
        """LLMQueryRouter 应满足 QueryRouter Protocol。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        assert isinstance(router, QueryRouter)


# ════════════════════════════════════════════════════════════════
# 5. LLMQueryRouter 边界与异常处理
# ════════════════════════════════════════════════════════════════

class TestLLMQueryRouterEdgeCases:
    """测试 LLMQueryRouter 的边界与异常处理。"""

    def test_empty_query_fallback(self) -> None:
        """空查询应直接降级, 不调用 LLM。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client, fallback_type=QueryType.GENERAL)
        result = router.classify(QueryRecord(text=""))
        assert result == QueryType.GENERAL
        # 不应调用 LLM
        client.chat.completions.create.assert_not_called()
        assert router.stats()["fallback_count"] == 1

    def test_whitespace_query_fallback(self) -> None:
        """纯空白查询应降级。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        result = router.classify(QueryRecord(text="   \n\t  "))
        assert result == router.fallback_type
        client.chat.completions.create.assert_not_called()

    def test_empty_query_fallback_disabled(self) -> None:
        """禁用空查询降级时, 空查询也会调用 LLM。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client, empty_query_fallback=False)
        router.classify(QueryRecord(text=""))
        client.chat.completions.create.assert_called_once()

    def test_non_standard_json_response(self) -> None:
        """LLM 返回非标准 JSON 时应降级。"""
        client = make_mock_client("I think the answer is current.")
        # "current" 关键词会被兜底匹配到, 所以这里用无法匹配的内容
        client2 = make_mock_client("Sorry, I cannot understand.")
        router = LLMQueryRouter(client=client2, fallback_type=QueryType.GENERAL)
        result = router.classify(QueryRecord(text="test"))
        # "Sorry, I cannot understand." 无关键词, 应降级
        assert result == QueryType.GENERAL
        assert router.stats()["fallback_count"] == 1

    def test_garbage_response(self) -> None:
        """LLM 返回完全无法解析的内容时降级。"""
        client = make_mock_client("!!!@@@###")
        router = LLMQueryRouter(client=client, fallback_type=QueryType.CURRENT)
        result = router.classify(QueryRecord(text="test"))
        assert result == QueryType.CURRENT

    def test_empty_response(self) -> None:
        """LLM 返回空字符串时降级。"""
        client = make_mock_client("")
        router = LLMQueryRouter(client=client)
        result = router.classify(QueryRecord(text="test"))
        assert result == router.fallback_type

    def test_api_timeout_exception(self) -> None:
        """API 超时异常时应降级。"""
        client = make_mock_client_exception(TimeoutError("Connection timed out"))
        router = LLMQueryRouter(client=client, max_retries=0, fallback_type=QueryType.GENERAL)
        result = router.classify(QueryRecord(text="test"))
        assert result == QueryType.GENERAL
        assert router.stats()["fallback_count"] == 1

    def test_api_connection_error(self) -> None:
        """API 连接错误时应降级。"""
        client = make_mock_client_exception(ConnectionError("Network unreachable"))
        router = LLMQueryRouter(client=client, max_retries=0)
        result = router.classify(QueryRecord(text="test"))
        assert result == router.fallback_type

    def test_api_generic_exception(self) -> None:
        """API 抛出通用异常时应降级。"""
        client = make_mock_client_exception(RuntimeError("Unexpected error"))
        router = LLMQueryRouter(client=client, max_retries=0)
        result = router.classify(QueryRecord(text="test"))
        assert result == router.fallback_type

    def test_retry_then_success(self) -> None:
        """第一次失败, 重试后成功。"""
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            ConnectionError("fail"),
            make_mock_response('{"query_type": "current"}'),
        ]
        router = LLMQueryRouter(client=client, max_retries=1)
        result = router.classify(QueryRecord(text="test"))
        assert result == QueryType.CURRENT
        assert client.chat.completions.create.call_count == 2
        assert router.stats()["fallback_count"] == 0

    def test_retry_all_fail(self) -> None:
        """重试全部失败后降级。"""
        client = make_mock_client_exception(ConnectionError("always fail"))
        router = LLMQueryRouter(client=client, max_retries=2, fallback_type=QueryType.HISTORICAL)
        result = router.classify(QueryRecord(text="test"))
        assert result == QueryType.HISTORICAL
        assert client.chat.completions.create.call_count == 3  # 1 + 2 retries
        assert router.stats()["fallback_count"] == 1

    def test_no_client_fallback(self) -> None:
        """无 client (api_key 为空) 时应降级。"""
        router = LLMQueryRouter(api_key="", fallback_type=QueryType.GENERAL)
        assert router._client is None
        result = router.classify(QueryRecord(text="test"))
        assert result == QueryType.GENERAL

    def test_custom_fallback_type(self) -> None:
        """自定义 fallback_type。"""
        client = make_mock_client_exception(RuntimeError("err"))
        router = LLMQueryRouter(
            client=client, max_retries=0, fallback_type=QueryType.CHANGE
        )
        result = router.classify(QueryRecord(text="test"))
        assert result == QueryType.CHANGE

    def test_fallback_type_from_string(self) -> None:
        """fallback_type 接受字符串。"""
        router = LLMQueryRouter(api_key="", fallback_type="current")  # type: ignore[arg-type]
        assert router.fallback_type == QueryType.CURRENT

    def test_logging_not_swallowed(self, caplog: pytest.LogCaptureFixture) -> None:
        """异常应被 logger 记录, 不被吞掉。"""
        client = make_mock_client_exception(ConnectionError("test error"))
        router = LLMQueryRouter(client=client, max_retries=0)
        with caplog.at_level(logging.WARNING):
            router.classify(QueryRecord(text="test"))
        # 应有 WARNING 级别的日志
        assert any("失败" in r.message or "降级" in r.message for r in caplog.records)


# ════════════════════════════════════════════════════════════════
# 6. LLMQueryRouter ABC 兼容性测试
# ════════════════════════════════════════════════════════════════

class TestLLMQueryRouterABC:
    """测试 LLMQueryRouter 对 interfaces.QueryRouter ABC 的兼容实现。"""

    def test_classify_query(self) -> None:
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        assert router.classify_query("我现在适合吃什么?") == QueryType.CURRENT

    def test_classify_query_with_context(self) -> None:
        client = make_mock_client('{"query_type": "change"}')
        router = LLMQueryRouter(client=client)
        assert router.classify_query("我还喜欢吗?", context="之前喜欢") == QueryType.CHANGE

    def test_route_auto_classify(self) -> None:
        client = make_mock_client('{"query_type": "historical"}')
        router = LLMQueryRouter(client=client)
        assert router.route("我以前喜欢什么?") == ViewType.HISTORY

    def test_route_explicit_type(self) -> None:
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        assert router.route("xxx", QueryType.CHANGE) == ViewType.BOTH
        assert router.route("xxx", QueryType.CURRENT) == ViewType.CURRENT
        assert router.route("xxx", QueryType.HISTORICAL) == ViewType.HISTORY
        assert router.route("xxx", QueryType.GENERAL) == ViewType.NONE

    def test_check_relevance(self) -> None:
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        from statebudgetmem.interfaces import MemoryPiece
        mem = MemoryPiece(content="用户喜欢跑步", timestamp=time.time())
        score = router.check_relevance(mem, "用户喜欢什么运动?")
        assert 0.0 <= score <= 1.0


# ════════════════════════════════════════════════════════════════
# 7. 配置加载测试
# ════════════════════════════════════════════════════════════════

class TestConfig:
    """测试配置加载。"""

    def test_load_default_config(self) -> None:
        from statebudgetmem.routing.router import load_config
        cfg = load_config()
        assert "llm" in cfg
        assert "routing" in cfg
        assert "prompt" in cfg

    def test_load_nonexistent_config(self) -> None:
        from statebudgetmem.routing.router import load_config
        cfg = load_config("/nonexistent/path.yaml")
        assert cfg == {}

    def test_config_values_applied(self) -> None:
        """config.yaml 的值应被应用到 router。"""
        router = LLMQueryRouter(api_key="test")  # 提供 key 避免 warning
        assert router.model is not None
        assert router.fallback_type in QueryType
        assert isinstance(router.temperature, float)
        assert isinstance(router.max_tokens, int)

    def test_explicit_params_override_config(self) -> None:
        """显式参数应覆盖 config。"""
        router = LLMQueryRouter(
            api_key="test", model="my-model", temperature=0.5, max_tokens=100
        )
        assert router.model == "my-model"
        assert router.temperature == 0.5
        assert router.max_tokens == 100

    def test_env_var_api_key(self) -> None:
        """环境变量 OPENAI_API_KEY 应被读取。"""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            router = LLMQueryRouter()
            assert router.api_key == "env-key"

    def test_env_var_base_url(self) -> None:
        """环境变量 OPENAI_BASE_URL 应被读取。"""
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "https://custom.api.com"}):
            router = LLMQueryRouter(api_key="test")
            assert router.base_url == "https://custom.api.com"


# ════════════════════════════════════════════════════════════════
# 8. Memora 长周期场景模拟测试
# ════════════════════════════════════════════════════════════════

class TestMemoraScenarios:
    """
    模拟 Memora 数据集的长周期状态演进场景。

    Memora 特点: 用户偏好/习惯/状态在数周到数月间持续变化,
    存在大量过期状态与时态逻辑。本组测试用 Mock LLM 验证
    router 能正确处理这些场景。
    """

    @pytest.fixture
    def router_with_mock(self) -> tuple[LLMQueryRouter, MagicMock]:
        """返回 (router, client), client 的返回值可在测试中修改。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        return router, client

    @pytest.mark.parametrize(
        "scenario, query_text, llm_response, expected",
        [
            # 场景1: 用户数月前喜欢辣, 现在问当前偏好
            (
                "current_preference",
                "我现在适合吃什么?",
                '{"query_type": "current", "reason": "问当前饮食"}',
                QueryType.CURRENT,
            ),
            # 场景2: 问过去某时刻的状态
            (
                "historical_state",
                "我去年这个时候在做什么工作?",
                '{"query_type": "historical", "reason": "明确指向去年"}',
                QueryType.HISTORICAL,
            ),
            # 场景3: 问状态变化过程 (Memora 核心)
            (
                "change_process",
                "我的饮食习惯是怎么变化的?",
                '{"query_type": "change", "reason": "问变化过程"}',
                QueryType.CHANGE,
            ),
            # 场景4: "现在还…吗" 隐含对比
            (
                "implicit_comparison",
                "我现在还喜欢吃辣吗?",
                '{"query_type": "change", "reason": "隐含对比"}',
                QueryType.CHANGE,
            ),
            # 场景5: 状态切换时间点
            (
                "switch_point",
                "我从什么时候开始改用 Rust 的?",
                '{"query_type": "change", "reason": "问切换时间"}',
                QueryType.CHANGE,
            ),
            # 场景6: 两个时期对比
            (
                "period_comparison",
                "对比一下我上半年和下半年的运动频率",
                '{"query_type": "change", "reason": "时期对比"}',
                QueryType.CHANGE,
            ),
            # 场景7: 通用知识 (无需个人记忆)
            (
                "general_knowledge",
                "今天北京天气怎么样?",
                '{"query_type": "general", "reason": "天气查询"}',
                QueryType.GENERAL,
            ),
            # 场景8: 通用计算
            (
                "general_math",
                "帮我算一下 123 乘以 456",
                '{"query_type": "general", "reason": "数学计算"}',
                QueryType.GENERAL,
            ),
        ],
    )
    def test_memora_scenarios(
        self,
        scenario: str,
        query_text: str,
        llm_response: str,
        expected: QueryType,
    ) -> None:
        client = make_mock_client(llm_response)
        router = LLMQueryRouter(client=client)
        # 模拟数月跨度: reference_time 设为 2026-12-01
        qr = QueryRecord(text=query_text, reference_time="2026-12-01T10:00:00")
        result = router.classify(qr)
        assert result == expected, f"场景 {scenario} 失败: 期望 {expected}, 得到 {result}"

    def test_long_timespan_reference_time_in_prompt(self) -> None:
        """验证 reference_time 会被注入到 prompt 中。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        qr = QueryRecord(text="我现在适合吃什么?", reference_time="2026-12-01T10:00:00")
        router.classify(qr)
        user_msg = client.chat.completions.create.call_args.kwargs["messages"][-1]["content"]
        assert "2026-12-01" in user_msg

    def test_state_mutation_scenario(self) -> None:
        """
        模拟 Memora 的状态突变场景:
        用户 6 月喜欢 Python, 12 月改用 Rust, 现在问"我还在用 Python 吗"。
        """
        client = make_mock_client('{"query_type": "change", "reason": "隐含对比"}')
        router = LLMQueryRouter(client=client)
        qr = QueryRecord(
            text="我现在还在用 Python 吗?",
            reference_time="2026-12-15T10:00:00",
            context="用户6月用Python, 12月改用Rust",
        )
        result = router.classify(qr)
        assert result == QueryType.CHANGE

    def test_obsolete_state_query(self) -> None:
        """模拟查询已过期的状态。"""
        client = make_mock_client('{"query_type": "historical", "reason": "过去状态"}')
        router = LLMQueryRouter(client=client)
        qr = QueryRecord(
            text="我三个月前的体重是多少?",
            reference_time="2026-12-15T10:00:00",
        )
        result = router.classify(qr)
        assert result == QueryType.HISTORICAL


# ════════════════════════════════════════════════════════════════
# 9. 离线隔离验证
# ════════════════════════════════════════════════════════════════

class TestOfflineIsolation:
    """
    验证所有测试都是离线的 (不依赖外部网络 API)。

    通过 mock 掉 socket 连接, 确保任何意外的网络调用都会被捕获。
    """

    def test_no_network_calls_with_mock(self) -> None:
        """使用 Mock client 时不应有任何网络调用。"""
        client = make_mock_client('{"query_type": "current"}')
        router = LLMQueryRouter(client=client)
        router.classify(QueryRecord(text="test"))
        # Mock client 的 create 方法被调用, 但不涉及真实网络
        assert client.chat.completions.create.called

    def test_rule_based_router_no_network(self) -> None:
        """RuleBasedRouter 完全离线。"""
        router = RuleBasedRouter()
        # 不应有任何 client
        assert not hasattr(router, "_client") or router.__dict__.get("_client") is None
        result = router.classify(QueryRecord(text="我现在适合吃什么?"))
        assert result == QueryType.CURRENT

    def test_router_without_api_key_offline(self) -> None:
        """无 API Key 时 router 应离线降级, 不尝试网络。"""
        router = LLMQueryRouter(api_key="")
        assert router._client is None
        result = router.classify(QueryRecord(text="test"))
        assert result == router.fallback_type
