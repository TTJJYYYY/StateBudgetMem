"""
routing.prompts — 查询分类的 Prompt 模板与 Few-shot 示例

针对 **Memora** 数据集特征调优:
    Memora 模拟用户数周/数月/数季度的长周期对话, 用户偏好、习惯、状态
    持续演进 (Memory Mutation), 存在大量过期状态与时态逻辑。

因此本 prompt 特别强调:
    1. 对"过去 / 现在 / 之前 / 现在 / 最近 / 以前"等时间线索高度敏感。
    2. 对"改了 / 换了 / 不再 / 现在 / 之前 vs 现在"等状态切换信号敏感。
    3. 区分"问当前有效状态" vs "问历史某时刻" vs "问变化过程"。
    4. 通用常识 / 天气 / 数学等不依赖个人记忆的查询归为 GENERAL。
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════════
# 系统提示词 (System Prompt)
# ════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
你是一个"时态感知查询分类器" (Temporal-Aware Query Classifier)。

## 背景
你服务于一个个性化智能体的长期记忆系统。该智能体记录了用户数周到数月的
对话历史, 用户的偏好、习惯、状态会随时间持续演进 (例如: 以前喜欢吃辣,
后来因为胃病改为清淡饮食; 以前用 Python, 现在改用 Rust)。

## 任务
把用户的查询分类为以下 4 种类型之一:

- **current**     : 查询"当前有效"的状态或偏好。关键词: 现在、目前、当前、
                    最近、现在还、现在是不是、我现在的…
- **historical**  : 查询"过去某一特定时刻"的状态。关键词: 以前、之前、
                    那时候、上周、上个月、去年、曾经、当时、过去…
- **change**      : 查询状态的"演进 / 对比 / 切换 / 变化过程"。
                    关键词: 变了没、改了、换成、对比、之前 vs 现在、
                    演变、转变、什么时候开始、从…到…、还…吗 (隐含对比)
- **general**     : 与个人历史记忆完全无关的通用事实 / 常识 / 天气 /
                    数学计算 / 世界知识。不需要检索个人记忆即可回答。

## 关键判别原则 (针对长周期状态演进场景)
1. **"现在还 X 吗?"** → 通常归 **change**: 隐含"之前 X, 现在是否仍 X"的对比。
2. **"我现在 X 吗?" / "我目前 X 是什么?"** → 归 **current**: 只关心当前有效状态。
3. **"我以前 X 吗?" / "我去年 X 是什么?"** → 归 **historical**: 明确指向过去。
4. **"我从什么时候开始 X?" / "我的 X 是怎么变化的?"** → 归 **change**: 关心演变过程。
5. **"今天天气怎么样?" / "1+1=?" / "法国首都是哪?"** → 归 **general**: 无需个人记忆。
6. 当查询同时涉及"过去和现在"或"变化"时, 优先 **change**。
7. 当查询既无明确时间词、又涉及个人偏好/状态时, 默认归 **current**
   (因为用户最常关心的是当前状态)。

## 输出格式 (兼容小模型)
你可以选择输出 JSON，也可以用 Markdown 或直接用自然语言回答。
但为了能被系统稳健解析，请在你的回答最后，用方括号包裹输出你的最终分类标签，格式如下：
[CURRENT] 或 [HISTORICAL] 或 [CHANGE] 或 [GENERAL]。

示例 1 (JSON 格式):
{"query_type": "current", "reason": "问当前饮食偏好"}

示例 2 (口语化格式):
因为用户问的是‘二月份’，明显指向过去的时间，所以属于历史查询。[HISTORICAL]
"""


# ════════════════════════════════════════════════════════════════
# Few-shot 示例 (针对 Memora 长周期状态演进场景调优)
# ════════════════════════════════════════════════════════════════
# 每条示例模拟 Memora 中典型的"数周到数月跨度"的用户状态变化。
# reference_time 模拟用户"当下提问"的时刻。

FEW_SHOT_EXAMPLES: list[dict[str, str]] = [
    # ── CURRENT 类 ──────────────────────────────────────────────
    {
        "text": "我现在适合吃什么?",
        "reference_time": "2026-09-15 12:00",
        "query_type": "current",
        "reason": "问当前适合的饮食, 关心现在有效状态。",
    },
    {
        "text": "我目前的运动习惯是什么?",
        "reference_time": "2026-10-01 09:00",
        "query_type": "current",
        "reason": "明确问'目前'的运动习惯, 关心当前状态。",
    },
    {
        "text": "我最近在用什么编程语言?",
        "reference_time": "2026-11-20 14:30",
        "query_type": "current",
        "reason": "'最近'指向当前有效状态。",
    },
    {
        "text": "推荐一些适合我现在的电影",
        "reference_time": "2026-12-05 20:00",
        "query_type": "current",
        "reason": "基于当前偏好推荐, 关心现在。",
    },
    # ── HISTORICAL 类 ───────────────────────────────────────────
    {
        "text": "我去年这个时候在做什么工作?",
        "reference_time": "2026-09-15 12:00",
        "query_type": "historical",
        "reason": "明确指向'去年这个时候', 是过去特定时刻。",
    },
    {
        "text": "我大学时候喜欢吃什么?",
        "reference_time": "2026-10-01 09:00",
        "query_type": "historical",
        "reason": "'大学时候'是明确的过去时期。",
    },
    {
        "text": "三个月前我的体重是多少?",
        "reference_time": "2026-11-20 14:30",
        "query_type": "historical",
        "reason": "'三个月前'指向过去特定时刻。",
    },
    {
        "text": "我之前用的手机型号是什么?",
        "reference_time": "2026-12-05 20:00",
        "query_type": "historical",
        "reason": "'之前用的'指向过去的设备状态。",
    },
    # ── CHANGE 类 (Memora 核心场景) ─────────────────────────────
    {
        "text": "我的饮食习惯是怎么变化的?",
        "reference_time": "2026-09-15 12:00",
        "query_type": "change",
        "reason": "明确问'怎么变化的', 关心演进过程。",
    },
    {
        "text": "我现在还喜欢吃辣吗?",
        "reference_time": "2026-10-01 09:00",
        "query_type": "change",
        "reason": "'现在还…吗'隐含之前吃辣 vs 现在是否仍吃辣的对比。",
    },
    {
        "text": "我从什么时候开始改用 Rust 的?",
        "reference_time": "2026-11-20 14:30",
        "query_type": "change",
        "reason": "问状态切换的发生时间, 关心变化节点。",
    },
    {
        "text": "对比一下我上半年和下半年的运动频率",
        "reference_time": "2026-12-05 20:00",
        "query_type": "change",
        "reason": "明确要求对比两个时期, 是变化类。",
    },
    {
        "text": "我的睡眠时间相比上个月有变化吗?",
        "reference_time": "2026-12-10 08:00",
        "query_type": "change",
        "reason": "'相比…有变化吗'是典型的状态对比。",
    },
    {
        "text": "我换工作之后作息有什么改变?",
        "reference_time": "2026-12-15 21:00",
        "query_type": "change",
        "reason": "'换工作之后…改变'关心状态切换前后的差异。",
    },
    # ── GENERAL 类 ──────────────────────────────────────────────
    {
        "text": "今天北京天气怎么样?",
        "reference_time": "2026-09-15 12:00",
        "query_type": "general",
        "reason": "天气查询, 不依赖个人记忆。",
    },
    {
        "text": "光合作用的化学方程式是什么?",
        "reference_time": "2026-10-01 09:00",
        "query_type": "general",
        "reason": "科学常识, 与个人历史无关。",
    },
    {
        "text": "帮我算一下 123 乘以 456",
        "reference_time": "2026-11-20 14:30",
        "query_type": "general",
        "reason": "数学计算, 不需要个人记忆。",
    },
    {
        "text": "法国的首都是哪里?",
        "reference_time": "2026-12-05 20:00",
        "query_type": "general",
        "reason": "世界知识, 与个人历史无关。",
    },
]


# ════════════════════════════════════════════════════════════════
# 用户消息模板
# ════════════════════════════════════════════════════════════════

USER_TEMPLATE = """\
请对以下查询进行时态分类。

查询文本: {text}
参考时间: {reference_time}
对话上下文: {context}

请输出 JSON: {{"query_type": "...", "reason": "..."}}
"""


def build_messages(
    query_text: str,
    reference_time_str: str,
    context: str = "",
    enable_few_shot: bool = True,
    few_shot_count: int = 8,
) -> list[dict[str, str]]:
    """
    构造发送给 LLM 的 messages 列表。

    Args:
        query_text        : 用户查询文本
        reference_time_str: 格式化后的参考时间字符串 (如 "2026-09-15 12:00")
        context           : 可选对话上下文
        enable_few_shot   : 是否注入 few-shot 示例
        few_shot_count    : 注入的示例数量

    Returns:
        OpenAI chat completions 格式的 messages 列表。
    """
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if enable_few_shot:
        examples = FEW_SHOT_EXAMPLES[:few_shot_count]
        for ex in examples:
            messages.append(
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        text=ex["text"],
                        reference_time=ex["reference_time"],
                        context="",
                    ),
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": '{{"query_type": "{qt}", "reason": "{rs}"}}'.format(
                        qt=ex["query_type"], rs=ex["reason"]
                    ),
                }
            )

    # 真实查询
    messages.append(
        {
            "role": "user",
            "content": USER_TEMPLATE.format(
                text=query_text,
                reference_time=reference_time_str,
                context=context or "(无)",
            ),
        }
    )
    return messages


__all__ = [
    "SYSTEM_PROMPT",
    "USER_TEMPLATE",
    "FEW_SHOT_EXAMPLES",
    "build_messages",
]
