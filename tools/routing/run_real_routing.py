#!/usr/bin/env python3
"""
run_real_routing.py — 真实 API 调用演示脚本

本脚本演示如何用你自己的 API Key 与模型, 真实运行 LLMQueryRouter
对一组模拟的 Memora 长周期查询进行分类。

═══════════════════════════════════════════════════════════════
使用方法 (3 种配置方式, 任选其一):
═══════════════════════════════════════════════════════════════

【方式 1: 环境变量 (推荐)】
    export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
    export OPENAI_BASE_URL="https://api.deepseek.com/v1"   # 可选
    export ROUTING_MODEL="deepseek-chat"                    # 可选
    python tools/routing/run_real_routing.py

【方式 2: 命令行参数】
    python tools/routing/run_real_routing.py \\
        --api-key "sk-xxx" \\
        --base-url "https://api.deepseek.com/v1" \\
        --model "deepseek-chat"

【方式 3: 直接修改脚本底部的 DEFAULT_* 常量】
    编辑本文件, 把 DEFAULT_API_KEY / DEFAULT_BASE_URL / DEFAULT_MODEL
    改成你的配置, 然后:
    python tools/routing/run_real_routing.py

═══════════════════════════════════════════════════════════════
支持的模型示例 (OpenAI 兼容 API):
═══════════════════════════════════════════════════════════════
    DeepSeek : base_url=https://api.deepseek.com/v1   model=deepseek-chat
    Qwen     : base_url=https://dashscope.aliyuncs.com/compatible-mode/v1  model=qwen-plus
    GLM      : base_url=https://open.bigmodel.cn/api/paas/v4  model=glm-4-plus
    OpenAI   : base_url=(留空)                          model=gpt-4o-mini

═══════════════════════════════════════════════════════════════
预期输出:
═══════════════════════════════════════════════════════════════
    脚本会打印每个查询的分类结果 (QueryType) 与 LLM 给出的理由,
    最后打印准确率统计。

    示例:
        [1/10] Q: 我现在适合吃什么?
              → CURRENT    | 理由: 问当前饮食偏好
        [2/10] Q: 我去年这个时候在做什么工作?
              → HISTORICAL | 理由: 明确指向去年
        ...
        ═══ 准确率: 10/10 (100.0%) ═══
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── 把项目 src 加入 path (使脚本可独立运行) ────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR, _SRC_DIR / "statebudgetmem"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ── 导入 routing 模块 ──────────────────────────────────────────
try:
    from statebudgetmem.routing import (  # type: ignore[import-not-found]
        LLMQueryRouter,
        QueryRecord,
        QueryType,
    )
except ImportError:
    from routing import (  # type: ignore[no-redef]
        LLMQueryRouter,
        QueryRecord,
        QueryType,
    )

# ════════════════════════════════════════════════════════════════
# 默认配置 (可被环境变量 / 命令行参数覆盖)
# ════════════════════════════════════════════════════════════════
DEFAULT_API_KEY = ""  # ← 也可直接填在这里
DEFAULT_BASE_URL = ""  # ← 留空则用 OpenAI 官方
DEFAULT_MODEL = "gpt-4o-mini"


# ════════════════════════════════════════════════════════════════
# 测试数据: 模拟 Memora 长周期场景的查询
# ════════════════════════════════════════════════════════════════
# 每条: (query_text, reference_time_str, expected_type, 场景说明)
MEMORA_TEST_CASES: list[tuple[str, str, QueryType, str]] = [
    # ── CURRENT: 查询当前有效状态 ──────────────────────────────
    ("我现在适合吃什么?", "2026-09-15 12:00", QueryType.CURRENT, "问当前饮食偏好"),
    ("我目前的工作是什么?", "2026-09-15 12:00", QueryType.CURRENT, "问当前职业"),
    ("我最近在学什么编程语言?", "2026-09-15 12:00", QueryType.CURRENT, "问当前学习状态"),

    # ── HISTORICAL: 查询过去某特定时刻 ────────────────────────
    ("我去年这个时候在做什么工作?", "2026-09-15 12:00", QueryType.HISTORICAL, "明确指向去年"),
    ("我大学时候喜欢吃什么?", "2026-09-15 12:00", QueryType.HISTORICAL, "指向大学时期"),
    ("三个月前我的体重是多少?", "2026-09-15 12:00", QueryType.HISTORICAL, "指向三个月前"),

    # ── CHANGE: 查询状态演进/对比/切换 ─────────────────────────
    ("我的饮食习惯是怎么变化的?", "2026-09-15 12:00", QueryType.CHANGE, "问变化过程"),
    ("我现在还喜欢吃辣吗?", "2026-09-15 12:00", QueryType.CHANGE, "隐含对比(之前辣现在?)"),
    ("我从什么时候开始改用 Rust 的?", "2026-09-15 12:00", QueryType.CHANGE, "问切换时间点"),
    ("对比一下我上半年和下半年的运动频率", "2026-09-15 12:00", QueryType.CHANGE, "时期对比"),

    # ── GENERAL: 通用知识, 与个人记忆无关 ──────────────────────
    ("今天北京天气怎么样?", "2026-09-15 12:00", QueryType.GENERAL, "天气查询"),
    ("帮我算一下 123 乘以 456", "2026-09-15 12:00", QueryType.GENERAL, "数学计算"),
    ("法国的首都是哪里?", "2026-09-15 12:00", QueryType.GENERAL, "常识问答"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="StateBudgetMem routing 真实 API 演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--api-key", default=None, help="API Key (默认读 OPENAI_API_KEY)")
    parser.add_argument("--base-url", default=None, help="API Base URL (默认读 OPENAI_BASE_URL)")
    parser.add_argument("--model", default=None, help="模型名 (默认读 ROUTING_MODEL 或 gpt-4o-mini)")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印 LLM 原始返回")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # ── 解析配置: 命令行 > 环境变量 > 默认值 ──────────────────
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or DEFAULT_API_KEY
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL
    model = args.model or os.environ.get("ROUTING_MODEL") or DEFAULT_MODEL

    if not api_key:
        print("❌ 未配置 API Key!")
        print("   请通过以下任一方式配置:")
        print("   1. 环境变量: export OPENAI_API_KEY='sk-xxx'")
        print("   2. 命令行:   --api-key 'sk-xxx'")
        print("   3. 编辑脚本: 修改 DEFAULT_API_KEY 常量")
        return 1

    print(f"═══ StateBudgetMem Routing 真实 API 演示 ═══")
    print(f"模型: {model}")
    print(f"Base URL: {base_url or '(OpenAI 默认)'}")
    print(f"测试用例: {len(MEMORA_TEST_CASES)} 条")
    print()

    # ── 初始化 Router ──────────────────────────────────────────
    router = LLMQueryRouter(
        api_key=api_key,
        base_url=base_url or None,
        model=model,
        log_raw_response=args.verbose,
    )

    # ── 逐条分类 ───────────────────────────────────────────────
    correct = 0
    total = len(MEMORA_TEST_CASES)
    now = datetime.now(timezone.utc)

    for i, (text, ref_time_str, expected, scenario) in enumerate(MEMORA_TEST_CASES, 1):
        query = QueryRecord(text=text, reference_time=ref_time_str)
        try:
            result = router.classify(query)
        except Exception as e:
            print(f"[{i}/{total}] ❌ 异常: {text}")
            print(f"        错误: {e}")
            result = router.fallback_type

        is_correct = result == expected
        if is_correct:
            correct += 1

        mark = "✓" if is_correct else "✗"
        print(f"[{i}/{total}] {mark} Q: {text}")
        print(f"        预期: {expected.name:<11} 实际: {result.name:<11} | {scenario}")

        # 礼貌延时, 避免 rate limit
        time.sleep(0.3)

    # ── 统计 ───────────────────────────────────────────────────
    print()
    print(f"═══ 准确率: {correct}/{total} ({correct / total * 100:.1f}%) ═══")

    # ── 打印统计信息 ───────────────────────────────────────────
    stats = router.get_stats()
    print(f"调用次数: {stats['total_calls']}  成功: {stats['successful']}  降级: {stats['fallbacks']}")

    return 0 if correct == total else 0  # 即使有错也返回 0, 方便用户看完整输出


if __name__ == "__main__":
    sys.exit(main())
