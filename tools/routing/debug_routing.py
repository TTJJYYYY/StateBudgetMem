#!/usr/bin/env python3
"""
debug_routing.py — Prompt 调试脚本

本脚本用于在开发阶段调试 LLMQueryRouter 的 prompt, 检查:
    1. 发送给 LLM 的完整 messages (含 few-shot 示例)
    2. LLM 的原始返回内容
    3. 解析后的 query_type
    4. 与预期类型的对比

═══════════════════════════════════════════════════════════════
使用方法:
═══════════════════════════════════════════════════════════════

    # 方式 1: 环境变量
    export OPENAI_API_KEY="sk-xxx"
    export OPENAI_BASE_URL="https://api.deepseek.com/v1"
    python tools/routing/debug_routing.py --model deepseek-chat

    # 方式 2: 命令行参数
    python tools/routing/debug_routing.py \\
        --api-key "sk-xxx" \\
        --base-url "https://api.deepseek.com/v1" \\
        --model deepseek-chat \\
        --query "我现在还喜欢吃辣吗?"

    # 方式 3: 交互模式 (逐条输入查询)
    python tools/routing/debug_routing.py --interactive --api-key "sk-xxx" ...

    # 方式 4: 只查看 prompt 不调用 API (离线)
    python tools/routing/debug_routing.py --dry-run --query "我现在还喜欢吃辣吗?"

═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── 路径设置 ────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
for _p in (_PROJECT_ROOT, _SRC_DIR, _SRC_DIR / "statebudgetmem"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from statebudgetmem.routing import (  # type: ignore[import-not-found]
        LLMQueryRouter,
        QueryRecord,
        QueryType,
        build_messages,
        parse_query_type_from_response,
    )
except ImportError:
    from routing import (  # type: ignore[no-redef]
        LLMQueryRouter,
        QueryRecord,
        QueryType,
        build_messages,
        parse_query_type_from_response,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Routing Prompt 调试工具")
    p.add_argument("--api-key", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--query", "-q", default=None, help="单条查询 (不指定则用默认用例集)")
    p.add_argument("--reference-time", "-t", default=None, help="参考时间 (如 2026-09-15 12:00)")
    p.add_argument("--interactive", "-i", action="store_true", help="交互模式")
    p.add_argument("--dry-run", action="store_true", help="只打印 prompt, 不调用 API")
    p.add_argument("--no-few-shot", action="store_true", help="禁用 few-shot 示例")
    p.add_argument("--few-shot-count", type=int, default=8, help="few-shot 示例数量")
    return p.parse_args()


# 默认调试用例
DEBUG_CASES = [
    "我现在适合吃什么?",
    "我去年这个时候在做什么工作?",
    "我现在还喜欢吃辣吗?",
    "我的饮食习惯是怎么变化的?",
    "今天北京天气怎么样?",
    "我从什么时候开始改用 Rust 的?",
    "我大学时候的室友叫什么?",
    "帮我算一下 123 乘以 456",
]


def show_messages(messages: list[dict]) -> None:
    """打印完整的 messages 列表。"""
    print("─" * 60)
    print("📋 发送给 LLM 的 messages:")
    print("─" * 60)
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]
        # 截断过长的内容
        if len(content) > 500:
            content = content[:500] + "...(截断)"
        print(f"\n[{i}] {role}:")
        print(f"    {content}")
    print("─" * 60)


def debug_single(
    router: LLMQueryRouter,
    text: str,
    ref_time: str | None,
    dry_run: bool,
) -> None:
    """调试单条查询。"""
    query = QueryRecord(text=text, reference_time=ref_time)

    # 1. 展示 messages
    messages = build_messages(
        query_text=query.text,
        reference_time_str=query.reference_time.strftime("%Y-%m-%d %H:%M"),
        context=query.context,
        enable_few_shot=router.enable_few_shot,
        few_shot_count=router.few_shot_count,
    )
    show_messages(messages)

    if dry_run:
        print("\n(dry-run 模式, 跳过 API 调用)")
        return

    # 2. 调用 API 并展示原始返回
    print("\n🤖 调用 LLM...")
    try:
        result = router.classify(query)
        print(f"   分类结果: {result.name}")
    except Exception as e:
        print(f"   ❌ 异常: {e}")

    # 3. 展示统计
    stats = router.get_stats()
    print(f"\n📊 统计: 总调用 {stats['total_calls']}, 成功 {stats['successful']}, 降级 {stats['fallbacks']}")


def main() -> int:
    args = parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "")

    # 初始化 router (dry-run 模式下不需要 api_key)
    router = LLMQueryRouter(
        api_key=api_key if not args.dry_run else "",
        base_url=base_url or None,
        model=args.model,
        enable_few_shot=not args.no_few_shot,
        few_shot_count=args.few_shot_count,
        log_raw_response=True,
    )

    if args.interactive:
        print("═══ 交互调试模式 (输入 quit 退出) ═══")
        while True:
            try:
                text = input("\n🔍 输入查询: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见!")
                break
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            debug_single(router, text, args.reference_time, args.dry_run)
        return 0

    if args.query:
        debug_single(router, args.query, args.reference_time, args.dry_run)
        return 0

    # 默认: 跑全部调试用例
    print("═══ 批量调试 (默认用例集) ═══\n")
    for text in DEBUG_CASES:
        print(f"\n{'='*60}")
        print(f"查询: {text}")
        print(f"{'='*60}")
        debug_single(router, text, args.reference_time, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())
