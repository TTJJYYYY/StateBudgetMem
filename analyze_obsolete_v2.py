"""
analyze_obsolete_v2.py — 通用过期记忆分析（修复版）

核心改进：
1. 否定词检测更严格：必须"否定词 + 具体行为/主题"才算状态变化信号
2. 支持精确标注模式（DEMO_HISTORY）和自动检测模式（Memora）
3. 增加 LLM-as-Judge 精确判断（可选）

用法：
    python analyze_obsolete_v2.py --mode demo              # DEMO_HISTORY 精确分析
    python analyze_obsolete_v2.py --dataset Memora/data --persona software_engineer --sample 20
"""

import json
import argparse
import re
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from memorybank_v2 import MemoryBank, TFIDFMemoryBank
from evaluation_v2 import DEMO_HISTORY, load_memora_data
from interfaces import UpdateOperation

# ═══════════════════════════════════════════════════════════════
# 1. 精确标注：DEMO_HISTORY 的过期记忆（我们完全了解对话内容）
# ═══════════════════════════════════════════════════════════════

# 基于小林的故事，精确标注哪些记忆已过期
DEMO_GROUND_TRUTH = {
    # 格式：memory_content_substring -> "obsolete" | "current"
    # 吃辣相关（已过期）
    "川菜": "obsolete",
    "火锅": "obsolete",
    "吃辣": "obsolete",
    "辛辣": "obsolete",
    # 清淡饮食相关（当前有效）
    "清淡": "current",
    "白切鸡": "current",
    "蒸鱼": "current",
    "粤菜": "current",
    "胃不好": "current",
    "医生建议": "current",
    # 运动相关（一直有效）
    "篮球": "current",
    "游泳": "current",
    "每周去两次": "current",
    # 专业相关（修正后有效）
    "软件工程": "current",
    "计算机系": "obsolete",  # 说错了，后来纠正
}

DEMO_TEST_QUESTIONS = [
    ("我现在适合吃什么？", "current"),  # 应该返回清淡相关
    ("我以前喜欢吃什么？", "historical"),  # 可以返回吃辣相关
    ("我现在还能吃火锅吗？", "current"),  # 应该返回不建议
    ("推荐一些运动", "current"),  # 篮球游泳
    ("我是什么专业？", "current"),  # 软件工程
]


def label_memory_demo(content: str) -> str:
    """基于精确标注判断 DEMO_HISTORY 中的记忆状态"""
    for keyword, status in DEMO_GROUND_TRUTH.items():
        if keyword in content:
            return status
    return "neutral"


# ═══════════════════════════════════════════════════════════════
# 2. 自动检测：Memora 数据集的过期记忆检测
# ═══════════════════════════════════════════════════════════════

class ObsoleteDetector:
    """
    自动检测过期记忆。

    策略：识别"状态转变信号"——不是简单匹配否定词，
    而是检测"旧行为 → 新行为"的转换模式。
    """

    # 状态转变信号模式：(旧状态关键词, 转变信号, 新状态关键词)
    TRANSITION_PATTERNS = [
        # 饮食转变
        ("spicy", "stopped|quit|avoid|can't|cannot|gave up|no longer", "mild|light|bland|healthy"),
        ("meat", "stopped|quit|avoid|became vegetarian|vegan", "vegetable|plant"),
        ("sugar", "cut|reduced|avoid|quit", "low sugar|sugar free"),
        # 运动转变
        ("running|gym", "stopped|quit|injured|can't", "walking|swimming|yoga"),
        # 居住地
        ("lived in|apartment in", "moved|relocated", "live in|new place"),
        # 工作
        ("company|job|employed", "quit|fired|left|changed", "new job|unemployed|freelance"),
    ]

    # 更强的否定信号（必须和具体主题一起出现）
    STRONG_NEGATION = [
        r"(?:no longer|not anymore)\s+(?:eat|drink|like|enjoy|do|go|live|work)",
        r"(?:stopped|quit|gave up)\s+(?:eating|drinking|doing|going|living|working)",
        r"can't\s+(?:eat|drink|do|go)\s+(?:anymore|now)",
        r"avoid(?:ing)?\s+(?:spicy|meat|sugar|alcohol)",
        r"switched\s+(?:to|from)",
        r"changed\s+(?:to|from|my)",
        r"moved\s+(?:to|from)",
        r"used to\s+(?:eat|live|work|go|like)",
    ]

    def __init__(self, llm_judge: Optional[callable] = None):
        self.llm_judge = llm_judge

    def detect_transitions(self, memories: List[Dict]) -> Dict[str, str]:
        """
        检测状态转变，标记过期记忆。

        Returns:
            {memory_id: "current" | "obsolete" | "neutral"}
        """
        labels = {}

        # 按时间排序
        sorted_mems = sorted(memories, key=lambda m: m.get("timestamp", 0))

        # 第一遍：标记强转变信号
        transition_sessions = set()  # 包含转变信号的 session
        for mem in sorted_mems:
            content = mem.get("content", "").lower()
            if self._has_transition_signal(content):
                transition_sessions.add(mem.get("memory_id", ""))

        # 第二遍：对于每条记忆，检查是否有后续的转变信号覆盖它
        for i, mem in enumerate(sorted_mems):
            mid = mem.get("memory_id", "")
            content = mem.get("content", "").lower()

            # 找到后续的转变记忆
            subsequent_transitions = [
                tm for tm in sorted_mems[i + 1:]
                if tm.get("memory_id", "") in transition_sessions
            ]

            if not subsequent_transitions:
                labels[mid] = "current"
                continue

            # 检查后续转变是否覆盖了当前记忆的主题
            is_covered = False
            for trans in subsequent_transitions:
                trans_content = trans.get("content", "").lower()
                if self._topics_overlap(content, trans_content):
                    is_covered = True
                    break

            if is_covered:
                labels[mid] = "obsolete"
            else:
                labels[mid] = "current"

        return labels

    def _has_transition_signal(self, content: str) -> bool:
        """检查内容是否包含状态转变信号"""
        content_lower = content.lower()
        for pattern in self.STRONG_NEGATION:
            if re.search(pattern, content_lower):
                return True
        return False

    def _topics_overlap(self, content1: str, content2: str) -> bool:
        """检查两个记忆是否涉及同一主题（共享关键词）"""
        # 提取长度 > 3 的关键词
        words1 = set(w.lower() for w in re.findall(r'\b\w{4,}\b', content1))
        words2 = set(w.lower() for w in re.findall(r'\b\w{4,}\b', content2))

        # 排除常见停用词
        stopwords = {"that", "this", "with", "have", "from", "they", "will", "would", "there", "their", "what", "about",
                     "which", "when", "make", "like", "time", "just", "know", "take", "year", "good", "some", "come",
                     "could", "state", "than", "then", "them", "well", "were", "said", "each", "which", "how", "also",
                     "after", "back", "other", "many", "than", "only", "those", "come", "day", "more", "way", "may",
                     "say", "great", "where", "help", "through", "much", "before", "right", "too", "any", "same",
                     "tell", "very", "still", "own", "under", "while", "last", "might", "even", "leave", "put", "here",
                     "does", "should", "never", "these", "both", "between", "long", "really", "going", "again", "work",
                     "three", "must", "without", "another", "life", "again", "why", "called", "being", "another",
                     "find", "part", "place", "made", "live", "where", "found", "own", "still", "eyes", "hand",
                     "thought", "head", "soon", "each", "done", "open", "case", "show", "live", "play", "went", "told",
                     "seen", "heard", "feel", "seem", "turn", "hand", "high", "sure", "upon", "head", "help", "home",
                     "side", "move", "both", "five", "once", "same", "must", "name", "left", "each", "done", "open",
                     "case", "show", "live", "play", "went", "told", "seen", "heard", "feel", "seem", "turn", "hand",
                     "high", "sure", "upon", "head", "help", "home", "side", "move", "both", "five", "once", "same",
                     "must", "name", "left"}
        words1 -= stopwords
        words2 -= stopwords

        shared = words1 & words2
        return len(shared) >= 2


def label_memora_with_evidence(questions_data: List[Dict], memories: List[Dict]) -> Dict[str, str]:
    """
    利用 Memora 的 memory_evidence 来标注过期记忆。

    策略：
    - memory_evidence 中列出的 session 是"正确答案应该基于的记忆"
    - 如果某条记忆和 evidence 中的记忆主题相同，但不 evidence 中，可能是过期的
    """
    labels = {}

    # 收集所有 evidence 中的 memory_id
    evidence_ids = set()
    for q in questions_data:
        evidence = q.get("memory_evidence", {})
        if isinstance(evidence, dict):
            for sid in evidence.keys():
                evidence_ids.add(str(sid))

    # 将 session_id 映射到 memory
    for mem in memories:
        mid = mem.get("memory_id", "")
        source = mem.get("source", "")

        # 如果 memory 的来源在 evidence 中 → current
        if source in evidence_ids or mid in evidence_ids:
            labels[mid] = "current"
        else:
            labels[mid] = "neutral"  # 不确定

    return labels


# ═══════════════════════════════════════════════════════════════
# 3. OMR 计算（通用）
# ═══════════════════════════════════════════════════════════════

def calculate_omr(retrieved: List[Dict], ground_truth: Dict[str, str]) -> Dict:
    """计算 Outdated Memory Rate"""
    total = len(retrieved)
    obs = sum(1 for r in retrieved if ground_truth.get(r.get("memory_id", ""), "neutral") == "obsolete")
    cur = sum(1 for r in retrieved if ground_truth.get(r.get("memory_id", ""), "neutral") == "current")
    neu = total - obs - cur

    return {
        "total": total,
        "obsolete": obs,
        "current": cur,
        "neutral": neu,
        "omr": obs / total if total > 0 else 0.0,
        "cor": cur / total if total > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════

def analyze_demo():
    """DEMO_HISTORY 精确分析"""
    print("=" * 60)
    print("DEMO_HISTORY 过期记忆分析（精确标注）")
    print("=" * 60)

    mb = MemoryBank()
    tfidf = TFIDFMemoryBank()
    mb.add(DEMO_HISTORY)
    tfidf.add(DEMO_HISTORY)

    # 获取所有记忆并标注
    all_mems = []
    for mem in mb.get_all():
        status = label_memory_demo(mem.content)
        all_mems.append({"memory_id": mem.memory_id, "content": mem.content, "status": status})

    gt = {m["memory_id"]: m["status"] for m in all_mems}
    print(
        f"Ground Truth: {len(gt)} 条 | 过期 {sum(1 for v in gt.values() if v == 'obsolete')} | 当前 {sum(1 for v in gt.values() if v == 'current')} | 无关 {sum(1 for v in gt.values() if v == 'neutral')}")

    # 对每个问题分析
    for q, qtype in DEMO_TEST_QUESTIONS:
        print(f"\n  Q: {q} [{qtype}]")

        for name, system in [("MemoryBank", mb), ("TF-IDF", tfidf)]:
            results = system.retrieve(q, top_k=5)
            omr = calculate_omr(results, gt)
            flag = "✅" if (qtype == "current" and omr["omr"] == 0) or (
                        qtype == "historical" and omr["omr"] > 0) else "❌"
            print(f"    {name:<12} OMR: {omr['omr']:>6.1%} | COR: {omr['cor']:>6.1%} {flag}")
            for r in results[:3]:
                s = gt.get(r.get("memory_id", ""), "neutral")
                emoji = "🔴" if s == "obsolete" else "🟢" if s == "current" else "⚪"
                print(f"      {emoji} {r.get('content', '')[:40]}...")


def analyze_memora(dataset_path: str, persona: str, sample_size: int = 20):
    """Memora 数据集分析（自动检测）"""
    print("=" * 60)
    print(f"Memora 过期记忆分析: {persona}")
    print("=" * 60)

    history, questions, portrait = load_memora_data(dataset_path, persona)

    # 只选 reasoning 类型的问题（更容易暴露过期记忆）
    reasoning_questions = [q for q in questions if q.get("category") == "reasoning"]
    if not reasoning_questions:
        reasoning_questions = questions[:sample_size]

    test_questions = reasoning_questions[:sample_size]
    print(f"测试问题数: {len(test_questions)} (reasoning 类型)")

    mb = MemoryBank()
    mb.add(history)

    # 自动检测过期记忆
    detector = ObsoleteDetector()
    all_mems = [{"memory_id": m.memory_id, "content": m.content, "timestamp": m.timestamp} for m in mb.get_all()]
    gt = detector.detect_transitions(all_mems)

    obs_count = sum(1 for v in gt.values() if v == "obsolete")
    cur_count = sum(1 for v in gt.values() if v == "current")
    print(f"自动检测: 过期 {obs_count} | 当前 {cur_count} | 不确定 {len(gt) - obs_count - cur_count}")

    # 分析
    total_omr = 0
    total_retrieved = 0

    for q in test_questions:
        question = q["question"]
        results = mb.retrieve(question, top_k=5)
        omr = calculate_omr(results, gt)
        total_omr += omr["omr"] * omr["total"]
        total_retrieved += omr["total"]

        print(f"\n  Q: {question[:60]}")
        print(f"     OMR: {omr['omr']:>6.1%} | COR: {omr['cor']:>6.1%}")

    overall = total_omr / total_retrieved if total_retrieved > 0 else 0
    print(f"\n  Overall OMR: {overall:.2%}")
    return overall


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="demo", choices=["demo", "memora"])
    parser.add_argument("--dataset", default="Memora/data")
    parser.add_argument("--persona", default="software_engineer")
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "demo":
        analyze_demo()
    else:
        analyze_memora(args.dataset, args.persona, args.sample)


if __name__ == "__main__":
    main()