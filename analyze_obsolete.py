"""
analyze_obsolete.py — 通用过期记忆分析

支持任意数据集，自动检测过期记忆，计算 OMR。

用法：
    python analyze_obsolete.py              # 用 DEMO_HISTORY 自动测试
    python analyze_obsolete.py --dataset Memora/data --persona software_engineer
"""

import json
import argparse
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from memorybank_v2 import MemoryBank, TFIDFMemoryBank
from evaluation_v2 import DEMO_HISTORY, load_memora_data
from interfaces import UpdateOperation, MemoryStatus, MemoryPiece


# ═══════════════════════════════════════════════════════════════
# 通用过期记忆检测器（不依赖硬编码关键词）
# ═══════════════════════════════════════════════════════════════

class ObsoleteDetector:
    """
    自动检测过期记忆。

    策略：基于否定词 + 语义聚类 + 时间排序。
    不依赖特定关键词，适用于任意数据集。
    """

    # 中文/英文否定/转变信号词
    NEGATION_WORDS = [
        "不再", "没有", "不是", "改了", "变为", "变成", "换成", "改为",
        "告别", "放弃", "停止", "取消", "暂时不", "不能", "不会",
        "no longer", "not", "stopped", "quit", "changed to", "switched to",
        "gave up", "avoid", "can't", "cannot",
    ]

    def __init__(self, llm_judge: Optional[callable] = None):
        """
        Args:
            llm_judge: 可选的 LLM 判断函数，用于精确检测冲突
                      签名: f(mem1_content, mem2_content) -> bool (是否冲突)
        """
        self.llm_judge = llm_judge

    def detect_by_negation(self, memories: List[dict]) -> List[Tuple[str, str]]:
        """
        基于否定词检测：如果新记忆包含否定词且与旧记忆语义相似，
        则旧记忆可能已过期。

        Returns:
            [(old_memory_id, new_memory_id), ...]  # 旧记忆被新记忆替代
        """
        supersede_pairs = []

        # 按时间排序
        sorted_mems = sorted(memories, key=lambda m: m.get("timestamp", 0))

        for i, new_mem in enumerate(sorted_mems):
            content = new_mem.get("content", "").lower()

            # 检查是否包含否定/转变信号
            has_negation = any(nw in content for nw in self.NEGATION_WORDS)
            if not has_negation:
                continue

            # 找语义相似的旧记忆
            for old_mem in sorted_mems[:i]:  # 只检查更早的记忆
                old_content = old_mem.get("content", "").lower()

                # 简单规则：共享至少 2 个关键词（长度 > 2 的词）
                old_words = set(w for w in old_content.split() if len(w) > 2)
                new_words = set(w for w in content.split() if len(w) > 2)
                shared = old_words & new_words

                if len(shared) >= 2:
                    # 可能冲突
                    if self.llm_judge:
                        # 用 LLM 精确判断
                        if self.llm_judge(old_mem.get("content", ""), new_mem.get("content", "")):
                            supersede_pairs.append((old_mem["memory_id"], new_mem["memory_id"]))
                    else:
                        # 无 LLM，直接用规则判断
                        supersede_pairs.append((old_mem["memory_id"], new_mem["memory_id"]))

        return supersede_pairs

    def detect_by_clustering(self, memories: List[dict],
                             similarity_threshold: float = 0.7) -> Dict[str, List[str]]:
        """
        基于语义聚类：把相似的记忆分到一组，每组只保留最新的。

        Returns:
            {topic_key: [memory_id_oldest, ..., memory_id_newest]}
        """
        from memorybank_v2 import MemoryBank

        # 用临时 MemoryBank 做向量聚类
        mb = MemoryBank()

        # 给每条记忆建一个临时 entry
        for mem in memories:
            mid = mem.get("memory_id", "")
            content = mem.get("content", "")
            ts = mem.get("timestamp", 0)
            if mid and content:
                mb._insert_memory(MemoryPiece(
                    content=content, timestamp=ts, memory_id=mid
                ))

        # 对每个记忆，检索最相似的
        clusters = defaultdict(list)
        visited = set()

        for mem in memories:
            mid = mem.get("memory_id", "")
            if mid in visited:
                continue

            # 检索相似记忆
            results = mb.retrieve(mem.get("content", ""), top_k=5)
            cluster = [r["memory_id"] for r in results
                       if r.get("composite_score", 0) > similarity_threshold]

            for cid in cluster:
                visited.add(cid)
                clusters[mid].append(cid)

        return dict(clusters)

    def build_ground_truth(self, memories: List[dict]) -> Dict[str, str]:
        """
        综合检测，构建每条记忆的"真实状态"标签。

        Returns:
            {memory_id: "current" | "obsolete" | "unknown"}
        """
        labels = {}

        # 1. 先基于否定词检测
        pairs = self.detect_by_negation(memories)
        superseded_ids = set(old_id for old_id, _ in pairs)

        for mem in memories:
            mid = mem.get("memory_id", "")
            if mid in superseded_ids:
                labels[mid] = "obsolete"
            else:
                labels[mid] = "current"

        return labels


# ═══════════════════════════════════════════════════════════════
# OMR 计算器（通用）
# ═══════════════════════════════════════════════════════════════

def calculate_omr(retrieved_memories: List[dict],
                  ground_truth: Dict[str, str]) -> Dict:
    """
    计算 Outdated Memory Rate。

    Args:
        retrieved_memories: 检索结果，每条包含 memory_id
        ground_truth: {memory_id: "current" | "obsolete" | "unknown"}

    Returns:
        {
            "total_retrieved": int,
            "obsolete_count": int,
            "current_count": int,
            "unknown_count": int,
            "omr": float,  # obsolete / total
            "cor": float,  # current / total (Current Object Rate)
        }
    """
    total = len(retrieved_memories)
    obsolete = 0
    current = 0
    unknown = 0

    for r in retrieved_memories:
        mid = r.get("memory_id", "")
        status = ground_truth.get(mid, "unknown")

        if status == "obsolete":
            obsolete += 1
        elif status == "current":
            current += 1
        else:
            unknown += 1

    return {
        "total_retrieved": total,
        "obsolete_count": obsolete,
        "current_count": current,
        "unknown_count": unknown,
        "omr": obsolete / total if total > 0 else 0.0,
        "cor": current / total if total > 0 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def analyze_system_generic(name: str, system,
                           history: List[tuple],
                           questions: List[str],
                           detector: ObsoleteDetector) -> Dict:
    """
    通用分析流程：
    1. 自动构建 ground truth（哪些记忆已过期）
    2. 对每个问题检索
    3. 计算 OMR
    """
    print(f"\n{'=' * 60}")
    print(f"分析: {name}")
    print(f"{'=' * 60}")

    # 灌入数据
    system.add(history)

    # 获取所有记忆，构建 ground truth
    all_mems = []
    for mem in system.get_all():
        all_mems.append({
            "memory_id": mem.memory_id,
            "content": mem.content,
            "timestamp": mem.timestamp,
        })

    # 自动检测过期记忆
    gt = detector.build_ground_truth(all_mems)

    obsolete_count = sum(1 for v in gt.values() if v == "obsolete")
    current_count = sum(1 for v in gt.values() if v == "current")
    print(f"Ground Truth: {len(gt)} 条记忆 | 过期 {obsolete_count} | 当前 {current_count}")

    # 逐个问题分析
    results = []
    total_omr = 0
    total_retrieved = 0

    for q in questions:
        retrieved = system.retrieve(q, top_k=5)
        omr_result = calculate_omr(retrieved, gt)
        results.append({
            "question": q,
            **omr_result,
            "retrieved_memories": [r.get("content", "")[:50] for r in retrieved],
        })
        total_omr += omr_result["omr"] * omr_result["total_retrieved"]
        total_retrieved += omr_result["total_retrieved"]

        print(f"\n  Q: {q}")
        print(f"     检索 {omr_result['total_retrieved']} 条 | "
              f"OMR: {omr_result['omr']:.2%} | "
              f"COR: {omr_result['cor']:.2%}")

    overall_omr = total_omr / total_retrieved if total_retrieved > 0 else 0

    print(f"\n  Overall OMR: {overall_omr:.2%}")

    return {
        "system": name,
        "questions": results,
        "overall_omr": overall_omr,
        "ground_truth_stats": {
            "total": len(gt),
            "obsolete": obsolete_count,
            "current": current_count,
        }
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None, help="数据集路径")
    parser.add_argument("--persona", default="software_engineer")
    parser.add_argument("--questions", default=None, help="问题列表 JSON 文件")
    args = parser.parse_args()

    # 加载数据
    if args.dataset and args.dataset != "demo":
        history, probe_questions, portrait = load_memora_data(
            args.dataset, persona=args.persona
        )
        questions = [p["question"] for p in probe_questions]
    else:
        history = DEMO_HISTORY
        questions = [
            "我现在适合吃什么类型的食物？",
            "我现在还能经常吃火锅吗？",
            "推荐一家餐厅",
        ]

    # 创建检测器
    detector = ObsoleteDetector()

    # 分析 MemoryBank
    mb = MemoryBank()
    report_mb = analyze_system_generic("MemoryBank", mb, history, questions, detector)

    # 分析 TF-IDF
    tfidf = TFIDFMemoryBank()
    report_tfidf = analyze_system_generic("TF-IDF", tfidf, history, questions, detector)

    # 汇总
    print(f"\n{'=' * 60}")
    print("汇总")
    print(f"{'=' * 60}")
    print(f"{'System':<20} {'Overall OMR':>12}")
    print(f"{'-' * 40}")
    print(f"{'MemoryBank':<20} {report_mb['overall_omr']:>11.2%}")
    print(f"{'TF-IDF':<20} {report_tfidf['overall_omr']:>11.2%}")

    # 保存
    with open("obsolete_analysis.json", "w") as f:
        json.dump({"memorybank": report_mb, "tfidf": report_tfidf}, f, indent=2)
    print("\n已保存: obsolete_analysis.json")


if __name__ == "__main__":
    main()