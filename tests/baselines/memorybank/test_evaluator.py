from __future__ import annotations

from dataclasses import dataclass

from statebudgetmem.baselines.memorybank import (
    EvaluationResult,
    MemoryEvaluator,
    summarize_results,
)


@dataclass
class FakeMemory:
    memory_id: str
    content: str
    timestamp: float


class FakeMemoryBank:
    def __init__(self, forgetting_threshold: float = 0.3) -> None:
        self.memories: list[FakeMemory] = []
        self.portrait = ""

    def add(self, messages, **kwargs):
        ids = []
        for index, (_role, content, _timestamp) in enumerate(messages):
            memory_id = f"m{len(self.memories) + index}"
            self.memories.append(FakeMemory(memory_id, content, float(len(self.memories))))
            ids.append(memory_id)
        return ids

    def build_augmented_prompt(self, query, current_time=None, top_k=5):
        context = "\n".join(memory.content for memory in self.memories[-top_k:])
        return {
            "prompt_template": f"相关历史记忆:\n{context}\n问题:{query}",
            "retrieved_count": min(top_k, len(self.memories)),
        }

    def update_user_portrait(self, portrait):
        self.portrait = portrait

    def get_stats(self):
        return {"total_memories": len(self.memories)}


def test_memory_evaluator_runs_with_injected_backend() -> None:
    def llm(prompt: str) -> str:
        return "小林" if "相关历史记忆" in prompt else "不知道"

    evaluator = MemoryEvaluator(
        llm_caller=llm,
        memory_bank_factory=FakeMemoryBank,
    )
    output = evaluator.run_evaluation(
        history=[("用户", "我叫小林", "2026-01-01")],
        probe_questions=[
            {
                "question": "我叫什么？",
                "expected_keywords": ["小林"],
                "category": "fact",
            }
        ],
    )
    assert output["summary"]["memory_avg"] == 1.0
    assert output["summary"]["baseline_avg"] == 0.0


def test_summarize_results_by_category() -> None:
    summary = summarize_results(
        [
            EvaluationResult(
                question="q",
                baseline_answer="b",
                memory_answer="m",
                baseline_score=0.0,
                memory_score=1.0,
                expected_keywords=["x"],
                baseline_hit_keywords=[],
                memory_hit_keywords=["x"],
                category="fact",
            )
        ]
    )
    assert summary["category_stats"]["fact"]["memory_avg"] == 1.0
