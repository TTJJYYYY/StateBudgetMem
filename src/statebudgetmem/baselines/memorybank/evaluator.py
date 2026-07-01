"""MemoryBank-versus-stateless evaluation utilities.

This module consolidates the useful parts of the former root-level
the former root-level evaluation scripts into an importable, testable
package module.  Heavy MemoryBank dependencies are imported lazily.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from statebudgetmem.baselines.memorybank.agents import BaselineAgent, MemoryAugmentedAgent
from statebudgetmem.baselines.memorybank.datasets import DEMO_HISTORY, DEMO_QUESTIONS, History, Probe

LLMCaller = Callable[[str], str]
MemoryBankFactory = Callable[..., Any]


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible text generation wrapper."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        max_tokens: int = 300,
        temperature: float = 0.7,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must not be empty")
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "LLM evaluation requires the optional 'llm' dependencies: "
                "pip install -e '.[llm]'"
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.OpenAI(**kwargs)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return str(response.choices[0].message.content or "")


class DeepSeekLLM(OpenAICompatibleLLM):
    """Convenience wrapper retaining the original experiment API."""

    def __init__(self, api_key: str, model: str = "deepseek-chat") -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.deepseek.com",
        )


class MockLLM:
    """Deterministic offline generator for smoke tests and demos."""

    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        memory_prompt = "相关历史记忆" in prompt or "历史记忆" in prompt
        if not memory_prompt:
            return "好的。请再告诉我一些相关信息。"

        rules = [
            (("名字", "小林"), "你叫小林。"),
            (("专业", "软件工程"), "你后来纠正说自己是软件工程专业。"),
            (("运动", "篮球"), "你喜欢篮球和游泳。"),
            (("放松", "音乐"), "你会听音乐、看电影，也会运动放松。"),
            (("书", "深度学习入门"), "我之前推荐过《深度学习入门》。"),
            (("游泳", "每周"), "你大约每周游泳两次。"),
            (("火锅", "胃"), "你胃不太好，医生建议清淡饮食，不建议经常吃火锅。"),
            (("吃什么", "清淡"), "你现在更适合清淡饮食，也尝试过粤菜。"),
            (("为什么", "饮食"), "因为胃部不适和医生建议，你改成了清淡饮食。"),
            (("暑假", "AI"), "你暑假计划留校参加 AI 竞赛，做图像识别项目。"),
        ]
        for required, answer in rules:
            if all(token in prompt for token in required):
                return answer
        return "我检索到了一些相关历史信息，但还需要进一步确认。"


@dataclass(slots=True)
class EvaluationResult:
    question: str
    baseline_answer: str
    memory_answer: str
    baseline_score: float
    memory_score: float
    expected_keywords: list[str]
    baseline_hit_keywords: list[str]
    memory_hit_keywords: list[str]
    category: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MemoryEvaluator:
    """Run paired stateless/MemoryBank answer-generation experiments."""

    def __init__(
        self,
        llm_caller: LLMCaller,
        *,
        judge_caller: LLMCaller | None = None,
        forgetting_threshold: float = 0.3,
        memory_bank_factory: MemoryBankFactory | None = None,
    ) -> None:
        self.llm = llm_caller
        self.judge = judge_caller
        self.forgetting_threshold = forgetting_threshold
        self.memory_bank_factory = memory_bank_factory

    def run_evaluation(
        self,
        history: History | None = None,
        probe_questions: list[Probe] | None = None,
        user_portrait: str = "",
        *,
        timestamp: str = "2026-06-27 10:00",
    ) -> dict[str, Any]:
        history = history or DEMO_HISTORY
        probe_questions = probe_questions or DEMO_QUESTIONS

        memory_bank = self._build_memory_bank()
        baseline_agent = BaselineAgent(llm_caller=self.llm)
        memory_agent = MemoryAugmentedAgent(
            memory_bank=memory_bank,
            llm_caller=self.llm,
        )
        memory_agent.batch_store_history(history)

        portrait = user_portrait or (
            "用户小林，软件工程专业大三学生，爱好篮球和游泳，"
            "因健康原因已改为清淡饮食。"
        )
        update_portrait = getattr(memory_bank, "update_user_portrait", None)
        if callable(update_portrait):
            update_portrait(portrait)

        results: list[EvaluationResult] = []
        for probe in probe_questions:
            question = str(probe.get("question", "")).strip()
            if not question:
                continue
            expected = [str(item) for item in probe.get("expected_keywords", [])]
            baseline_answer = baseline_agent.chat(question, timestamp)
            memory_answer = memory_agent.chat(question, timestamp)
            baseline_score, baseline_hits = self._score_answer(
                question, baseline_answer, expected
            )
            memory_score, memory_hits = self._score_answer(
                question, memory_answer, expected
            )
            results.append(
                EvaluationResult(
                    question=question,
                    baseline_answer=baseline_answer,
                    memory_answer=memory_answer,
                    baseline_score=baseline_score,
                    memory_score=memory_score,
                    expected_keywords=expected,
                    baseline_hit_keywords=baseline_hits,
                    memory_hit_keywords=memory_hits,
                    category=str(probe.get("category", "")),
                    note=str(probe.get("note", "")),
                )
            )

        summary = summarize_results(results)
        return {
            "results": results,
            "summary": summary,
            "memory_stats": memory_agent.get_memory_stats(),
        }

    def export_results(self, results_data: dict[str, Any], filepath: str | Path) -> Path:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": results_data["summary"],
            "memory_stats": results_data.get("memory_stats", {}),
            "details": [
                result.to_dict() if isinstance(result, EvaluationResult) else result
                for result in results_data["results"]
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _build_memory_bank(self) -> Any:
        if self.memory_bank_factory is not None:
            return self.memory_bank_factory(
                forgetting_threshold=self.forgetting_threshold
            )
        from statebudgetmem.baselines.memorybank.system import MemoryBank

        return MemoryBank(forgetting_threshold=self.forgetting_threshold)

    def _llm_judge(self, question: str, answer: str, expected_info: str) -> float:
        """Compatibility method from the original ``evaluation_v2.py`` API."""
        caller = self.judge or self.llm
        prompt = f"""你是一个严格的评判者。请评判模型回答是否准确利用了历史记忆。

【问题】{question}
【历史记忆中的相关信息】{expected_info}
【模型回答】{answer[:800]}

只输出 0 到 1 之间的分数数字。"""
        try:
            raw = caller(prompt).strip()
            match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", raw)
            if match:
                return max(0.0, min(1.0, float(match.group(0))))
        except Exception:
            pass
        return 0.0

    def _summarize(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Compatibility wrapper for the original evaluator method."""
        return summarize_results(results)

    def _print_summary(
        self, summary: dict[str, Any], memory_stats: dict[str, Any]
    ) -> None:
        """Compatibility wrapper for the original detailed console report."""
        print_summary(summary, memory_stats)

    def _score_answer(
        self,
        question: str,
        answer: str,
        expected_keywords: list[str],
    ) -> tuple[float, list[str]]:
        answer_lower = answer.lower()
        hits = [
            keyword
            for keyword in expected_keywords
            if keyword and keyword.lower() in answer_lower
        ]
        if self.judge is None:
            score = len(hits) / len(expected_keywords) if expected_keywords else 0.0
            return score, hits

        expected_info = ", ".join(expected_keywords[:5]) or "（无特定期望）"
        prompt = (
            "你是严格的评判者。请判断回答是否准确利用了给定历史信息。\n"
            f"问题：{question}\n"
            f"历史信息：{expected_info}\n"
            f"回答：{answer[:800]}\n"
            "只输出 0 到 1 之间的分数。"
        )
        try:
            raw = self.judge(prompt).strip()
            match = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)(?!\d)", raw)
            if match:
                score = max(0.0, min(1.0, float(match.group(0))))
                return score, hits
        except Exception:
            pass

        score = len(hits) / len(expected_keywords) if expected_keywords else 0.0
        return score, hits


def summarize_results(results: list[EvaluationResult]) -> dict[str, Any]:
    total = len(results)
    baseline_avg = sum(item.baseline_score for item in results) / total if total else 0.0
    memory_avg = sum(item.memory_score for item in results) / total if total else 0.0
    improvement = (
        (memory_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else None
    )

    category_stats: dict[str, dict[str, float | int]] = {}
    for item in results:
        category = item.category or "未分类"
        stats = category_stats.setdefault(
            category,
            {
                "count": 0,
                "baseline_total": 0.0,
                "memory_total": 0.0,
                "baseline_correct": 0,
                "memory_correct": 0,
            },
        )
        stats["count"] = int(stats["count"]) + 1
        stats["baseline_total"] = float(stats["baseline_total"]) + item.baseline_score
        stats["memory_total"] = float(stats["memory_total"]) + item.memory_score
        stats["baseline_correct"] = int(stats["baseline_correct"]) + int(
            item.baseline_score >= 0.8
        )
        stats["memory_correct"] = int(stats["memory_correct"]) + int(
            item.memory_score >= 0.8
        )

    for stats in category_stats.values():
        count = int(stats["count"])
        stats["baseline_avg"] = float(stats.pop("baseline_total")) / count
        stats["memory_avg"] = float(stats.pop("memory_total")) / count

    return {
        "total": total,
        "baseline_avg": baseline_avg,
        "memory_avg": memory_avg,
        "improvement": improvement,
        "baseline_correct": sum(item.baseline_score >= 0.8 for item in results),
        "memory_correct": sum(item.memory_score >= 0.8 for item in results),
        "category_stats": category_stats,
    }


__all__ = [
    "DeepSeekLLM",
    "OpenAICompatibleLLM",
    "MockLLM",
    "EvaluationResult",
    "MemoryEvaluator",
    "summarize_results",
    "print_summary",
    "run_memora_batch",
]


def print_summary(summary: dict[str, Any], memory_stats: dict[str, Any] | None = None) -> None:
    """Print the detailed category summary used by the original v2 evaluator."""
    print("=" * 60)
    print("实验结果汇总")
    print("=" * 60)
    print(f"探针问题: {summary.get('total', 0)}")
    improvement = summary.get("improvement")
    improvement_text = "N/A" if improvement is None else f"{improvement:.1%}"
    print(
        f"基线均分: {summary.get('baseline_avg', 0.0):.3f} | "
        f"MemoryBank: {summary.get('memory_avg', 0.0):.3f} | "
        f"提升: {improvement_text}"
    )
    print(
        f"基线答对: {summary.get('baseline_correct', 0)}/{summary.get('total', 0)} | "
        f"MemoryBank: {summary.get('memory_correct', 0)}/{summary.get('total', 0)}"
    )
    print("-" * 60)
    for category, stats in summary.get("category_stats", {}).items():
        print(
            f"{category} ({stats.get('count', 0)}题): "
            f"基线 {stats.get('baseline_avg', 0.0):.2f} → "
            f"MemoryBank {stats.get('memory_avg', 0.0):.2f}"
        )
    if memory_stats:
        print("\n[MemoryBank 统计]")
        for key, value in memory_stats.items():
            print(f"  {key}: {value}")


def run_memora_batch(
    *,
    data_dir: str | Path,
    llm_caller: LLMCaller,
    personas: list[str] | None = None,
    period: str = "weekly",
    output_dir: str | Path = "results/memorybank",
    judge_caller: LLMCaller | None = None,
) -> dict[str, Any]:
    """Run the original multi-persona Memora evaluation as a reusable function."""
    from statebudgetmem.baselines.memorybank.datasets import load_memora_data

    personas = personas or [
        "software_engineer",
        "academic_researcher",
        "business_executive",
        "financial_analyst",
        "startup_founder",
    ]
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    evaluator = MemoryEvaluator(llm_caller=llm_caller, judge_caller=judge_caller)
    summaries: dict[str, Any] = {}

    for persona in personas:
        try:
            history, questions, portrait = load_memora_data(
                data_dir, persona=persona, period=period
            )
            result = evaluator.run_evaluation(history, questions, portrait)
            evaluator.export_results(
                result, output_root / f"evaluation_results_{persona}.json"
            )
            summaries[persona] = result["summary"]
        except Exception as exc:  # keep batch runs going, matching the original script
            summaries[persona] = {"error": str(exc)}

    (output_root / "evaluation_results_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summaries
