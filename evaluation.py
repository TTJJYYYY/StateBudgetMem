"""
MemoryBank 对比实验框架

用于评估"无记忆基线"vs"MemoryBank增强"的效果差异。

评测方式：
1. 先灌入一段"历史对话"模拟长期使用
2. 提出需要依赖历史记忆的"探针问题"
3. 分别用基线Agent和MemoryBank Agent回答
4. 对比回答的准确率、一致性和连贯性
"""

import json
import time
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass

from memorybank import MemoryBank, MemoryAugmentedAgent, BaselineAgent


# ═══════════════════════════════════════════════════════════════
# 模拟 LLM（用于测试，实际使用时替换为真实 LLM）
# ═══════════════════════════════════════════════════════════════

class DeepSeekLLM:
    """
    DeepSeek API 调用封装

    需要安装: pip install openai
    需要设置 API Key（在创建实例时传入）
    """

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        import openai
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[API Error] {e}")
            return "[调用失败，请检查 API Key 和网络]"


# 保留 MockLLM 用于离线测试
class MockLLM:
    """
    模拟 LLM——用于在没有真实 API 的情况下测试流程
    """

    def __init__(self):
        self.call_count = 0

    def __call__(self, prompt: str) -> str:
        self.call_count += 1

        prompt_lower = prompt.lower()

        if "相关历史记忆" in prompt:
            if "篮球" in prompt or "游泳" in prompt:
                return "我记得你喜欢打篮球和游泳！"
            if "考试" in prompt or "压力" in prompt:
                return "你之前提到过在准备期末考试，压力比较大。"
            if "小明" in prompt:
                return "你好小明！之前我们聊过你喜欢运动和准备考试的事情。"
            if "书" in prompt or "深度学习" in prompt:
                return "我之前推荐过《深度学习入门》给你。"

        return "好的，我明白了。有什么我可以帮你的吗？"


# ═══════════════════════════════════════════════════════════════
# 评测数据集
# ═══════════════════════════════════════════════════════════════

# 模拟历史对话：用户 "小明" 与 AI 的多天对话记录
SIMULATED_HISTORY = [
    # Day 1: 2026-06-20
    ("用户", "你好，我叫小明", "2026-06-20 10:00"),
    ("AI", "你好小明！很高兴认识你。", "2026-06-20 10:00"),
    ("用户", "我喜欢打篮球和游泳，周末经常运动", "2026-06-20 10:05"),
    ("AI", "篮球和游泳都是很棒的运动！你一般去哪里运动？", "2026-06-20 10:05"),
    ("用户", "我在学校体育馆打篮球，去市游泳馆游泳", "2026-06-20 10:10"),
    ("AI", "看来你是运动爱好者啊！", "2026-06-20 10:10"),

    # Day 2: 2026-06-21
    ("用户", "最近期末考试要来了，压力很大", "2026-06-21 15:00"),
    ("AI", "考试加油！需要我帮你做复习计划吗？", "2026-06-21 15:00"),
    ("用户", "不用了，我想自己安排。推荐一本好点的参考书吧", "2026-06-21 15:05"),
    ("AI", "我推荐你看看《深度学习入门》，非常适合初学者。", "2026-06-21 15:05"),
    ("用户", "好，我去看看", "2026-06-21 15:10"),

    # Day 3: 2026-06-22
    ("用户", "今天状态不太好，感觉很累", "2026-06-22 09:00"),
    ("AI", "注意休息，可以适当放松一下。你平时喜欢怎么放松？", "2026-06-22 09:00"),
    ("用户", "听听音乐或者看看电影", "2026-06-22 09:05"),
    ("AI", "这些都是很好的放松方式。", "2026-06-22 09:05"),

    # Day 4: 2026-06-23
    ("用户", "昨天那本《深度学习入门》看了一部分，还不错", "2026-06-23 20:00"),
    ("AI", " glad to hear that! 有什么不懂的地方可以随时问我。", "2026-06-23 20:00"),
    ("用户", "我对神经网络那章比较感兴趣", "2026-06-23 20:05"),
    ("AI", "那章确实很有意思，是整本书的核心部分。", "2026-06-23 20:05"),

    # Day 5: 2026-06-24
    ("用户", "游泳卡快到期了，要不要续费呢", "2026-06-24 18:00"),
    ("AI", "看你平时去的频率吧，经常去的话续费比较划算。", "2026-06-24 18:00"),
    ("用户", "我大概每周去两次", "2026-06-24 18:05"),
    ("AI", "那频率还挺高的，续费应该值得。", "2026-06-24 18:05"),
]

# 探针问题：需要依赖历史记忆才能回答的问题
PROBE_QUESTIONS = [
    {
        "question": "我叫什么名字？",
        "expected_keywords": ["小明"],
        "required_memory": "Day 1 的自我介绍"
    },
    {
        "question": "我喜欢什么运动？",
        "expected_keywords": ["篮球", "游泳"],
        "required_memory": "Day 1 的运动爱好"
    },
    {
        "question": "你之前推荐过什么书给我？",
        "expected_keywords": ["深度学习入门"],
        "required_memory": "Day 2 的图书推荐"
    },
    {
        "question": "我最近压力大是因为什么？",
        "expected_keywords": ["考试", "期末"],
        "required_memory": "Day 2 的考试压力"
    },
    {
        "question": "我平时喜欢怎么放松？",
        "expected_keywords": ["音乐", "电影"],
        "required_memory": "Day 3 的放松方式"
    },
    {
        "question": "我对书的哪一章比较感兴趣？",
        "expected_keywords": ["神经网络"],
        "required_memory": "Day 4 的阅读偏好"
    },
    {
        "question": "我每周游泳几次？",
        "expected_keywords": ["两次", "2次"],
        "required_memory": "Day 5 的游泳频率"
    },
    {
        "question": "推荐一些适合我的运动",
        "expected_keywords": ["篮球", "游泳"],
        "required_memory": "需要综合 Day 1 和 Day 5 的运动偏好"
    },
]


# ═══════════════════════════════════════════════════════════════
# 评测器
# ═══════════════════════════════════════════════════════════════

@dataclass
class EvaluationResult:
    """单条评测结果"""
    question: str
    baseline_answer: str
    memory_answer: str
    baseline_score: float  # 0-1
    memory_score: float    # 0-1
    expected_keywords: List[str]
    baseline_hit_keywords: List[str]
    memory_hit_keywords: List[str]


class MemoryEvaluator:
    """
    记忆系统评测器

    对比 baseline（无记忆）和 memory-enhanced（有 MemoryBank）的效果
    """

    def __init__(
        self,
        llm_caller: Callable[[str], str],
        forgetting_threshold: float = 0.3
    ):
        self.llm = llm_caller
        self.forgetting_threshold = forgetting_threshold

    def run_evaluation(
        self,
        history: List[Tuple[str, str, str]] = None,
        probe_questions: List[Dict] = None,
        user_portrait: str = ""
    ) -> Dict:
        """
        执行完整对比评测

        流程：
        1. 创建两个 Agent：baseline（无记忆）和 memory（有 MemoryBank）
        2. 给两个 Agent 灌入相同的历史对话
        3. 用探针问题分别提问
        4. 对比回答质量

        Returns:
            {
                "baseline_agent": {...},
                "memory_agent": {...},
                "results": [EvaluationResult, ...],
                "summary": {...}
            }
        """
        history = history or SIMULATED_HISTORY
        probe_questions = probe_questions or PROBE_QUESTIONS

        print("=" * 60)
        print("MemoryBank 对比实验")
        print("=" * 60)

        # 1. 创建两个 Agent
        print("\n[1] 初始化 Agent...")
        baseline_agent = BaselineAgent(llm_caller=self.llm)
        memory_bank = MemoryBank(forgetting_threshold=self.forgetting_threshold)
        memory_agent = MemoryAugmentedAgent(
            memory_bank=memory_bank,
            llm_caller=self.llm
        )

        # 2. 灌入历史对话
        print(f"[2] 灌入历史对话（{len(history)} 条）...")
        baseline_agent.chat("初始化对话", "2026-06-19 00:00")  # 占位
        baseline_agent.dialog_history = []  # 清空，确保测试时是干净的

        memory_agent.batch_store_history(history)

        # 设置用户画像
        if user_portrait:
            memory_agent.memory.update_user_portrait(user_portrait)
        else:
            memory_agent.memory.update_user_portrait(
                "用户小明，大学生，爱好篮球和游泳，最近在准备期末考试"
            )

        # 3. 执行探针测试
        print(f"[3] 执行探针测试（{len(probe_questions)} 题）...\n")
        results = []

        for i, probe in enumerate(probe_questions, 1):
            question = probe["question"]
            expected = probe["expected_keywords"]

            print(f"--- 探针 {i}/{len(probe_questions)} ---")
            print(f"问题: {question}")
            print(f"期望关键词: {expected}")

            # Baseline 回答
            baseline_answer = baseline_agent.chat(question, "2026-06-25 10:00")

            # Memory 回答
            memory_answer = memory_agent.chat(question, "2026-06-25 10:00")

            # 评估
            b_score, b_hits = self._score_answer(baseline_answer, expected)
            m_score, m_hits = self._score_answer(memory_answer, expected)

            result = EvaluationResult(
                question=question,
                baseline_answer=baseline_answer,
                memory_answer=memory_answer,
                baseline_score=b_score,
                memory_score=m_score,
                expected_keywords=expected,
                baseline_hit_keywords=b_hits,
                memory_hit_keywords=m_hits
            )
            results.append(result)

            print(f"基线回答: {baseline_answer[:80]}...")
            print(f"基线得分: {b_score:.1f} (命中: {b_hits})")
            print(f"记忆回答: {memory_answer[:80]}...")
            print(f"记忆得分: {m_score:.1f} (命中: {m_hits})")
            print()

        # 4. 汇总结果
        summary = self._summarize(results)

        # 5. 统计信息
        print("=" * 60)
        print("实验结果汇总")
        print("=" * 60)
        print(f"探针问题数: {len(results)}")
        print(f"基线平均得分: {summary['baseline_avg_score']:.3f}")
        print(f"记忆平均得分: {summary['memory_avg_score']:.3f}")
        print(f"提升幅度: {summary['improvement']:.1%}")
        print(f"基线答对题数: {summary['baseline_correct']}/{len(results)}")
        print(f"记忆答对题数: {summary['memory_correct']}/{len(results)}")
        print(f"\nMemoryBank 统计:")
        for k, v in memory_agent.get_memory_stats().items():
            print(f"  {k}: {v}")

        return {
            "baseline_agent": baseline_agent,
            "memory_agent": memory_agent,
            "results": results,
            "summary": summary
        }

    def _score_answer(self, answer: str, expected_keywords: List[str]) -> Tuple[float, List[str]]:
        """
        评分：回答中包含多少个期望关键词

        Returns:
            (score, hit_keywords)
            score = 命中关键词数 / 总关键词数
        """
        answer_lower = answer.lower()
        hits = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        score = len(hits) / len(expected_keywords) if expected_keywords else 0
        return score, hits

    def _summarize(self, results: List[EvaluationResult]) -> Dict:
        """汇总统计"""
        b_scores = [r.baseline_score for r in results]
        m_scores = [r.memory_score for r in results]

        baseline_avg = sum(b_scores) / len(b_scores) if b_scores else 0
        memory_avg = sum(m_scores) / len(m_scores) if m_scores else 0

        # 完全答对（score=1）的数量
        baseline_correct = sum(1 for s in b_scores if s >= 0.99)
        memory_correct = sum(1 for s in m_scores if s >= 0.99)

        improvement = (memory_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else float('inf')

        return {
            "baseline_avg_score": baseline_avg,
            "memory_avg_score": memory_avg,
            "improvement": improvement,
            "baseline_correct": baseline_correct,
            "memory_correct": memory_correct,
            "total_questions": len(results)
        }

    def export_results(self, results_data: Dict, filepath: str):
        """导出评测结果到 JSON"""
        export = {
            "summary": results_data["summary"],
            "detailed_results": [
                {
                    "question": r.question,
                    "baseline_answer": r.baseline_answer,
                    "memory_answer": r.memory_answer,
                    "baseline_score": r.baseline_score,
                    "memory_score": r.memory_score,
                    "expected_keywords": r.expected_keywords,
                    "baseline_hits": r.baseline_hit_keywords,
                    "memory_hits": r.memory_hit_keywords
                }
                for r in results_data["results"]
            ]
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)

        print(f"\n结果已导出到: {filepath}")


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("MemoryBank 对比实验框架")
    print("-" * 60)

    # ===== 选择 LLM =====

    # 方式 1：使用 DeepSeek API（推荐，需要填入你的 API Key）
    llm = DeepSeekLLM(api_key="sk-e3de338dd55545569fc90b4444e9ead1")

    # 方式 2：使用 MockLLM（离线测试，无需 API Key）
    #  llm = MockLLM()

    evaluator = MemoryEvaluator(llm_caller=llm)

    # 运行评测
    results = evaluator.run_evaluation()

    # 导出结果
    evaluator.export_results(results, "evaluation_results.json")

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)
    print("\n下一步:")
    print("1. 将 MockLLM 替换为真实的 LLM API（OpenAI / Qwen 等）")
    print("2. 扩充探针问题和模拟历史数据")
    print("3. 设计更精细的评分标准（人工评分或 GPT-4 评分）")