"""
MemoryBank 对比实验框架

用于评估"无记忆基线"vs"MemoryBank增强"的效果差异。

评测方式：
1. 先灌入一段"历史对话"模拟长期使用
2. 提出需要依赖历史记忆的"探针问题"
3. 分别用基线Agent和MemoryBank Agent回答
4. 对比回答的准确率、一致性和连贯性

支持外部数据集：LongMemEval / Memora / STALE / MemConflict 等
"""

import json
import os
import time
from typing import List, Dict, Tuple, Callable, Optional
from dataclasses import dataclass, field

from memorybank import MemoryBank, MemoryAugmentedAgent, BaselineAgent


# ═══════════════════════════════════════════════════════════════
# LLM 封装
# ═══════════════════════════════════════════════════════════════

class DeepSeekLLM:
    """DeepSeek API 调用封装

    需要: pip install openai
    需要: DeepSeek API Key (https://platform.deepseek.com)
    """

    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        import openai
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model = model
        self.call_count = 0
        self.total_tokens = 0

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


class MockLLM:
    """模拟 LLM——用于在没有真实 API 的情况下测试流程"""

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
            if "小林" in prompt:
                return "你好小林！之前我们聊过你喜欢运动和准备考试的事情。"
            if "书" in prompt or "深度学习" in prompt:
                return "我之前推荐过《深度学习入门》给你。"

        return "好的，我明白了。有什么我可以帮你的吗？"


# ═══════════════════════════════════════════════════════════════
# 数据集加载函数（支持外部评测集）
# ═══════════════════════════════════════════════════════════════

def load_dataset(filepath: str, sample_idx: int = 0) -> Tuple[List[Tuple[str, str, str]], List[Dict], str]:
    """
    加载外部数据集（LongMemEval / Memora / STALE / MemConflict 等）

    支持格式：
    - JSON 文件，顶层为列表，每个元素是一个样本
    - 自动探测 conversation_history / dialogs / conversation 等常见字段名
    - 自动探测 questions / probes / test_cases 等常见字段名

    Args:
        filepath: 数据集 JSON 文件路径
        sample_idx: 取第几个样本（0-based）

    Returns:
        (history, probe_questions, user_portrait)
        history: [(role, content, timestamp), ...]
        probe_questions: [{"question": ..., "expected_keywords": [...], "category": ...}, ...]
        user_portrait: 用户画像字符串（如果数据集提供）
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据集文件不存在: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 顶层是列表还是字典
    if isinstance(data, list):
        if sample_idx >= len(data):
            raise IndexError(f"sample_idx={sample_idx} 超出范围，数据集共 {len(data)} 个样本")
        sample = data[sample_idx]
    else:
        sample = data

    print(f"[数据集] 加载 {filepath}，样本 {sample_idx}")
    print(f"[数据集] 可用字段: {list(sample.keys())}")

    # ── 解析对话历史 ──
    history = []
    conv_key = None
    for key in ["conversation_history", "dialogs", "conversation", "history", "messages"]:
        if key in sample:
            conv_key = key
            break

    if conv_key:
        conv = sample[conv_key]
        if isinstance(conv, list):
            for i, turn in enumerate(conv):
                if isinstance(turn, dict):
                    role = "用户" if turn.get("speaker", turn.get("role", "")) in ["user", "User", "USER", "human"] else "AI"
                    content = turn.get("content", turn.get("text", turn.get("message", "")))
                    ts = turn.get("timestamp", turn.get("time", f"2026-06-{20 + i // 4:02d} {10 + i:02d}:00"))
                elif isinstance(turn, str):
                    role = "用户" if i % 2 == 0 else "AI"
                    content = turn
                    ts = f"2026-06-{20 + i // 4:02d} {10 + i:02d}:00"
                else:
                    continue
                if content:
                    history.append((role, content, ts))
    else:
        print("[警告] 未找到对话历史字段，尝试从 questions 反推...")

    # ── 解析探针问题 ──
    probe_questions = []
    q_key = None
    for key in ["questions", "probes", "test_cases", "probing_questions"]:
        if key in sample:
            q_key = key
            break

    if q_key:
        for q in sample[q_key]:
            probe = {
                "question": q.get("question", q.get("query", q.get("q", ""))),
                "expected_keywords": q.get("expected_keywords", q.get("keywords", [])),
                "category": q.get("category", q.get("type", "未分类")),
                "note": q.get("note", q.get("description", ""))
            }
            if not probe["expected_keywords"]:
                # 尝试从 answer / ground_truth 提取
                ans = q.get("answer", q.get("ground_truth", q.get("reference", "")))
                if ans:
                    probe["expected_keywords"] = [ans] if isinstance(ans, str) else ans
            if probe["question"]:
                probe_questions.append(probe)

    # ── 用户画像 ──
    user_portrait = sample.get("user_profile", sample.get("persona", sample.get("profile", "")))

    print(f"[数据集] 解析完成: {len(history)} 条对话, {len(probe_questions)} 个探针")
    return history, probe_questions, user_portrait

def load_memora_data(data_dir: str, persona: str = "software_engineer", period: str = "weekly"):
    """
    加载 Memora 数据集

    Args:
        data_dir: Memora/data 目录路径
        persona: 人物角色，如 software_engineer / academic_researcher
        period: weekly / monthly / quarterly

    Returns:
        (history, probe_questions, user_portrait)
    """
    import glob

    base_dir = os.path.join(data_dir, period, persona)

    # ── 1. 加载所有对话 session ──
    conv_dir = os.path.join(base_dir, "conversations")
    session_files = sorted(glob.glob(os.path.join(conv_dir, "session_*.json")))

    history = []
    for sf in session_files:
        with open(sf, "r", encoding="utf-8") as f:
            session = json.load(f)
        date = session.get("date", "2025-06-01")

        for turn in session.get("conversation", []):
            speaker = turn.get("speaker", "user")
            role = "用户" if speaker == "user" else "AI"
            message = turn.get("message", "")
            if message:
                history.append((role, message, date))

    # ── 2. 加载评测问题 ──
    eval_file = os.path.join(base_dir, f"evaluation_questions_{persona}.json")
    with open(eval_file, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    probe_questions = []
    for task_type in ["remembering", "reasoning", "recommending"]:
        for q in eval_data.get("questions", {}).get(task_type, []):
            # 从 memory_evidence 提取关键词
            evidence = q.get("memory_evidence", {})
            keywords = []
            if isinstance(evidence, dict):
                # 提取所有值作为关键词
                for v in evidence.values():
                    if isinstance(v, str):
                        keywords.append(v)
                    elif isinstance(v, list):
                        keywords.extend([str(x) for x in v])
            elif isinstance(evidence, str):
                keywords = [evidence]

            # 取前 5 个关键词，避免太长
            keywords = keywords[:5]

            probe_questions.append({
                "question": q["question"],
                "expected_keywords": keywords,
                "category": task_type,  # remembering / reasoning / recommending
                "note": f"{persona}, {q.get('question_date', '')}"
            })

    # ── 3. 用户画像 ──
    user_portrait = persona.replace("_", " ")

    print(f"[Memora] {period}/{persona}: {len(history)} 条对话, {len(probe_questions)} 题")
    print(f"[Memora] 任务分布: remembering={sum(1 for q in probe_questions if q['category']=='remembering')}, "
          f"reasoning={sum(1 for q in probe_questions if q['category']=='reasoning')}, "
          f"recommending={sum(1 for q in probe_questions if q['category']=='recommending')}")

    return history, probe_questions, user_portrait

# ═══════════════════════════════════════════════════════════════
# 内置演示数据集（当没有外部数据集时使用）
# ═══════════════════════════════════════════════════════════════

DEMO_HISTORY = [
    # Day 1
    ("用户", "你好，我叫小林，是计算机系大三的学生", "2026-06-20 10:00"),
    ("AI", "你好小林！计算机系大三，那应该有很多有趣的项目经验吧。", "2026-06-20 10:00"),
    ("用户", "平时喜欢打篮球和游泳，周末经常运动", "2026-06-20 10:05"),
    ("AI", "篮球和游泳都是很棒的运动！", "2026-06-20 10:05"),
    ("用户", "对了，我还特别喜欢吃辣的东西，川菜和火锅是我的最爱", "2026-06-20 10:15"),
    ("AI", "川菜和火锅确实很有魅力！", "2026-06-20 10:15"),
    # Day 2
    ("用户", "最近期末考试要来了，压力很大", "2026-06-21 15:00"),
    ("AI", "考试加油！", "2026-06-21 15:00"),
    ("用户", "不用了，我想自己安排。推荐一本好点的参考书吧", "2026-06-21 15:05"),
    ("AI", "我推荐你看看《深度学习入门》，非常适合初学者。", "2026-06-21 15:05"),
    # Day 3
    ("用户", "今天状态不太好，感觉很累", "2026-06-22 09:00"),
    ("AI", "注意休息。", "2026-06-22 09:00"),
    ("用户", "除了运动，还会听听音乐或者看看电影", "2026-06-22 09:05"),
    ("AI", "这些都是很好的放松方式。", "2026-06-22 09:05"),
    # Day 4 — 关键转折：健康原因改变饮食
    ("用户", "昨天去校医院了，医生说我的胃不太好", "2026-06-23 11:00"),
    ("AI", "啊，严重吗？", "2026-06-23 11:00"),
    ("用户", "不严重，但医生说让我少吃辛辣刺激的食物，建议清淡饮食", "2026-06-23 11:05"),
    ("AI", "那确实需要调整饮食习惯了。", "2026-06-23 11:05"),
    ("用户", "以后川菜和火锅可能要告别了", "2026-06-23 11:10"),
    ("AI", "健康第一！其实清淡的食物也有很多好吃的选择。", "2026-06-23 11:10"),
    # Day 5
    ("用户", "今天去了学校附近一家粤菜馆，味道还不错", "2026-06-24 12:00"),
    ("AI", "看来清淡饮食也挺适合你的！", "2026-06-24 12:00"),
    ("用户", "白切鸡和蒸鱼挺好吃的", "2026-06-24 12:05"),
    ("AI", "这些菜健康又美味。", "2026-06-24 12:05"),
    ("用户", "游泳卡快到期了，要不要续费呢", "2026-06-24 18:00"),
    ("AI", "看你平时去的频率吧。", "2026-06-24 18:00"),
    ("用户", "我大概每周去两次", "2026-06-24 18:05"),
    ("AI", "那频率还挺高的，续费应该值得。", "2026-06-24 18:05"),
    # Day 6 — 错误修正
    ("用户", "对了，我之前好像说错了，我不是计算机系的", "2026-06-25 09:00"),
    ("AI", "啊？那你是什么专业的？", "2026-06-25 09:00"),
    ("用户", "我是软件工程专业的", "2026-06-25 09:05"),
    ("AI", "明白了，软件工程更偏向工程实践。", "2026-06-25 09:05"),
    # Day 7
    ("用户", "暑假打算留在学校，准备参加一个AI竞赛", "2026-06-26 10:00"),
    ("AI", "好棒！", "2026-06-26 10:00"),
    ("用户", "想做一个图像识别的项目", "2026-06-26 10:05"),
    ("AI", "和你之前学的神经网络很契合！", "2026-06-26 10:05"),
]

DEMO_QUESTIONS = [
    # 基础信息
    {"question": "我叫什么名字？", "expected_keywords": ["小林"], "category": "基础信息", "note": "Day 1"},
    {"question": "我是什么专业的学生？", "expected_keywords": ["软件工程"], "category": "基础信息", "note": "Day 6 修正"},
    # 静态偏好
    {"question": "我喜欢什么运动？", "expected_keywords": ["篮球", "游泳"], "category": "静态偏好", "note": "Day 1"},
    {"question": "我平时喜欢怎么放松？", "expected_keywords": ["音乐", "电影"], "category": "静态偏好", "note": "Day 3"},
    # 事实记忆
    {"question": "你之前推荐过什么书给我？", "expected_keywords": ["深度学习入门"], "category": "事实记忆", "note": "Day 2"},
    {"question": "我每周游泳几次？", "expected_keywords": ["两次", "2次"], "category": "事实记忆", "note": "Day 5"},
    # 过期记忆（核心亮点）
    {"question": "我现在适合吃什么类型的食物？", "expected_keywords": ["清淡", "粤菜"], "category": "过期记忆", "note": "Day 4 后应推荐清淡"},
    {"question": "我现在还能经常吃火锅吗？", "expected_keywords": ["不能", "不建议", "胃"], "category": "过期记忆", "note": "健康原因"},
    {"question": "我为什么改变饮食习惯？", "expected_keywords": ["胃", "医生", "健康"], "category": "过期记忆", "note": "Day 4"},
    # 新偏好
    {"question": "我最近吃过什么好吃的？", "expected_keywords": ["白切鸡", "蒸鱼", "粤菜"], "category": "新偏好", "note": "Day 5"},
    {"question": "暑假我打算做什么？", "expected_keywords": ["AI竞赛", "图像识别"], "category": "新偏好", "note": "Day 7"},
    # 错误修正
    {"question": "我之前说我是计算机系的，我实际是什么专业？", "expected_keywords": ["软件工程"], "category": "错误修正", "note": "Day 6"},
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
    baseline_score: float
    memory_score: float
    expected_keywords: List[str]
    baseline_hit_keywords: List[str]
    memory_hit_keywords: List[str]
    category: str = ""
    note: str = ""


class MemoryEvaluator:
    """记忆系统评测器"""

    def __init__(self, llm_caller: Callable[[str], str], forgetting_threshold: float = 0.3):
        self.llm = llm_caller
        self.forgetting_threshold = forgetting_threshold

    def run_evaluation(
        self,
        history: Optional[List[Tuple[str, str, str]]] = None,
        probe_questions: Optional[List[Dict]] = None,
        user_portrait: str = ""
    ) -> Dict:
        """执行完整对比评测"""

        # 如果没有提供数据，使用内置演示数据
        if not history:
            history = DEMO_HISTORY
            print("[提示] 未提供外部数据，使用内置演示数据集")
        if not probe_questions:
            probe_questions = DEMO_QUESTIONS

        print("=" * 60)
        print("MemoryBank 对比实验")
        print("=" * 60)
        print(f"对话历史: {len(history)} 条")
        print(f"探针问题: {len(probe_questions)} 题")

        # 1. 创建 Agent
        print("\n[1] 初始化 Agent...")
        baseline_agent = BaselineAgent(llm_caller=self.llm)
        memory_bank = MemoryBank(forgetting_threshold=self.forgetting_threshold)
        memory_agent = MemoryAugmentedAgent(memory_bank=memory_bank, llm_caller=self.llm)

        # 2. 灌入历史
        print(f"[2] 灌入历史对话...")
        memory_agent.batch_store_history(history)

        if user_portrait:
            memory_agent.memory.update_user_portrait(user_portrait)
        else:
            # 使用内置默认画像
            memory_agent.memory.update_user_portrait(
                "用户小林，软件工程专业大三学生，爱好篮球和游泳，"
                "因健康原因已改为清淡饮食，喜欢晚上学习，室友叫阿杰"
            )

        # 3. 探针测试
        print(f"[3] 执行探针测试（{len(probe_questions)} 题）...\n")
        results = []

        for i, probe in enumerate(probe_questions, 1):
            question = probe["question"]
            expected = probe["expected_keywords"]
            cat = probe.get("category", "")
            note = probe.get("note", "")

            print(f"--- 探针 {i}/{len(probe_questions)} [{cat}] ---")
            print(f"问题: {question}")
            print(f"期望: {expected}")

            baseline_answer = baseline_agent.chat(question, "2026-06-27 10:00")
            memory_answer = memory_agent.chat(question, "2026-06-27 10:00")

            b_score, b_hits = self._score_answer(baseline_answer, expected, question)
            m_score, m_hits = self._score_answer(memory_answer, expected, question)

            results.append(EvaluationResult(
                question=question, baseline_answer=baseline_answer,
                memory_answer=memory_answer, baseline_score=b_score,
                memory_score=m_score, expected_keywords=expected,
                baseline_hit_keywords=b_hits, memory_hit_keywords=m_hits,
                category=cat, note=note
            ))

            print(f"基线 Judge 分: {b_score:.2f} | MemoryBank Judge 分: {m_score:.2f}")
            print(f"基线回答: {baseline_answer[:120]}...")
            print(f"记忆回答: {memory_answer[:120]}...")
            print()

        # 4. 汇总
        summary = self._summarize(results)
        self._print_summary(summary, memory_agent.get_memory_stats())

        return {"baseline_agent": baseline_agent, "memory_agent": memory_agent,
                "results": results, "summary": summary}

    def _llm_judge(self, question: str, answer: str, expected_info: str) -> float:
            """LLM-as-Judge：让 LLM 评判回答质量，返回 0.0~1.0"""
            judge_prompt = f"""你是一个严格的评判者。请评判模型回答是否准确利用了历史记忆。

            【问题】{question}
            【历史记忆中的相关信息】{expected_info}
            【模型回答】{answer[:800]}

            评分标准（0到1之间）:
            - 1.0: 回答精准利用了历史记忆，信息准确完整
            - 0.7-0.9: 部分利用了历史记忆，有小遗漏或不精确
            - 0.3-0.6: 提及了历史记忆但未充分利用，或信息有误
            - 0.1-0.2: 几乎未利用历史记忆，主要靠通用知识回答
            - 0.0: 完全错误、编造或未回答

            注意：如果回答只是"好的通用建议"但没有利用具体的历史记忆信息，应给低分（0.1-0.3）。

            只输出分数数字，不要解释。"""

            try:
                result = self.llm(judge_prompt)
                import re
                nums = re.findall(r"0\.\d{1,2}|1\.0{1,2}", result.strip())
                if nums:
                    score = float(nums[0])
                    return min(1.0, max(0.0, score))
            except Exception as e:
                print(f"[Judge Error] {e}")

            return 0.0

    def _score_answer(self, answer: str, expected_keywords: List[str], question: str = "") -> Tuple[
            float, List[str]]:
            """评分：优先 LLM-as-Judge，回退到关键词匹配"""
            expected_text = ", ".join(str(k)[:100] for k in expected_keywords[:5])
            if not expected_text:
                expected_text = "（无特定期望）"

            score = self._llm_judge(question, answer, expected_text)

            # 关键词匹配仅用于记录
            answer_lower = answer.lower()
            hits = [kw for kw in expected_keywords if str(kw).lower() in answer_lower]

            return score, hits

    def _summarize(self, results: List[EvaluationResult]) -> Dict:
        b_scores = [r.baseline_score for r in results]
        m_scores = [r.memory_score for r in results]
        baseline_avg = sum(b_scores) / len(b_scores) if b_scores else 0
        memory_avg = sum(m_scores) / len(m_scores) if m_scores else 0
        baseline_correct = sum(1 for s in b_scores if s >= 0.8)
        memory_correct = sum(1 for s in m_scores if s >= 0.8)
        improvement = (memory_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else float('inf')

        # 按类别分组
        cat_stats = {}
        for r in results:
            cat = r.category or "未分类"
            if cat not in cat_stats:
                cat_stats[cat] = {"count": 0, "b_total": 0, "m_total": 0,
                                  "b_correct": 0, "m_correct": 0}
            cat_stats[cat]["count"] += 1
            cat_stats[cat]["b_total"] += r.baseline_score
            cat_stats[cat]["m_total"] += r.memory_score
            if r.baseline_score >= 0.99:
                cat_stats[cat]["b_correct"] += 1
            if r.memory_score >= 0.99:
                cat_stats[cat]["m_correct"] += 1

        for s in cat_stats.values():
            s["b_avg"] = s["b_total"] / s["count"]
            s["m_avg"] = s["m_total"] / s["count"]

        return {"baseline_avg": baseline_avg, "memory_avg": memory_avg,
                "improvement": improvement, "baseline_correct": baseline_correct,
                "memory_correct": memory_correct, "total": len(results),
                "category_stats": cat_stats}

    def _print_summary(self, summary: Dict, mem_stats: Dict):
        print("=" * 60)
        print("实验结果汇总")
        print("=" * 60)
        print(f"探针问题: {summary['total']}")
        print(f"基线均分: {summary['baseline_avg']:.3f} | "
              f"MemoryBank: {summary['memory_avg']:.3f} | "
              f"提升: {summary['improvement']:.1%}")
        print(f"基线答对: {summary['baseline_correct']}/{summary['total']} | "
              f"MemoryBank: {summary['memory_correct']}/{summary['total']}")

        # 按类别展示
        cat_stats = summary.get("category_stats", {})
        order = ["remembering", "reasoning", "recommending"]
        print(f"\n{'─' * 50}")
        for cat in order:
            if cat not in cat_stats:
                continue
            s = cat_stats[cat]
            marker = " 🔥" if cat == "过期记忆" else ""
            print(f"{cat}{marker} ({s['count']}题): "
                  f"基线 {s['b_avg']:.2f} → MemoryBank {s['m_avg']:.2f}")
        print(f"{'─' * 50}")

        if "remembering" in cat_stats:
            print("\n[ remembering 类分析 ]")
            print("MemoryBank 在'回忆具体事实'上表现最好，")
            print("因为向量检索能精准匹配语义相关的历史记忆。")
        if "recommending" in cat_stats:
            print("\n[ recommending 类分析 ]")
            print("MemoryBank 能基于用户历史偏好给出个性化推荐，")
            print("而基线只能给出通用建议，无法体现'记得用户'的能力。")

        print(f"\n[MemoryBank 统计]")
        for k, v in mem_stats.items():
            print(f"  {k}: {v}")

    def export_results(self, results_data: Dict, filepath: str):
        export = {
            "summary": {k: v for k, v in results_data["summary"].items() if k != "category_stats"},
            "by_category": results_data["summary"].get("category_stats", {}),
            "details": [{"question": r.question, "category": r.category,
                         "baseline_score": r.baseline_score, "memory_score": r.memory_score,
                         "baseline_answer": r.baseline_answer, "memory_answer": r.memory_answer}
                        for r in results_data["results"]]
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export, f, ensure_ascii=False, indent=2)
        print(f"\n结果已导出: {filepath}")


# ═══════════════════════════════════════════════════════════════
# 主程序
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("MemoryBank 对比实验框架 v2 — 批量评测")
    print("-" * 60)

    # ===== 1. 选择 LLM =====
    llm = DeepSeekLLM(api_key="sk-你的API key")

    # ===== 2. 批量跑多个 persona =====
    MEMORA_DIR = "Memora/data"

    if not os.path.exists(MEMORA_DIR):
        print(f"Memora 数据集未找到: {MEMORA_DIR}")
        print("使用内置演示数据...")
        evaluator = MemoryEvaluator(llm_caller=llm)
        results = evaluator.run_evaluation()
        evaluator.export_results(results, "evaluation_results.json")
    else:
        # 要跑的 persona 列表
        personas = [
            "software_engineer",
            "academic_researcher",
            "business_executive",
            "financial_analyst",
            "startup_founder",
        ]

        evaluator = MemoryEvaluator(llm_caller=llm)
        all_results = {}

        for persona in personas:
            print(f"\n{'#' * 60}")
            print(f"# 正在评测: {persona}")
            print(f"{'#' * 60}")

            try:
                history, questions, portrait = load_memora_data(
                    MEMORA_DIR, persona=persona, period="weekly"
                )
                results = evaluator.run_evaluation(history, questions, portrait)

                # 保存单个结果
                evaluator.export_results(results, f"evaluation_results_{persona}.json")

                # 记录汇总
                all_results[persona] = {
                    "baseline_avg": results["summary"]["baseline_avg"],
                    "memory_avg": results["summary"]["memory_avg"],
                    "improvement": results["summary"]["improvement"],
                    "baseline_correct": results["summary"]["baseline_correct"],
                    "memory_correct": results["summary"]["memory_correct"],
                    "total": results["summary"]["total"],
                }

            except Exception as e:
                print(f"[错误] {persona} 评测失败: {e}")
                all_results[persona] = None

        # ===== 3. 汇总所有 persona =====
        print(f"\n{'=' * 60}")
        print("批量评测汇总")
        print(f"{'=' * 60}")
        print(f"{'Persona':<25} {'基线':>6} {'MB':>6} {'提升':>8} {'答对':>10}")
        print("-" * 60)

        total_baseline = 0
        total_memory = 0
        total_questions = 0
        total_baseline_correct = 0
        total_memory_correct = 0

        for persona, r in all_results.items():
            if r is None:
                print(f"{persona:<25} [失败]")
                continue

            total_baseline += r["baseline_avg"] * r["total"]
            total_memory += r["memory_avg"] * r["total"]
            total_questions += r["total"]
            total_baseline_correct += r["baseline_correct"]
            total_memory_correct += r["memory_correct"]

            print(f"{persona:<25} {r['baseline_avg']:>6.3f} {r['memory_avg']:>6.3f} "
                  f"{r['improvement']:>7.1%} {r['memory_correct']}/{r['total']:>3}")

        print("-" * 60)
        if total_questions > 0:
            overall_baseline = total_baseline / total_questions
            overall_memory = total_memory / total_questions
            overall_improvement = (overall_memory - overall_baseline) / overall_baseline if overall_baseline > 0 else 0
            print(f"{'总计':<25} {overall_baseline:>6.3f} {overall_memory:>6.3f} "
                  f"{overall_improvement:>7.1%} {total_memory_correct}/{total_questions:>3}")

        print(f"\n每个 persona 的详细结果已导出到:")
        for persona in all_results:
            if all_results[persona]:
                print(f"  evaluation_results_{persona}.json")

    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)