"""Dataset adapters used by memory-system comparison experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

History = list[tuple[str, str, str]]
Probe = dict[str, Any]


def load_json_dataset(
    filepath: str | Path,
    sample_idx: int = 0,
) -> tuple[History, list[Probe], str]:
    """Load one sample from a generic long-memory JSON dataset.

    The adapter accepts several common field spellings used by LongMemEval-like
    or locally converted datasets.  It returns the normalized conversation
    history, probe questions, and an optional user portrait.
    """

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"dataset file does not exist: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not 0 <= sample_idx < len(data):
            raise IndexError(
                f"sample_idx={sample_idx} is out of range for {len(data)} samples"
            )
        sample = data[sample_idx]
    elif isinstance(data, dict):
        sample = data
    else:
        raise ValueError("dataset root must be a JSON object or array")

    if not isinstance(sample, dict):
        raise ValueError("selected dataset sample must be a JSON object")

    history: History = []
    conversation = _first_present(
        sample,
        "conversation_history",
        "dialogs",
        "conversation",
        "history",
        "messages",
        default=[],
    )
    if isinstance(conversation, list):
        for index, turn in enumerate(conversation):
            if isinstance(turn, dict):
                raw_role = str(turn.get("speaker", turn.get("role", ""))).lower()
                role = "用户" if raw_role in {"user", "human"} else "AI"
                content = str(
                    turn.get("content", turn.get("text", turn.get("message", "")))
                ).strip()
                timestamp = str(
                    turn.get(
                        "timestamp",
                        turn.get("time", f"2026-06-{20 + index // 4:02d} 10:00"),
                    )
                )
            elif isinstance(turn, str):
                role = "用户" if index % 2 == 0 else "AI"
                content = turn.strip()
                timestamp = f"2026-06-{20 + index // 4:02d} 10:00"
            else:
                continue
            if content:
                history.append((role, content, timestamp))

    probes: list[Probe] = []
    questions = _first_present(
        sample,
        "questions",
        "probes",
        "test_cases",
        "probing_questions",
        default=[],
    )
    if isinstance(questions, list):
        for raw_question in questions:
            if not isinstance(raw_question, dict):
                continue
            question = str(
                raw_question.get(
                    "question", raw_question.get("query", raw_question.get("q", ""))
                )
            ).strip()
            if not question:
                continue
            expected = raw_question.get(
                "expected_keywords", raw_question.get("keywords", [])
            )
            if not expected:
                expected = raw_question.get(
                    "answer",
                    raw_question.get(
                        "ground_truth", raw_question.get("reference", "")
                    ),
                )
            if isinstance(expected, str):
                expected_keywords = [expected] if expected else []
            elif isinstance(expected, list):
                expected_keywords = [str(item) for item in expected]
            else:
                expected_keywords = []

            probes.append(
                {
                    "question": question,
                    "expected_keywords": expected_keywords,
                    "category": str(
                        raw_question.get("category", raw_question.get("type", "未分类"))
                    ),
                    "note": str(
                        raw_question.get("note", raw_question.get("description", ""))
                    ),
                }
            )

    portrait = str(
        sample.get("user_profile", sample.get("persona", sample.get("profile", "")))
    )
    return history, probes, portrait


def load_memora_data(
    data_dir: str | Path,
    persona: str = "software_engineer",
    period: str = "weekly",
) -> tuple[History, list[Probe], str]:
    """Load one Memora persona from its original directory layout."""

    base_dir = Path(data_dir) / period / persona
    conversation_dir = base_dir / "conversations"
    evaluation_file = base_dir / f"evaluation_questions_{persona}.json"

    if not conversation_dir.exists():
        raise FileNotFoundError(f"Memora conversation directory not found: {conversation_dir}")
    if not evaluation_file.exists():
        raise FileNotFoundError(f"Memora evaluation file not found: {evaluation_file}")

    history: History = []
    for session_file in sorted(conversation_dir.glob("session_*.json")):
        session = json.loads(session_file.read_text(encoding="utf-8"))
        date = str(session.get("date", "2025-06-01"))
        for turn in session.get("conversation", []):
            if not isinstance(turn, dict):
                continue
            role = "用户" if turn.get("speaker", "user") == "user" else "AI"
            message = str(turn.get("message", "")).strip()
            if message:
                history.append((role, message, date))

    evaluation = json.loads(evaluation_file.read_text(encoding="utf-8"))
    probes: list[Probe] = []
    for task_type in ("remembering", "reasoning", "recommending"):
        for raw_question in evaluation.get("questions", {}).get(task_type, []):
            evidence = raw_question.get("memory_evidence", {})
            keywords: list[str] = []
            if isinstance(evidence, dict):
                for value in evidence.values():
                    if isinstance(value, str):
                        keywords.append(value)
                    elif isinstance(value, list):
                        keywords.extend(str(item) for item in value)
            elif isinstance(evidence, str):
                keywords.append(evidence)

            probes.append(
                {
                    "question": str(raw_question.get("question", "")),
                    "expected_keywords": keywords[:5],
                    "category": task_type,
                    "note": f"{persona}, {raw_question.get('question_date', '')}",
                    "memory_evidence": evidence,
                }
            )

    return history, probes, persona.replace("_", " ")


def _first_present(mapping: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


DEMO_HISTORY: History = [
    ("用户", "你好，我叫小林，是计算机系大三的学生", "2026-06-20 10:00"),
    ("AI", "你好小林！计算机系大三，那应该有很多有趣的项目经验吧。", "2026-06-20 10:00"),
    ("用户", "平时喜欢打篮球和游泳，周末经常运动", "2026-06-20 10:05"),
    ("AI", "篮球和游泳都是很棒的运动！", "2026-06-20 10:05"),
    ("用户", "对了，我还特别喜欢吃辣的东西，川菜和火锅是我的最爱", "2026-06-20 10:15"),
    ("AI", "川菜和火锅确实很有魅力！", "2026-06-20 10:15"),
    ("用户", "最近期末考试要来了，压力很大", "2026-06-21 15:00"),
    ("AI", "考试加油！", "2026-06-21 15:00"),
    ("用户", "不用了，我想自己安排。推荐一本好点的参考书吧", "2026-06-21 15:05"),
    ("AI", "我推荐你看看《深度学习入门》，非常适合初学者。", "2026-06-21 15:05"),
    ("用户", "今天状态不太好，感觉很累", "2026-06-22 09:00"),
    ("AI", "注意休息。", "2026-06-22 09:00"),
    ("用户", "除了运动，还会听听音乐或者看看电影", "2026-06-22 09:05"),
    ("AI", "这些都是很好的放松方式。", "2026-06-22 09:05"),
    ("用户", "昨天去校医院了，医生说我的胃不太好", "2026-06-23 11:00"),
    ("AI", "啊，严重吗？", "2026-06-23 11:00"),
    ("用户", "不严重，但医生说让我少吃辛辣刺激的食物，建议清淡饮食", "2026-06-23 11:05"),
    ("AI", "那确实需要调整饮食习惯了。", "2026-06-23 11:05"),
    ("用户", "以后川菜和火锅可能要告别了", "2026-06-23 11:10"),
    ("AI", "健康第一！其实清淡的食物也有很多好吃的选择。", "2026-06-23 11:10"),
    ("用户", "今天去了学校附近一家粤菜馆，味道还不错", "2026-06-24 12:00"),
    ("AI", "看来清淡饮食也挺适合你的！", "2026-06-24 12:00"),
    ("用户", "白切鸡和蒸鱼挺好吃的", "2026-06-24 12:05"),
    ("AI", "这些菜健康又美味。", "2026-06-24 12:05"),
    ("用户", "游泳卡快到期了，要不要续费呢", "2026-06-24 18:00"),
    ("AI", "看你平时去的频率吧。", "2026-06-24 18:00"),
    ("用户", "我大概每周去两次", "2026-06-24 18:05"),
    ("AI", "那频率还挺高的，续费应该值得。", "2026-06-24 18:05"),
    ("用户", "对了，我之前好像说错了，我不是计算机系的", "2026-06-25 09:00"),
    ("AI", "啊？那你是什么专业的？", "2026-06-25 09:00"),
    ("用户", "我是软件工程专业的", "2026-06-25 09:05"),
    ("AI", "明白了，软件工程更偏向工程实践。", "2026-06-25 09:05"),
    ("用户", "暑假打算留在学校，准备参加一个AI竞赛", "2026-06-26 10:00"),
    ("AI", "好棒！", "2026-06-26 10:00"),
    ("用户", "想做一个图像识别的项目", "2026-06-26 10:05"),
    ("AI", "和你之前学的神经网络很契合！", "2026-06-26 10:05"),
]

DEMO_QUESTIONS: list[Probe] = [
    {"question": "我叫什么名字？", "expected_keywords": ["小林"], "category": "基础信息", "note": "Day 1"},
    {"question": "我是什么专业的学生？", "expected_keywords": ["软件工程"], "category": "错误修正", "note": "Day 6"},
    {"question": "我喜欢什么运动？", "expected_keywords": ["篮球", "游泳"], "category": "静态偏好", "note": "Day 1"},
    {"question": "我平时喜欢怎么放松？", "expected_keywords": ["音乐", "电影"], "category": "静态偏好", "note": "Day 3"},
    {"question": "你之前推荐过什么书给我？", "expected_keywords": ["深度学习入门"], "category": "事实记忆", "note": "Day 2"},
    {"question": "我每周游泳几次？", "expected_keywords": ["两次", "2次"], "category": "事实记忆", "note": "Day 5"},
    {"question": "我现在适合吃什么类型的食物？", "expected_keywords": ["清淡", "粤菜"], "category": "过期记忆", "note": "Day 4 后"},
    {"question": "我现在还能经常吃火锅吗？", "expected_keywords": ["不能", "不建议", "胃"], "category": "过期记忆", "note": "健康原因"},
    {"question": "我为什么改变饮食习惯？", "expected_keywords": ["胃", "医生", "健康"], "category": "过期记忆", "note": "Day 4"},
    {"question": "我最近吃过什么好吃的？", "expected_keywords": ["白切鸡", "蒸鱼", "粤菜"], "category": "新偏好", "note": "Day 5"},
    {"question": "暑假我打算做什么？", "expected_keywords": ["AI竞赛", "图像识别"], "category": "新偏好", "note": "Day 7"},
    {"question": "我之前说我是计算机系的，我实际是什么专业？", "expected_keywords": ["软件工程"], "category": "错误修正", "note": "Day 6"},
]


# Original public name retained for notebooks and scripts.
load_dataset = load_json_dataset

__all__ = [
    "History",
    "Probe",
    "ReproductionProbe",
    "ReproductionUser",
    "DEMO_HISTORY",
    "DEMO_QUESTIONS",
    "load_json_dataset",
    "load_dataset",
    "load_memora_data",
    "load_reproduction_dataset",
    "load_user_file",
]

# ── Phase 1 Reproduction Dataset ────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ReproductionUser:
    """One user in the Phase 1 reproduction dataset."""

    user_id: str
    profile: dict = field(default_factory=dict)
    days: list[dict] = field(default_factory=list)
    global_summary: str = ""
    user_portrait: str = ""
    global_memory_ids: dict[str, str] = field(default_factory=dict)


@dataclass
class ReproductionProbe:
    """One probing question with gold labels."""

    query_id: str
    user_id: str
    question: str
    reference_answer: str = ""
    gold_memory_ids: list[str] = field(default_factory=list)
    expected_keywords: list[str] = field(default_factory=list)
    question_type: str = "memory_recall"
    query_timestamp: str = ""


def load_reproduction_dataset(
    dataset_dir: str | Path,
) -> tuple[list[ReproductionUser], list[ReproductionProbe]]:
    """Load the Phase 1 reproduction dataset.

    Expects:
        {dataset_dir}/
        ├── users/
        │   ├── user_001.json
        │   └── ...
        └── probing_questions.jsonl

    Returns ``(users, probes)`` after format validation.
    """
    root = Path(dataset_dir)
    users_dir = root / "users"
    probes_path = root / "probing_questions.jsonl"

    if not users_dir.is_dir():
        raise FileNotFoundError(f"Users directory not found: {users_dir}")
    if not probes_path.is_file():
        raise FileNotFoundError(f"Probing questions not found: {probes_path}")

    users: list[ReproductionUser] = []
    for user_file in sorted(users_dir.glob("*.json")):
        users.append(load_user_file(user_file))

    user_ids = {u.user_id for u in users}
    probes: list[ReproductionProbe] = []
    with probes_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            probe = ReproductionProbe(
                query_id=str(data["query_id"]),
                user_id=str(data["user_id"]),
                question=str(data["question"]),
                reference_answer=str(data.get("reference_answer", "")),
                gold_memory_ids=[
                    str(mid) for mid in data.get("gold_memory_ids", [])
                ],
                expected_keywords=[
                    str(kw) for kw in data.get("expected_keywords", [])
                ],
                question_type=str(data.get("question_type", "memory_recall")),
                query_timestamp=str(data.get("query_timestamp", "")),
            )
            if probe.user_id not in user_ids:
                raise ValueError(
                    f"Probe '{probe.query_id}' references unknown user "
                    f"'{probe.user_id}'"
                )
            probes.append(probe)

    # Validate at least one negative question exists per user
    _validate_probe_types(probes, user_ids)

    return users, probes


def load_user_file(path: str | Path) -> ReproductionUser:
    """Load a single user JSON file into a ReproductionUser."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    user = ReproductionUser(
        user_id=str(data.get("user_id", "")),
        profile=data.get("profile", {}),
        days=data.get("days", []),
        global_summary=str(data.get("global_event_summary", "")),
        user_portrait=str(data.get("global_user_portrait", "")),
        global_memory_ids={
            str(key): str(value)
            for key, value in data.get("global_memory_ids", {}).items()
        },
    )
    if not user.user_id:
        raise ValueError(f"User file {path} missing 'user_id' field")
    return user


def _validate_probe_types(
    probes: list[ReproductionProbe],
    user_ids: set[str],
) -> None:
    """Check required question types exist for each user."""
    for uid in user_ids:
        user_probes = [p for p in probes if p.user_id == uid]
        types = {p.question_type for p in user_probes}
        if "negative_memory" not in types:
            print(
                f"[WARNING] User '{uid}' has no 'negative_memory' probe — "
                f"consider adding one"
            )
