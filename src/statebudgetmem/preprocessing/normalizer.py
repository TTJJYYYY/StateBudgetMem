from __future__ import annotations

import re
import unicodedata

_SPACE_RE = re.compile(r"\s+")
_SPLIT_RE = re.compile(r"[。；;!?！？\n]+")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

_FILLERS = [
    "嗯",
    "呃",
    "额",
    "其实",
    "说实话",
    "老实说",
    "我觉得",
    "我感觉",
    "就是",
    "然后",
]

_ATTRIBUTE_ALIASES = {
    "住处": "home_location",
    "住址": "home_location",
    "地址": "home_location",
    "城市": "home_location",
    "居住地": "home_location",
    "早餐": "breakfast",
    "早饭": "breakfast",
    "午餐": "lunch",
    "午饭": "lunch",
    "晚餐": "dinner",
    "晚饭": "dinner",
    "过敏": "allergy",
    "公司": "company",
    "学校": "school",
    "工作": "job",
}


def normalize_text(text: str) -> str:
    """轻量文本清洗，不改变原意。"""
    cleaned = unicodedata.normalize("NFKC", text).strip()
    for filler in _FILLERS:
        cleaned = cleaned.replace(filler, "")
    cleaned = _SPACE_RE.sub(" ", cleaned)
    return cleaned.strip(" ,，.;；!?！？")


def split_clauses(text: str) -> list[str]:
    """把一条长文本切成若干可抽取的小句。"""
    text = normalize_text(text)
    clauses: list[str] = []

    for part in _SPLIT_RE.split(text):
        part = part.strip(" ,，")
        if not part:
            continue

        if any(marker in part for marker in ["现在", "目前", "以前", "之前", "改成", "改为", "换成", "搬到"]):
            clauses.extend(item.strip(" ,，") for item in re.split(r"[,，]", part) if item.strip(" ,，"))
        else:
            clauses.append(part)

    return clauses


def clean_value(value: str | None) -> str:
    """清洗抽取出的 value。"""
    if value is None:
        return "unknown"

    value = normalize_text(value)
    value = re.sub(r"^(是|为|在|到|成|去|了|改成|改为|换成)", "", value)
    value = re.sub(r"(了|啦|呢|。)$", "", value)
    return value.strip(" ,，.;；!?！？") or "unknown"


def canonical_attribute(attribute: str) -> str:
    """把自然语言属性名统一成稳定 key。"""
    attribute = normalize_text(attribute)

    if attribute in _ATTRIBUTE_ALIASES:
        return _ATTRIBUTE_ALIASES[attribute]

    attribute = attribute.lower()
    attribute = re.sub(r"[^\w\u4e00-\u9fff]+", "_", attribute)
    return attribute.strip("_") or "attribute"


def infer_memory_type(attribute: str) -> str:
    """根据 attribute 粗略判断 memory_type。"""
    if attribute == "allergy":
        return "health"
    if attribute in {"preference", "food_preference", "drink_preference"}:
        return "preference"
    if attribute in {"breakfast", "lunch", "dinner", "habit"}:
        return "habit"
    return "profile"


def estimate_token_cost(text: str) -> int:
    """不用 tokenizer 的简易 token 估算，保证离线可跑。"""
    return max(1, len(_CJK_RE.findall(text)) + len(_WORD_RE.findall(text)))
