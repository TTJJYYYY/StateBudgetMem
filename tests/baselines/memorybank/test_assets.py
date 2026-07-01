from pathlib import Path

from statebudgetmem.baselines.memorybank.datasets import DEMO_HISTORY, DEMO_QUESTIONS


ROOT = Path(__file__).resolve().parents[3]


def test_original_main_demo_dataset_is_complete() -> None:
    assert len(DEMO_HISTORY) == 36
    assert len(DEMO_QUESTIONS) == 12
    assert any("计算机系" in question["question"] for question in DEMO_QUESTIONS)


def test_full_gradio_demo_content_is_preserved() -> None:
    demo_text = (
        ROOT / "src/statebudgetmem/baselines/memorybank/demo.py"
    ).read_text(encoding="utf-8")
    assert "PRESET_ANSWERS" in demo_text
    assert "无记忆基线" in demo_text
    assert "MemoryBank 增强" in demo_text
