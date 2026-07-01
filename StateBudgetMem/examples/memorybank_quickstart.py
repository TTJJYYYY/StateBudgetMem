"""Original MemoryBank quick-start, moved from the root implementation."""

from statebudgetmem.baselines import MemoryBank


def main() -> None:
    memory = MemoryBank()
    memory.store_dialog("用户", "你好，我叫小明", "2026-06-20 10:00")
    memory.store_dialog("用户", "我喜欢打篮球和游泳", "2026-06-20 10:05")
    memory.store_dialog("用户", "我在准备期末考试，最近压力很大", "2026-06-21 15:00")
    memory.store_dialog("AI", "我推荐的是《深度学习入门》。", "2026-06-22 09:00")
    memory.update_user_portrait("用户小明，爱好篮球和游泳，近期在准备期末考试")

    results = memory.retrieve(
        "推荐一些运动相关的活动",
        top_k=3,
        current_time="2026-06-25 10:00",
    )
    for result in results:
        print(
            f"[{result['memory_type']}] {result['content']} "
            f"score={result['composite_score']:.3f}"
        )


if __name__ == "__main__":
    main()
