from statebudgetmem.baselines.memorybank import (
    ObsoleteDetector,
    calculate_outdated_memory_rate,
    label_demo_memory,
)


def test_staleness_metrics_and_demo_labels() -> None:
    assert label_demo_memory("用户最喜欢川菜和火锅") == "obsolete"
    result = calculate_outdated_memory_rate(
        [{"memory_id": "m1"}, {"memory_id": "m2"}],
        {"m1": "obsolete", "m2": "current"},
    )
    assert result["omr"] == 0.5
    assert result["cor"] == 0.5


def test_transition_detector_marks_older_topic_obsolete() -> None:
    detector = ObsoleteDetector()
    labels = detector.detect_transitions(
        [
            {"memory_id": "old", "content": "I like eating spicy food", "timestamp": 1},
            {"memory_id": "new", "content": "I no longer eat spicy food", "timestamp": 2},
        ]
    )
    assert labels["old"] == "obsolete"
    assert labels["new"] == "current"
