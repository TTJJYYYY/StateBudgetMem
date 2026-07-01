from statebudgetmem.interfaces import MemoryRecord, UpdateOperation, VersionManager
from statebudgetmem.schemas import MemoryStatus
from statebudgetmem.versioning import StateKey, VersioningEngine


def make_record(
    memory_id: str,
    value: str,
    event_time: str,
    *,
    operation_hint: str | None = None,
) -> MemoryRecord:
    metadata = {}
    if operation_hint is not None:
        metadata["operation_hint"] = operation_hint
    return MemoryRecord(
        memory_id=memory_id,
        subject="user",
        attribute="diet.spice",
        value=value,
        text=value,
        event_time=event_time,
        status=MemoryStatus.CURRENT,
        memory_type="state",
        importance=1.0,
        confidence=1.0,
        token_cost=8,
        metadata=metadata,
    )


def main() -> None:
    engine: VersionManager = VersioningEngine()
    engine.ingest(make_record("m1", "no_spicy_food", "2026-01-01"))
    update = engine.ingest(
        make_record(
            "m2",
            "mild_spicy_food_is_ok",
            "2026-03-01",
            operation_hint="supersede",
        )
    )

    assert update.results[0].decision.operation is UpdateOperation.SUPERSEDE
    key = StateKey(subject="user", attribute="diet.spice")
    print(engine.resolve_current(key))
    print(engine.history(key))


if __name__ == "__main__":
    main()
