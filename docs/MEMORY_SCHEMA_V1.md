# Memory Schema V1

## Contract

`MemoryRecord` is the public memory input that algorithms may read. Each record represents one atomic state fact by default. If one source sentence contains multiple facts, split it into multiple `MemoryRecord` objects.

`subject` identifies who or what the memory is about. `attribute` identifies the state dimension being described. `value` stores the observed value for that attribute.

`dimensions` stores additional string dimensions that participate in state identity and condition checks, such as scope, location, device, health condition, or time-of-day condition. `metadata` stores extra information that does not participate in state matching.

`event_time` is when the source event happened. `valid_from` is when the state starts being valid. `valid_to` is when the state stops being valid. Validity intervals use left-closed, right-open semantics: `[valid_from, valid_to)`.

Raw inputs and gold annotations must be separated. Algorithms must not read answer fields such as gold status, gold operation, `supersedes`, or `temporarily_invalidates`. Evaluation code may use separate annotation records.

`status`, `supersedes`, and `temporarily_invalidates` remain in `MemoryRecord` only as legacy data-compatibility fields. New gold annotations should use `MemoryAnnotation`. Algorithm implementations must not read these legacy compatibility fields when making versioning, retrieval, or routing decisions.

Each module should use an Adapter to convert `MemoryRecord` into its own internal structure. Future schema changes should only add fields or add Adapters; they must not change existing field semantics.

## Examples

### Plain Unconditional State

```json
{
  "memory_id": "M1",
  "subject": "user",
  "attribute": "diet.preference.spicy",
  "value": "avoid",
  "text": "The user avoids spicy food.",
  "event_time": "2026-06-01",
  "valid_from": "2026-06-01",
  "valid_to": null,
  "status": "CURRENT",
  "memory_type": "preference",
  "importance": 0.8,
  "confidence": 0.9,
  "token_cost": 8,
  "dimensions": {},
  "supersedes": [],
  "temporarily_invalidates": [],
  "metadata": {
    "source": "note"
  }
}
```

### Conditional State With Multiple Dimensions

```json
{
  "memory_id": "M2",
  "subject": "user",
  "attribute": "transport.preference",
  "value": "subway",
  "text": "On weekday mornings in Shanghai, the user prefers the subway.",
  "event_time": "2026-06-10",
  "valid_from": "2026-06-10",
  "valid_to": null,
  "status": "CURRENT",
  "memory_type": "preference",
  "importance": 0.6,
  "confidence": 0.85,
  "token_cost": 12,
  "dimensions": {
    "city": "Shanghai",
    "day_type": "weekday",
    "time_of_day": "morning"
  },
  "supersedes": [],
  "temporarily_invalidates": [],
  "metadata": {
    "source": "conversation"
  }
}
```

### Multiple Facts From One Sentence

Source sentence: "The user lives in Hangzhou and works remotely on Fridays."

```json
{
  "memory_id": "M3",
  "subject": "user",
  "attribute": "location.home_city",
  "value": "Hangzhou",
  "text": "The user lives in Hangzhou.",
  "event_time": "2026-06-12",
  "valid_from": "2026-06-12",
  "valid_to": null,
  "status": "CURRENT",
  "memory_type": "profile",
  "importance": 0.7,
  "confidence": 0.9,
  "token_cost": 7,
  "dimensions": {},
  "supersedes": [],
  "temporarily_invalidates": [],
  "metadata": {
    "source_sentence_id": "S1"
  }
}
```

```json
{
  "memory_id": "M4",
  "subject": "user",
  "attribute": "work.location",
  "value": "remote",
  "text": "The user works remotely on Fridays.",
  "event_time": "2026-06-12",
  "valid_from": "2026-06-12",
  "valid_to": null,
  "status": "CURRENT",
  "memory_type": "schedule",
  "importance": 0.6,
  "confidence": 0.9,
  "token_cost": 7,
  "dimensions": {
    "day_of_week": "Friday"
  },
  "supersedes": [],
  "temporarily_invalidates": [],
  "metadata": {
    "source_sentence_id": "S1"
  }
}
```
