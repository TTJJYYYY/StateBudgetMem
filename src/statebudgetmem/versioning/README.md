# Versioning

This module turns already structured, atomic `MemoryRecord` objects into an auditable temporal version graph. It does **not** perform natural-language extraction. The current public contract assumes one `MemoryRecord` represents one atomic state fact.

## Pipeline

```text
MemoryRecord
  -> MemoryRecordAdapter
  -> StateObservation
  -> StructuredStateMatcher
  -> RuleBasedOperationClassifier
  -> VersionUpdater
  -> VersionGraphValidator
  -> VersionResolver
```

`VersioningEngine` orchestrates the full pipeline and applies updates atomically.

## Operations

- `ADD`: create an independent state node.
- `MERGE`: replace an exact state with a pre-computed merged state.
- `SUPERSEDE`: permanently replace an exact active state.
- `TEMP_INVALIDATE`: temporarily suppress an exact or broader active state.
- `RESTORE`: end a temporary override and create a new restored version.
- `DELETE`: delete matching state nodes and add an auditable deletion event.
- `NOOP`: preserve the graph because the input is a duplicate or has no effect.

## Version relations

- `SUPERSEDES`
- `TEMP_INVALIDATES`
- `RESTORES`
- `MERGES_INTO`
- `DELETES`

Edges always point from an older state node to the newer event/version node.

## Matcher

The matcher finds candidate old states only; it never chooses the update operation. The deterministic matcher uses:

- exact subject matching;
- exact attribute matching;
- scope comparison over `StateKey.dimensions`;
- event-time filtering;
- computed-status filtering;
- deterministic score, dimension-distance, time, and ID ordering.

The four scope relations are `EXACT_SLOT`, `BROADER_SCOPE`, `NARROWER_SCOPE`, and `COMPATIBLE_SCOPE`.

## Classifier

The default classifier is conservative. Without upstream hints it uses:

- equivalent exact active value -> `NOOP`;
- different exact active value -> `SUPERSEDE`;
- bounded/temporary observation with an active target -> `TEMP_INVALIDATE`;
- value returning to a temporarily invalidated source -> `RESTORE`;
- no exact active target -> `ADD`.

`MERGE` and `DELETE` require explicit observable intent because arbitrary string values cannot be merged or deleted safely from structure alone.

Optional observable metadata keys are:

- `versioning_intent`: one of the seven operation names;
- `versioning_target_ids`: explicit target IDs;
- `temporary`: boolean;
- `temporal_type`: `TEMPORARY` or `BOUNDED`;
- `restore_signal`: boolean;
- `delete_request`: boolean;
- `merge_request`: boolean;
- `versioning_delete_scope`: `exact` or `attribute`.

These are upstream semantic hints, not evaluation labels. The versioning module never reads `status`, `supersedes`, `temporarily_invalidates`, `MemoryAnnotation`, query gold IDs, `scenario_category`, or `state_role`.

## Temporal semantics

Intervals are left-closed and right-open: `[valid_from, valid_to)`. A temporary invalidation does not permanently close the underlying long-term state. The resolver suppresses it only while the temporary successor is active. When the temporary state expires, the long-term state becomes resolvable again automatically. An explicit recovery creates a new version linked by `RESTORES`.

## Main API

```python
engine = VersioningEngine()
engine.ingest(memory)
engine.ingest_many(memories)
engine.resolve_current(state_key, reference_time=...)
engine.resolve_at(state_key, reference_time)
engine.history(state_key)
engine.lineage(memory_id)
engine.current_view(reference_time=...)
engine.validate()
engine.snapshot()
```

All graph updates are performed on a candidate graph, validated, and committed only if no invariant error is found.
