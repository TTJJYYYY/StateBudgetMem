#!/usr/bin/env python3
"""Deterministic Memora -> MemoryRecord adapter for StateBudgetMem Versioning.

The released Memora sessions already contain structured ``operation`` and
``operation_details`` fields.  This adapter intentionally consumes those fields
instead of running another LLM extraction step, so it can be used to evaluate
and demonstrate the Versioning module in isolation.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

from statebudgetmem.schemas import MemoryRecord, MemoryStatus


_OPERATION_MAP = {
    "add": "ADD",
    "set": "ADD",
    "create": "ADD",
    "insert": "ADD",
    "update": "SUPERSEDE",
    "replace": "SUPERSEDE",
    "edit": "SUPERSEDE",
    "modify": "SUPERSEDE",
    "complete": "SUPERSEDE",
    "delete": "DELETE",
    "remove": "DELETE",
}

_ID_KEYS = (
    "id",
    "item_id",
    "task_id",
    "event_id",
    "record_id",
    "content_id",
    "goal_id",
    "preference_id",
    "uuid",
)

# Fields that frequently identify an item and normally survive an update.
_STABLE_KEYS = (
    "created_at",
    "task_type",
    "event_type",
    "expense_type",
    "amount",
    "name",
    "title",
    "description",
    "date",
    "start_date",
    "platform",
)

# Common wrappers used by update-shaped payloads.
_BEFORE_KEYS = ("old_item", "previous_item", "before", "old_value", "previous_value")
_AFTER_KEYS = ("new_item", "updated_item", "after", "new_value", "value")


@dataclass(slots=True)
class ConversionIssue:
    source_file: str
    session_id: str
    reason: str
    operation: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "session_id": self.session_id,
            "reason": self.reason,
            "operation": self.operation,
            "details": self.details,
        }


@dataclass(slots=True)
class ConversionResult:
    records: list[MemoryRecord]
    issues: list[ConversionIssue]
    scanned_sessions: int
    skipped_no_memory: int


class MemoraVersioningAdapter:
    """Convert released Memora conversation sessions into atomic MemoryRecord objects."""

    def __init__(self, *, subject_prefix: str = "memora") -> None:
        self.subject_prefix = subject_prefix.strip() or "memora"

    def convert_persona(
        self,
        data_dir: str | Path,
        *,
        period: str,
        persona: str,
        limit_sessions: int | None = None,
    ) -> ConversionResult:
        base_dir = Path(data_dir) / period / persona
        conversation_dir = base_dir / "conversations"
        if not conversation_dir.exists():
            raise FileNotFoundError(
                "Memora conversation directory not found: "
                f"{conversation_dir}. Pass the path to Memora/data via --memora-dir."
            )

        files = sorted(conversation_dir.glob("session_*.json"))
        if limit_sessions is not None:
            if limit_sessions < 1:
                raise ValueError("limit_sessions must be positive")
            files = files[:limit_sessions]

        records: list[MemoryRecord] = []
        issues: list[ConversionIssue] = []
        skipped_no_memory = 0

        for source_file in files:
            try:
                session = self._load_session(source_file)
                record, issue = self.convert_session(session, source_file=source_file)
            except Exception as exc:  # noqa: BLE001 - keep long dataset runs alive
                issues.append(
                    ConversionIssue(
                        source_file=str(source_file),
                        session_id=source_file.stem.removeprefix("session_"),
                        reason="session conversion failed",
                        details={
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                )
                continue

            if record is not None:
                records.append(record)
            elif issue is not None:
                if issue.reason == "no memory operation":
                    skipped_no_memory += 1
                else:
                    issues.append(issue)

        records.sort(key=lambda item: (item.event_time, item.memory_id))
        return ConversionResult(
            records=records,
            issues=issues,
            scanned_sessions=len(files),
            skipped_no_memory=skipped_no_memory,
        )

    def convert_session(
        self,
        session: dict[str, Any],
        *,
        source_file: str | Path = "<memory>",
    ) -> tuple[MemoryRecord | None, ConversionIssue | None]:
        source_path = str(source_file)
        session_id = str(session.get("session_id", session.get("id", "unknown")))
        raw_operation = session.get("operation")
        operation = str(raw_operation).strip().lower() if raw_operation is not None else ""
        details = session.get("operation_details") or {}
        if not operation:
            return None, ConversionIssue(
                source_file=source_path,
                session_id=session_id,
                reason="no memory operation",
                operation=None,
            )
        if operation not in _OPERATION_MAP:
            return None, ConversionIssue(
                source_file=source_path,
                session_id=session_id,
                reason="unsupported Memora operation",
                operation=operation,
                details=details if isinstance(details, dict) else {"raw": details},
            )
        if not isinstance(details, dict) or not details:
            return None, ConversionIssue(
                source_file=source_path,
                session_id=session_id,
                reason="missing operation_details",
                operation=operation,
            )

        persona = str(session.get("persona", "unknown_persona")).strip() or "unknown_persona"
        memory_type = self._normalize_memory_type(session.get("session_type", session.get("type")))
        category = self._category(session, details, memory_type)
        event_date = self._coerce_date(session.get("date"))

        state_payload = self._state_payload(details, operation)
        identity_payload = self._identity_payload(details, operation)
        item_key = self._item_key(identity_payload, details, memory_type, category)
        dimensions = self._dimensions(details, memory_type, item_key)

        text = self._memory_text(session)
        if not text:
            text = f"{operation.upper()} {category}: {self._display_value(state_payload)}"

        state_value = self._display_value(state_payload)
        memory_id = self._memory_id(persona, session_id, category, operation, state_payload)
        versioning_intent = _OPERATION_MAP[operation]

        metadata: dict[str, Any] = {
            "source": "memora",
            "source_file": source_path,
            "memora_session_id": session_id,
            "memora_session_type": memory_type,
            "memora_operation": operation,
            "memora_category": category,
            "operation_details": details,
            "versioning_intent": versioning_intent,
        }
        if versioning_intent == "DELETE":
            metadata["delete_request"] = True

        return (
            MemoryRecord(
                memory_id=memory_id,
                subject=f"{self.subject_prefix}:{persona}",
                attribute=category,
                value=state_value,
                text=text,
                event_time=event_date,
                valid_from=event_date,
                valid_to=None,
                status=MemoryStatus.CURRENT,
                memory_type=memory_type,
                importance=0.5,
                confidence=1.0,
                token_cost=max(1, (len(text) + 3) // 4),
                dimensions=dimensions,
                metadata=metadata,
            ),
            None,
        )

    @staticmethod
    def _load_session(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Memora session must be a JSON object: {path}")
        return payload

    @staticmethod
    def _normalize_memory_type(value: Any) -> str:
        raw = str(value or "memory").strip().lower()
        aliases = {
            "activity": "activity_memory",
            "preference": "preference_memory",
            "content": "content_memory",
            "goal": "goal_memory",
        }
        return aliases.get(raw, raw)

    @staticmethod
    def _category(
        session: dict[str, Any],
        details: dict[str, Any],
        memory_type: str,
    ) -> str:
        value = details.get("category", session.get("category"))
        if value is None:
            value = memory_type.removesuffix("_memory")
        normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip()).strip("_")
        return normalized or "uncategorized"

    @staticmethod
    def _coerce_date(value: Any) -> date:
        if isinstance(value, date):
            return value
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("Memora session is missing date")
        # Released files use YYYY-MM-DD.  The split keeps the adapter tolerant
        # of ISO timestamps without introducing a third-party date parser.
        try:
            return date.fromisoformat(raw[:10])
        except ValueError as exc:
            raise ValueError(f"invalid Memora session date: {raw!r}") from exc

    @staticmethod
    def _first_present(details: dict[str, Any], keys: Iterable[str]) -> Any | None:
        for key in keys:
            if key in details and details[key] is not None:
                return details[key]
        return None

    def _state_payload(self, details: dict[str, Any], operation: str) -> Any:
        if operation in {"update", "replace", "edit", "modify", "complete"}:
            after = self._first_present(details, _AFTER_KEYS)
            if after is not None:
                return after
        if "item" in details:
            item = details["item"]
            # Preference and content payloads carry important fields beside item.
            siblings = {
                key: value
                for key, value in details.items()
                if key not in {"category", "item"} and value is not None
            }
            if siblings:
                return {"item": item, **siblings}
            return item
        payload = {
            key: value
            for key, value in details.items()
            if key not in {"category"} and value is not None
        }
        return payload or details

    def _identity_payload(self, details: dict[str, Any], operation: str) -> Any:
        if operation in {"update", "replace", "edit", "modify", "complete"}:
            before = self._first_present(details, _BEFORE_KEYS)
            if before is not None:
                return before
        return details.get("item", self._state_payload(details, operation))

    def _item_key(
        self,
        payload: Any,
        details: dict[str, Any],
        memory_type: str,
        category: str,
    ) -> str | None:
        # Goals are modeled as one state slot per category, so updates supersede
        # earlier targets instead of creating one slot per numeric value.
        if memory_type == "goal_memory":
            return None

        if isinstance(payload, (str, int, float, bool)):
            return self._short_key(str(payload))

        if isinstance(payload, dict):
            for key in _ID_KEYS:
                if key in payload and payload[key] not in (None, ""):
                    return self._short_key(f"{key}:{payload[key]}")

            stable_parts = [
                f"{key}={self._compact(payload[key])}"
                for key in _STABLE_KEYS
                if key in payload and payload[key] not in (None, "")
            ]
            if stable_parts:
                return self._short_key("|".join(stable_parts))

        # Preference memories usually expose the preferred object as item and
        # its taxonomy as subcategory.  Content memories expose a string item id.
        if "subcategory" in details and details.get("item") is not None:
            return self._short_key(
                f"{details['subcategory']}:{self._compact(details['item'])}"
            )

        # Fallback: deterministic fingerprint.  This keeps add/delete pairs
        # matchable when their operation_details are repeated verbatim.
        fingerprint = self._compact(payload)
        if not fingerprint:
            fingerprint = f"{memory_type}:{category}"
        return self._short_key(fingerprint)

    @staticmethod
    def _dimensions(
        details: dict[str, Any],
        memory_type: str,
        item_key: str | None,
    ) -> dict[str, str]:
        dimensions: dict[str, str] = {"memory_type": memory_type}
        if item_key is not None:
            dimensions["item_key"] = item_key
        subcategory = details.get("subcategory")
        if subcategory not in (None, ""):
            dimensions["subcategory"] = str(subcategory)
        return dimensions

    @staticmethod
    def _memory_text(session: dict[str, Any]) -> str:
        messages: list[str] = []
        for turn in session.get("conversation", []):
            if not isinstance(turn, dict) or not turn.get("share_memory", False):
                continue
            message = str(turn.get("message", "")).strip()
            if message:
                messages.append(message)
        return " ".join(messages)

    @staticmethod
    def _display_value(value: Any) -> str:
        if isinstance(value, str):
            return value.strip() or "<empty>"
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _compact(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _short_key(value: str, *, max_prefix: int = 48) -> str:
        normalized = re.sub(r"\s+", " ", value.strip()).casefold()
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
        prefix = re.sub(r"[^a-z0-9_.-]+", "_", normalized)[:max_prefix].strip("_")
        return f"{prefix or 'item'}-{digest}"

    @staticmethod
    def _memory_id(
        persona: str,
        session_id: str,
        category: str,
        operation: str,
        payload: Any,
    ) -> str:
        seed = json.dumps(
            {
                "persona": persona,
                "session_id": session_id,
                "category": category,
                "operation": operation,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
        safe_persona = re.sub(r"[^a-zA-Z0-9_-]+", "_", persona)
        return f"memora_{safe_persona}_{int(session_id):04d}_{digest}" if session_id.isdigit() else f"memora_{safe_persona}_{session_id}_{digest}"


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_memory_records(path: str | Path) -> Iterator[MemoryRecord]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield MemoryRecord.model_validate_json(line)
            except Exception as exc:  # noqa: BLE001 - enrich file/line context
                raise ValueError(f"invalid MemoryRecord at {source}:{line_number}: {exc}") from exc

_UPDATE_OPERATIONS = {"update", "replace", "edit", "modify", "complete"}
_ADD_OPERATIONS = {"add", "set", "create", "insert"}
_DELETE_OPERATIONS = {"delete", "remove"}

_TEMPORARY_CUES = (
    "temporary",
    "temporarily",
    "for now",
    "until ",
    "for the next",
    "for two weeks",
    "for a week",
    "for a month",
    "暂时",
    "临时",
    "这几天",
    "这两周",
    "未来两周",
    "直到",
)
_RESTORE_CUES = (
    "restore",
    "resume",
    "back to",
    "return to",
    "again",
    "recovered",
    "no longer",
    "恢复",
    "重新",
    "回到",
    "已经好了",
    "不再",
)
_DATE_KEYS = (
    "valid_to",
    "end_date",
    "until",
    "temporary_until",
    "expires_at",
    "expiry_date",
)


@dataclass(slots=True)
class _SlotState:
    current: MemoryRecord | None = None
    restore_source: MemoryRecord | None = None
    temporary: MemoryRecord | None = None


class SemanticMemoraVersioningAdapter:
    """Refine Memora records into conservative Versioning intents.

    The released Memora operation is preserved in metadata.  The refined intent
    is based on observable payload changes and temporal/restore cues.  Low-
    confidence cases are marked with ``adapter_requires_review``.
    """

    def __init__(self, *, subject_prefix: str = "memora") -> None:
        self.base = MemoraVersioningAdapter(subject_prefix=subject_prefix)

    def convert_persona(
        self,
        data_dir: str | Path,
        *,
        period: str,
        persona: str,
        limit_sessions: int | None = None,
    ) -> ConversionResult:
        result = self.base.convert_persona(
            data_dir,
            period=period,
            persona=persona,
            limit_sessions=limit_sessions,
        )
        return ConversionResult(
            records=self.refine_records(result.records),
            issues=result.issues,
            scanned_sessions=result.scanned_sessions,
            skipped_no_memory=result.skipped_no_memory,
        )

    def refine_records(self, records: Iterable[MemoryRecord]) -> list[MemoryRecord]:
        slots: dict[tuple[str, str, tuple[tuple[str, str], ...]], _SlotState] = {}
        refined: list[MemoryRecord] = []

        for record in sorted(records, key=lambda item: (item.event_time, item.memory_id)):
            key = self._slot_key(record)
            slot = slots.setdefault(key, _SlotState())
            updated, intent = self._refine_record(record, slot)
            refined.append(updated)
            self._advance_slot(slot, updated, intent)

        return refined

    def _refine_record(
        self,
        record: MemoryRecord,
        slot: _SlotState,
    ) -> tuple[MemoryRecord, str]:
        metadata = dict(record.metadata)
        raw_operation = str(metadata.get("memora_operation", "")).strip().casefold()
        details = metadata.get("operation_details")
        if not isinstance(details, dict):
            details = {}
        text = record.text or ""
        valid_to = self._extract_valid_to(details, text, record.event_time)

        intent, targets, reason, confidence, needs_review = self._infer(
            record=record,
            raw_operation=raw_operation,
            details=details,
            text=text,
            slot=slot,
            valid_to=valid_to,
        )

        metadata.update(
            {
                "adapter_mode": "semantic",
                "adapter_inferred_intent": intent,
                "adapter_inference_reason": reason,
                "adapter_inference_confidence": confidence,
                "adapter_requires_review": needs_review,
                "versioning_intent": intent,
            }
        )
        if targets:
            metadata["versioning_target_ids"] = targets
        else:
            metadata.pop("versioning_target_ids", None)

        if intent == "DELETE":
            metadata["delete_request"] = True
        else:
            metadata.pop("delete_request", None)
        if intent == "MERGE":
            metadata["merge_request"] = True
        else:
            metadata.pop("merge_request", None)
        if intent == "TEMP_INVALIDATE":
            metadata["temporary"] = True
            metadata["temporal_type"] = "BOUNDED" if valid_to else "TEMPORARY"
        else:
            metadata.pop("temporary", None)
            metadata.pop("temporal_type", None)
        if intent == "RESTORE":
            metadata["restore_signal"] = True
        else:
            metadata.pop("restore_signal", None)

        return (
            record.model_copy(
                deep=True,
                update={
                    "valid_to": valid_to if intent == "TEMP_INVALIDATE" else record.valid_to,
                    "metadata": metadata,
                },
            ),
            intent,
        )

    def _infer(
        self,
        *,
        record: MemoryRecord,
        raw_operation: str,
        details: dict[str, Any],
        text: str,
        slot: _SlotState,
        valid_to: date | None,
    ) -> tuple[str, list[str], str, float, bool]:
        current = slot.current
        normalized_new = self._normalized_value(record.value)

        if raw_operation in _DELETE_OPERATIONS:
            if current is None:
                return (
                    "NOOP",
                    [],
                    "Memora requested deletion, but no active exact-slot state exists",
                    0.90,
                    True,
                )
            return (
                "DELETE",
                [current.memory_id],
                "Memora explicitly requested deletion of the active state",
                0.99,
                False,
            )

        if current is not None and normalized_new == self._normalized_value(current.value):
            return (
                "NOOP",
                [current.memory_id],
                "Incoming value is equivalent to the active exact-slot value",
                0.99,
                False,
            )

        restore_cue = self._contains_any(text, _RESTORE_CUES) or self._contains_any(
            json.dumps(details, ensure_ascii=False), _RESTORE_CUES
        )
        if slot.restore_source is not None and slot.temporary is not None:
            returns_to_source = normalized_new == self._normalized_value(
                slot.restore_source.value
            )
            if restore_cue or returns_to_source:
                return (
                    "RESTORE",
                    [slot.temporary.memory_id, slot.restore_source.memory_id],
                    "Incoming state restores the value that existed before a temporary override",
                    0.97 if returns_to_source else 0.88,
                    not returns_to_source,
                )

        temporary_cue = (
            valid_to is not None
            or self._contains_any(text, _TEMPORARY_CUES)
            or self._contains_any(json.dumps(details, ensure_ascii=False), _TEMPORARY_CUES)
            or bool(details.get("temporary", False))
        )
        if temporary_cue and current is not None:
            return (
                "TEMP_INVALIDATE",
                [current.memory_id],
                "Bounded or explicitly temporary information overrides the active state",
                0.95 if valid_to else 0.82,
                valid_to is None,
            )

        if current is not None and self._is_strict_extension(current.value, record.value):
            return (
                "MERGE",
                [current.memory_id],
                "New payload preserves the active payload and adds information",
                0.94,
                False,
            )

        if current is None:
            review = raw_operation in _UPDATE_OPERATIONS
            return (
                "ADD",
                [],
                "No active exact-slot state exists; create an independent state",
                0.99 if raw_operation in _ADD_OPERATIONS else 0.76,
                review,
            )

        if raw_operation in _UPDATE_OPERATIONS:
            return (
                "SUPERSEDE",
                [current.memory_id],
                "Memora update changes an existing exact-slot value without merge/temporary evidence",
                0.96,
                False,
            )

        if raw_operation in _ADD_OPERATIONS:
            return (
                "SUPERSEDE",
                [current.memory_id],
                "A new value was written into an already occupied exact state slot",
                0.82,
                True,
            )

        return (
            "ADD",
            [],
            "Unsupported or missing source label; preserve the record as an independent state",
            0.60,
            True,
        )

    @staticmethod
    def _advance_slot(slot: _SlotState, record: MemoryRecord, intent: str) -> None:
        if intent == "NOOP":
            return
        if intent == "DELETE":
            slot.current = None
            slot.restore_source = None
            slot.temporary = None
            return
        if intent == "TEMP_INVALIDATE":
            slot.restore_source = slot.current
            slot.temporary = record
            slot.current = record
            return
        if intent == "RESTORE":
            slot.current = record
            slot.restore_source = None
            slot.temporary = None
            return
        slot.current = record
        slot.restore_source = None
        slot.temporary = None

    @staticmethod
    def _slot_key(record: MemoryRecord) -> tuple[str, str, tuple[tuple[str, str], ...]]:
        return (record.subject, record.attribute, tuple(sorted(record.dimensions.items())))

    @staticmethod
    def _normalized_value(value: str) -> str:
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return re.sub(r"\s+", " ", str(value).strip()).casefold()
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _is_strict_extension(cls, old_value: str, new_value: str) -> bool:
        try:
            old_payload = json.loads(old_value)
            new_payload = json.loads(new_value)
        except (json.JSONDecodeError, TypeError):
            return False
        if not isinstance(old_payload, dict) or not isinstance(new_payload, dict):
            return False
        if len(new_payload) <= len(old_payload):
            return False
        return all(key in new_payload and new_payload[key] == value for key, value in old_payload.items())

    @classmethod
    def _extract_valid_to(
        cls,
        details: dict[str, Any],
        text: str,
        event_time: date,
    ) -> date | None:
        for key in _DATE_KEYS:
            value = cls._find_nested(details, key)
            parsed = cls._parse_date(value)
            if parsed is not None and parsed >= event_time:
                return parsed

        duration_days = cls._find_nested(details, "duration_days")
        if isinstance(duration_days, (int, float)) and duration_days > 0:
            return event_time + timedelta(days=int(duration_days))

        lowered = text.casefold()
        duration_match = re.search(
            r"for\s+(?:the\s+next\s+)?(\d+)\s*(day|days|week|weeks|month|months)",
            lowered,
        )
        if duration_match:
            amount = int(duration_match.group(1))
            unit = duration_match.group(2)
            multiplier = 1 if unit.startswith("day") else 7 if unit.startswith("week") else 30
            return event_time + timedelta(days=amount * multiplier)

        zh_match = re.search(r"(?:未来|接下来)?\s*(\d+)\s*(天|周|个月|月)", text)
        if zh_match:
            amount = int(zh_match.group(1))
            unit = zh_match.group(2)
            multiplier = 1 if unit == "天" else 7 if unit == "周" else 30
            return event_time + timedelta(days=amount * multiplier)
        return None

    @staticmethod
    def _find_nested(payload: Any, key: str) -> Any | None:
        if isinstance(payload, dict):
            if key in payload:
                return payload[key]
            for value in payload.values():
                found = SemanticMemoraVersioningAdapter._find_nested(value, key)
                if found is not None:
                    return found
        elif isinstance(payload, list):
            for value in payload:
                found = SemanticMemoraVersioningAdapter._find_nested(value, key)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        try:
            return date.fromisoformat(raw[:10])
        except ValueError:
            return None

    @staticmethod
    def _contains_any(text: str, cues: Iterable[str]) -> bool:
        lowered = text.casefold()
        return any(cue.casefold() in lowered for cue in cues)


def build_full_capability_records(
    *,
    subject: str = "controlled:demo_user",
) -> list[MemoryRecord]:
    """Return a deterministic seven-step scenario covering every operation."""

    def make(
        memory_id: str,
        value: str,
        event_time: str,
        *,
        valid_to: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        text = value.replace("_", " ")
        return MemoryRecord(
            memory_id=memory_id,
            subject=subject,
            attribute="diet.preference",
            value=value,
            text=text,
            event_time=date.fromisoformat(event_time),
            valid_from=date.fromisoformat(event_time),
            valid_to=date.fromisoformat(valid_to) if valid_to else None,
            status=MemoryStatus.CURRENT,
            memory_type="controlled_showcase",
            importance=0.9,
            confidence=1.0,
            token_cost=max(1, (len(text) + 3) // 4),
            dimensions={"scope": "default"},
            metadata={
                "source": "controlled_full_capability_showcase",
                "controlled": True,
                **(metadata or {}),
            },
        )

    return [
        make("showcase_add", "accepts_mild_spicy", "2026-01-01"),
        make("showcase_noop", " accepts_mild_spicy ", "2026-01-05"),
        make(
            "showcase_merge",
            "accepts_mild_spicy_and_prefers_low_oil",
            "2026-01-10",
            metadata={"merge_request": True},
        ),
        make("showcase_supersede", "vegetarian", "2026-02-01"),
        make(
            "showcase_temp",
            "temporarily_avoid_dairy",
            "2026-02-10",
            valid_to="2026-03-01",
        ),
        make("showcase_restore", "vegetarian", "2026-02-20"),
        make(
            "showcase_delete",
            "delete_diet_preference",
            "2026-03-10",
            metadata={"delete_request": True},
        ),
    ]

__all__ = [
    "ConversionIssue",
    "ConversionResult",
    "MemoraVersioningAdapter",
    "SemanticMemoraVersioningAdapter",
    "build_full_capability_records",
    "read_memory_records",
    "write_jsonl",
]
