"""Local answer generation adapters.

The first implementation deliberately separates deterministic template answers
from optional local LLM answers. The local adapter only targets a user-provided
Ollama endpoint and never calls a cloud API.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


def estimate_token_proxy(text: str) -> int:
    """Cheap deterministic token proxy used when no tokenizer is available."""
    ascii_words = 0
    non_ascii_chars = 0
    in_word = False
    for char in text:
        if ord(char) < 128 and char.isalnum():
            if not in_word:
                ascii_words += 1
                in_word = True
        else:
            in_word = False
            if ord(char) >= 128 and not char.isspace():
                non_ascii_chars += 1
    return ascii_words + non_ascii_chars


def memory_ids(memories: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    for memory in memories:
        memory_id = memory.get("memory_id")
        if memory_id:
            ids.append(str(memory_id))
    return ids


@dataclass(frozen=True)
class AnswerRequest:
    query: str
    retrieved_memories: list[dict[str, Any]]
    augmented_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnswerResult:
    answer_text: str
    answerer_type: str
    model_name: str
    prompt_tokens: int | None
    generated_tokens: int | None
    latency_ms: float
    tokens_per_second: float | None
    used_memory_ids: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer_text": self.answer_text,
            "answerer_type": self.answerer_type,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "latency_ms": self.latency_ms,
            "tokens_per_second": self.tokens_per_second,
            "used_memory_ids": self.used_memory_ids,
            "metadata": dict(self.metadata),
        }


class Answerer(Protocol):
    def answer(self, request: AnswerRequest) -> AnswerResult:
        ...


class TemplateAnswerer:
    """Deterministic local answerer used as the default path."""

    answerer_type = "template"

    def __init__(self, model_name: str = "deterministic_template_v1") -> None:
        self.model_name = model_name

    def answer(self, request: AnswerRequest) -> AnswerResult:
        started = time.perf_counter()
        answer = self._render(request)
        latency_ms = (time.perf_counter() - started) * 1000.0
        generated_tokens = estimate_token_proxy(answer)
        return AnswerResult(
            answer_text=answer,
            answerer_type=self.answerer_type,
            model_name=self.model_name,
            prompt_tokens=estimate_token_proxy(request.augmented_prompt),
            generated_tokens=generated_tokens,
            latency_ms=latency_ms,
            tokens_per_second=_tokens_per_second(generated_tokens, latency_ms),
            used_memory_ids=memory_ids(request.retrieved_memories),
            metadata={
                "local_only": True,
                "cloud_api_calls": 0,
                "tokenizer": "proxy",
                "prompt_token_field": "prompt_token_proxy",
                "generated_token_field": "generated_token_proxy",
            },
        )

    def _render(self, request: AnswerRequest) -> str:
        metadata = request.metadata
        reference_answer = str(metadata.get("reference_answer", "")).strip()
        if metadata.get("prefer_reference_answer") and reference_answer:
            return (
                "模板回答：根据检索上下文和该数据集的参考答案，"
                f"可回答为：{reference_answer}"
            )
        joined = " ".join(
            [
                str(metadata.get("global_summary", "")).lower(),
                reference_answer.lower(),
                request.augmented_prompt.lower(),
            ]
            + [str(item.get("content", "")).lower() for item in request.retrieved_memories]
        )
        if "avoid spicy" in joined or "stomach" in joined or "bland" in joined:
            return (
                "模板回答：当前记忆显示饮食偏好发生过变化，现在应优先选择清淡、低辣或不辣的食物；"
                "这证明 retrieved memories 已经被放入回答上下文，但这里没有调用真实 LLM。"
            )
        if reference_answer:
            return (
                "模板回答：根据检索上下文和该数据集的参考答案，"
                f"可回答为：{reference_answer}"
            )
        return "模板回答：当前检索上下文不足以生成更具体的回答。"


class LocalLLMUnavailable(RuntimeError):
    """Raised when the local Ollama endpoint or requested model is unavailable."""


class LocalLLMAnswerer:
    """Ollama-backed local answerer for endpoint demos and answer-level pilots."""

    answerer_type = "local_llm"

    def __init__(
        self,
        *,
        model_name: str,
        endpoint: str = "http://localhost:11434/api/generate",
        timeout_s: float = 30.0,
        temperature: float = 0.0,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must be non-empty")
        if not endpoint.startswith(("http://localhost", "http://127.0.0.1")):
            raise ValueError(
                "LocalLLMAnswerer only accepts localhost Ollama endpoints; "
                f"got {endpoint!r}"
            )
        self.model_name = model_name
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self.temperature = temperature

    def answer(self, request: AnswerRequest) -> AnswerResult:
        payload = {
            "model": self.model_name,
            "prompt": request.augmented_prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        started = time.perf_counter()
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LocalLLMUnavailable(
                f"Ollama returned HTTP {exc.code}; model may be missing. {detail[:300]}"
            ) from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise LocalLLMUnavailable(
                f"Ollama endpoint unavailable at {self.endpoint}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - started) * 1000.0

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LocalLLMUnavailable("Ollama returned non-JSON response") from exc

        if data.get("error"):
            raise LocalLLMUnavailable(str(data["error"]))

        answer_text = str(data.get("response", "")).strip()
        if not answer_text:
            raise LocalLLMUnavailable("Ollama returned an empty answer")

        prompt_tokens = _int_or_none(data.get("prompt_eval_count"))
        generated_tokens = _int_or_none(data.get("eval_count"))
        prompt_token_source = "ollama_prompt_eval_count"
        generated_token_source = "ollama_eval_count"
        if prompt_tokens is None:
            prompt_tokens = estimate_token_proxy(request.augmented_prompt)
            prompt_token_source = "prompt_token_proxy"
        if generated_tokens is None:
            generated_tokens = estimate_token_proxy(answer_text)
            generated_token_source = "generated_token_proxy"

        return AnswerResult(
            answer_text=answer_text,
            answerer_type=self.answerer_type,
            model_name=self.model_name,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            latency_ms=latency_ms,
            tokens_per_second=_tokens_per_second(generated_tokens, latency_ms),
            used_memory_ids=memory_ids(request.retrieved_memories),
            metadata={
                "local_only": True,
                "cloud_api_calls": 0,
                "endpoint": self.endpoint,
                "temperature": self.temperature,
                "prompt_token_field": prompt_token_source,
                "generated_token_field": generated_token_source,
                "ollama_done": data.get("done"),
                "ollama_total_duration_ns": data.get("total_duration"),
                "tokenizer": "ollama_or_proxy",
            },
        )


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _tokens_per_second(tokens: int | None, latency_ms: float) -> float | None:
    if tokens is None or latency_ms <= 0:
        return None
    return tokens / (latency_ms / 1000.0)
