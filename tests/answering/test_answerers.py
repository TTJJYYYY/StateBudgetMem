from __future__ import annotations

import pytest

from statebudgetmem.answering import (
    AnswerRequest,
    LocalLLMAnswerer,
    LocalLLMUnavailable,
    TemplateAnswerer,
)


def test_template_answerer_returns_local_metrics() -> None:
    request = AnswerRequest(
        query="What should I eat tonight?",
        retrieved_memories=[
            {
                "memory_id": "m1",
                "content": "User has stomach discomfort and should avoid spicy food.",
            }
        ],
        augmented_prompt="Use memory m1 to answer. The user should avoid spicy food.",
        metadata={"reference_answer": "Eat bland food.", "global_summary": "stomach"},
    )

    result = TemplateAnswerer().answer(request)

    assert result.answerer_type == "template"
    assert result.model_name == "deterministic_template_v1"
    assert result.prompt_tokens is not None
    assert result.generated_tokens is not None
    assert result.latency_ms >= 0
    assert result.used_memory_ids == ["m1"]
    assert result.metadata["local_only"] is True
    assert result.metadata["cloud_api_calls"] == 0


def test_local_llm_rejects_non_localhost_endpoint() -> None:
    with pytest.raises(ValueError, match="localhost"):
        LocalLLMAnswerer(
            model_name="demo",
            endpoint="https://api.example.com/v1/chat/completions",
        )


def test_local_llm_unavailable_raises_clear_error() -> None:
    answerer = LocalLLMAnswerer(
        model_name="missing-model",
        endpoint="http://127.0.0.1:9/api/generate",
        timeout_s=0.1,
    )

    with pytest.raises(LocalLLMUnavailable, match="Ollama endpoint unavailable"):
        answerer.answer(
            AnswerRequest(
                query="hello",
                retrieved_memories=[],
                augmented_prompt="hello",
            )
        )
