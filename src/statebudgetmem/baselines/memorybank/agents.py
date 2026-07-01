"""Small agent wrappers used by MemoryBank comparison experiments.

The wrappers deliberately keep generation separate from memory management: an
LLM is injected as ``Callable[[str], str]`` and the memory backend is injected
through a minimal protocol.  This makes the experiment code reusable with the
FAISS MemoryBank implementation as well as future StateBudgetMem pipelines.
"""

from __future__ import annotations

from typing import Callable, Protocol


LLMCaller = Callable[[str], str]


class PromptMemory(Protocol):
    """Memory backend capabilities required by :class:`MemoryAugmentedAgent`."""

    def build_augmented_prompt(
        self,
        query: str,
        current_time: str | None = None,
        top_k: int = 5,
    ) -> dict:
        ...

    def add(self, messages: list[tuple[str, str, str]], **kwargs) -> list[str]:
        ...

    def get_stats(self) -> dict:
        ...


class BaselineAgent:
    """Stateless generation baseline used in paired evaluations."""

    def __init__(self, llm_caller: LLMCaller | None = None) -> None:
        self.llm_caller = llm_caller or (lambda _: "[placeholder response]")
        self.dialog_history: list[dict] = []

    def chat(self, user_input: str, timestamp: str | None = None) -> str:
        prompt = f"请回答用户的问题：\n\n{user_input}"
        response = self.llm_caller(prompt)
        self.dialog_history.append(
            {"user": user_input, "ai": response, "timestamp": timestamp}
        )
        return response


class MemoryAugmentedAgent:
    """Retrieve memory, augment a prompt, generate, then store the new turn."""

    def __init__(
        self,
        memory_bank: PromptMemory,
        llm_caller: LLMCaller | None = None,
    ) -> None:
        self.memory = memory_bank
        self.llm_caller = llm_caller or (lambda _: "[placeholder response]")
        self.dialog_history: list[dict] = []

    def chat(self, user_input: str, timestamp: str | None = None) -> str:
        context = self.memory.build_augmented_prompt(
            query=user_input,
            current_time=timestamp,
            top_k=5,
        )
        prompt = str(context.get("prompt_template", user_input))
        response = self.llm_caller(prompt)

        store_dialog = getattr(self.memory, "store_dialog", None)
        if callable(store_dialog):
            store_dialog("用户", user_input, timestamp)
            store_dialog("AI", response, timestamp)
        else:
            self.memory.add(
                [
                    ("用户", user_input, timestamp or ""),
                    ("AI", response, timestamp or ""),
                ]
            )
        self.dialog_history.append(
            {
                "user": user_input,
                "ai": response,
                "retrieved_memories": int(context.get("retrieved_count", 0)),
                "timestamp": timestamp,
            }
        )
        return response

    def batch_store_history(self, dialogs: list[tuple[str, str, str]]) -> list[str]:
        store_dialog = getattr(self.memory, "store_dialog", None)
        if callable(store_dialog):
            ids: list[str] = []
            for role, content, timestamp in dialogs:
                memory = store_dialog(role, content, timestamp)
                memory_id = getattr(memory, "memory_id", "")
                if memory_id:
                    ids.append(memory_id)
            return ids
        return self.memory.add(dialogs)

    def get_memory_stats(self) -> dict:
        return self.memory.get_stats()
