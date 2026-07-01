"""Tests for TfidfRetriever."""

import pytest

from statebudgetmem.retrieval.tfidf import TfidfRetriever


class TestTfidfRetriever:

    @pytest.fixture
    def retriever(self):
        return TfidfRetriever()

    def test_empty_items(self, retriever):
        assert retriever.retrieve("hello", [], top_k=5) == []

    def test_top_k_zero(self, retriever):
        items = [{"memory_id": "1", "text": "hello world"}]
        assert retriever.retrieve("hello", items, top_k=0) == []

    def test_exact_match_top(self, retriever):
        items = [
            {"memory_id": "a", "text": "apple banana"},
            {"memory_id": "b", "text": "cat dog"},
            {"memory_id": "c", "text": "apple pie"},
        ]
        result = retriever.retrieve("apple", items, top_k=2)
        assert len(result) == 2
        scores = [s for _, _, s in result]
        assert scores[0] >= scores[1]

    def test_score_range(self, retriever):
        items = [{"memory_id": "1", "text": "hello world"}]
        _, _, score = retriever.retrieve("hello", items)[0]
        assert 0.0 <= score <= 1.0

    def test_deterministic(self, retriever):
        items = [
            {"memory_id": "1", "text": "a b c"},
            {"memory_id": "2", "text": "d e f"},
            {"memory_id": "3", "text": "a b d"},
        ]
        r1 = retriever.retrieve("a", items)
        r2 = retriever.retrieve("a", items)
        assert r1 == r2

    def test_chinese_text(self, retriever):
        items = [
            {"memory_id": "1", "text": "我喜欢吃辣的食物"},
            {"memory_id": "2", "text": "猫和狗是宠物"},
        ]
        result = retriever.retrieve("喜欢吃辣", items)
        assert len(result) > 0
        assert result[0][1]["memory_id"] == "1"

    def test_content_key_fallback(self, retriever):
        items = [{"memory_id": "1", "content": "hello world", "text": ""}]
        result = retriever.retrieve("hello", items)
        assert len(result) == 1

    def test_empty_query(self, retriever):
        items = [{"memory_id": "1", "text": "hello"}]
        result = retriever.retrieve("", items)
        assert len(result) <= 1

    def test_single_item(self, retriever):
        items = [{"memory_id": "only", "text": "the only item"}]
        result = retriever.retrieve("anything", items)
        assert len(result) == 1
