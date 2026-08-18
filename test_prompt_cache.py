"""Tests for tiny-prompt-cache."""

import time
import pytest
from prompt_cache import PromptCache, cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == 1.0

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == -1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])


class TestPromptCache:
    def test_basic_get_set(self):
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.set("hello", "world")
        assert cache.get("hello") == "world"

    def test_get_or_call_miss_then_hit(self):
        cache = PromptCache(max_size=10, ttl_seconds=60)
        calls = []

        def getter():
            calls.append(1)
            return "response"

        result = cache.get_or_call("prompt", getter)
        assert result == "response"
        assert len(calls) == 1

        result = cache.get_or_call("prompt", getter)
        assert result == "response"
        assert len(calls) == 1  # Called only once

    def test_lru_eviction(self):
        cache = PromptCache(max_size=3, ttl_seconds=60)
        for i in range(5):
            cache.set(f"key{i}", f"value{i}")

        # First 2 should be evicted
        assert cache.get("key0") is None
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_ttl_expiry(self):
        cache = PromptCache(max_size=10, ttl_seconds=0.05)
        cache.set("k", "v")
        assert cache.get("k") == "v"
        time.sleep(0.1)
        assert cache.get("k") is None

    def test_invalidate(self):
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.set("k", "v")
        assert cache.invalidate("k") is True
        assert cache.get("k") is None
        assert cache.invalidate("k") is False

    def test_clear(self):
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_stats(self):
        cache = PromptCache(max_size=10, ttl_seconds=60)
        cache.get_or_call("p1", lambda: "r1")
        cache.get_or_call("p1", lambda: "r1")
        cache.get_or_call("p2", lambda: "r2")

        stats = cache.stats()
        assert stats["total"] == 3
        assert stats["cache_hits"] == 1
        assert stats["cache_misses"] == 2
        assert stats["hit_rate"] == pytest.approx(1 / 3)

    def test_semantic_requires_embedding_fn(self):
        with pytest.raises(ValueError):
            PromptCache(semantic_threshold=0.9)

    def test_semantic_match(self):
        mock_embeddings = {"a": [1.0, 0.0], "b": [0.99, 0.01]}

        cache = PromptCache(
            max_size=10,
            ttl_seconds=60,
            semantic_threshold=0.99,
            embedding_fn=lambda p: mock_embeddings.get(p, [0.0, 1.0]),
        )

        cache.set("a", "response_from_a")
        # "b" is semantically similar to "a"
        result = cache.get_or_call("b", lambda: "SHOULD NOT BE CALLED")
        assert result == "response_from_a"

    def test_semantic_threshold_not_met(self):
        mock_embeddings = {"a": [1.0, 0.0], "c": [0.0, 1.0]}

        cache = PromptCache(
            max_size=10,
            ttl_seconds=60,
            semantic_threshold=0.95,
            embedding_fn=lambda p: mock_embeddings.get(p, [0.5, 0.5]),
        )

        cache.set("a", "response_a")
        result = cache.get_or_call("c", lambda: "response_c")
        assert result == "response_c"  # Not similar enough, called the fn

    def test_thread_safety(self):
        import threading

        cache = PromptCache(max_size=100, ttl_seconds=60)
        results = []

        def worker(i):
            r = cache.get_or_call(f"key{i % 10}", lambda i=i: f"response_{i}")
            results.append(r)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 50
        # At least some hits since we only have 10 unique keys across 50 calls
        stats = cache.stats()
        assert stats["cache_hits"] > 0
