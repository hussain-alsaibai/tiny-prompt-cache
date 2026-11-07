"""Tests for tiny_prompt_cache. Run with: python -m pytest test_tiny_prompt_cache.py -v

or, if pytest isn't installed:
    python -m unittest test_tiny_prompt_cache.py -v
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from tiny_prompt_cache import (
    CacheStore,
    FileCacheStore,
    InMemoryCacheStore,
    InvalidConfigError,
    PromptCache,
    PromptCacheError,
    StoreError,
    estimate_tokens,
    get_default_cache,
)


def _echo(text: str) -> str:
    """A no-op LLM stand-in: returns the input verbatim with a marker."""
    return f"echo:{text}"


def _slow_echo(text: str) -> str:
    time.sleep(0.001)
    return _echo(text)


class TestKeyGeneration(unittest.TestCase):
    def test_same_prompts_same_key(self) -> None:
        cache = PromptCache(prefix_mode="prefix_chars", prefix_length=200)
        k1 = cache.make_key("sys", "user")
        k2 = cache.make_key("sys", "user")
        self.assertEqual(k1, k2)

    def test_different_system_different_key(self) -> None:
        cache = PromptCache(prefix_mode="prefix_chars", prefix_length=200)
        self.assertNotEqual(
            cache.make_key("sys-a", "user"),
            cache.make_key("sys-b", "user"),
        )

    def test_system_only_mode(self) -> None:
        cache = PromptCache(prefix_mode="system_only")
        # User message content is ignored entirely in this mode.
        self.assertEqual(
            cache.make_key("sys", "user-aaaaaaaaaa"),
            cache.make_key("sys", "user-bbbbbbbbbb"),
        )

    def test_prefix_chars_truncates(self) -> None:
        cache = PromptCache(prefix_mode="prefix_chars", prefix_length=5)
        # Both messages share the first 5 chars -> same key.
        self.assertEqual(
            cache.make_key("sys", "hello world"),
            cache.make_key("sys", "hello there"),
        )
        # Differ on the first 5 chars -> different keys.
        self.assertNotEqual(
            cache.make_key("sys", "hello world"),
            cache.make_key("sys", "howdy world"),
        )

    def test_prefix_words_truncates(self) -> None:
        cache = PromptCache(prefix_mode="prefix_words", prefix_length=3)
        # First 3 words are shared -> same key.
        self.assertEqual(
            cache.make_key("sys", "one two three four"),
            cache.make_key("sys", "one two three five"),
        )
        # Differ on a word in the prefix -> different keys.
        self.assertNotEqual(
            cache.make_key("sys", "one two three four"),
            cache.make_key("sys", "one two OTHER three"),
        )

    def test_prefix_modes_are_disjoint(self) -> None:
        # Same prompts but different modes -> different keys.
        a = PromptCache(prefix_mode="prefix_chars", prefix_length=10).make_key(
            "sys", "user message"
        )
        b = PromptCache(prefix_mode="prefix_words", prefix_length=10).make_key(
            "sys", "user message"
        )
        c = PromptCache(prefix_mode="system_only").make_key("sys", "user message")
        self.assertEqual(len({a, b, c}), 3)


class TestGetOrCompute(unittest.TestCase):
    def test_first_call_is_miss(self) -> None:
        cache = PromptCache(ttl=60)
        result = cache.get_or_compute(
            "sys", "user", compute_fn=lambda: _echo("user")
        )
        self.assertEqual(result, "echo:user")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["bypasses"], 0)

    def test_second_call_is_hit(self) -> None:
        cache = PromptCache(ttl=60)
        cache.get_or_compute("sys", "user", compute_fn=lambda: _echo("user"))
        call_count = {"n": 0}

        def _compute() -> str:
            call_count["n"] += 1
            return _echo("user")

        result = cache.get_or_compute("sys", "user", compute_fn=_compute)
        self.assertEqual(result, "echo:user")
        self.assertEqual(call_count["n"], 0, "compute_fn should not be called on hit")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)

    def test_prefix_chars_share_cache(self) -> None:
        cache = PromptCache(prefix_mode="prefix_chars", prefix_length=5, ttl=60)
        cache.get_or_compute("sys", "hello world", compute_fn=lambda: _echo("hello world"))
        # Different suffix, same first 5 chars -> hit.
        result = cache.get_or_compute(
            "sys", "hello there", compute_fn=lambda: _echo("hello there")
        )
        self.assertEqual(result, "echo:hello world")

    def test_prefix_words_share_cache(self) -> None:
        cache = PromptCache(prefix_mode="prefix_words", prefix_length=2, ttl=60)
        cache.get_or_compute("sys", "alpha beta gamma", compute_fn=lambda: "first")
        result = cache.get_or_compute(
            "sys", "alpha beta delta", compute_fn=lambda: "second"
        )
        # Should hit and return the first value.
        self.assertEqual(result, "first")

    def test_bypass_skips_lookup_and_store(self) -> None:
        cache = PromptCache(ttl=60)
        # Warm.
        cache.get_or_compute("sys", "user", compute_fn=lambda: "cached")
        # Bypass -> compute_fn runs and the cache is NOT mutated.
        result = cache.get_or_compute(
            "sys", "user", compute_fn=lambda: "fresh", bypass=True
        )
        self.assertEqual(result, "fresh")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["bypasses"], 1)
        # Cache still holds the original "cached" value:
        self.assertEqual(cache.get("sys", "user"), "cached")

    def test_default_bypass(self) -> None:
        cache = PromptCache(ttl=60, default_bypass=True)
        calls: list[int] = []

        def _compute() -> str:
            calls.append(1)
            return "v" + str(len(calls))

        # default_bypass=True: every call invokes _compute and ignores the
        # cache entirely.
        cache.get_or_compute("sys", "user", compute_fn=_compute)
        cache.get_or_compute("sys", "user", compute_fn=_compute)
        cache.get_or_compute("sys", "user", compute_fn=_compute)
        self.assertEqual(len(calls), 3)
        self.assertEqual(cache.stats()["bypasses"], 3)
        # Cache is empty because bypass never writes:
        self.assertEqual(cache.stats()["size"], 0)


class TestInvalidateAndClear(unittest.TestCase):
    def test_invalidate_removes_entry(self) -> None:
        cache = PromptCache(ttl=60)
        cache.get_or_compute("sys", "user", compute_fn=lambda: "value")
        removed = cache.invalidate("sys", "user")
        self.assertTrue(removed)
        # Next call should miss.
        result = cache.get_or_compute("sys", "user", compute_fn=lambda: "new")
        self.assertEqual(result, "new")
        self.assertEqual(cache.stats()["misses"], 2)

    def test_invalidate_returns_false_when_missing(self) -> None:
        cache = PromptCache(ttl=60)
        self.assertFalse(cache.invalidate("sys", "user"))

    def test_clear_wipes_cache_and_stats(self) -> None:
        cache = PromptCache(ttl=60)
        cache.get_or_compute("sys", "user", compute_fn=lambda: "x")
        cache.get_or_compute("sys", "user2", compute_fn=lambda: "y")
        cache.clear()
        self.assertEqual(cache.stats()["size"], 0)
        self.assertEqual(cache.stats()["hits"], 0)
        self.assertEqual(cache.stats()["misses"], 0)


class TestTTL(unittest.TestCase):
    def test_entry_expires_after_ttl(self) -> None:
        cache = PromptCache(ttl=1)
        cache.get_or_compute("sys", "user", compute_fn=lambda: "fresh")
        time.sleep(1.1)
        # Should be expired.
        entry = cache.store.get(cache.make_key("sys", "user"))
        # store.get returns None if missing OR expired; here should be expired.
        if entry is not None:
            _, expires_at = entry
            self.assertLessEqual(expires_at, time.time())
        # get() returns None on miss:
        self.assertIsNone(cache.get("sys", "user"))

    def test_zero_ttl_means_no_expiry(self) -> None:
        cache = PromptCache(ttl=0)
        cache.get_or_compute("sys", "user", compute_fn=lambda: "v")
        # Even after a brief delay, the entry should still be present.
        time.sleep(0.05)
        self.assertEqual(cache.get("sys", "user"), "v")


class TestStats(unittest.TestCase):
    def test_hit_rate(self) -> None:
        cache = PromptCache(ttl=60)
        # 1 miss + 2 hits = 0.667
        cache.get_or_compute("sys", "user", compute_fn=lambda: "x")
        cache.get_or_compute("sys", "user", compute_fn=lambda: "x")
        cache.get_or_compute("sys", "user", compute_fn=lambda: "x")
        stats = cache.stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)
        self.assertAlmostEqual(stats["hit_rate"], 2 / 3, places=4)

    def test_bytes_and_tokens_saved(self) -> None:
        cache = PromptCache(ttl=60)
        # First call caches a value; then 2 hits count savings.
        cache.get_or_compute("sys", "user", compute_fn=lambda: "hello world")
        cache.get_or_compute("sys", "user", compute_fn=lambda: "should not run")
        cache.get_or_compute("sys", "user", compute_fn=lambda: "should not run")
        stats = cache.stats()
        # "hello world" is 11 bytes -> _record_savings adds 11 each hit.
        self.assertEqual(stats["bytes_saved"], 11 * 2)
        # len("hello world") // 4 = 2 tokens per hit, * 2 hits = 4.
        self.assertEqual(stats["tokens_saved"], 4)

    def test_reset_stats_preserves_entries(self) -> None:
        cache = PromptCache(ttl=60)
        cache.get_or_compute("sys", "user", compute_fn=lambda: "v")
        cache.get_or_compute("sys", "user", compute_fn=lambda: "v")  # hit
        cache.reset_stats()
        self.assertEqual(cache.stats()["hits"], 0)
        self.assertEqual(cache.stats()["misses"], 0)
        # Entries preserved:
        self.assertEqual(cache.get("sys", "user"), "v")


class TestStores(unittest.TestCase):
    def test_in_memory_max_entries_eviction(self) -> None:
        store = InMemoryCacheStore(max_entries=2)
        cache = PromptCache(store=store, ttl=60)
        cache.get_or_compute("sys", "a", compute_fn=lambda: "A")
        cache.get_or_compute("sys", "b", compute_fn=lambda: "B")
        cache.get_or_compute("sys", "c", compute_fn=lambda: "C")
        # At most 2 entries retained.
        self.assertLessEqual(len(store), 2)

    def test_file_store_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.json"
            store = FileCacheStore(path)
            cache = PromptCache(store=store, ttl=60)
            cache.get_or_compute("sys", "user", compute_fn=lambda: "persisted")
            # Force flush and reload.
            store.flush()
            with path.open() as f:
                payload = json.load(f)
            self.assertGreater(len(payload), 0)
            # Reload in a new store.
            store2 = FileCacheStore(path)
            self.assertEqual(store2.get(cache.make_key("sys", "user"))[0], "persisted")

    def test_file_store_clear(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.json"
            store = FileCacheStore(path)
            store.set("k", "v", time.time() + 60)
            store.clear()
            self.assertEqual(len(store), 0)
            self.assertIsNone(store.get("k"))

    def test_file_store_invalid_json_raises(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.json"
            path.write_text("not-json{")
            with self.assertRaises(StoreError):
                FileCacheStore(path)

    def test_file_store_top_level_object_required(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.json"
            path.write_text("[1, 2, 3]")
            with self.assertRaises(StoreError):
                FileCacheStore(path)

    def test_in_memory_delete(self) -> None:
        store = InMemoryCacheStore()
        store.set("k", "v", time.time() + 60)
        store.delete("k")
        self.assertIsNone(store.get("k"))
        # Deleting missing keys is a no-op.
        store.delete("nope")


class TestConcurrency(unittest.TestCase):
    def test_thread_safety(self) -> None:
        cache = PromptCache(ttl=60)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            try:
                for j in range(50):
                    cache.get_or_compute(
                        "sys", f"user-{i // 5}", compute_fn=lambda: f"v{i}-{j}"
                    )
            except BaseException as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        stats = cache.stats()
        # Misses limited to one per (i // 5) -> 5 misses total at most.
        self.assertLessEqual(stats["misses"], 5)
        self.assertGreaterEqual(stats["size"], 1)


class TestDecorator(unittest.TestCase):
    def test_decorator_caches_result(self) -> None:
        cache = PromptCache(prefix_mode="prefix_chars", prefix_length=5, ttl=60)
        calls: list[int] = []

        @cache.cached()
        def ask(system_prompt: str, user_message: str) -> str:
            calls.append(1)
            return f"answer:{user_message}"

        ask(system_prompt="sys", user_message="hello world")
        # First 5 chars match ("hello") -> hit, no recompute.
        ask(system_prompt="sys", user_message="hello there")
        self.assertEqual(len(calls), 1)
        self.assertGreaterEqual(cache.stats()["hits"], 1)

    def test_decorator_custom_arg_names(self) -> None:
        cache = PromptCache(prefix_mode="system_only", ttl=60)

        @cache.cached(system_arg="sys", user_arg="usr")
        def ask(sys: str, usr: str) -> str:
            return f"{sys}|{usr}"

        self.assertEqual(ask(sys="S", usr="U"), "S|U")
        # Same call -> hit.
        ask(sys="S", usr="U")
        self.assertEqual(cache.stats()["hits"], 1)

    def test_decorator_bypass_kwarg(self) -> None:
        cache = PromptCache(ttl=60)
        calls = []

        @cache.cached(bypass_arg="force")
        def ask(system_prompt: str, user_message: str, force: bool = False) -> str:
            calls.append(user_message)
            return f"r{len(calls)}"

        ask(system_prompt="sys", user_message="hi")
        ask(system_prompt="sys", user_message="hi", force=True)
        # The bypassed call still goes through compute_fn.
        self.assertGreaterEqual(len(calls), 2)
        # First non-bypass should have hit on second non-bypass invocation.
        ask(system_prompt="sys", user_message="hi")
        self.assertEqual(cache.stats()["hits"], 1)


class TestErrors(unittest.TestCase):
    def test_invalid_prefix_mode(self) -> None:
        with self.assertRaises(InvalidConfigError):
            PromptCache(prefix_mode="nonsense")

    def test_negative_ttl(self) -> None:
        with self.assertRaises(InvalidConfigError):
            PromptCache(ttl=-1)

    def test_invalid_prefix_length(self) -> None:
        with self.assertRaises(InvalidConfigError):
            PromptCache(prefix_mode="prefix_chars", prefix_length=0)

    def test_non_callable_compute_fn(self) -> None:
        cache = PromptCache(ttl=60)
        with self.assertRaises(InvalidConfigError):
            cache.get_or_compute("sys", "user", compute_fn="not-callable")  # type: ignore[arg-type]

    def test_get_or_compute_returns_fresh_value(self) -> None:
        cache = PromptCache(ttl=60)
        # Compute fn returns non-hashable values - still works.
        v = cache.get_or_compute("sys", "u", compute_fn=lambda: {"a": [1, 2, 3]})
        self.assertEqual(v, {"a": [1, 2, 3]})
        v2 = cache.get_or_compute("sys", "u", compute_fn=lambda: {"a": [9, 9]})
        # Hit -> first value preserved.
        self.assertEqual(v2, {"a": [1, 2, 3]})


class TestUtilities(unittest.TestCase):
    def test_estimate_tokens_basic(self) -> None:
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("a"), 1)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertEqual(estimate_tokens("abcde"), 1)
        self.assertEqual(estimate_tokens("abcdefgh"), 2)

    def test_store_protocol_is_runtime_checkable(self) -> None:
        # The class types themselves satisfy the protocol.
        self.assertTrue(isinstance(InMemoryCacheStore(), CacheStore))
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(isinstance(FileCacheStore(Path(d) / "c.json"), CacheStore))

    def test_get_default_cache_uses_inmemory(self) -> None:
        cache = get_default_cache(ttl=30)
        result = cache.get_or_compute("s", "u", compute_fn=lambda: "ok")
        self.assertEqual(result, "ok")
        self.assertIsInstance(cache.store, InMemoryCacheStore)


if __name__ == "__main__":
    unittest.main()
