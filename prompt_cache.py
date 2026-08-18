"""
tiny-prompt-cache — Semantic + LRU + TTL cache for LLM prompts.

Zero dependencies. Exact hash match + optional semantic similarity.
Works with any LLM provider.

Usage:
    from prompt_cache import PromptCache

    cache = PromptCache(max_size=1000, ttl_seconds=3600)

    response = cache.get_or_call(
        prompt="Explain quantum entanglement",
        call_fn=lambda: llm("Explain quantum entanglement"),
    )
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

__version__ = "0.1.0"
__all__ = ["PromptCache", "cosine_similarity"]


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Vector lengths must match: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------


@dataclass
class CacheEntry:
    prompt: str
    response: str
    created_at: float
    last_accessed: float
    hash: str
    embedding: list[float] | None = None
    token_estimate: int = 0

    def is_expired(self, ttl: float) -> bool:
        return time.time() - self.created_at > ttl


# ----------------------------------------------------------------------
# Core Cache
# ----------------------------------------------------------------------


class PromptCache:
    """
    LLM prompt cache with exact hash + optional semantic matching.

    Args:
        max_size: Maximum entries before LRU eviction.
        ttl_seconds: Time-to-live per entry in seconds.
        semantic_threshold: Cosine similarity threshold (1.0 = exact only).
                           Set < 1.0 to enable semantic deduplication.
        embedding_fn: Optional function: (prompt: str) -> list[float].
                      Required when semantic_threshold < 1.0.
        cost_per_1k_tokens: Cost estimate for savings reporting.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,
        semantic_threshold: float = 1.0,
        embedding_fn: Callable[[str], list[float]] | None = None,
        cost_per_1k_tokens: float = 0.01,
    ):
        if semantic_threshold < 1.0 and embedding_fn is None:
            raise ValueError(
                "embedding_fn is required when semantic_threshold < 1.0. "
                "Provide a function: (prompt: str) -> list[float]"
            )
        if not 0.0 <= semantic_threshold <= 1.0:
            raise ValueError("semantic_threshold must be between 0.0 and 1.0")

        self._max_size = max_size
        self._ttl = ttl_seconds
        self._semantic_threshold = semantic_threshold
        self._embedding_fn = embedding_fn
        self._cost_per_1k = cost_per_1k_tokens

        self._lock = threading.RLock()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats_hits = 0
        self._stats_misses = 0
        self._stats_total_tokens = 0

    # ── Public API ────────────────────────────────────────────────────

    def get_or_call(
        self, prompt: str, call_fn: Callable[[], str]
    ) -> str:
        """
        Return cached response for prompt, or call call_fn() and cache the result.

        Args:
            prompt: The input prompt string.
            call_fn: A callable that returns the LLM response string.
        Returns:
            The response string (from cache or call_fn).
        """
        cached = self._get(prompt)
        if cached is not None:
            return cached

        # Cache miss — call the LLM
        response = call_fn()

        # Estimate tokens (rough: ~4 chars per token)
        token_estimate = max(1, len(response) // 4)
        self._set(prompt, response, token_estimate=token_estimate)
        return response

    def get(self, prompt: str) -> str | None:
        """Return cached response or None."""
        return self._get(prompt)

    def set(self, prompt: str, response: str, token_estimate: int = 0) -> None:
        """Manually set a cache entry."""
        self._set(prompt, response, token_estimate)

    def invalidate(self, prompt: str) -> bool:
        """Remove an entry by prompt. Returns True if found."""
        return self._invalidate(prompt)

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self._stats_hits = 0
            self._stats_misses = 0
            self._stats_total_tokens = 0

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self._stats_hits + self._stats_misses
            hit_rate = self._stats_hits / total if total > 0 else 0.0
            estimated_cost = (self._stats_total_tokens / 1000) * self._cost_per_1k
            savings = (self._stats_hits / total * estimated_cost) if total > 0 else 0.0
            return {
                "total": total,
                "cache_hits": self._stats_hits,
                "cache_misses": self._stats_misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
                "semantic_threshold": self._semantic_threshold,
                "estimated_cost_usd": estimated_cost,
                "estimated_savings_usd": savings,
            }

    # ── Internal ───────────────────────────────────────────────────────

    def _get(self, prompt: str) -> str | None:
        """Find and return cached response, or None."""
        prompt_hash = _hash_prompt(prompt)

        with self._lock:
            # Fast path: exact hash match
            entry = self._cache.get(prompt_hash)
            if entry is not None:
                if entry.is_expired(self._ttl):
                    del self._cache[prompt_hash]
                else:
                    # Refresh LRU position
                    self._cache.move_to_end(prompt_hash)
                    entry.last_accessed = time.time()
                    self._stats_hits += 1
                    return entry.response

            # Semantic match (only if enabled)
            if self._semantic_threshold < 1.0 and self._embedding_fn is not None:
                embedding = self._embedding_fn(prompt)
                best_score = 0.0
                best_entry: CacheEntry | None = None

                for candidate in self._cache.values():
                    if candidate.embedding is None:
                        continue
                    if candidate.is_expired(self._ttl):
                        continue
                    score = cosine_similarity(embedding, candidate.embedding)
                    if score >= self._semantic_threshold and score > best_score:
                        best_score = score
                        best_entry = candidate

                if best_entry is not None:
                    self._cache.move_to_end(best_entry.hash)
                    best_entry.last_accessed = time.time()
                    self._stats_hits += 1
                    return best_entry.response

            self._stats_misses += 1
            return None

    def _set(self, prompt: str, response: str, token_estimate: int = 0) -> None:
        """Store a response in the cache."""
        prompt_hash = _hash_prompt(prompt)
        embedding = self._embedding_fn(prompt) if self._embedding_fn else None

        entry = CacheEntry(
            prompt=prompt,
            response=response,
            created_at=time.time(),
            last_accessed=time.time(),
            hash=prompt_hash,
            embedding=embedding,
            token_estimate=token_estimate,
        )

        with self._lock:
            # Evict LRU if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[prompt_hash] = entry
            self._stats_total_tokens += token_estimate

    def _invalidate(self, prompt: str) -> bool:
        prompt_hash = _hash_prompt(prompt)
        with self._lock:
            if prompt_hash in self._cache:
                del self._cache[prompt_hash]
                return True
            return False
