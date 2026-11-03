"""tiny_prompt_cache — zero-dependency prompt prefix cache for LLM calls.

A single-file library that deduplicates LLM API calls where the same system prompt
or context prefix is reused with different user inputs. Caches responses keyed by
the system prompt and a configurable prefix of the user message. Zero dependencies,
stdlib only.

Usage:
    from tiny_prompt_cache import PromptCache, InMemoryCacheStore, FileCacheStore

    cache = PromptCache(store=InMemoryCacheStore(), ttl=3600,
                        prefix_mode="prefix_chars", prefix_length=200)

    text = cache.get_or_compute(
        system_prompt="You are a helpful assistant.",
        user_message="What is Python?",
        compute_fn=lambda: call_llm("You are a helpful assistant.",
                                     "What is Python?"),
    )

    # Drop-in decorator
    @cache.cached()
    def ask_llm(system: str, user: str) -> str:
        return call_llm(system, user)
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable


__version__ = "0.1.0"


# ---------- Errors ----------


class PromptCacheError(Exception):
    """Base exception raised by tiny_prompt_cache."""


class StoreError(PromptCacheError):
    """Raised when a cache store fails to read or write."""


class InvalidConfigError(PromptCacheError):
    """Raised when PromptCache or the decorator are configured incorrectly."""


# ---------- Types ----------


PrefixMode = str  # "system_only" | "prefix_words" | "prefix_chars"

ComputeFn = Callable[..., Any]
ComputeKwFn = Callable[..., Any]


# ---------- Cache Store Protocol ----------


@runtime_checkable
class CacheStore(Protocol):
    """Pluggable persistence layer for cached prompt completions.

    Implementations must be safe to call from multiple threads.
    """

    def get(self, key: str) -> Optional[tuple[Any, float]]:
        """Return (value, expires_at_epoch) if present, else None.

        ``expires_at_epoch`` is a Unix timestamp (seconds). Entries whose
        timestamp is in the past are considered expired and should not be
        returned.
        """
        ...

    def set(self, key: str, value: Any, expires_at: float) -> None:
        """Store ``value`` under ``key`` until ``expires_at`` (epoch seconds)."""
        ...

    def delete(self, key: str) -> None:
        """Remove a single key. Missing keys are ignored."""
        ...

    def clear(self) -> None:
        """Remove every key in the store."""
        ...


# ---------- In-memory store ----------


class InMemoryCacheStore:
    """Default in-process cache store backed by a dict + RLock.

    Suitable for single-process applications. Safe for use from multiple
    threads. Lost on process restart.

    Args:
        max_entries: Soft cap on the number of cached entries. When the cap is
            exceeded, the oldest entries are evicted. ``0`` (the default) means
            "no cap".
    """

    def __init__(self, max_entries: int = 0) -> None:
        if max_entries < 0:
            raise ValueError("max_entries must be >= 0")
        self._data: dict[str, tuple[Any, float]] = {}
        self._lock = threading.RLock()
        self._max_entries = max_entries

    def get(self, key: str) -> Optional[tuple[Any, float]]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            return entry

    def set(self, key: str, value: Any, expires_at: float) -> None:
        with self._lock:
            self._data[key] = (value, expires_at)
            if self._max_entries > 0:
                while len(self._data) > self._max_entries:
                    # Evict oldest by insertion order (oldest key first).
                    oldest_key = next(iter(self._data))
                    if oldest_key == key:
                        # Refusing to evict the entry we just wrote would
                        # deadlock the loop, so stop at exactly max_entries.
                        break
                    del self._data[oldest_key]

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


# ---------- File store ----------


class FileCacheStore:
    """JSON-file cache store with atomic writes.

    Each :class:`FileCacheStore` owns a single file. Writes are performed via
    a temp file + ``os.replace`` so partial writes cannot corrupt the cache.
    Safe for use from multiple threads within a single process.

    Args:
        path: Path to the JSON cache file. Parent directories are created on
            first write.
        autosave: If ``True`` (default), each :meth:`set` is flushed to disk
            immediately. If ``False``, changes are buffered in memory until
            :meth:`flush` is called explicitly. ``autosave=True`` is safest
            but slowest.
    """

    def __init__(self, path: str | os.PathLike[str], autosave: bool = True) -> None:
        self._path = Path(path)
        self._autosave = autosave
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    # ----- public -----

    def get(self, key: str) -> Optional[tuple[Any, float]]:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            value = entry.get("value")
            expires_at = float(entry.get("expires_at", 0.0))
            return (value, expires_at)

    def set(self, key: str, value: Any, expires_at: float) -> None:
        with self._lock:
            self._data[key] = {"value": value, "expires_at": float(expires_at)}
            if self._autosave:
                self._write_atomic()

    def delete(self, key: str) -> None:
        with self._lock:
            removed = self._data.pop(key, None) is not None
            if removed and self._autosave:
                self._write_atomic()

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            if self._autosave:
                self._write_atomic()

    def flush(self) -> None:
        """Force pending changes to disk (only relevant when ``autosave=False``)."""
        with self._lock:
            self._write_atomic()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    # ----- internal -----

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            raise StoreError(
                f"failed to read cache file {self._path!s}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise StoreError(
                f"cache file {self._path!s} is not a JSON object at the top level"
            )
        # Defensive: filter malformed entries.
        clean: dict[str, dict[str, Any]] = {}
        for k, v in payload.items():
            if isinstance(v, dict) and "expires_at" in v:
                clean[k] = v
        self._data = clean

    def _write_atomic(self) -> None:
        parent = self._path.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(self._data, ensure_ascii=False)
            # Use a sibling temp file for an atomic rename.
            fd, tmp_path = tempfile.mkstemp(
                prefix=self._path.name + ".",
                suffix=".tmp",
                dir=str(parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self._path)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as exc:
            raise StoreError(
                f"failed to write cache file {self._path!s}: {exc}"
            ) from exc


# ---------- Utilities ----------


# A reasonable approximation; matches what most LLM cost estimators use.
_TOKEN_RE = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Return a rough token count for ``text``.

    Uses the simple ``len(text) // 4`` heuristic, which is conservative
    enough to be useful for cache sizing and cost estimates without pulling
    in a tokenizer.
    """
    if not text:
        return 0
    # Standard rule of thumb: ~4 chars per token on average for English text.
    return max(1, len(text) // 4)


def _split_words(text: str) -> list[str]:
    return [w for w in _TOKEN_RE.split(text.strip()) if w]


def _now() -> float:
    return time.time()


# ---------- Cache ----------


class PromptCache:
    """Prompt-prefix cache for LLM completions.

    A single :class:`PromptCache` instance is safe to share across threads.
    It owns a :class:`CacheStore` (defaulting to :class:`InMemoryCacheStore`),
    a TTL, and a prefix-key strategy.

    Args:
        store: Backing store. Defaults to a fresh :class:`InMemoryCacheStore`.
        ttl: Seconds an entry lives before being treated as stale. ``0`` means
            "never expire" (entries live until the store is cleared or the
            process restarts).
        prefix_mode: ``"system_only"`` keys by the system prompt only;
            ``"prefix_words"`` keys by the system prompt plus the first
            ``prefix_length`` whitespace-delimited tokens of the user message;
            ``"prefix_chars"`` keys by the system prompt plus the first
            ``prefix_length`` characters of the user message.
        prefix_length: Number of words/chars (depending on ``prefix_mode``)
            to include in the cache key. Ignored for ``"system_only"``.
        default_bypass: If ``True``, every call bypasses the cache (a global
            "off switch" useful for testing).
    """

    def __init__(
        self,
        store: Optional[CacheStore] = None,
        ttl: int = 3600,
        prefix_mode: PrefixMode = "prefix_chars",
        prefix_length: int = 200,
        default_bypass: bool = False,
    ) -> None:
        if ttl < 0:
            raise InvalidConfigError("ttl must be >= 0")
        if prefix_mode not in ("system_only", "prefix_words", "prefix_chars"):
            raise InvalidConfigError(
                f"unknown prefix_mode {prefix_mode!r}; "
                "expected one of: system_only, prefix_words, prefix_chars"
            )
        if prefix_mode != "system_only" and prefix_length <= 0:
            raise InvalidConfigError(
                "prefix_length must be > 0 when prefix_mode is not 'system_only'"
            )
        self._store: CacheStore = store if store is not None else InMemoryCacheStore()
        self._ttl = int(ttl)
        self._prefix_mode: PrefixMode = prefix_mode
        self._prefix_length = int(prefix_length)
        self._default_bypass = bool(default_bypass)

        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._bytes_saved = 0
        self._tokens_saved = 0
        self._bypasses = 0

    # ----- configuration helpers -----

    @property
    def store(self) -> CacheStore:
        """The configured cache store."""
        return self._store

    @property
    def ttl(self) -> int:
        """Configured TTL in seconds. ``0`` means "never expire"."""
        return self._ttl

    @property
    def prefix_mode(self) -> PrefixMode:
        """Active keying strategy."""
        return self._prefix_mode

    @property
    def prefix_length(self) -> int:
        """Number of words/chars used to build a key (mode-dependent)."""
        return self._prefix_length

    # ----- key generation -----

    def _user_prefix(self, user_message: str) -> str:
        if self._prefix_mode == "system_only":
            return ""
        if self._prefix_mode == "prefix_chars":
            return user_message[: self._prefix_length]
        # prefix_words
        words = _split_words(user_message)
        return " ".join(words[: self._prefix_length])

    def make_key(self, system_prompt: str, user_message: str) -> str:
        """Build a deterministic cache key from a system + user prompt pair.

        The key is a SHA-256 hex digest of the prefix mode, system prompt,
        and the user-message prefix chosen by :attr:`prefix_mode`. Useful for
        inspection and for cross-process coordination.

        Note: only the prefix of the user message is used (per
        :attr:`prefix_mode`). Two requests with different suffixes but
        matching prefixes will share the same key by design.
        """
        user_prefix = self._user_prefix(user_message)
        # Include the mode so a key in one mode never collides with another.
        payload = (
            f"{self._prefix_mode}|"
            f"{len(system_prompt)}|{system_prompt}|"
            f"{len(user_prefix)}|{user_prefix}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ----- core operations -----

    def get(self, system_prompt: str, user_message: str) -> Optional[Any]:
        """Return the cached value if present and unexpired, else ``None``.

        Expired entries are evicted lazily on read.
        """
        with self._lock:
            key = self.make_key(system_prompt, user_message)
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if self._ttl > 0 and expires_at > 0 and expires_at <= _now():
                self._store.delete(key)
                return None
            return value

    def set(self, system_prompt: str, user_message: str, value: Any) -> None:
        """Store ``value`` for this prompt pair using the configured TTL."""
        with self._lock:
            key = self.make_key(system_prompt, user_message)
            expires_at = (_now() + self._ttl) if self._ttl > 0 else 0.0
            self._store.set(key, value, expires_at)

    def get_or_compute(
        self,
        system_prompt: str,
        user_message: str,
        compute_fn: ComputeFn,
        bypass: bool = False,
    ) -> Any:
        """Return cached value or invoke ``compute_fn`` and cache its result.

        Args:
            system_prompt: The system prompt used by the underlying LLM call.
            user_message: The user message used by the underlying LLM call.
            compute_fn: Zero-arg callable that performs the actual LLM call.
                Invoked only on a cache miss (or when caching is bypassed).
            bypass: If ``True``, skip both the lookup and the store: the
                cache is neither read nor written for this call. Useful for
                forcing a fresh response from the underlying model.

        Returns:
            The cached or freshly computed value.
        """
        if not callable(compute_fn):
            raise InvalidConfigError("compute_fn must be callable")
        with self._lock:
            if bypass or self._default_bypass:
                self._bypasses += 1
                return compute_fn()

            key = self.make_key(system_prompt, user_message)
            entry = self._store.get(key)
            if entry is not None:
                value, expires_at = entry
                if self._ttl == 0 or expires_at == 0 or expires_at > _now():
                    # Cache hit.
                    self._hits += 1
                    self._record_savings(value)
                    return value
                # Expired; drop and continue to recompute.
                self._store.delete(key)

            # Cache miss.
            self._misses += 1
            value = compute_fn()
            # Set with TTL>=0 (0 means "no expiry").
            expires_at = (_now() + self._ttl) if self._ttl > 0 else 0.0
            self._store.set(key, value, expires_at)
            return value

    def invalidate(self, system_prompt: str, user_message: str) -> bool:
        """Remove the cached entry for a specific prompt pair.

        Returns ``True`` if an entry was present and removed, ``False``
        otherwise.
        """
        with self._lock:
            key = self.make_key(system_prompt, user_message)
            had = self._store.get(key) is not None
            self._store.delete(key)
            return had

    def clear(self) -> None:
        """Remove every cached entry and reset stats."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._bytes_saved = 0
            self._tokens_saved = 0
            self._bypasses = 0

    # ----- decorators -----

    def cached(
        self,
        system_arg: str = "system_prompt",
        user_arg: str = "user_message",
        bypass_arg: Optional[str] = None,
    ) -> Callable[[ComputeKwFn], ComputeKwFn]:
        """Return a decorator that caches a function's return value.

        The decorated function must accept the system prompt and user message
        as keyword arguments (names are configurable). Results are cached per
        (system_prompt, user_prefix) pair.

        Args:
            system_arg: Keyword name of the system prompt argument.
            user_arg: Keyword name of the user message argument.
            bypass_arg: Optional keyword name that, when truthy, bypasses the
                cache for the current call. Pass ``None`` to disable.
        """
        if not system_arg or not user_arg:
            raise InvalidConfigError("system_arg and user_arg must be non-empty")

        cache = self

        def decorator(fn: ComputeKwFn) -> ComputeKwFn:
            if not callable(fn):
                raise InvalidConfigError("decorator target must be callable")

            @functools.wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Find system/user messages. Prefer kwargs; fall back to positional.
                system_prompt, user_message = _resolve_prompt_args(
                    fn, args, kwargs, system_arg, user_arg
                )
                bypass = False
                if bypass_arg is not None and bypass_arg in kwargs:
                    bypass = bool(kwargs[bypass_arg])

                def _compute() -> Any:
                    return fn(*args, **kwargs)

                return cache.get_or_compute(
                    system_prompt,
                    user_message,
                    _compute,
                    bypass=bypass,
                )

            # Expose the underlying cache for inspection.
            wrapper.__wrapped__ = fn  # type: ignore[attr-defined]
            wrapper.__prompt_cache__ = cache  # type: ignore[attr-defined]
            return wrapper

        return decorator

    # ----- introspection -----

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of cache statistics.

        All counters are cumulative since construction or the last
        :meth:`clear`. :meth:`reset_stats` zeroes them without dropping
        cached values.
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "bypasses": self._bypasses,
                "hit_rate": hit_rate,
                "bytes_saved": self._bytes_saved,
                "tokens_saved": self._tokens_saved,
                "size": len(self._store),
            }

    def reset_stats(self) -> None:
        """Zero all statistics counters without touching cached values."""
        with self._lock:
            self._hits = 0
            self._misses = 0
            self._bypasses = 0
            self._bytes_saved = 0
            self._tokens_saved = 0

    # ----- internals -----

    def _record_savings(self, value: Any) -> None:
        text = _stringify_for_metrics(value)
        if text is None:
            return
        self._bytes_saved += len(text.encode("utf-8"))
        self._tokens_saved += estimate_tokens(text)


def _stringify_for_metrics(value: Any) -> Optional[str]:
    """Convert a cached value to text for byte/token accounting.

    Returns ``None`` if no meaningful text representation is available.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None


def _resolve_prompt_args(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    system_arg: str,
    user_arg: str,
) -> tuple[str, str]:
    """Extract the system + user prompts from a decorated call."""
    system_prompt: Optional[Any] = kwargs.get(system_arg)
    user_message: Optional[Any] = kwargs.get(user_arg)

    if system_prompt is None or user_message is None:
        # Fall back to positional args by inspecting the function signature.
        try:
            import inspect

            sig = inspect.signature(fn)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()
        except (TypeError, ValueError):
            bound = None
        if bound is not None:
            system_prompt = bound.arguments.get(system_arg, system_prompt)
            user_message = bound.arguments.get(user_arg, user_message)

    if system_prompt is None or user_message is None:
        raise InvalidConfigError(
            f"could not locate system/user prompts for {fn.__name__!r}; "
            f"expected kwargs {system_arg!r} and {user_arg!r}"
        )
    return str(system_prompt), str(user_message)


# ---------- Module-level helpers ----------


def get_default_cache(**kwargs: Any) -> PromptCache:
    """Build a :class:`PromptCache` with default in-memory store.

    Convenience wrapper for ``PromptCache(InMemoryCacheStore(), **kwargs)``.
    """
    store = kwargs.pop("store", None)
    if store is None:
        store = InMemoryCacheStore()
    return PromptCache(store=store, **kwargs)


__all__ = [
    "__version__",
    "PromptCacheError",
    "StoreError",
    "InvalidConfigError",
    "CacheStore",
    "InMemoryCacheStore",
    "FileCacheStore",
    "PromptCache",
    "estimate_tokens",
    "get_default_cache",
]
