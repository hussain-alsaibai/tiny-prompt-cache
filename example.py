"""Example usage of tiny_prompt_cache. Run with: python example.py"""

from __future__ import annotations

import time

from tiny_prompt_cache import (
    FileCacheStore,
    InMemoryCacheStore,
    PromptCache,
    estimate_tokens,
)


def fake_llm_call(system_prompt: str, user_message: str) -> str:
    """Stand-in for a real LLM call. Pretends to be slow & expensive."""
    time.sleep(0.05)
    return (
        f"[{len(system_prompt)}B system / {len(user_message)}B user] "
        f"=> {estimate_tokens(user_message)} estimated tokens, "
        f"'{user_message[:40]}...'"
    )


def main() -> None:
    print("=== tiny-prompt-cache example ===\n")

    # ----- 1. Basic in-memory cache -----
    print("[1] In-memory cache, prefix_chars=20, ttl=60s")
    cache = PromptCache(
        store=InMemoryCacheStore(),
        ttl=60,
        prefix_mode="prefix_chars",
        prefix_length=20,
    )

    system = "You are a pirate captain with twenty years at sea."
    calls = []

    for user in [
        "Tell me a joke about Python decorators.",
        "Tell me a joke about Python generators.",
        "Tell me a joke about Ruby on Rails.",
        "Tell me a joke about JavaScript frameworks.",
    ]:
        start = time.perf_counter()
        out = cache.get_or_compute(system, user, compute_fn=lambda: fake_llm_call(system, user))
        calls.append(time.perf_counter() - start)
        print(f"  user='{user[:40]}...' -> hit={'yes' if out.startswith('[') and len(calls)>1 else 'no'}")

    print(f"  stats: {cache.stats()}\n")

    # ----- 2. Decorator usage -----
    print("[2] @cache.cached() decorator")
    cache2 = PromptCache(prefix_mode="prefix_words", prefix_length=3)

    @cache2.cached()
    def ask(system_prompt: str, user_message: str) -> str:
        return fake_llm_call(system_prompt, user_message)

    ask("sys", "What is the capital of France?")
    ask("sys", "What is the capital of Germany?")          # miss
    ask("sys", "What is the capital of France?")           # hit
    ask("sys", "What is the capital of France in summer?") # hit
    print(f"  stats: {cache2.stats()}\n")

    # ----- 3. File-backed persistent cache -----
    print("[3] FileCacheStore (persistent)")
    cache3 = PromptCache(store=FileCacheStore("/tmp/example-tpc.json"), ttl=3600)
    cache3.get_or_compute("sys", "What is Python?", compute_fn=lambda: "Python is a snake. 🐍")
    print(f"  wrote to /tmp/example-tpc.json")
    print(f"  stats: {cache3.stats()}\n")

    # ----- 4. Bypass -----
    print("[4] bypass=True (no read, no write)")
    cache4 = PromptCache()
    cache4.get_or_compute("sys", "u1", compute_fn=lambda: "cached")
    fresh = cache4.get_or_compute("sys", "u1", compute_fn=lambda: "fresh!", bypass=True)
    print(f"  bypassed result='{fresh}', size={cache4.stats()['size']} (still 1)\n")

    # ----- 5. Invalidate -----
    print("[5] invalidate()")
    cache4.invalidate("sys", "u1")
    print(f"  after invalidate, size={cache4.stats()['size']}\n")

    print("Done.")


if __name__ == "__main__":
    main()
