"""Real-time repetition detection for LLM streaming.

Fed token-by-token (or chunk-by-chunk) and signals when generation
should be aborted due to detected repetition loops.

Three detection strategies run in parallel:

1. **N-Gram Frequency** -- Tracks all n-grams; fires when one appears too often.
2. **Suffix Loop** -- Checks whether the current text tail is repeating itself.
3. **Stagnation** -- Detects when the last N tokens consist entirely of
   previously seen phrases (the model is "going in circles").

Example:
    >>> from dgxarley.integration.streaming_repetition_guard import RepetitionGuard
    >>>
    >>> guard = RepetitionGuard()
    >>>
    >>> for chunk in llm_stream:
    ...     token = chunk.choices[0].delta.content or ""
    ...     result = guard.feed(token)
    ...     if result.should_stop:
    ...         print(f"STOP: {result.reason}")
    ...         break
    ...     print(token, end="", flush=True)
    >>>
    >>> clean = guard.get_clean_text()
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Generator, Iterator, TypeVar

# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────

_T = TypeVar("_T")


class StopReason(Enum):
    """Reason why the guard triggered a stop.

    Attributes:
        NGRAM_FLOOD: A single n-gram exceeded the maximum allowed count.
        SUFFIX_LOOP: The text tail contains a block-level repeating pattern.
        STAGNATION: Recent tokens consist almost entirely of recycled phrases.
    """

    NGRAM_FLOOD = auto()
    SUFFIX_LOOP = auto()
    STAGNATION = auto()


@dataclass
class FeedResult:
    """Result of a single :meth:`RepetitionGuard.feed` call.

    Attributes:
        should_stop: Whether generation should be aborted.
        reason: The :class:`StopReason` if ``should_stop`` is True, else None.
        detail: Human-readable description of the trigger condition.
        tokens_seen: Total number of word-level tokens processed so far.
        worst_ngram_count: Highest occurrence count among all tracked n-grams.
        loop_confidence: Exponentially smoothed loop confidence (0.0--1.0).
        diagnostics: Detailed trigger info for false-positive analysis. Only
            populated when ``should_stop`` is True. Keys depend on the
            :class:`StopReason`:

            **NGRAM_FLOOD**: ``top_ngrams`` (top 10 with counts),
            ``effective_max``, ``ratio``, ``total_tokens``.

            **SUFFIX_LOOP**: ``pattern_text`` (truncated to 300 chars),
            ``pattern_length``, ``repetitions``, ``tail_length``.

            **STAGNATION**: ``recycled_ratio``, ``recycled_count``,
            ``recent_count``, ``sample_recycled`` (up to 10 recycled n-grams),
            ``sample_new`` (up to 5 n-grams only in recent window).
    """

    should_stop: bool = False
    reason: StopReason | None = None
    detail: str = ""

    tokens_seen: int = 0
    worst_ngram_count: int = 0
    loop_confidence: float = 0.0
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class GuardConfig:
    """Configuration for all detection thresholds.

    Sensible defaults for outputs up to ~1k tokens. For longer outputs
    (e.g. 8k+), consider increasing ``ngram_max_count`` and
    ``min_tokens_before_check``.

    Attributes:
        ngram_n: Length of n-grams for frequency checking. 4 is a good
            trade-off: long enough to avoid false positives, short enough
            to catch repetition early.
        ngram_max_count: Base maximum allowed occurrences of any single
            n-gram before triggering a stop. This is the floor; the
            effective threshold grows with output length via
            ``ngram_count_scale_tokens``.
        ngram_count_scale_tokens: Every this many tokens, allow one
            additional n-gram repeat. Prevents false positives on longer
            outputs where domain-specific phrases naturally recur. Set to
            0 to disable scaling (use fixed ``ngram_max_count``). With the
            default of 100, effective limit at 400 tokens is 8 + 4 = 12.
        ngram_min_ratio: Minimum density (fraction of total tokens) the
            worst n-gram must reach before triggering. Acts as a second
            gate alongside the count threshold to prevent false positives
            on structured enumerations (REST API endpoints, CLI flags)
            where a domain-specific prefix recurs naturally.
            Computed as ``count * ngram_n / total_tokens``. Default 0.14
            means the n-gram's instances must cover >14% of all tokens.
        suffix_window: Number of trailing characters to consider for
            suffix loop detection. Larger values catch longer loops but
            require more text before detection.
        suffix_min_pattern: Shortest pattern (in characters) that counts
            as a loop. Below 30 there are too many false positives with
            natural language.
        suffix_min_reps: Minimum number of fuzzy-matched repetitions of the
            pattern required to trigger a stop. Default 4 avoids false
            positives on structured/tabular data (DNS records, code
            listings, table rows) and mathematical notation (LaTeX
            formulas reusing ``\\frac``, ``\\left``, ``\\right``) where
            three adjacent blocks can share >90% character similarity
            by coincidence.
        stagnation_window: Number of trailing tokens to consider for
            stagnation detection.
        stagnation_threshold: Fraction (0.0--1.0) of tokens in the window
            that must consist of previously seen n-grams to trigger
            stagnation. 0.85 means 85% of recent tokens are recycled.
        min_tokens_before_check: Minimum number of tokens before any
            checking begins. Prevents false positives on short outputs.
        cooldown_tokens: Number of tokens to skip after a near-trigger
            before checking again. Prevents flicker.
        check_every_n: Only run checks every N ``feed()`` calls.
            1 = every call, 5 = every 5th call. Performance tuning.
    """

    ngram_n: int = 4
    ngram_max_count: int = 8
    ngram_count_scale_tokens: int = 100
    ngram_min_ratio: float = 0.14

    suffix_window: int = 600
    suffix_min_pattern: int = 30
    suffix_min_reps: int = 4

    stagnation_window: int = 80
    stagnation_threshold: float = 0.85

    min_tokens_before_check: int = 40
    cooldown_tokens: int = 0
    check_every_n: int = 3


# ──────────────────────────────────────────────────────────────
# Guard Class
# ──────────────────────────────────────────────────────────────


class RepetitionGuard:
    """Streaming-capable repetition watchdog.

    Designed to be inserted into a token streaming pipeline. Lightweight
    enough to be called on every token.

    Example:
        >>> guard = RepetitionGuard()
        >>> # or with custom config:
        >>> guard = RepetitionGuard(GuardConfig(ngram_max_count=3, suffix_window=400))
        >>>
        >>> for token in stream:
        ...     result = guard.feed(token)
        ...     if result.should_stop:
        ...         break
    """

    def __init__(self, config: GuardConfig | None = None) -> None:
        """Initialize the guard with optional custom configuration.

        Args:
            config: Detection thresholds. Uses :class:`GuardConfig`
                defaults if not provided.
        """
        self.config: GuardConfig = config or GuardConfig()
        self._reset_state()

    def _reset_state(self) -> None:
        """Reset all internal state to initial values."""
        self._full_text: str = ""
        self._tokens: list[str] = []
        self._token_count: int = 0
        self._feed_count: int = 0

        self._ngram_counts: Counter[str] = Counter()
        self._worst_ngram_count: int = 0

        self._early_ngrams: set[str] = set()
        self._early_ngrams_frozen: bool = False

        self._loop_confidence: float = 0.0
        self._cooldown_remaining: int = 0
        self._last_clean_pos: int = 0

    def reset(self) -> None:
        """Reset state for a new generation request."""
        self._reset_state()

    # ──────────────────────────────────────────────────────
    # Main Method
    # ──────────────────────────────────────────────────────

    def feed(self, chunk: str) -> FeedResult:
        """Ingest a token or text chunk from the LLM stream.

        Runs the configured detection checks (n-gram flood, suffix loop,
        stagnation) and returns a result indicating whether generation
        should be stopped.

        Args:
            chunk: The next token or text chunk from the stream. Can be a
                single token (``"Hello"``) or a multi-token chunk
                (``" is an important"``).

        Returns:
            A :class:`FeedResult` with ``should_stop=True`` if generation
            should be aborted.
        """
        if not chunk:
            return FeedResult(tokens_seen=self._token_count)

        self._full_text += chunk
        self._feed_count += 1

        new_words: list[str] = re.findall(r"\b\w+\b", chunk.lower())
        self._tokens.extend(new_words)
        self._token_count += len(new_words)

        self._update_ngrams(new_words)

        if self._token_count < self.config.min_tokens_before_check:
            return FeedResult(
                tokens_seen=self._token_count,
                worst_ngram_count=self._worst_ngram_count,
            )

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= len(new_words)
            return FeedResult(
                tokens_seen=self._token_count,
                worst_ngram_count=self._worst_ngram_count,
                loop_confidence=self._loop_confidence,
            )

        if self._feed_count % self.config.check_every_n != 0:
            return FeedResult(
                tokens_seen=self._token_count,
                worst_ngram_count=self._worst_ngram_count,
                loop_confidence=self._loop_confidence,
            )

        # Run checks
        result = self._check_ngram_flood()
        if result:
            return result

        result = self._check_suffix_loop()
        if result:
            return result

        result = self._check_stagnation()
        if result:
            return result

        return FeedResult(
            tokens_seen=self._token_count,
            worst_ngram_count=self._worst_ngram_count,
            loop_confidence=self._loop_confidence,
        )

    # ──────────────────────────────────────────────────────
    # N-Gram Tracking
    # ──────────────────────────────────────────────────────

    def _update_ngrams(self, new_words: list[str]) -> None:
        """Incrementally update n-gram counts with newly arrived tokens.

        Only computes n-grams that include at least one of the new tokens,
        avoiding redundant work on the full token history.

        Args:
            new_words: Newly extracted word tokens from the latest chunk.
        """
        n: int = self.config.ngram_n
        tokens: list[str] = self._tokens

        if len(tokens) < n:
            return

        start: int = max(0, len(tokens) - len(new_words) - n + 1)
        for i in range(start, len(tokens) - n + 1):
            gram: str = " ".join(tokens[i : i + n])
            self._ngram_counts[gram] += 1

            if self._ngram_counts[gram] > self._worst_ngram_count:
                self._worst_ngram_count = self._ngram_counts[gram]

        # Freeze early n-grams once minimum token count is reached
        if not self._early_ngrams_frozen:
            if self._token_count >= self.config.min_tokens_before_check:
                self._early_ngrams = set(self._ngram_counts.keys())
                self._early_ngrams_frozen = True

    # ──────────────────────────────────────────────────────
    # Check 1: N-Gram Flood
    # ──────────────────────────────────────────────────────

    def _effective_ngram_max(self) -> int:
        """Compute the dynamic n-gram count threshold.

        Scales with token count so that longer outputs tolerate more
        natural repetition of domain-specific phrases.

        Returns:
            The effective maximum allowed count for any single n-gram.
        """
        base: int = self.config.ngram_max_count
        scale: int = self.config.ngram_count_scale_tokens
        if scale <= 0:
            return base
        return base + self._token_count // scale

    def _check_ngram_flood(self) -> FeedResult | None:
        """Check whether any n-gram has exceeded the maximum allowed count.

        Two conditions must BOTH be met:
        1. The worst n-gram count exceeds the dynamic threshold.
        2. The worst n-gram's density (``count * n / total_tokens``)
           exceeds ``ngram_min_ratio``, ensuring short domain phrases
           that recur naturally in long outputs don't trigger falsely.

        Returns:
            A :class:`FeedResult` with ``should_stop=True`` if both
            thresholds are exceeded, or ``None`` otherwise.
        """
        effective_max: int = self._effective_ngram_max()
        if self._worst_ngram_count < effective_max:
            return None

        # Ratio gate: is this n-gram dense enough to be a real problem?
        n: int = self.config.ngram_n
        ratio: float = (self._worst_ngram_count * n) / max(self._token_count, 1)
        if ratio < self.config.ngram_min_ratio:
            return None

        worst_gram: tuple[str, int] = self._ngram_counts.most_common(1)[0]
        top_ngrams: list[tuple[str, int]] = self._ngram_counts.most_common(10)

        self._mark_loop_start()
        return FeedResult(
            should_stop=True,
            reason=StopReason.NGRAM_FLOOD,
            detail=f"N-Gram '{worst_gram[0]}' appeared {worst_gram[1]}× "
            f"(limit: {effective_max}, ratio: {ratio:.1%})",
            tokens_seen=self._token_count,
            worst_ngram_count=self._worst_ngram_count,
            loop_confidence=1.0,
            diagnostics={
                "trigger": "NGRAM_FLOOD",
                "worst_ngram": worst_gram[0],
                "worst_count": worst_gram[1],
                "effective_max": effective_max,
                "ratio": round(ratio, 4),
                "total_tokens": self._token_count,
                "top_ngrams": {gram: count for gram, count in top_ngrams},
            },
        )

    # ──────────────────────────────────────────────────────
    # Check 2: Suffix Loop
    # ──────────────────────────────────────────────────────

    def _check_suffix_loop(self) -> FeedResult | None:
        """Check whether the text tail contains a repeating block pattern.

        Takes the last ``suffix_window`` characters and tries pattern lengths
        from ``suffix_min_pattern`` upward. For each length, checks whether
        the last block matches the preceding block (with 90% fuzzy matching).

        Complexity is O(suffix_window²) worst case, but bounded by the
        window size and ``check_every_n`` throttling.

        Returns:
            A :class:`FeedResult` with ``should_stop=True`` if a repeating
            pattern is found, or ``None`` otherwise.
        """
        text: str = self._full_text
        window: int = self.config.suffix_window
        min_pat: int = self.config.suffix_min_pattern

        tail: str = text[-window:] if len(text) > window else text
        tail_len: int = len(tail)

        if tail_len < min_pat * 2:
            return None

        best_reps: int = 0
        best_pat_len: int = 0

        for pat_len in range(min_pat, tail_len // 2 + 1, 5):
            pattern: str = tail[-pat_len:]

            reps: int = 1
            pos: int = tail_len - pat_len * 2
            while pos >= 0:
                block: str = tail[pos : pos + pat_len]
                matches: int = sum(a == b for a, b in zip(block, pattern))
                if matches / pat_len >= 0.9:
                    reps += 1
                    pos -= pat_len
                else:
                    break

            if reps >= self.config.suffix_min_reps and reps > best_reps:
                best_reps = reps
                best_pat_len = pat_len

        if best_reps >= self.config.suffix_min_reps:
            raw_confidence: float = min(best_reps / (self.config.suffix_min_reps + 1), 1.0)
            self._loop_confidence = max(self._loop_confidence, raw_confidence)

            if best_pat_len >= min_pat:
                pattern_text: str = tail[-best_pat_len:]

                # Skip markdown structural patterns (table separators, thematic
                # breaks, horizontal rules). These are inherently repetitive
                # (e.g. "---|---|---") but are not content loops.
                stripped: str = pattern_text.strip()
                if re.fullmatch(r"[\s|:*_\-=]+", stripped):
                    return None

                self._mark_loop_start()
                return FeedResult(
                    should_stop=True,
                    reason=StopReason.SUFFIX_LOOP,
                    detail=f"Loop detected: {best_pat_len} char pattern × {best_reps} " f"(tail of {tail_len} chars)",
                    tokens_seen=self._token_count,
                    worst_ngram_count=self._worst_ngram_count,
                    loop_confidence=self._loop_confidence,
                    diagnostics={
                        "trigger": "SUFFIX_LOOP",
                        "pattern_text": pattern_text[:300] + ("..." if len(pattern_text) > 300 else ""),
                        "pattern_length": best_pat_len,
                        "repetitions": best_reps,
                        "tail_length": tail_len,
                        "total_tokens": self._token_count,
                    },
                )
        else:
            self._loop_confidence *= 0.95

        return None

    # ──────────────────────────────────────────────────────
    # Check 3: Stagnation
    # ──────────────────────────────────────────────────────

    def _check_stagnation(self) -> FeedResult | None:
        """Detect when the model is only recycling previously seen phrases.

        Less aggressive than the other checks -- catches the subtler
        "going in circles" behaviour where the text is not exactly repeated
        but consists entirely of previously seen vocabulary.

        Compares n-grams in the recent window against n-grams from the
        early phase of generation. If the overlap exceeds
        ``stagnation_threshold``, the model is considered stagnant.

        Returns:
            A :class:`FeedResult` with ``should_stop=True`` if stagnation
            is detected, or ``None`` otherwise.
        """
        if not self._early_ngrams_frozen:
            return None

        n: int = self.config.ngram_n
        w: int = self.config.stagnation_window
        tokens: list[str] = self._tokens

        if len(tokens) < w + n:
            return None

        recent_tokens: list[str] = tokens[-w:]
        recent_grams: set[str] = set()
        for i in range(len(recent_tokens) - n + 1):
            gram: str = " ".join(recent_tokens[i : i + n])
            recent_grams.add(gram)

        if not recent_grams:
            return None

        recycled: set[str] = recent_grams & self._early_ngrams
        new_only: set[str] = recent_grams - self._early_ngrams
        recycled_ratio: float = len(recycled) / len(recent_grams)

        if recycled_ratio >= self.config.stagnation_threshold:
            self._mark_loop_start()
            return FeedResult(
                should_stop=True,
                reason=StopReason.STAGNATION,
                detail=f"Stagnation: {recycled_ratio:.0%} of recent {n}-grams "
                f"are recycled ({len(recycled)}/{len(recent_grams)})",
                tokens_seen=self._token_count,
                worst_ngram_count=self._worst_ngram_count,
                loop_confidence=recycled_ratio,
                diagnostics={
                    "trigger": "STAGNATION",
                    "recycled_ratio": round(recycled_ratio, 4),
                    "recycled_count": len(recycled),
                    "recent_count": len(recent_grams),
                    "stagnation_window": w,
                    "total_tokens": self._token_count,
                    "sample_recycled": sorted(recycled)[:10],
                    "sample_new": sorted(new_only)[:5],
                },
            )

        return None

    # ──────────────────────────────────────────────────────
    # Text Cleanup
    # ──────────────────────────────────────────────────────

    def _mark_loop_start(self) -> None:
        """Record the approximate position where the repetition began.

        Uses a heuristic: the loop probably started around
        ``text_length - suffix_window``. Snaps backward to the nearest
        sentence boundary for a clean cut.
        """
        text: str = self._full_text
        approx_start: int = max(0, len(text) - self.config.suffix_window)

        for end_marker in [". ", ".\n", "!\n", "? ", "?\n"]:
            pos: int = text.rfind(end_marker, 0, approx_start + 100)
            if pos != -1:
                self._last_clean_pos = max(self._last_clean_pos, pos + len(end_marker))
                return

        pos = text.rfind("\n", 0, approx_start + 100)
        if pos != -1:
            self._last_clean_pos = max(self._last_clean_pos, pos + 1)

    def get_clean_text(self) -> str:
        """Return the text truncated before the detected repetition.

        Useful for showing the user the "good" part without the
        repetition loop at the end.

        Returns:
            The accumulated text up to the point where repetition was
            first detected, trimmed of trailing whitespace.
        """
        if self._last_clean_pos > 0:
            return self._full_text[: self._last_clean_pos].rstrip()
        return self._full_text.rstrip()

    def get_full_text(self) -> str:
        """Return the full accumulated text including any repetitions.

        Returns:
            The complete text as received via :meth:`feed` calls.
        """
        return self._full_text

    def get_stats(self) -> dict[str, int | float | dict[str, int]]:
        """Return debug information about the current guard state.

        Returns:
            A dictionary with keys: ``tokens_seen``, ``feeds``,
            ``text_length``, ``worst_ngram_count``, ``loop_confidence``,
            ``clean_text_length``, and ``top_ngrams`` (the 5 most
            frequent n-grams with their counts).
        """
        top_ngrams: list[tuple[str, int]] = self._ngram_counts.most_common(5)
        return {
            "tokens_seen": self._token_count,
            "feeds": self._feed_count,
            "text_length": len(self._full_text),
            "worst_ngram_count": self._worst_ngram_count,
            "ngram_effective_max": self._effective_ngram_max(),
            "loop_confidence": round(self._loop_confidence, 3),
            "clean_text_length": self._last_clean_pos or len(self._full_text),
            "top_ngrams": {gram: count for gram, count in top_ngrams},
        }


# ──────────────────────────────────────────────────────────────
# Convenience: Wrapper for OpenAI-compatible streams
# ──────────────────────────────────────────────────────────────


def guarded_stream(
    stream_iterator: Iterator[_T],
    config: GuardConfig | None = None,
    on_stop: Callable[[FeedResult, RepetitionGuard, str], None] | None = None,
    extract_token: Callable[[_T], tuple[str, str]] | None = None,
) -> Generator[_T, None, None]:
    """Generator wrapper that guards an LLM stream against repetition.

    Wraps an existing stream iterator and yields chunks through until
    repetition is detected. Monitors both ``delta.content`` and
    ``delta.reasoning_content`` (thinking tokens) with independent
    guards so that repetition in the thinking phase is caught
    independently of content.

    Args:
        stream_iterator: An iterator yielding stream chunks. Typically
            the response from ``openai.chat.completions.create(stream=True)``
            or an SGLang streaming response.
        config: Detection thresholds. Uses :class:`GuardConfig` defaults
            if not provided.
        on_stop: Callback invoked when repetition is detected. Receives
            ``(result, guard, source)`` where ``source`` is ``"content"``
            or ``"reasoning"``. Useful for logging or retrying with
            higher ``repetition_penalty``.
        extract_token: Function that extracts a ``(content, reasoning)``
            tuple from a stream chunk. Defaults to the OpenAI-compatible
            format using ``delta.content`` and ``delta.reasoning_content``.

    Yields:
        The original stream chunks, as long as no repetition is detected.

    Example:
        >>> import openai
        >>>
        >>> client = openai.OpenAI(base_url="http://localhost:30000/v1")
        >>> response = client.chat.completions.create(
        ...     model="Qwen3-235B-A22B-Thinking-2507-AWQ",
        ...     messages=[{"role": "user", "content": "Explain quantum physics"}],
        ...     stream=True,
        ... )
        >>>
        >>> for chunk in guarded_stream(response):
        ...     token = chunk.choices[0].delta.content or ""
        ...     print(token, end="", flush=True)
    """
    _extract: Callable[[_T], tuple[str, str]]
    if extract_token is not None:
        _extract = extract_token
    else:

        def _extract(chunk: _T) -> tuple[str, str]:
            try:
                delta = chunk.choices[0].delta  # type: ignore[attr-defined]
                content: str = getattr(delta, "content", None) or ""
                reasoning: str = getattr(delta, "reasoning_content", None) or ""
                return content, reasoning
            except (AttributeError, IndexError):
                return "", ""

    content_guard: RepetitionGuard = RepetitionGuard(config)
    reasoning_guard: RepetitionGuard = RepetitionGuard(config)

    for chunk in stream_iterator:
        content, reasoning = _extract(chunk)

        if reasoning:
            result: FeedResult = reasoning_guard.feed(reasoning)
            if result.should_stop:
                if on_stop:
                    on_stop(result, reasoning_guard, "reasoning")
                return

        if content:
            result = content_guard.feed(content)
            if result.should_stop:
                if on_stop:
                    on_stop(result, content_guard, "content")
                return

        yield chunk


# ──────────────────────────────────────────────────────────────
# Demo / Test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Simulated LLM stream that falls into a loop
    simulated_tokens: str = (
        # Normal start
        "Quantum physics is a fascinating field of modern physics. "
        "It deals with the behaviour of particles at the subatomic level. "
        "The wave function describes the state of a quantum system and "
        "contains all measurable information. The superposition principle "
        "states that a particle can exist in multiple states simultaneously. "
        # Loop begins (typical LLM behaviour)
        "It is important to understand that quantum physics has many "
        "practical applications. Quantum physics has many practical "
        "applications in modern technology. "
        "It is important to understand that quantum physics has many "
        "practical applications. Quantum physics has many practical "
        "applications in modern technology. "
        "It is important to understand that quantum physics has many "
        "practical applications. Quantum physics has many practical "
        "applications in modern technology. "
        "It is important to understand that quantum physics has many "
        "practical applications. Quantum physics has many practical "
        "applications in modern technology. "
    )

    print("=" * 60)
    print("STREAMING REPETITION GUARD — DEMO")
    print("=" * 60)
    print()

    guard = RepetitionGuard(
        GuardConfig(
            ngram_max_count=4,
            min_tokens_before_check=20,
        )
    )

    words: list[str] = simulated_tokens.split(" ")
    stopped: bool = False

    for i, word in enumerate(words):
        token: str = word + " "
        result: FeedResult = guard.feed(token)

        if result.should_stop:
            print(f"\n\n{'─' * 40}")
            print(f"STOPPED at token #{result.tokens_seen}")
            print(f"   Reason:     {result.reason.name if result.reason else 'unknown'}")
            print(f"   Detail:     {result.detail}")
            print(f"   Confidence: {result.loop_confidence:.0%}")
            if result.diagnostics:
                import json as _json

                print(f"   Diagnostics:")
                for k, v in result.diagnostics.items():
                    print(
                        f"      {k}: {_json.dumps(v, ensure_ascii=False, default=str) if isinstance(v, (dict, list)) else v}"
                    )
            stopped = True
            break
        else:
            print(word, end=" ", flush=True)

    if not stopped:
        print("\n\n(Stream ended without triggering guard)")

    print(f"\n{'─' * 40}")
    print(f"Stats: {guard.get_stats()}")

    clean: str = guard.get_clean_text()
    print(f"\nClean text ({len(clean)} chars):")
    print(f'  "{clean[:120]}..."')
    print()
