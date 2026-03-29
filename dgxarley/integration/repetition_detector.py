"""Detect various forms of repetition in LLM outputs.

This module provides three detection layers that can be used independently
or combined via the convenience function :func:`detect_repetition`:

1. **N-Gram Repetition** -- Repeated word groups (e.g. identical 4-word phrases).
2. **Sentence Repetition** -- Semantically identical or near-identical sentences.
3. **Loop Detection** -- Repeating text blocks (the typical LLM "stuck in a loop").

Example:
    >>> from dgxarley.integration.repetition_detector import detect_repetition
    >>> text = "This is a test. This is a test. Really a test."
    >>> report = detect_repetition(text)
    >>> print(report.summary())
    [LOW] score=0.12 — N-Gram 'this is a test' x2
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

# ──────────────────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────────────────


@dataclass
class NGramHit:
    """A repeated n-gram with frequency metadata.

    Attributes:
        ngram: The repeated phrase as a space-joined lowercase string.
        n: Length of the n-gram in tokens.
        count: Number of occurrences in the text.
        ratio: Fraction of total tokens consumed by this n-gram
            (``count * n / total_tokens``).
    """

    ngram: str
    n: int
    count: int
    ratio: float


@dataclass
class SentenceHit:
    """A pair of sentences with high textual similarity.

    Attributes:
        sentence_a: First sentence of the pair.
        sentence_b: Second sentence of the pair.
        similarity: Similarity score between 0.0 and 1.0 as computed
            by :class:`difflib.SequenceMatcher`.
        index_a: Zero-based index of the first sentence in the sentence list.
        index_b: Zero-based index of the second sentence.
    """

    sentence_a: str
    sentence_b: str
    similarity: float
    index_a: int
    index_b: int


@dataclass
class LoopHit:
    """A detected loop block -- a text segment that repeats consecutively.

    Attributes:
        pattern: The repeating text block (truncated to 200 chars for display).
        length_chars: Length of the repeating block in characters.
        repetitions: Number of consecutive occurrences.
        start_pos: Character offset where the loop begins in the source text.
    """

    pattern: str
    length_chars: int
    repetitions: int
    start_pos: int


@dataclass
class RepetitionReport:
    """Combined result of all three repetition analyses.

    Attributes:
        ngram_score: N-gram repetition score (0.0 = none, 1.0 = extreme).
        sentence_score: Sentence repetition score (0.0 = none, 1.0 = extreme).
        loop_score: Loop detection score (0.0 = none, 1.0 = extreme).
        overall_score: Weighted combination of all three scores.
        severity: Human-readable severity level -- one of
            ``"none"``, ``"low"``, ``"medium"``, ``"high"``, ``"critical"``.
        ngram_hits: List of detected repeated n-grams, sorted by impact.
        sentence_hits: List of similar sentence pairs, sorted by similarity.
        loop_hits: List of detected loops, sorted by wasted characters.
        total_tokens: Total number of word-level tokens in the analysed text.
        total_sentences: Total number of sentences in the analysed text.
    """

    ngram_score: float = 0.0
    sentence_score: float = 0.0
    loop_score: float = 0.0
    overall_score: float = 0.0

    severity: str = "none"

    ngram_hits: list[NGramHit] = field(default_factory=list)
    sentence_hits: list[SentenceHit] = field(default_factory=list)
    loop_hits: list[LoopHit] = field(default_factory=list)

    total_tokens: int = 0
    total_sentences: int = 0

    def summary(self) -> str:
        """Return a human-readable one-line summary of the report.

        Returns:
            A string like ``[HIGH] score=0.42 — N-Gram 'foo bar' ×7; Loop (84 chars) ×3``.
        """
        parts: list[str] = []
        if self.ngram_hits:
            top = self.ngram_hits[0]
            parts.append(f"N-Gram '{top.ngram}' ×{top.count}")
        if self.sentence_hits:
            parts.append(f"{len(self.sentence_hits)} similar sentence pairs")
        if self.loop_hits:
            top_loop = self.loop_hits[0]
            parts.append(f"Loop ({top_loop.length_chars} chars) ×{top_loop.repetitions}")
        detail = "; ".join(parts) if parts else "no issues"
        return f"[{self.severity.upper()}] score={self.overall_score:.2f} — {detail}"


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens using whitespace and punctuation boundaries.

    This is a simple tokenizer sufficient for repetition detection --
    no BPE or SentencePiece required.

    Args:
        text: The input text to tokenize.

    Returns:
        A list of lowercase word tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation boundaries.

    Works reasonably well for common European languages.

    Args:
        text: The input text to split.

    Returns:
        A list of sentence strings, each at least 6 characters long.
    """
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 5]


# ──────────────────────────────────────────────────────────────
# 1) N-Gram Repetition
# ──────────────────────────────────────────────────────────────


def detect_ngram_repetition(
    text: str,
    ns: tuple[int, ...] = (3, 4, 5, 6, 8),
    min_count: int = 3,
    top_k: int = 10,
) -> tuple[float, list[NGramHit]]:
    """Find repeated n-grams and compute an impact score.

    For each n-gram length in ``ns``, counts all occurrences. N-grams
    appearing at least ``min_count`` times are reported. Shorter n-grams
    that are substrings of already-reported longer n-grams are deduplicated.

    The score represents the fraction of "redundant" tokens:
    ``sum((count - 1) * n / total_tokens)`` for each hit, capped at 1.0.

    Args:
        text: The text to analyse.
        ns: N-gram lengths to check.
        min_count: Minimum number of occurrences to qualify as a hit.
        top_k: Maximum number of hits to return, sorted by impact.

    Returns:
        A tuple of ``(score, hits)`` where score is 0.0--1.0 and hits is a
        list of :class:`NGramHit` sorted by descending impact.
    """
    tokens: list[str] = _tokenize(text)
    total: int = len(tokens)

    if total < max(ns):
        return 0.0, []

    all_hits: list[NGramHit] = []

    for n in ns:
        ngrams: list[str] = [" ".join(tokens[i : i + n]) for i in range(total - n + 1)]
        counts: Counter[str] = Counter(ngrams)

        for gram, count in counts.items():
            if count >= min_count:
                ratio: float = (count * n) / total
                all_hits.append(
                    NGramHit(
                        ngram=gram,
                        n=n,
                        count=count,
                        ratio=ratio,
                    )
                )

    # Deduplicate: if "a b c d" is already reported, skip "a b c"
    all_hits.sort(key=lambda h: (h.n, h.count), reverse=True)
    seen_substrings: set[str] = set()
    filtered: list[NGramHit] = []
    for hit in all_hits:
        if not any(hit.ngram in s for s in seen_substrings):
            filtered.append(hit)
            seen_substrings.add(hit.ngram)

    filtered.sort(key=lambda h: (h.count - 1) * h.n, reverse=True)
    filtered = filtered[:top_k]

    score: float = sum((h.count - 1) * h.n / total for h in filtered)
    score = min(score, 1.0)

    return score, filtered


# ──────────────────────────────────────────────────────────────
# 2) Sentence Repetition
# ──────────────────────────────────────────────────────────────


def detect_sentence_repetition(
    text: str,
    similarity_threshold: float = 0.75,
    max_comparisons: int = 50_000,
) -> tuple[float, list[SentenceHit]]:
    """Find sentence pairs with high textual similarity.

    Compares all sentence pairs using :class:`difflib.SequenceMatcher`.
    The score is the fraction of sentences involved in at least one
    similar pair, weighted by average similarity.

    Args:
        text: The text to analyse.
        similarity_threshold: Minimum similarity ratio (0.0--1.0) for a
            pair to be reported. 0.75 means "nearly identical".
        max_comparisons: Safety limit on pairwise comparisons for very
            long texts (O(n²) complexity).

    Returns:
        A tuple of ``(score, hits)`` where score is 0.0--1.0 and hits is a
        list of :class:`SentenceHit` sorted by descending similarity.
    """
    sentences: list[str] = _split_sentences(text)
    n: int = len(sentences)
    hits: list[SentenceHit] = []

    if n < 2:
        return 0.0, []

    involved: set[int] = set()
    comparison_count: int = 0

    for i in range(n):
        for j in range(i + 1, n):
            comparison_count += 1
            if comparison_count > max_comparisons:
                break

            sim: float = SequenceMatcher(None, sentences[i].lower(), sentences[j].lower()).ratio()

            if sim >= similarity_threshold:
                hits.append(
                    SentenceHit(
                        sentence_a=sentences[i],
                        sentence_b=sentences[j],
                        similarity=sim,
                        index_a=i,
                        index_b=j,
                    )
                )
                involved.update([i, j])

    if not hits:
        return 0.0, hits

    avg_sim: float = sum(h.similarity for h in hits) / len(hits)
    affected_ratio: float = len(involved) / n
    score: float = min(affected_ratio * avg_sim, 1.0)

    hits.sort(key=lambda h: h.similarity, reverse=True)
    return score, hits


# ──────────────────────────────────────────────────────────────
# 3) Loop Detection (consecutively repeating blocks)
# ──────────────────────────────────────────────────────────────


def detect_loops(
    text: str,
    min_pattern_len: int = 20,
    max_pattern_len: int = 500,
    min_repetitions: int = 2,
    step: int = 10,
) -> tuple[float, list[LoopHit]]:
    """Detect loop patterns: text blocks that repeat consecutively.

    This catches the typical LLM failure mode where the model generates
    the same paragraph multiple times in a row.

    The algorithm uses a sliding window over various pattern lengths.
    For each position and length, it checks whether the next block is
    identical, then counts the chain of consecutive repetitions.

    The score represents the fraction of text "wasted" by loop
    repetitions: ``sum(length * (reps - 1)) / text_length``.

    Args:
        text: The text to analyse.
        min_pattern_len: Shortest block (in characters) that qualifies
            as a loop pattern.
        max_pattern_len: Longest block to check.
        min_repetitions: Minimum consecutive repetitions to qualify.
        step: Step size for pattern lengths (performance tuning).

    Returns:
        A tuple of ``(score, hits)`` where score is 0.0--1.0 and hits is a
        list of :class:`LoopHit` sorted by descending wasted characters.
    """
    text_len: int = len(text)
    hits: list[LoopHit] = []
    covered_ranges: list[tuple[int, int]] = []

    for plen in range(min_pattern_len, min(max_pattern_len, text_len // 2) + 1, step):
        pos: int = 0
        while pos <= text_len - plen * 2:
            pattern: str = text[pos : pos + plen]

            if not pattern.strip():
                pos += 1
                continue

            reps: int = 1
            check_pos: int = pos + plen
            while check_pos + plen <= text_len:
                if text[check_pos : check_pos + plen] == pattern:
                    reps += 1
                    check_pos += plen
                else:
                    break

            if reps >= min_repetitions:
                loop_start: int = pos
                loop_end: int = pos + plen * reps
                already_covered: bool = any(s <= loop_start and e >= loop_end for s, e in covered_ranges)
                if not already_covered:
                    hits.append(
                        LoopHit(
                            pattern=pattern[:200] + ("..." if len(pattern) > 200 else ""),
                            length_chars=plen,
                            repetitions=reps,
                            start_pos=pos,
                        )
                    )
                    covered_ranges.append((loop_start, loop_end))
                pos = check_pos
            else:
                pos += 1

    wasted_chars: int = sum(h.length_chars * (h.repetitions - 1) for h in hits)
    score: float = min(wasted_chars / max(text_len, 1), 1.0)

    hits.sort(key=lambda h: h.length_chars * h.repetitions, reverse=True)
    return score, hits


# ──────────────────────────────────────────────────────────────
# Main Function
# ──────────────────────────────────────────────────────────────


def detect_repetition(
    text: str,
    *,
    ngram_ns: tuple[int, ...] = (3, 4, 5, 6, 8),
    ngram_min_count: int = 3,
    sentence_similarity_threshold: float = 0.75,
    loop_min_pattern_len: int = 20,
    loop_max_pattern_len: int = 500,
    loop_min_repetitions: int = 2,
    weights: tuple[float, float, float] = (0.3, 0.3, 0.4),
) -> RepetitionReport:
    """Run all three detection layers and produce a combined report.

    This is the main entry point for analysing LLM output. It runs
    n-gram, sentence, and loop detection, then combines their scores
    into a single weighted overall score with a severity classification.

    Severity levels:
        - ``"none"``:     overall_score < 0.05 -- No issues.
        - ``"low"``:      overall_score < 0.15 -- Mild repetition, usually acceptable.
        - ``"medium"``:   overall_score < 0.30 -- Noticeable, consider tuning sampling.
        - ``"high"``:     overall_score < 0.50 -- Clear problem.
        - ``"critical"``: overall_score >= 0.50 -- Text is essentially broken.

    Args:
        text: The LLM output to analyse.
        ngram_ns: N-gram lengths to check. See :func:`detect_ngram_repetition`.
        ngram_min_count: Minimum n-gram occurrences to qualify as a hit.
        sentence_similarity_threshold: Similarity threshold for sentence
            pairs. See :func:`detect_sentence_repetition`.
        loop_min_pattern_len: Shortest loop pattern in characters.
            See :func:`detect_loops`.
        loop_max_pattern_len: Longest loop pattern in characters.
        loop_min_repetitions: Minimum consecutive repetitions for loops.
        weights: Score weights as ``(ngram, sentence, loop)``. Default
            gives loops the highest weight (0.4) since they represent
            the most severe failure mode.

    Returns:
        A :class:`RepetitionReport` with scores, hits, and severity.

    Example:
        >>> report = detect_repetition(llm_output)
        >>> print(report.summary())
        [MEDIUM] score=0.27 — N-Gram 'is an important field' x7; Loop (84 chars) x3
        >>> if report.overall_score > 0.2:
        ...     print("Consider increasing repetition_penalty!")
    """
    tokens: list[str] = _tokenize(text)
    sentences: list[str] = _split_sentences(text)

    ngram_score, ngram_hits = detect_ngram_repetition(
        text,
        ns=ngram_ns,
        min_count=ngram_min_count,
    )
    sentence_score, sentence_hits = detect_sentence_repetition(
        text,
        similarity_threshold=sentence_similarity_threshold,
    )
    loop_score, loop_hits = detect_loops(
        text,
        min_pattern_len=loop_min_pattern_len,
        max_pattern_len=loop_max_pattern_len,
        min_repetitions=loop_min_repetitions,
    )

    w_ng, w_sent, w_loop = weights
    overall: float = w_ng * ngram_score + w_sent * sentence_score + w_loop * loop_score
    overall = min(overall, 1.0)

    if overall < 0.05:
        severity = "none"
    elif overall < 0.15:
        severity = "low"
    elif overall < 0.30:
        severity = "medium"
    elif overall < 0.50:
        severity = "high"
    else:
        severity = "critical"

    return RepetitionReport(
        ngram_score=ngram_score,
        sentence_score=sentence_score,
        loop_score=loop_score,
        overall_score=overall,
        severity=severity,
        ngram_hits=ngram_hits,
        sentence_hits=sentence_hits,
        loop_hits=loop_hits,
        total_tokens=len(tokens),
        total_sentences=len(sentences),
    )


# ──────────────────────────────────────────────────────────────
# CLI / Quick Test
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_text = """
    Artificial intelligence is an important field of computer science.
    It encompasses machine learning and deep learning.
    Artificial intelligence is an important field of modern research.
    Machine learning is a subfield of artificial intelligence.
    Artificial intelligence is an important field of computer science.
    The model learns from data and improves continuously.
    The model learns from data and keeps improving steadily.
    The model learns from data and improves over time.
    It is important to note that AI has many applications.
    It is important to note that AI offers many advantages.
    It is important to note that AI will shape the future.
    In summary, it can be said that artificial intelligence
    is an important field. In summary, it can be said that
    artificial intelligence is an important field.
    In summary, it can be said that artificial intelligence
    is an important field.
    """

    report = detect_repetition(test_text)

    print("=" * 60)
    print("REPETITION REPORT")
    print("=" * 60)
    print(f"\n  {report.summary()}\n")

    print(f"  Tokens: {report.total_tokens}  |  Sentences: {report.total_sentences}")
    print(f"  N-Gram Score:   {report.ngram_score:.3f}")
    print(f"  Sentence Score: {report.sentence_score:.3f}")
    print(f"  Loop Score:     {report.loop_score:.3f}")
    print(f"  Overall Score:  {report.overall_score:.3f}")
    print(f"  Severity:       {report.severity}")

    if report.ngram_hits:
        print("\n  Top N-Gram Hits:")
        for nh in report.ngram_hits[:5]:
            print(f"    '{nh.ngram}' — {nh.count}× (n={nh.n})")

    if report.sentence_hits:
        print("\n  Top Sentence Similarities:")
        for sh in report.sentence_hits[:5]:
            print(f'    [{sh.similarity:.0%}] "{sh.sentence_a[:60]}..."')
            print(f'         ↔ "{sh.sentence_b[:60]}..."')

    if report.loop_hits:
        print("\n  Loops:")
        for lh in report.loop_hits[:3]:
            print(f"    {lh.length_chars} chars × {lh.repetitions} reps @ pos {lh.start_pos}")
            print(f'    Pattern: "{lh.pattern[:80]}..."')

    print()
