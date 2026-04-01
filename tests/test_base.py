"""Tests for dgxarley."""

import dgxarley
from dgxarley.integration.repetition_detector import (
    RepetitionReport,
    detect_loops,
    detect_ngram_repetition,
    detect_repetition,
    detect_sentence_repetition,
)
from dgxarley.integration.streaming_repetition_guard import (
    FeedResult,
    GuardConfig,
    RepetitionGuard,
    StopReason,
)

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------


def test_version_exists() -> None:
    """Verify that the package has a version string."""
    assert hasattr(dgxarley, "__version__")
    assert isinstance(dgxarley.__version__, str)
    assert len(dgxarley.__version__) > 0


def test_version_format() -> None:
    """Verify version follows semver pattern."""
    version = dgxarley.__version__
    parts = version.split(".")
    assert len(parts) >= 2, "Version should have at least major.minor"
    for part in parts:
        assert part.isdigit() or part[0].isdigit(), f"Version part '{part}' should start with a digit"


# ---------------------------------------------------------------------------
# Repetition Detector — N-Gram
# ---------------------------------------------------------------------------


def test_ngram_no_repetition() -> None:
    """Unique text should produce zero score."""
    text = "The quick brown fox jumps over the lazy dog near a quiet river."
    score, hits = detect_ngram_repetition(text)
    assert score == 0.0
    assert hits == []


def test_ngram_detects_repeated_phrase() -> None:
    """Repeated phrase should be detected."""
    text = "This is a test sentence. This is a test sentence. " "This is a test sentence. This is a test sentence."
    score, hits = detect_ngram_repetition(text, ns=(4,), min_count=2)
    assert score > 0.0
    assert len(hits) > 0
    assert hits[0].count >= 2


def test_ngram_short_text_returns_empty() -> None:
    """Text shorter than the n-gram size should return empty."""
    score, hits = detect_ngram_repetition("hello world", ns=(8,))
    assert score == 0.0
    assert hits == []


# ---------------------------------------------------------------------------
# Repetition Detector — Sentence
# ---------------------------------------------------------------------------


def test_sentence_no_repetition() -> None:
    """Distinct sentences should not match."""
    text = "The sun is bright. Rain falls in winter. Birds fly south."
    score, hits = detect_sentence_repetition(text)
    assert score == 0.0
    assert hits == []


def test_sentence_detects_similar_pair() -> None:
    """Near-identical sentences should be detected."""
    text = (
        "Artificial intelligence is an important field of computer science. "
        "Artificial intelligence is an important field of modern research."
    )
    score, hits = detect_sentence_repetition(text, similarity_threshold=0.7)
    assert score > 0.0
    assert len(hits) == 1
    assert hits[0].similarity >= 0.7


# ---------------------------------------------------------------------------
# Repetition Detector — Loop
# ---------------------------------------------------------------------------


def test_loop_no_repetition() -> None:
    """Non-repeating text should produce zero score."""
    text = "A completely unique paragraph with no repeating blocks whatsoever in it."
    score, hits = detect_loops(text)
    assert score == 0.0
    assert hits == []


def test_loop_detects_repeated_block() -> None:
    """Consecutively repeated block should be detected."""
    block = "This is a repeated block of text that keeps appearing. "
    text = block * 5
    score, hits = detect_loops(text, min_pattern_len=20, min_repetitions=2)
    assert score > 0.0
    assert len(hits) > 0
    assert hits[0].repetitions >= 2


# ---------------------------------------------------------------------------
# Repetition Detector — Combined
# ---------------------------------------------------------------------------


def test_detect_repetition_clean_text() -> None:
    """Clean text should have severity 'none'."""
    text = (
        "Python is a versatile language used in web development. "
        "Rust provides memory safety without garbage collection. "
        "Go excels at building concurrent networked services."
    )
    report = detect_repetition(text)
    assert isinstance(report, RepetitionReport)
    assert report.severity == "none"
    assert report.overall_score < 0.05


def test_detect_repetition_loopy_text() -> None:
    """Heavily repeated text should have high severity."""
    block = "The model keeps generating the same text over and over again. "
    text = block * 20
    report = detect_repetition(text)
    assert report.severity in ("high", "critical")
    assert report.overall_score > 0.3


def test_report_summary() -> None:
    """Summary should be a non-empty string with severity."""
    report = RepetitionReport(severity="low", overall_score=0.1)
    summary = report.summary()
    assert "[LOW]" in summary
    assert "0.10" in summary


# ---------------------------------------------------------------------------
# Streaming Repetition Guard
# ---------------------------------------------------------------------------


def test_guard_no_repetition() -> None:
    """Normal text should not trigger the guard."""
    guard = RepetitionGuard(GuardConfig(min_tokens_before_check=5, check_every_n=1))
    tokens = "The quick brown fox jumps over the lazy dog in the park".split()
    for token in tokens:
        result = guard.feed(token + " ")
        assert not result.should_stop


def test_guard_detects_ngram_flood() -> None:
    """Repeated phrase should trigger NGRAM_FLOOD."""
    guard = RepetitionGuard(
        GuardConfig(
            ngram_max_count=3,
            ngram_min_ratio=0.01,
            min_tokens_before_check=10,
            check_every_n=1,
        )
    )
    phrase = "quantum physics has many practical applications in technology "
    stopped = False
    for _ in range(30):
        result = guard.feed(phrase)
        if result.should_stop:
            assert result.reason == StopReason.NGRAM_FLOOD
            stopped = True
            break
    assert stopped, "Guard should have triggered NGRAM_FLOOD"


def test_guard_reset() -> None:
    """Reset should clear all state."""
    guard = RepetitionGuard()
    guard.feed("some tokens to build state ")
    guard.reset()
    stats = guard.get_stats()
    assert stats["tokens_seen"] == 0
    assert stats["feeds"] == 0


def test_guard_get_clean_text() -> None:
    """get_clean_text returns accumulated text."""
    guard = RepetitionGuard()
    guard.feed("Hello ")
    guard.feed("world")
    assert guard.get_clean_text() == "Hello world"


def test_guard_get_full_text() -> None:
    """get_full_text returns all accumulated text."""
    guard = RepetitionGuard()
    guard.feed("Hello ")
    guard.feed("world")
    assert guard.get_full_text() == "Hello world"


def test_feed_empty_chunk() -> None:
    """Empty chunk should be a no-op."""
    guard = RepetitionGuard()
    result = guard.feed("")
    assert not result.should_stop
    assert result.tokens_seen == 0


def test_guard_config_defaults() -> None:
    """GuardConfig defaults should be sensible."""
    cfg = GuardConfig()
    assert cfg.ngram_n == 4
    assert cfg.ngram_max_count == 8
    assert cfg.suffix_min_reps == 4
    assert cfg.min_tokens_before_check == 40
    assert cfg.check_every_n == 3


def test_guard_suffix_loop_ignores_structured_data() -> None:
    """Structured/tabular data with 2 similar lines should NOT trigger SUFFIX_LOOP.

    Regression test for false positive on DNS root server NS records where
    adjacent lines like '.  518400  IN  NS  a.root-servers.net' and
    '.  518400  IN  NS  b.root-servers.net' share >90% character similarity.
    """
    guard = RepetitionGuard(
        GuardConfig(
            min_tokens_before_check=5,
            check_every_n=1,
            # Disable ngram/stagnation to isolate suffix loop
            ngram_max_count=999,
            stagnation_threshold=1.0,
        )
    )
    # Simulate DNS records — each line is ~55 chars, differ only in server letter
    preamble = "Here are the root servers for the DNS root zone:\n"
    records = (
        ".            518400    IN    NS    a.root-servers.net.\n"
        ".            518400    IN    NS    b.root-servers.net.\n"
    )
    text = preamble + records
    for word in text.split():
        result = guard.feed(word + " ")
        assert not result.should_stop, f"False positive on structured data: {result.detail}"


def test_guard_suffix_loop_detects_three_reps() -> None:
    """Three repetitions of a pattern should trigger SUFFIX_LOOP."""
    guard = RepetitionGuard(
        GuardConfig(
            min_tokens_before_check=5,
            check_every_n=1,
            # Disable ngram/stagnation to isolate suffix loop
            ngram_max_count=999,
            stagnation_threshold=1.0,
        )
    )
    block = "This is a block that repeats verbatim in the output stream. "
    text = "Some preamble text here. " + block * 4
    stopped = False
    for word in text.split():
        result = guard.feed(word + " ")
        if result.should_stop:
            assert result.reason == StopReason.SUFFIX_LOOP
            stopped = True
            break
    assert stopped, "Guard should have triggered SUFFIX_LOOP on 4 repetitions"


# ---------------------------------------------------------------------------
# Thinking Parser
# ---------------------------------------------------------------------------

from dgxarley.integration.thinking_parser import ThinkingParser


def test_thinking_parser_reasoning_content_passthrough() -> None:
    """Server-separated reasoning_content should pass through directly."""
    p = ThinkingParser()
    r = p.feed(reasoning_content="Let me think...")
    assert r.thinking == "Let me think..."
    assert r.content == ""
    assert r.is_thinking is True

    r = p.feed(content="The answer is 42.")
    assert r.thinking == ""
    assert r.content == "The answer is 42."
    assert r.is_thinking is False


def test_thinking_parser_inline_tags() -> None:
    """Inline <think>...</think> in content should be separated."""
    p = ThinkingParser()
    r = p.feed(content="<think>Hmm let me consider</think>The answer is 42.")
    assert r.thinking == "Hmm let me consider"
    assert r.content == "The answer is 42."
    assert r.is_thinking is False


def test_thinking_parser_split_across_chunks() -> None:
    """Tags split across chunk boundaries should be handled."""
    p = ThinkingParser()
    r1 = p.feed(content="<think>thinking")
    assert r1.thinking == "thinking"
    assert r1.is_thinking is True

    r2 = p.feed(content=" more thoughts</think>answer")
    assert r2.thinking == " more thoughts"
    assert r2.content == "answer"
    assert r2.is_thinking is False


def test_thinking_parser_partial_tag_boundary() -> None:
    """Partial tag at chunk boundary should buffer correctly."""
    p = ThinkingParser()
    r1 = p.feed(content="<think>thoughts</thi")
    assert r1.thinking == "thoughts"
    assert r1.is_thinking is True

    r2 = p.feed(content="nk>content here")
    assert r2.content == "content here"
    assert r2.is_thinking is False


def test_thinking_parser_no_tags() -> None:
    """Plain content without tags should pass through as content."""
    p = ThinkingParser()
    r = p.feed(content="Just regular content")
    assert r.thinking == ""
    assert r.content == "Just regular content"
    assert r.is_thinking is False


def test_thinking_parser_token_estimates() -> None:
    """Token estimates should accumulate correctly."""
    p = ThinkingParser()
    p.feed(content="<think>" + "x" * 40 + "</think>" + "y" * 80)
    assert p.thinking_chars == 40
    assert p.content_chars == 80
    assert p.thinking_tokens_est == 10
    assert p.content_tokens_est == 20
    assert p.total_tokens_est == 30


def test_thinking_parser_reset() -> None:
    """Reset should clear all state."""
    p = ThinkingParser()
    p.feed(content="<think>stuff</think>more")
    p.reset()
    assert p.thinking_chars == 0
    assert p.content_chars == 0
    assert p._in_thinking is False


def test_thinking_parser_empty_chunks() -> None:
    """Empty chunks should be no-ops."""
    p = ThinkingParser()
    r = p.feed(content="", reasoning_content="")
    assert r.thinking == ""
    assert r.content == ""
