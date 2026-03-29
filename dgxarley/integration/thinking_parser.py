"""Unified thinking/content token classifier for LLM streaming responses.

Provides a stateful parser that classifies incoming SSE stream tokens as
either "thinking" or "content", regardless of whether the server uses the
``reasoning_content`` field (proper parser like ``qwen3`` or ``minimax``)
or inlines ``<think>...</think>`` tags in the ``content`` field (broken
parsers like ``minimax-append-think``).

Usage::

    parser = ThinkingParser()

    for chunk in sse_stream:
        delta = chunk["choices"][0]["delta"]
        result = parser.feed(
            content=delta.get("content", ""),
            reasoning_content=delta.get("reasoning_content", ""),
        )
        # result.thinking  — thinking text from this chunk (may be "")
        # result.content   — content text from this chunk (may be "")
        # result.is_thinking — True if currently inside a thinking block

The parser handles:
    - ``reasoning_content`` field populated (server-side separation): passes
      through directly, no tag parsing needed.
    - ``<think>...</think>`` tags inline in ``content``: strips tags and
      routes text to the correct field.
    - Tags split across chunk boundaries (e.g. ``"</thi"`` + ``"nk>"``).
    - Nested or repeated ``<think>`` blocks.
    - Mixed mode: some chunks use ``reasoning_content``, others use inline
      tags (gracefully handled, though unlikely in practice).
"""

from dataclasses import dataclass


@dataclass
class ThinkingResult:
    """Result of classifying a single stream chunk.

    Attributes:
        thinking: Thinking/reasoning text extracted from this chunk.
        content: Content/output text extracted from this chunk.
        is_thinking: Whether the parser is currently inside a thinking block
            after processing this chunk.
    """

    thinking: str = ""
    content: str = ""
    is_thinking: bool = False


class ThinkingParser:
    """Stateful parser that separates thinking from content in LLM streams.

    Handles two modes transparently:

    1. **Server-separated** (``reasoning_content`` field populated):
       passes through directly, no tag parsing.
    2. **Inline tags** (``<think>...</think>`` in ``content``):
       strips tags and routes text to ``thinking`` vs ``content``.

    The parser tracks state across chunk boundaries, so partial tags
    like ``"</thi"`` + ``"nk>"`` are handled correctly via an internal
    buffer.

    Attributes:
        thinking_chars: Cumulative character count of thinking text.
        content_chars: Cumulative character count of content text.
    """

    THINK_START: str = "<think>"
    THINK_END: str = "</think>"

    def __init__(self) -> None:
        """Initialize the parser with empty state."""
        self._in_thinking: bool = False
        self._buffer: str = ""  # Only holds partial tags (max 7 chars) — str is fine here
        self._saw_reasoning_content: bool = False
        self.thinking_chars: int = 0
        self.content_chars: int = 0

    @property
    def thinking_tokens_est(self) -> int:
        """Estimated thinking token count (chars // 4)."""
        return self.thinking_chars // 4

    @property
    def content_tokens_est(self) -> int:
        """Estimated content token count (chars // 4)."""
        return self.content_chars // 4

    @property
    def total_tokens_est(self) -> int:
        """Estimated total token count (chars // 4)."""
        return (self.thinking_chars + self.content_chars) // 4

    def reset(self) -> None:
        """Reset all state for a new stream."""
        self._in_thinking = False
        self._buffer = ""
        self._saw_reasoning_content = False
        self.thinking_chars = 0
        self.content_chars = 0

    def feed(
        self,
        content: str = "",
        reasoning_content: str = "",
    ) -> ThinkingResult:
        """Classify a single stream chunk into thinking and content.

        If ``reasoning_content`` is non-empty, it is used directly (the
        server already separated thinking from content).  Otherwise,
        ``content`` is scanned for ``<think>...</think>`` tags and split
        accordingly.

        Args:
            content: The ``delta.content`` field from the SSE chunk.
            reasoning_content: The ``delta.reasoning_content`` field from
                the SSE chunk.  Empty or ``None`` means the server did not
                separate thinking tokens.

        Returns:
            A :class:`ThinkingResult` with the classified text and current
            thinking state.
        """
        # Mode 1: server provides reasoning_content — pass through
        if reasoning_content:
            self._saw_reasoning_content = True
            self._in_thinking = True
            self.thinking_chars += len(reasoning_content)
            # Content may arrive in the same chunk (transition from thinking to content)
            if content:
                self._in_thinking = False
                self.content_chars += len(content)
            return ThinkingResult(
                thinking=reasoning_content,
                content=content,
                is_thinking=self._in_thinking,
            )

        # If we previously saw reasoning_content, content-only chunks mean
        # thinking is done — pass content through without tag parsing
        if self._saw_reasoning_content:
            if content:
                self._in_thinking = False
                self.content_chars += len(content)
            return ThinkingResult(
                content=content,
                is_thinking=self._in_thinking,
            )

        # Mode 2: parse <think>...</think> tags from content stream
        if not content:
            return ThinkingResult(is_thinking=self._in_thinking)

        return self._parse_tags(content)

    def _parse_tags(self, text: str) -> ThinkingResult:
        """Parse ``<think>``/``</think>`` tags from inline content.

        Handles tags split across chunk boundaries by buffering partial
        tag matches.  Processes the text character-by-character through
        the buffer when a potential tag prefix is detected.

        Args:
            text: Raw content text that may contain inline thinking tags.

        Returns:
            A :class:`ThinkingResult` with thinking and content separated.
        """
        thinking_out: list[str] = []
        content_out: list[str] = []
        buf = self._buffer + text
        self._buffer = ""
        pos = 0

        while pos < len(buf):
            if self._in_thinking:
                # Look for </think>
                end_idx = buf.find(self.THINK_END, pos)
                if end_idx == -1:
                    # Check for partial </think> at the end
                    for trim in range(len(self.THINK_END) - 1, 0, -1):
                        if buf.endswith(self.THINK_END[:trim]):
                            thinking_out.append(buf[pos : len(buf) - trim])
                            self._buffer = buf[len(buf) - trim :]
                            pos = len(buf)
                            break
                    else:
                        thinking_out.append(buf[pos:])
                        pos = len(buf)
                else:
                    thinking_out.append(buf[pos:end_idx])
                    self._in_thinking = False
                    pos = end_idx + len(self.THINK_END)
            else:
                # Look for <think>
                start_idx = buf.find(self.THINK_START, pos)
                if start_idx == -1:
                    # Check for partial <think> at the end
                    for trim in range(len(self.THINK_START) - 1, 0, -1):
                        if buf.endswith(self.THINK_START[:trim]):
                            content_out.append(buf[pos : len(buf) - trim])
                            self._buffer = buf[len(buf) - trim :]
                            pos = len(buf)
                            break
                    else:
                        content_out.append(buf[pos:])
                        pos = len(buf)
                else:
                    content_out.append(buf[pos:start_idx])
                    self._in_thinking = True
                    pos = start_idx + len(self.THINK_START)

        thinking = "".join(thinking_out)
        content = "".join(content_out)
        self.thinking_chars += len(thinking)
        self.content_chars += len(content)

        return ThinkingResult(
            thinking=thinking,
            content=content,
            is_thinking=self._in_thinking,
        )
