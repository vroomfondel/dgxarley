"""Integration test suite and LLM client library for the dgxarley cluster.

This package provides:

Modules:
    openwebui_integration_test: OpenWebUI and base LLM client with preset
        management and streaming.
    sglang_integration_test: Direct SGLang client, sequential and parallel
        load testing with live Rich TUI.
    sglang_raw: Interactive SSE stream viewer with dual-panel Rich display
        (interpreted output + raw JSON chunks).
    ollama_integration_test: Ollama API health, model, embedding, and chat
        completions tests.
    repetition_detector: Offline n-gram, sentence, and loop repetition
        analysis for completed LLM outputs.
    streaming_repetition_guard: Real-time repetition detection for token
        streams with configurable thresholds.
    thinking_parser: Unified thinking/content token classifier that handles
        both server-separated reasoning_content and inline <think> tags.
"""
