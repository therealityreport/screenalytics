from __future__ import annotations

from apps.api.services.log_formatter import LogFormatter


def test_log_formatter_does_not_warn_on_provider_line() -> None:
    formatter = LogFormatter(episode_id="ep-test", operation="faces_embed")
    line = (
        "[LOCAL MODE] ONNX providers: ['CoreMLExecutionProvider', 'CPUExecutionProvider'] "
        "(resolved_device=coreml, available=['CoreMLExecutionProvider', 'CPUExecutionProvider'])"
    )
    formatted = formatter.format_line(line)
    assert formatted is not None
    assert formatted.startswith("[LOCAL MODE] ONNX providers:")
    assert "available=" not in formatted
    assert formatter.state.onnx_warning_count == 0


def test_log_formatter_shows_single_fallback_warning_with_details() -> None:
    formatter = LogFormatter(episode_id="ep-test", operation="faces_embed")
    warn_line = "WARNING:onnxruntime: Some operators are not supported and will be executed on CPU."
    formatted_first = formatter.format_line(warn_line)
    assert formatted_first is not None
    assert formatted_first.startswith("[WARN] Some ops are falling back to CPU")
    assert "details:" in formatted_first
    formatted_second = formatter.format_line(warn_line)
    assert formatted_second is None


def test_log_formatter_includes_providers_line_on_fallback_warning() -> None:
    formatter = LogFormatter(episode_id="ep-test", operation="faces_embed")
    provider_line = (
        "[LOCAL MODE] ONNX providers: ['CoreMLExecutionProvider', 'CPUExecutionProvider'] "
        "(resolved_device=coreml, available=['CoreMLExecutionProvider', 'CPUExecutionProvider'])"
    )
    formatted_provider = formatter.format_line(provider_line)
    assert formatted_provider is not None
    assert "available=" not in formatted_provider

    warn_line = "WARNING:onnxruntime: Some operators are not supported and will be executed on CPU."
    formatted_warn = formatter.format_line(warn_line)
    assert formatted_warn is not None
    assert "(providers: [LOCAL MODE] ONNX providers:" in formatted_warn
    assert "available=" not in formatted_warn
