from __future__ import annotations

from tests.ui._helpers_loader import load_ui_helpers_module


def test_ensure_media_url_local_file(tmp_path):
    helpers = load_ui_helpers_module()
    sample = tmp_path / "frame.jpg"
    sample.write_bytes(b"\xff\xd8\xff\xdbtestjpegbytes")
    data_url = helpers.ensure_media_url(sample)
    assert isinstance(data_url, str)
    assert data_url.startswith("data:image/jpeg;base64,")
    https_url = "https://example.com/frame.jpg"
    assert helpers.ensure_media_url(https_url) == https_url
    assert helpers.ensure_media_url(None) is None


def test_thumb_html_placeholder_uses_data_url():
    helpers = load_ui_helpers_module()
    html_snippet = helpers.thumb_html(None, alt="missing thumb")
    assert "data:image/svg+xml;base64," in html_snippet
    assert 'alt="missing thumb"' in html_snippet
