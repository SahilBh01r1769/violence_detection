import requests

import alerts.telegram_alert as telegram_module
from alerts.telegram_alert import send_telegram_alert


class FakeResponse:
    def __init__(self, *, ok, status_code, payload):
        self.ok = ok
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


def test_telegram_submits_local_screenshot_as_photo(monkeypatch, tmp_path):
    screenshot = tmp_path / "event.jpg"
    screenshot.write_bytes(b"jpeg-data")
    captured = {}

    def accept(url, data, files, timeout):
        captured.update(url=url, data=data, files=files, timeout=timeout)
        assert files["photo"][1].read() == b"jpeg-data"
        return FakeResponse(ok=True, status_code=200, payload={"ok": True})

    monkeypatch.setattr(telegram_module.requests, "post", accept)

    result = send_telegram_alert(
        bot_token="secret-token",
        chat_id="12345",
        detected_class="violence",
        confidence=0.91,
        screenshot_path=screenshot,
        location="Camera-01",
    )

    assert result.accepted
    assert result.error is None
    assert captured["url"].endswith("/botsecret-token/sendPhoto")
    assert captured["data"]["chat_id"] == "12345"
    assert "91.0%" in captured["data"]["caption"]
    assert captured["timeout"] == 15.0


def test_telegram_api_rejection_retains_provider_description(monkeypatch, tmp_path):
    screenshot = tmp_path / "event.jpg"
    screenshot.write_bytes(b"jpeg-data")
    monkeypatch.setattr(
        telegram_module.requests,
        "post",
        lambda *_args, **_kwargs: FakeResponse(
            ok=False,
            status_code=400,
            payload={"ok": False, "description": "Bad Request: chat not found"},
        ),
    )

    result = send_telegram_alert(
        bot_token="secret-token",
        chat_id="invalid",
        detected_class="violence",
        confidence=0.8,
        screenshot_path=screenshot,
        location="Camera-01",
    )

    assert not result.accepted
    assert result.error == "Bad Request: chat not found"


def test_telegram_transport_error_does_not_leak_token(monkeypatch, tmp_path):
    screenshot = tmp_path / "event.jpg"
    screenshot.write_bytes(b"jpeg-data")

    def fail(url, **_kwargs):
        raise requests.ConnectionError(f"failed request to {url}")

    monkeypatch.setattr(telegram_module.requests, "post", fail)

    result = send_telegram_alert(
        bot_token="secret-token",
        chat_id="12345",
        detected_class="violence",
        confidence=0.8,
        screenshot_path=screenshot,
        location="Camera-01",
    )

    assert not result.accepted
    assert result.error == "Telegram request failed (ConnectionError)"
    assert "secret-token" not in result.error


def test_telegram_requires_existing_screenshot():
    result = send_telegram_alert(
        bot_token="secret-token",
        chat_id="12345",
        detected_class="violence",
        confidence=0.8,
        screenshot_path=None,
        location="Camera-01",
    )

    assert not result.accepted
    assert result.error == "event screenshot is unavailable"
