from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.server as server


class FakeDetector:
    confidence = 0.55
    frame_consistency = 5
    negative_release_frames = 1
    event_active = False
    positive_run = 2
    negative_run = 0
    def set_frame_consistency(self, value):
        self.frame_consistency = value
    def set_negative_release_frames(self, value):
        self.negative_release_frames = value


class FakeAlertManager:
    history = [
        SimpleNamespace(
            id=7,
            notification_status="failed",
            notification_channel="telegram",
            notification_completed_at="2026-09-06 10:00:00",
            notification_error="telegram unavailable",
            notification_suppression_reason=None,
        )
    ]
    cooldown = 30
    seconds_until_next_alert = 0
    enable_telegram = True
    telegram_bot_token = "test-token"
    telegram_chat_id = "test-chat"
    accepted_notifications = 0


class FakePipeline:
    def __init__(self, **kwargs):
        self._running = False
        self.frames_processed = 0
        self.events_recorded = 1
        self.current_event_ids = [7]
        self.uptime = 0
        self.fps = 0
        self.source_state = "disconnected"
        self.last_error = SimpleNamespace(
            as_dict=lambda: {
                "stage": "source",
                "message": "video source disconnected",
                "timestamp": "2026-09-06T10:00:00+00:00",
            }
        )
        self.detector = FakeDetector()
        self.alert_manager = FakeAlertManager()
    def run(self, source=None):
        self._running = False
    def stop(self):
        self._running = False


def test_health():
    assert TestClient(server.app).get("/health").status_code == 200


def test_alert_page_limit_rejects_200():
    assert TestClient(server.app).get("/alerts?per_page=200").status_code == 422


def test_status_exposes_runtime_and_notification_failures(monkeypatch):
    monkeypatch.setattr(server, "_pipeline", FakePipeline())

    data = TestClient(server.app).get("/status").json()

    assert data["source_state"] == "disconnected"
    assert data["event_active"] is False
    assert data["positive_run"] == 2
    assert data["negative_run"] == 0
    assert data["last_error"]["stage"] == "source"
    assert data["events_recorded"] == 1
    assert data["notifications_accepted"] == 0
    assert data["latest_notification"]["status"] == "failed"
    assert data["latest_notification"]["error"] == "telegram unavailable"
    assert data["telegram_enabled"] is True
    assert data["telegram_configured"] is True


def test_current_alerts_excludes_persisted_events_from_older_runs(monkeypatch):
    pipeline = FakePipeline()
    old = SimpleNamespace(id=6, detected_class="violence")
    current = SimpleNamespace(id=7, detected_class="violence")
    monkeypatch.setattr(server, "_pipeline", pipeline)
    monkeypatch.setattr(server, "_history_records", lambda: [current, old])

    data = TestClient(server.app).get("/alerts/current").json()

    assert data["total"] == 1
    assert [event["id"] for event in data["alerts"]] == [7]


def test_current_alerts_is_empty_before_first_run(monkeypatch):
    monkeypatch.setattr(server, "_pipeline", None)

    data = TestClient(server.app).get("/alerts/current").json()

    assert data == {"alerts": [], "total": 0}


def test_status_retains_latest_notification_failure_while_pipeline_is_idle(
    monkeypatch,
):
    monkeypatch.setattr(server, "_pipeline", None)
    monkeypatch.setattr(server, "_history_records", lambda: FakeAlertManager.history)

    data = TestClient(server.app).get("/status").json()

    assert data["source_state"] == "idle"
    assert data["latest_notification"]["status"] == "failed"


def test_config_updates_all_supported_settings(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(server, "_pipeline", pipeline)
    client = TestClient(server.app)
    response = client.post(
        "/pipeline/config",
        json={
            "confidence": 0.7,
            "frame_consistency": 7,
            "negative_release_frames": 3,
            "cooldown_seconds": 15,
            "enable_telegram": False,
        },
    )
    assert response.status_code == 200
    assert pipeline.detector.confidence == 0.7
    assert pipeline.detector.frame_consistency == 7
    assert pipeline.detector.negative_release_frames == 3
    assert pipeline.alert_manager.cooldown == 15
    assert not pipeline.alert_manager.enable_telegram
