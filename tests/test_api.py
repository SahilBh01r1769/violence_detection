from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.server as server


class FakeDetector:
    confidence = 0.55
    frame_consistency = 5
    negative_release_frames = 1
    event_active = False
    def set_frame_consistency(self, value):
        self.frame_consistency = value
    def set_negative_release_frames(self, value):
        self.negative_release_frames = value


class FakeAlertManager:
    history = [
        SimpleNamespace(
            id=7,
            status="failed",
            completed_at="2026-09-06 10:00:00",
            error="smtp unavailable",
        )
    ]
    cooldown = 30
    seconds_until_next_alert = 0
    enable_email = True
    enable_whatsapp = True


class FakePipeline:
    def __init__(self, **kwargs):
        self._running = False
        self.frames_processed = 0
        self.alerts_fired = 0
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


def test_status_exposes_runtime_and_alert_failures(monkeypatch):
    monkeypatch.setattr(server, "_pipeline", FakePipeline())

    data = TestClient(server.app).get("/status").json()

    assert data["source_state"] == "disconnected"
    assert data["event_active"] is False
    assert data["last_error"]["stage"] == "source"
    assert data["latest_alert_delivery"]["status"] == "failed"
    assert data["latest_alert_delivery"]["error"] == "smtp unavailable"


def test_status_retains_latest_alert_failure_while_pipeline_is_idle(monkeypatch):
    monkeypatch.setattr(server, "_pipeline", None)
    monkeypatch.setattr(server, "_history_records", lambda: FakeAlertManager.history)

    data = TestClient(server.app).get("/status").json()

    assert data["source_state"] == "idle"
    assert data["latest_alert_delivery"]["status"] == "failed"


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
            "enable_email": False,
            "enable_whatsapp": False,
        },
    )
    assert response.status_code == 200
    assert pipeline.detector.confidence == 0.7
    assert pipeline.detector.frame_consistency == 7
    assert pipeline.detector.negative_release_frames == 3
    assert pipeline.alert_manager.cooldown == 15
    assert not pipeline.alert_manager.enable_email
    assert not pipeline.alert_manager.enable_whatsapp
