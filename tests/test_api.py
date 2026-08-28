from fastapi.testclient import TestClient

import api.server as server


class FakeDetector:
    confidence = 0.55
    frame_consistency = 5
    def set_frame_consistency(self, value):
        self.frame_consistency = value


class FakeAlertManager:
    history = []
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


def test_config_updates_all_supported_settings(monkeypatch):
    pipeline = FakePipeline()
    server._pipeline = pipeline
    client = TestClient(server.app)
    response = client.post("/pipeline/config", json={"confidence": 0.7, "frame_consistency": 7, "cooldown_seconds": 15, "enable_email": False, "enable_whatsapp": False})
    assert response.status_code == 200
    assert pipeline.detector.confidence == 0.7
    assert pipeline.detector.frame_consistency == 7
    assert pipeline.alert_manager.cooldown == 15
    assert not pipeline.alert_manager.enable_email
    assert not pipeline.alert_manager.enable_whatsapp
