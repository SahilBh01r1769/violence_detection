import json

import alerts.alert_manager as alert_module
from alerts.alert_manager import AlertManager


class DeferredThread:
    instances = []

    def __init__(self, target, args, daemon):
        self.target = target
        self.args = args
        self.daemon = daemon
        self.instances.append(self)

    def start(self):
        pass

    def run(self):
        self.target(*self.args)


class ImmediateThread(DeferredThread):
    def start(self):
        self.run()


class RejectingThread(DeferredThread):
    def start(self):
        raise RuntimeError("thread unavailable")


def test_alert_is_queued_until_delivery_completes(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module, "ALERT_LOG_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alert_module.threading, "Thread", DeferredThread)
    monkeypatch.setattr(alert_module, "send_email_alert", lambda **_kwargs: True)
    manager = AlertManager(cooldown_seconds=30, enable_email=True, enable_whatsapp=False)

    assert manager.trigger("violence", 0.91)
    assert manager.history[0].status == "queued"
    assert not manager.can_trigger

    DeferredThread.instances[-1].run()

    record = manager.history[0]
    assert record.status == "delivered"
    assert record.email_sent
    assert record.completed_at is not None
    assert not manager.can_trigger
    assert manager.seconds_until_next_alert > 0


def test_failed_delivery_does_not_consume_cooldown(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module, "ALERT_LOG_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alert_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(alert_module, "send_email_alert", lambda **_kwargs: False)
    manager = AlertManager(cooldown_seconds=30, enable_email=True, enable_whatsapp=False)

    assert manager.trigger("violence", 0.82)

    record = manager.history[0]
    assert record.status == "failed"
    assert record.error == "all enabled delivery channels failed"
    assert manager.can_trigger
    assert manager.seconds_until_next_alert == 0


def test_delivery_exception_is_recorded_as_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module, "ALERT_LOG_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alert_module.threading, "Thread", ImmediateThread)

    def fail(**_kwargs):
        raise RuntimeError("smtp unavailable")

    monkeypatch.setattr(alert_module, "send_email_alert", fail)
    manager = AlertManager(cooldown_seconds=30, enable_email=True, enable_whatsapp=False)

    assert manager.trigger("violence", 0.82)

    record = manager.history[0]
    assert record.status == "failed"
    assert record.error == "smtp unavailable"
    assert manager.can_trigger


def test_queue_failure_releases_pending_alert(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module, "ALERT_LOG_FILE", tmp_path / "alerts.json")
    monkeypatch.setattr(alert_module.threading, "Thread", RejectingThread)
    manager = AlertManager(cooldown_seconds=30)

    assert not manager.trigger("violence", 0.82)

    record = manager.history[0]
    assert record.status == "failed"
    assert record.error == "could not queue delivery: thread unavailable"
    assert manager.can_trigger


def test_legacy_history_is_migrated_to_explicit_outcome(monkeypatch, tmp_path):
    history_path = tmp_path / "alerts.json"
    history_path.write_text(
        json.dumps(
            [
                {
                    "id": 4,
                    "timestamp": "2026-09-01 10:00:00",
                    "detected_class": "violence",
                    "confidence": 0.9,
                    "location": "Camera-01",
                    "screenshot_path": None,
                    "email_sent": True,
                    "whatsapp_sent": False,
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(alert_module, "ALERT_LOG_FILE", history_path)

    manager = AlertManager()

    assert manager.history[0].status == "delivered"
    assert manager.history[0].completed_at == "2026-09-01 10:00:00"
