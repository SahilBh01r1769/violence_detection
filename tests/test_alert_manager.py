import json

import pytest

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


@pytest.fixture(autouse=True)
def reset_deferred_threads():
    DeferredThread.instances.clear()


def manager_with_history(monkeypatch, tmp_path, **kwargs):
    monkeypatch.setattr(alert_module, "EVENT_LOG_FILE", tmp_path / "events.json")
    return AlertManager(**kwargs)


def test_event_is_persisted_before_notification_is_queued(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module.threading, "Thread", DeferredThread)
    monkeypatch.setattr(alert_module, "send_email_alert", lambda **_kwargs: True)
    manager = manager_with_history(
        monkeypatch,
        tmp_path,
        cooldown_seconds=30,
        enable_email=True,
        enable_whatsapp=False,
    )

    event = manager.record_event(
        "violence",
        0.91,
        screenshot_path=tmp_path / "event.jpg",
        source="sample.mp4",
    )

    persisted = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert event.notification_status == "not_attempted"
    assert persisted[0]["id"] == event.id
    assert persisted[0]["source"] == "sample.mp4"
    assert persisted[0]["notification_status"] == "not_attempted"

    assert manager.request_notification(event.id, "violence", 0.91)
    assert manager.history[0].notification_status == "queued"
    assert not manager.can_notify

    DeferredThread.instances[-1].run()

    record = manager.history[0]
    assert record.notification_status == "accepted"
    assert record.notification_channel == "email"
    assert record.notification_completed_at is not None
    assert record.notification_error is None
    assert not manager.can_notify
    assert manager.accepted_notifications == 1


def test_cooldown_suppresses_notification_but_not_event(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(alert_module, "send_email_alert", lambda **_kwargs: True)
    manager = manager_with_history(
        monkeypatch,
        tmp_path,
        cooldown_seconds=30,
        enable_email=True,
        enable_whatsapp=False,
    )
    first = manager.record_event("violence", 0.91)
    assert manager.request_notification(first.id, "violence", 0.91)

    second = manager.record_event("violence", 0.82)
    assert not manager.request_notification(second.id, "violence", 0.82)

    assert manager.total_events == 2
    record = manager.history[0]
    assert record.id == second.id
    assert record.notification_status == "not_attempted"
    assert record.notification_suppression_reason == "cooldown"
    persisted = json.loads((tmp_path / "events.json").read_text(encoding="utf-8"))
    assert len(persisted) == 2
    assert persisted[-1]["notification_suppression_reason"] == "cooldown"


def test_pending_delivery_suppresses_only_the_second_notification(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(alert_module.threading, "Thread", DeferredThread)
    manager = manager_with_history(
        monkeypatch,
        tmp_path,
        enable_email=True,
        enable_whatsapp=False,
    )
    first = manager.record_event("violence", 0.91)
    assert manager.request_notification(first.id, "violence", 0.91)

    second = manager.record_event("violence", 0.82)
    assert not manager.request_notification(second.id, "violence", 0.82)

    assert manager.total_events == 2
    assert manager.history[0].notification_suppression_reason == "delivery_pending"


def test_no_channel_still_preserves_event(monkeypatch, tmp_path):
    manager = manager_with_history(
        monkeypatch,
        tmp_path,
        enable_email=False,
        enable_whatsapp=False,
    )

    event = manager.record_event("violence", 0.82)

    assert not manager.request_notification(event.id, "violence", 0.82)
    assert manager.total_events == 1
    assert manager.history[0].notification_status == "not_attempted"
    assert manager.history[0].notification_suppression_reason == "no_channel_enabled"


def test_failed_delivery_does_not_consume_cooldown(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(alert_module, "send_email_alert", lambda **_kwargs: False)
    manager = manager_with_history(
        monkeypatch,
        tmp_path,
        cooldown_seconds=30,
        enable_email=True,
        enable_whatsapp=False,
    )
    event = manager.record_event("violence", 0.82)

    assert manager.request_notification(event.id, "violence", 0.82)

    record = manager.history[0]
    assert record.notification_status == "failed"
    assert record.notification_error == "all enabled delivery channels failed"
    assert manager.can_notify
    assert manager.seconds_until_next_alert == 0


def test_delivery_exception_is_recorded_as_failure(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module.threading, "Thread", ImmediateThread)

    def fail(**_kwargs):
        raise RuntimeError("smtp unavailable")

    monkeypatch.setattr(alert_module, "send_email_alert", fail)
    manager = manager_with_history(
        monkeypatch,
        tmp_path,
        enable_email=True,
        enable_whatsapp=False,
    )
    event = manager.record_event("violence", 0.82)

    assert manager.request_notification(event.id, "violence", 0.82)

    record = manager.history[0]
    assert record.notification_status == "failed"
    assert record.notification_error == "smtp unavailable"
    assert manager.can_notify


def test_queue_failure_is_attached_to_existing_event(monkeypatch, tmp_path):
    monkeypatch.setattr(alert_module.threading, "Thread", RejectingThread)
    manager = manager_with_history(monkeypatch, tmp_path)
    event = manager.record_event("violence", 0.82)

    assert not manager.request_notification(event.id, "violence", 0.82)

    record = manager.history[0]
    assert manager.total_events == 1
    assert record.notification_status == "failed"
    assert record.notification_error == "could not queue delivery: thread unavailable"
    assert manager.can_notify


def test_legacy_sender_success_is_migrated_as_accepted_not_received(
    monkeypatch,
    tmp_path,
):
    history_path = tmp_path / "events.json"
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
    monkeypatch.setattr(alert_module, "EVENT_LOG_FILE", history_path)

    manager = AlertManager()

    record = manager.history[0]
    assert record.notification_status == "accepted"
    assert record.notification_channel == "legacy_email"
    assert record.notification_completed_at == "2026-09-01 10:00:00"


def test_legacy_queued_record_is_not_treated_as_success(monkeypatch, tmp_path):
    history_path = tmp_path / "events.json"
    history_path.write_text(
        json.dumps(
            [
                {
                    "id": 5,
                    "timestamp": "2026-09-01 10:00:00",
                    "detected_class": "violence",
                    "confidence": 0.9,
                    "location": "Camera-01",
                    "screenshot_path": None,
                    "status": "queued",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(alert_module, "EVENT_LOG_FILE", history_path)

    manager = AlertManager()

    assert manager.history[0].notification_status == "failed"
    assert manager.history[0].notification_error == (
        "delivery outcome unavailable after restart"
    )
