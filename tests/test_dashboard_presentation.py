from datetime import date

from dashboard.presentation import (
    compact_source_label,
    filter_history,
    format_duration,
    history_display_rows,
    notification_label,
    runtime_state_label,
    temporal_phase,
)


def test_compact_source_label_hides_local_directories():
    assert compact_source_label(r"F:\Projects\Violence\runtime_uploads\clip.mp4") == "clip.mp4"
    assert compact_source_label("sample_videos/nonviolence.mp4") == "nonviolence.mp4"
    assert compact_source_label("0") == "0"


def test_compact_source_label_does_not_display_network_credentials():
    assert compact_source_label("rtsp://user:***@camera.test/stream") == "Network stream"


def test_notification_label_explains_suppressed_submission():
    event = {
        "notification_status": "not_attempted",
        "notification_suppression_reason": "telegram_disabled",
    }

    assert notification_label(event) == "Not attempted — telegram disabled"
    assert notification_label({"notification_status": "accepted"}) == "Accepted"


def test_history_rows_use_a_stable_readable_column_set():
    rows = history_display_rows(
        [
            {
                "id": 9,
                "timestamp": "2026-09-11 21:38:14",
                "detected_class": "violence",
                "confidence": 0.8394,
                "location": "Camera-01",
                "source": r"F:\Projects\Violence\runtime_uploads\clip.mp4",
                "screenshot_path": r"F:\Projects\Violence\screenshots\event.jpg",
                "notification_status": "queued",
            }
        ]
    )

    assert rows == [
        {
            "ID": 9,
            "Time": "2026-09-11 21:38:14",
            "Class": "violence",
            "Confidence": 0.8394,
            "Location": "Camera-01",
            "Source": "clip.mp4",
            "Telegram": "Queued",
        }
    ]


def test_runtime_presentation_helpers_distinguish_error_and_temporal_phase():
    assert format_duration(65.8) == "01:05"
    assert format_duration(3661) == "1:01:01"
    assert runtime_state_label({"source_state": "connected"}) == ("Connected", "normal")
    assert runtime_state_label({"source_state": "connected", "last_error": {"message": "boom"}}) == ("Runtime error", "danger")
    assert temporal_phase({"event_active": False, "negative_run": 0}) == "Inactive"
    assert temporal_phase({"event_active": True, "negative_run": 0}) == "Active"
    assert temporal_phase({"event_active": True, "negative_run": 1}) == "Releasing"


def test_history_filters_date_source_location_notification_and_sorts_newest():
    events = [
        {"id": 1, "timestamp": "2026-09-10 08:00:00", "source": "/clips/a.mp4", "location": "Gate", "notification_status": "failed"},
        {"id": 3, "timestamp": "2026-09-12 08:00:00", "source": "/clips/b.mp4", "location": "Hall", "notification_status": "accepted"},
        {"id": 2, "timestamp": "2026-09-11 08:00:00", "source": "/clips/a.mp4", "location": "Gate", "notification_status": "accepted"},
    ]

    assert [event["id"] for event in filter_history(events)] == [3, 2, 1]
    assert [event["id"] for event in filter_history(events, dates=(date(2026, 9, 11), date(2026, 9, 12)))] == [3, 2]
    assert [event["id"] for event in filter_history(events, source_or_location="Gate")] == [2, 1]
    assert [event["id"] for event in filter_history(events, source_or_location="b.mp4", notification="accepted")] == [3]
