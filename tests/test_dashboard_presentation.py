from dashboard.presentation import (
    compact_source_label,
    history_display_rows,
    notification_label,
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
