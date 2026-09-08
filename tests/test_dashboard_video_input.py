from pathlib import Path

import pytest

from dashboard.video_input import persist_uploaded_video


def test_uploaded_video_uses_safe_content_addressed_name(tmp_path: Path):
    path = persist_uploaded_video("../../Fight Clip.MP4", b"video", tmp_path)

    assert path == (tmp_path / "Fight-Clip-0cab1c961740.mp4").resolve()
    assert path.read_bytes() == b"video"


def test_repeated_upload_reuses_same_file(tmp_path: Path):
    first = persist_uploaded_video("clip.mp4", b"video", tmp_path)
    second = persist_uploaded_video("clip.mp4", b"video", tmp_path)

    assert first == second
    assert len(list(tmp_path.iterdir())) == 1


@pytest.mark.parametrize("filename", ["clip.exe", "clip.jpg", "clip"])
def test_uploaded_video_rejects_unsupported_type(filename, tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported video type"):
        persist_uploaded_video(filename, b"content", tmp_path)


def test_uploaded_video_rejects_empty_content(tmp_path: Path):
    with pytest.raises(ValueError, match="empty"):
        persist_uploaded_video("clip.mp4", b"", tmp_path)
