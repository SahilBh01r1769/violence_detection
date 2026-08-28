from core.pipeline import normalise_source


def test_webcam_source_string_becomes_integer():
    assert normalise_source("0") == 0


def test_video_path_stays_string():
    assert normalise_source("sample.mp4") == "sample.mp4"
