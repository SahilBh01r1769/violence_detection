import pytest

from core.temporal import TemporalEventFilter


def test_filter_reports_trigger_active_state_and_release():
    temporal_filter = TemporalEventFilter(positive_frames=2, negative_frames=2)

    decisions = [
        temporal_filter.update(value)
        for value in [True, True, False, True, False, False]
    ]

    assert [decision.triggered for decision in decisions] == [
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    assert [decision.event_active for decision in decisions] == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]
    assert [decision.released for decision in decisions] == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]


@pytest.mark.parametrize(
    ("positive_frames", "negative_frames"),
    [(0, 1), (1, 0)],
)
def test_filter_rejects_non_positive_settings(positive_frames, negative_frames):
    with pytest.raises(ValueError):
        TemporalEventFilter(positive_frames, negative_frames)
