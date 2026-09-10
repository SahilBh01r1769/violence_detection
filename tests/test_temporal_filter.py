import pytest

from core.temporal import RollingWindowEventFilter, TemporalEventFilter


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


def test_rolling_window_tolerates_an_intermittent_negative():
    temporal_filter = RollingWindowEventFilter(
        minimum_positives=3,
        window_size=4,
        negative_frames=2,
    )

    decisions = [
        temporal_filter.update(value)
        for value in [True, True, False, True]
    ]

    assert [decision.triggered for decision in decisions] == [
        False,
        False,
        False,
        True,
    ]
    assert temporal_filter.positive_count == 3


def test_rolling_window_release_clears_old_evidence_before_rearming():
    temporal_filter = RollingWindowEventFilter(2, 3, negative_frames=2)

    decisions = [
        temporal_filter.update(value)
        for value in [True, True, False, False, True, False, True]
    ]

    assert [decision.triggered for decision in decisions] == [
        False,
        True,
        False,
        False,
        False,
        False,
        True,
    ]
    assert decisions[3].released


@pytest.mark.parametrize(
    ("minimum_positives", "window_size", "negative_frames", "message"),
    [
        (0, 3, 1, "minimum_positives"),
        (4, 3, 1, "window_size"),
        (2, 3, 0, "negative_release_frames"),
    ],
)
def test_rolling_window_rejects_invalid_settings(
    minimum_positives,
    window_size,
    negative_frames,
    message,
):
    with pytest.raises(ValueError, match=message):
        RollingWindowEventFilter(
            minimum_positives,
            window_size,
            negative_frames,
        )
