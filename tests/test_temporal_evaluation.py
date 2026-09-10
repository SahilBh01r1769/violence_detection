import pytest

from evaluation.temporal import (
    FrameObservation,
    compare_operating_points,
    compare_temporal_settings,
    compare_thresholds,
    evaluate_threshold,
    replay_rolling_window_filter,
    trigger_indices,
)


def observations(predictions, truth, fps=10):
    return [
        FrameObservation(
            frame_id=index + 1,
            timestamp_seconds=index / fps,
            is_violent=prediction,
            confidence=0.8 if prediction else 0.1,
            ground_truth_violent=actual,
            capture_confidence_floor=0.1,
        )
        for index, (prediction, actual) in enumerate(zip(predictions, truth))
    ]


def test_threshold_comparison_exposes_false_trigger_delay_tradeoff():
    trace = observations(
        predictions=[True, False, False, True, True, True, True, True],
        truth=[False, False, False, True, True, True, True, True],
    )

    one_frame, three_frame, five_frame = compare_thresholds(trace, [1, 3, 5])

    assert one_frame.false_triggers == 1
    assert one_frame.mean_alert_delay_seconds == 0.0
    assert three_frame.false_triggers == 0
    assert three_frame.mean_alert_delay_seconds == 0.2
    assert five_frame.false_triggers == 0
    assert five_frame.mean_alert_delay_seconds == 0.4


def test_multiple_triggers_in_one_event_are_counted_as_duplicates():
    trace = observations(
        [True, True, True, False, True, True, True],
        [True] * 7,
    )
    result = evaluate_threshold(trace, 3)

    assert result.total_triggers == 2
    assert result.detected_events == 1
    assert result.duplicate_triggers == 1


def test_continuous_positive_run_is_one_raw_trigger():
    trace = observations([True] * 6, [True] * 6)

    assert trigger_indices(trace, 3) == [2]


def test_event_shorter_than_threshold_is_reported_as_missed():
    trace = observations([True, True, False], [True, True, False])
    result = evaluate_threshold(trace, 3)

    assert result.detected_events == 0
    assert result.missed_events == 1
    assert result.mean_alert_delay_seconds is None


def test_negative_prediction_resets_positive_run():
    trace = observations(
        [True, True, False, True, True, True],
        [False] * 6,
    )
    assert trigger_indices(trace, 3) == [5]


def test_release_comparison_reuses_trace_and_exposes_fragmentation_tradeoff():
    trace = observations(
        [True, True, True, False, True, True, True],
        [True] * 7,
    )

    one_negative, three_negative = compare_temporal_settings(trace, [3], [1, 3])

    assert one_negative.negative_release_frames == 1
    assert one_negative.total_triggers == 2
    assert one_negative.duplicate_triggers == 1
    assert three_negative.negative_release_frames == 3
    assert three_negative.total_triggers == 1
    assert three_negative.duplicate_triggers == 0


def test_release_requires_positive_integer():
    trace = observations([True], [True])

    with pytest.raises(ValueError, match="negative_release_frames"):
        trigger_indices(trace, 1, negative_release_frames=0)


def test_release_comparison_measures_end_delay_in_video_time():
    trace = observations(
        [True, True, True, False, False, False],
        [True, True, True, False, False, False],
    )

    one_negative, three_negative = compare_temporal_settings(trace, [3], [1, 3])

    assert one_negative.mean_release_delay_seconds == 0.0
    assert three_negative.mean_release_delay_seconds == 0.2


def test_release_comparison_marks_ground_truth_events_merged_by_active_run():
    trace = observations(
        [True, True, True, False, False, True, True, True, False, False, False],
        [True, True, True, False, False, True, True, True, False, False, False],
    )

    one_negative, three_negative = compare_temporal_settings(trace, [3], [1, 3])

    assert one_negative.total_triggers == 2
    assert one_negative.merged_ground_truth_events == 0
    assert three_negative.total_triggers == 1
    assert three_negative.merged_ground_truth_events == 1


def test_confidence_and_temporal_grid_reuses_recorded_scores():
    trace = [
        FrameObservation(1, 0.0, True, 0.45, False, 0.25),
        FrameObservation(2, 0.1, True, 0.58, False, 0.25),
        FrameObservation(3, 0.2, True, 0.59, False, 0.25),
        FrameObservation(4, 0.3, True, 0.82, True, 0.25),
        FrameObservation(5, 0.4, True, 0.84, True, 0.25),
        FrameObservation(6, 0.5, True, 0.86, True, 0.25),
    ]

    results = compare_operating_points(trace, [0.55, 0.8], [1, 3], [1])

    assert len(results) == 4
    low_one, low_three, high_one, high_three = results
    assert low_one.confidence_threshold == 0.55
    assert low_one.positive_frames == 1
    assert low_one.false_triggers == 1
    assert low_three.false_triggers == 0
    assert high_one.detected_events == 1
    assert high_three.mean_alert_delay_seconds == 0.2


def test_confidence_replay_rejects_threshold_below_capture_floor():
    trace = [FrameObservation(1, 0.0, True, 0.4, False, 0.25)]

    with pytest.raises(ValueError, match="below the trace capture floor"):
        trigger_indices(trace, 1, detector_confidence=0.2)


def test_confidence_replay_rejects_legacy_trace_without_capture_floor():
    trace = [FrameObservation(1, 0.0, True, 0.8, False)]

    with pytest.raises(ValueError, match="capture_confidence_floor"):
        trigger_indices(trace, 1, detector_confidence=0.55)


def test_zero_confidence_without_a_detection_is_not_positive():
    trace = [FrameObservation(1, 0.0, False, 0.0, False, 0.0)]

    assert trigger_indices(trace, 1, detector_confidence=0.0) == []


def test_confidence_replay_rejects_inconsistent_capture_floors():
    trace = [
        FrameObservation(1, 0.0, True, 0.5, False, 0.2),
        FrameObservation(2, 0.1, True, 0.5, False, 0.25),
    ]

    with pytest.raises(ValueError, match="inconsistent"):
        trigger_indices(trace, 1, detector_confidence=0.55)


def test_rolling_replay_uses_the_shared_window_filter_and_confidence():
    trace = observations(
        [True, True, False, True, False, False],
        [True] * 6,
    )

    replay = replay_rolling_window_filter(
        trace,
        minimum_positives=3,
        window_size=4,
        negative_release_frames=2,
        detector_confidence=0.7,
    )

    assert replay.trigger_indices == (3,)
    assert replay.active_intervals == ((3, 4),)
    assert replay.release_delays_seconds == pytest.approx((0.1,))
