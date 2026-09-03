from evaluation.temporal import (
    FrameObservation,
    compare_thresholds,
    evaluate_threshold,
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
    trace = observations([True] * 6, [True] * 6)
    result = evaluate_threshold(trace, 3)

    assert result.total_triggers == 2
    assert result.detected_events == 1
    assert result.duplicate_triggers == 1


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
