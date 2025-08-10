"""
Event detection, evaluation, and metrics for transient detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


import logging
import numpy as np
import jax.numpy as jnp

from data import TransientExample
from model import (
    ExperimentHyperparameters,
    TransientDetectorParameters,
    transient_detector,
    loss_function,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    params: TransientDetectorParameters
    loss: float
    threshold: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    accuracy: float
    recall: float


def detect_events_from_prediction(
    preds_np,
    threshold: float,
    sample_rate: float,
    window_s: float,
) -> list[float]:
    """
    Detect event times from a prediction array using upward threshold crossings with latching and a refractory period.
    Returns a list of event times (in seconds).
    """
    refractory_s = window_s
    pred_times: list[float] = []
    last_det_time = -1e9
    went_below_since_last = True  # allow first crossing
    for i in range(1, len(preds_np)):
        t_cur = i / sample_rate
        if preds_np[i] <= threshold:
            went_below_since_last = True
        if (
            preds_np[i - 1] <= threshold
            and preds_np[i] > threshold
            and went_below_since_last
            and (t_cur - last_det_time) >= refractory_s
        ):
            pred_times.append(t_cur)
            last_det_time = t_cur
            went_below_since_last = False
    return pred_times


def evaluate_model_for_threshold(
    hparams: ExperimentHyperparameters,
    params: TransientDetectorParameters,
    data: List[TransientExample],
    threshold: float,
) -> EvaluationResult:
    # Event-based evaluation: use detection function for event times
    tol_s = hparams.window_s / 2.0

    total_event_tp = 0
    total_event_fp = 0
    total_event_fn = 0

    # Sample-based confusion (for optional accuracy reporting)
    sample_tp = 0
    sample_fp = 0
    sample_tn = 0
    sample_fn = 0

    total_loss_sum = 0.0
    total_count = 0

    for ex in data:
        sr = ex.sample_rate
        preds = transient_detector(
            params,
            jnp.asarray(ex.audio),
            sr,
            do_debug=False,
            is_training=False,
            hyperparams=hparams,
        )
        preds_np = np.asarray(preds)

        # Sample-wise confusion
        y_true = ex.label_array > 0.5
        y_pred = preds_np > threshold
        sample_tp += int(np.sum(np.logical_and(y_true, y_pred)))
        sample_fp += int(np.sum(np.logical_and(~y_true, y_pred)))
        sample_tn += int(np.sum(np.logical_and(~y_true, ~y_pred)))
        sample_fn += int(np.sum(np.logical_and(y_true, ~y_pred)))

        # Loss (weighted by sample count)
        loss_val = float(loss_function(jnp.asarray(ex.label_array), preds, hparams))
        total_loss_sum += loss_val * len(ex.label_array)
        total_count += len(ex.label_array)

        # Use new detection function
        pred_times = detect_events_from_prediction(
            preds_np, threshold, sr, hparams.window_s
        )

        # Event matching to ground truth
        gt_times = list(ex.transient_times)
        matched_gt = [False] * len(gt_times)
        event_tp = 0
        event_fp = 0

        for pt in pred_times:
            # find nearest unmatched gt within tolerance
            best_j = -1
            best_dt = float("inf")
            for j, gt in enumerate(gt_times):
                if matched_gt[j]:
                    continue
                dt = abs(pt - gt)
                if dt < best_dt:
                    best_dt = dt
                    best_j = j
            if best_j != -1 and best_dt <= tol_s:
                matched_gt[best_j] = True
                event_tp += 1
            else:
                event_fp += 1

        event_fn = matched_gt.count(False)

        total_event_tp += event_tp
        total_event_fp += event_fp
        total_event_fn += event_fn

    # Aggregate metrics
    mean_loss = (total_loss_sum / max(1, total_count)) if total_count > 0 else 0.0

    # Event-based recall and a simple "accuracy" notion over events
    recall = (
        total_event_tp / max(1, (total_event_tp + total_event_fn))
        if (total_event_tp + total_event_fn) > 0
        else 0.0
    )
    accuracy = (
        total_event_tp / max(1, (total_event_tp + total_event_fp + total_event_fn))
        if (total_event_tp + total_event_fp + total_event_fn) > 0
        else 0.0
    )

    return EvaluationResult(
        params=params,
        loss=mean_loss,
        threshold=threshold,
        false_positives=total_event_fp,
        false_negatives=total_event_fn,
        true_positives=total_event_tp,
        true_negatives=0,  # TN not well-defined for event detection; leave as 0
        accuracy=accuracy,
        recall=recall,
    )


def evaluate_model(
    hparams: ExperimentHyperparameters,
    params: TransientDetectorParameters,
    data: List[TransientExample],
) -> List[EvaluationResult]:
    logger.info("Evaluating model")
    # Evaluate across a sweep of thresholds
    thresholds = np.linspace(0.1, 0.9, 9, dtype=np.float32)
    results: List[EvaluationResult] = []
    for th in thresholds:
        results.append(evaluate_model_for_threshold(hparams, params, data, float(th)))
    logger.info("Done evaluating model")
    return results
