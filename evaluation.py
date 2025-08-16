
"""
Event detection, evaluation, and metrics for transient detection.
"""

from __future__ import annotations
import json
from pathlib import Path
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



# Common metrics dataclass
@dataclass
class EvaluationMetrics:
    loss: float
    threshold: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    accuracy: float
    recall: float


@dataclass
class EvaluationResult:
    params: TransientDetectorParameters
    train_result: EvaluationMetrics
    val_result: EvaluationMetrics


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


def evaluate_metrics_for_threshold(
    hparams: ExperimentHyperparameters,
    params: TransientDetectorParameters,
    data: List[TransientExample],
    threshold: float,
) -> EvaluationMetrics:
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

    return EvaluationMetrics(
        loss=mean_loss,
        threshold=threshold,
        false_positives=total_event_fp,
        false_negatives=total_event_fn,
        true_positives=total_event_tp,
        true_negatives=0,  # TN not well-defined for event detection; leave as 0
        accuracy=accuracy,
        recall=recall,
    )




def evaluate_metrics(
    hparams: ExperimentHyperparameters,
    params: TransientDetectorParameters,
    data: List[TransientExample],
) -> list[EvaluationMetrics]:
    logger.info("Evaluating metrics")
    thresholds = np.linspace(0.1, 0.9, 9, dtype=np.float32)
    results: list[EvaluationMetrics] = []
    for th in thresholds:
        results.append(evaluate_metrics_for_threshold(hparams, params, data, float(th)))
    logger.info("Done evaluating metrics")
    return results



# --- Evaluate both train and validation sets ---
def evaluate_train_and_val(
    hparams: ExperimentHyperparameters,
    params: TransientDetectorParameters,
    train_data: List[TransientExample],
    val_data: List[TransientExample],
) -> list[EvaluationResult]:
    """
    Evaluate model on both train and validation sets for each threshold.
    Returns a list of EvaluationResult, one per threshold.
    """
    logger.info("Evaluating train set...")
    train_metrics = evaluate_metrics(hparams, params, train_data)
    logger.info("Evaluating validation set...")
    val_metrics = evaluate_metrics(hparams, params, val_data)
    # Assume thresholds are the same for both
    results = []
    for train_result, val_result in zip(train_metrics, val_metrics):
        results.append(EvaluationResult(params=params, train_result=train_result, val_result=val_result))
    return results


def save_results(run_name: str, results: List[EvaluationResult]):
    """
    Save a list of EvaluationResult as JSON files in data/results/.
    Each file is named {accuracy:.3f}_th{threshold:.2f}_{run_name}.json for easy sorting.
    Args:
        run_name: The name of the run (used as filename suffix).
        results: List of EvaluationResult objects.
    """
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    from dataclasses import asdict, is_dataclass
    import jax.numpy as jnp
    import numpy as np


    def make_serializable(obj):
        # Recursively convert to serializable types
        if isinstance(obj, type):
            # Don't serialize types/classes
            return str(obj)
        if is_dataclass(obj) and not isinstance(obj, type):
            # Only call asdict on dataclass instances, not types
            return {k: make_serializable(v) for k, v in asdict(obj).items()}
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (jnp.ndarray, np.ndarray)):
            return obj.tolist()
        elif hasattr(obj, 'item') and callable(getattr(obj, 'item', None)) and not isinstance(obj, type):
            # Handles 0-d arrays, but not types
            try:
                return obj.item()
            except Exception:
                return str(obj)
        elif isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        else:
            # Try to convert to float if possible (e.g., JAX scalars), but not on types
            try:
                if not isinstance(obj, type):
                    return float(obj)
                else:
                    return str(obj)
            except Exception:
                return str(obj)

    for res in results:
        res_dict = make_serializable(res)
        # Use validation accuracy/threshold for filename
        acc = float(res.val_result.accuracy)
        th = float(res.val_result.threshold)
        filename = f"{acc:.3f}_th{th:.2f}_{run_name}.json"
        out_path = results_dir / filename
        with open(out_path, "w") as f:
            json.dump(res_dict, f, indent=2)
        paths.append(str(out_path))
    return paths
