from typing import List
import logging
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp

from data import (
    load_dataset,
    chunkify_examples,
)
from model import (
    ExperimentHyperparameters,
    TransientDetectorParameters,
    optimize_transient_detector,
)
from evaluation import evaluate_model
from plotting import plot_predictions
from typing import Optional

logger = logging.getLogger(__name__)


def train_model(hyperparams: ExperimentHyperparameters, train_set, val_set):
    if not train_set:
        logger.error("No training data found. Exiting.")
        return
    logger.info(
        f"Loaded {len(train_set)} training and {len(val_set)} validation examples from dataset."
    )

    # Optionally chunkify all training and validation examples (if needed)
    train_chunks = []
    for ex in train_set:
        train_chunks.extend(chunkify_examples(ex, hyperparams))
    val_chunks = []
    for ex in val_set:
        val_chunks.extend(chunkify_examples(ex, hyperparams))
    logger.info(f"Total chunks: {len(train_chunks)} train, {len(val_chunks)} val")

    # Optimize parameters
    logger.info("Optimizing transient detector parameters on training set...")
    opt_params = optimize_transient_detector(train_chunks, hyperparams)
    logger.info(f"Optimized parameters: {opt_params}")

    # Switch to CPU for evaluation (we run the IIR filters in evaluation, which are very slow on the GPU)
    old_device = hyperparams.device
    hyperparams.device = "cpu"
    cpu = jax.devices("cpu")[0]
    opt_params_cpu = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, cpu) if isinstance(x, jax.Array) else x,
        opt_params,
    )

    with jax.default_device(cpu):
        # Evaluate optimized model on validation set across thresholds
        eval_results = evaluate_model(hyperparams, opt_params_cpu, val_chunks)
        # Print concise summary
        logger.info("Validation results by threshold:")
        for r in eval_results:
            logger.info(
                f"  th={r.threshold:.2f} | loss={r.loss:.4f} | TP={r.true_positives} FP={r.false_positives} FN={r.false_negatives} | acc={r.accuracy:.3f} rec={r.recall:.3f}"
            )
        # Select best by accuracy and recall
        best_by_acc = (
            max(eval_results, key=lambda r: r.accuracy) if eval_results else None
        )
        best_by_rec = (
            max(eval_results, key=lambda r: r.recall) if eval_results else None
        )
        if best_by_acc:
            logger.info(
                f"Best accuracy: th={best_by_acc.threshold:.2f}, acc={best_by_acc.accuracy:.3f}, rec={best_by_acc.recall:.3f}"
            )
        if best_by_rec:
            logger.info(
                f"Best recall:   th={best_by_rec.threshold:.2f}, acc={best_by_rec.accuracy:.3f}, rec={best_by_rec.recall:.3f}"
            )

        # Plot predictions after optimization (force CPU for IIR eval)
        plot_predictions(
            val_chunks,
            opt_params_cpu,
            output_dir="data/plots/chunk_preds_optimized",
            is_training=False,
            hyperparams=hyperparams,
            do_debug=True,
            prefix="",
            print_prefix="Optimized: ",
        )

    return eval_results


def main():
    logging.basicConfig(level=logging.INFO)
    hyperparams = ExperimentHyperparameters()
    train_set, val_set = load_dataset(Path("data/export"), hyperparams, split=0.5)
    train_model(hyperparams, train_set, val_set)


if __name__ == "__main__":
    main()
