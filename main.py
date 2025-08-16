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
from evaluation import EvaluationResult, evaluate_train_and_val, save_results
from plotting import plot_predictions
from typing import Optional

logger = logging.getLogger(__name__)


def train_model(hyperparams: ExperimentHyperparameters, train_set, val_set) -> List[EvaluationResult]:
    if not train_set:
        logger.error("No training data found. Exiting.")
        raise ValueError("No training data provided")

    logger.info(
        f"Loaded {len(train_set)} training and {len(val_set)} validation examples from dataset."
    )

    # plot_predictions(
    #     train_set,
    #     TransientDetectorParameters(),
    #     output_dir="data/plots/chunk_preds_preoptimized",
    #     is_training=True,
    #     hyperparams=hyperparams,
    #     do_debug=True,
    #     prefix="",
    #     print_prefix="Pre-optimized: ",
    # )

    # Optimize parameters
    logger.info("Optimizing transient detector parameters on training set...")
    opt_params = optimize_transient_detector(train_set, hyperparams)
    logger.info(f"Optimized parameters: {opt_params}")

    # Switch to CPU for evaluation (we run the IIR filters in evaluation, which are very slow on the GPU)
    hyperparams.device = "cpu"
    cpu = jax.devices("cpu")[0]
    opt_params_cpu = jax.tree_util.tree_map(
        lambda x: jax.device_put(x, cpu) if isinstance(x, jax.Array) else x,
        opt_params,
    )

    with jax.default_device(cpu):
        # Evaluate optimized model on both train and validation sets across thresholds
        eval_results = evaluate_train_and_val(hyperparams, opt_params_cpu, train_set, val_set)
        # Print concise summary
        logger.info("Results by threshold:")
        for r in eval_results:
            logger.info(
                f"  th={r.val_result.threshold:.2f} | train: acc={r.train_result.accuracy:.3f} rec={r.train_result.recall:.3f} | val: acc={r.val_result.accuracy:.3f} rec={r.val_result.recall:.3f}"
            )
        # Select best by validation accuracy and recall
        best_by_acc = (
            max(eval_results, key=lambda r: r.val_result.accuracy) if eval_results else None
        )
        best_by_rec = (
            max(eval_results, key=lambda r: r.val_result.recall) if eval_results else None
        )
        if best_by_acc:
            logger.info(
                f"Best val accuracy: th={best_by_acc.val_result.threshold:.2f}, acc={best_by_acc.val_result.accuracy:.3f}, rec={best_by_acc.val_result.recall:.3f}"
            )
        if best_by_rec:
            logger.info(
                f"Best val recall:   th={best_by_rec.val_result.threshold:.2f}, acc={best_by_rec.val_result.accuracy:.3f}, rec={best_by_rec.val_result.recall:.3f}"
            )

        # Plot predictions after optimization (force CPU for IIR eval)
        plot_predictions(
            val_set,
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
    from copy import deepcopy
    from evaluation import save_results
    from itertools import product

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Prepare data once
    base_hyperparams = ExperimentHyperparameters()
    train_set, val_set = load_dataset(Path("data/export"), base_hyperparams, split=0.2)

    logger.info(f"Loaded {len(train_set)} training and {len(val_set)} validation examples from dataset.")

    # Define grid
    sweep_space = {
        "num_channels": [2],
        "disable_filters": [True],
    }

    # Generate all combinations
    for num_channels, disable_filters in product(sweep_space["num_channels"], sweep_space["disable_filters"]):
        h = deepcopy(base_hyperparams)
        h.num_channels = num_channels
        h.disable_filters = disable_filters
        run_name = f"{'nofilter' if disable_filters else 'filter'}_{num_channels}ch"
        logger.info(f"=== Running sweep: {run_name} ===")

        logger.info(f"Training model with {num_channels} channels, filters {'disabled' if disable_filters else 'enabled'}")
        results = train_model(h, train_set, val_set)
        if results:
            save_results(run_name, results)
        logger.info(f"=== Done: {run_name} ===\n")


if __name__ == "__main__":
    main()
