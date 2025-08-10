from typing import List
import logging
import numpy as np
import jax
import jax.numpy as jnp

from data import (
    load_transient_example,
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

def train_model(hyperparams: ExperimentHyperparameters):
    # Load and plot an example
    base_path = "data/export/DarkIllusion_ElecGtr5DI"
    example = load_transient_example(base_path, hyperparams)
    chunks = chunkify_examples(example, hyperparams)
    logger.info(f"Loaded {len(chunks)} chunks from example {base_path}")

    params = TransientDetectorParameters()

    # Plot predictions before optimization (eval kernel)
    # plot_predictions(
    #     chunks,
    #     params,
    #     output_dir="data/plots/chunk_preds",
    #     is_training=False,
    #     hyperparams=hyperparams,
    #     do_debug=True,
    #     prefix="",
    #     print_prefix="",
    # )

    # Optimize parameters

    logger.info("Optimizing transient detector parameters...")
    opt_params = optimize_transient_detector(chunks, hyperparams)
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
        # Evaluate optimized model across thresholds
        eval_results = evaluate_model(hyperparams, opt_params_cpu, chunks)
        # Print concise summary
        logger.info("Evaluation results by threshold:")
        for r in eval_results:
            logger.info(
                f"  th={r.threshold:.2f} | loss={r.loss:.4f} | TP={r.true_positives} FP={r.false_positives} FN={r.false_negatives} | acc={r.accuracy:.3f} rec={r.recall:.3f}"
            )
        # Select best by accuracy and recall
        best_by_acc = max(eval_results, key=lambda r: r.accuracy) if eval_results else None
        best_by_rec = max(eval_results, key=lambda r: r.recall) if eval_results else None
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
            chunks,
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
    train_model(hyperparams)


if __name__ == "__main__":
    main()
