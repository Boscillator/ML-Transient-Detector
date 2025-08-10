from typing import List
import numpy as np
import jax.numpy as jnp

from data import (
    TransientExample,
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
import matplotlib.pyplot as plt  # kept for any future in-main plotting
from evaluation import EvaluationResult
from typing import Dict, Union
from plotting import plot_chunks
from plotting import plot_chunk_with_prediction, plot_predictions
from model import transient_detector, loss_function
from evaluation import EvaluationResult, evaluate_model_for_threshold

import numpy as np  # re-import for usage below


def main() -> None:
    hyperparams = ExperimentHyperparameters()
    # Load and plot an example
    base_path = "data/export/DarkIllusion_ElecGtr5DI"
    example = load_transient_example(base_path, hyperparams)
    chunks = chunkify_examples(example, hyperparams)
    chunks = chunks[:5]  # Limit to first 5 chunks for testing

    params = TransientDetectorParameters()

    # Plot predictions before optimization (eval kernel)
    plot_predictions(
        chunks,
        params,
        output_dir="data/plots/chunk_preds",
        is_training=False,
        hyperparams=hyperparams,
        do_debug=True,
        prefix="",
        print_prefix="",
    )

    # Optimize parameters
    print("Optimizing transient detector parameters...")
    opt_params = optimize_transient_detector(chunks, hyperparams)
    print(f"Optimized parameters: {opt_params}")

    # Evaluate optimized model across thresholds
    eval_results = evaluate_model(hyperparams, opt_params, chunks)
    # Print concise summary
    print("Evaluation results by threshold:")
    for r in eval_results:
        print(
            f"  th={r.threshold:.2f} | loss={r.loss:.4f} | TP={r.true_positives} FP={r.false_positives} FN={r.false_negatives} | acc={r.accuracy:.3f} rec={r.recall:.3f}"
        )
    # Select best by accuracy and recall
    best_by_acc = max(eval_results, key=lambda r: r.accuracy) if eval_results else None
    best_by_rec = max(eval_results, key=lambda r: r.recall) if eval_results else None
    if best_by_acc:
        print(
            f"Best accuracy: th={best_by_acc.threshold:.2f}, acc={best_by_acc.accuracy:.3f}, rec={best_by_acc.recall:.3f}"
        )
    if best_by_rec:
        print(
            f"Best recall:   th={best_by_rec.threshold:.2f}, acc={best_by_rec.accuracy:.3f}, rec={best_by_rec.recall:.3f}"
        )

    # Plot predictions after optimization
    plot_predictions(
        chunks,
        opt_params,
        output_dir="data/plots/chunk_preds_optimized",
        is_training=False,
        hyperparams=hyperparams,
        do_debug=True,
        prefix="",
        print_prefix="Optimized: ",
    )


if __name__ == "__main__":
    main()
