import json
from typing import Optional
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from pathlib import Path
from main import (
    EvaluationResult,
    ResultsSummary,
    transient_detector,
    Hyperparameters,
    Params,
    Chunk,
    FORCE_SAMPLE_RATE,
)


def load_summary(path: Path) -> ResultsSummary:
    with open(path, "r") as f:
        summary_dict = json.load(f)

    def from_serializable(cls, obj):
        # Recursively reconstruct dataclasses and arrays
        if cls is Hyperparameters:
            return Hyperparameters(**obj)
        elif cls is Params:
            # Convert lists to jnp arrays for array fields
            return Params(
                window_size_sec=jnp.array(obj["window_size_sec"]),
                weights=jnp.array(obj["weights"]),
                filter_f0s=jnp.array(obj["filter_f0s"]),
                filter_qs=jnp.array(obj["filter_qs"]),
                bias=obj["bias"],
                post_gain=obj["post_gain"],
                post_bias=obj["post_bias"],
                compressor_window_size_sec=obj["compressor_window_size_sec"],
                compressor_makeup_gain=obj["compressor_makeup_gain"],
                compressor_threshold=obj["compressor_threshold"],
            )
        elif cls is EvaluationResult:
            return EvaluationResult(**obj)
        elif cls is ResultsSummary:
            # Only keep EvaluationResult objects in lists
            tr = [
                from_serializable(EvaluationResult, r) for r in obj["training_results"]
            ]
            tr = [r for r in tr if isinstance(r, EvaluationResult)]
            vr = [
                from_serializable(EvaluationResult, r)
                for r in obj["validation_results"]
            ]
            vr = [r for r in vr if isinstance(r, EvaluationResult)]
            summary = ResultsSummary(
                hyperparameters=from_serializable(
                    Hyperparameters, obj["hyperparameters"]
                ),
                parameters=from_serializable(Params, obj["parameters"]),
                training_results=tr,
                validation_results=vr,
            )
            # Attach loss_history if present
            if "loss_history" in obj:
                summary.loss_history = obj["loss_history"]
            return summary
        else:
            return obj

    result = from_serializable(ResultsSummary, summary_dict)
    if not isinstance(result, ResultsSummary):
        raise TypeError("Deserialized object is not a ResultsSummary")
    return result


def simple_detector_explainer():
    # Load audio
    wav_path = Path("data/Gtr2.wav")
    # wav_path = Path("data/export/DarkIllusion_Kick.wav")
    sample_rate, audio_np = wavfile.read(wav_path)
    assert sample_rate == FORCE_SAMPLE_RATE, (
        f"Sample rate {sample_rate} != {FORCE_SAMPLE_RATE}"
    )
    # Convert to float32 and normalize to +/-1.0
    if audio_np.dtype == np.int16:
        audio_np = audio_np.astype(np.float32) / 32768.0
    elif audio_np.dtype == np.int32:
        audio_np = audio_np.astype(np.float32) / 2147483648.0
    elif audio_np.dtype == np.uint8:
        audio_np = (audio_np.astype(np.float32) - 128) / 128.0
    else:
        audio_np = audio_np.astype(np.float32)
    # If stereo, convert to mono
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=1)
    audio = jnp.array(audio_np)

    # Dummy labels (no ground truth for this file)
    labels = jnp.zeros_like(audio)

    # Setup hyperparameters and params (use reasonable defaults)
    hyper = Hyperparameters(
        num_channels=2, enable_filters=False, enable_compressor=False
    )
    params = Params(
        window_size_sec=jnp.array([0.02, 0.1]),
        weights=jnp.array([1.0, 1.0]),
        filter_f0s=jnp.array([2000.0, 4000.0]),
        filter_qs=jnp.array([1.0, 1.0]),
        bias=0.0,
        post_gain=10.0,
        post_bias=0.0,
        compressor_window_size_sec=0.005,
        compressor_makeup_gain=1.0,
        compressor_threshold=0.5,
    )

    # Run transient_detector with aux data
    _, aux = transient_detector(
        hyper, params, audio, is_training=False, return_aux=True
    )
    raw_envelopes = np.array(aux["raw_envelopes"])  # shape: (channels, samples)
    audio_np = np.array(audio)

    # Use xlim to crop to area around first transient (or just first second if no label)
    t = np.arange(len(audio_np)) / FORCE_SAMPLE_RATE
    label_indices = np.where(np.array(labels) > 0)[0]
    if len(label_indices) > 0:
        first_transient_idx = label_indices[0]
        crop_start_t = max(0, t[first_transient_idx] - 0.25)
        crop_end_t = min(t[-1], t[first_transient_idx] + 1000)
        # crop_end_t = t[-1]
    else:
        crop_start_t = 0.0
        crop_end_t = min(t[-1], 1.0)
        # crop_end_t = t[-1]

    fig, (ax_audio, ax_env) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot audio in first subplot
    ax_audio.plot(t, audio_np, label="Input", color="C0", alpha=0.7)
    ax_audio.set_ylabel("Audio amplitude", color="C0")
    ax_audio.tick_params(axis="y", labelcolor="C0")
    ax_audio.set_xlim(crop_start_t, crop_end_t)

    # If there are at least 2 envelopes, plot thresholded diff on a separate scale
    threshold = 0.05  # Example threshold, adjust as needed
    if raw_envelopes.shape[0] >= 2:
        diff = raw_envelopes[0] - raw_envelopes[1]
        above_thresh = (diff > threshold).astype(float)
        ax_audio_diff = ax_audio.twinx()
        ax_audio_diff.plot(
            t, above_thresh, label=f"Diff > {threshold}", color="C3", alpha=0.7
        )
        ax_audio_diff.set_ylabel("Diff > threshold", color="C3")
        ax_audio_diff.set_xlim(crop_start_t, crop_end_t)
        ax_audio_diff.tick_params(axis="y", labelcolor="C3")
        ax_audio_diff.axhline(y=threshold, color="C3", linestyle="--", alpha=0.7)
        # Combine legends
        lines_audio, labels_audio = ax_audio.get_legend_handles_labels()
        lines_diff, labels_diff = ax_audio_diff.get_legend_handles_labels()
        ax_audio.legend(
            lines_audio + lines_diff, labels_audio + labels_diff, loc="upper left"
        )
    else:
        ax_audio.legend(loc="upper left")

    # Plot envelopes on main axis of second subplot
    for i in range(raw_envelopes.shape[0]):
        ax_env.plot(
            t,
            raw_envelopes[i],
            label=f"Envelope period={int(params.window_size_sec[i] * 1000)}ms",
            linestyle="--",
            alpha=0.8,
        )
    ax_env.set_ylabel("Envelope")
    ax_env.set_xlim(crop_start_t, crop_end_t)
    ax_env.set_xlabel("Time (s)")

    # Add annotation arrows if there are at least 2 envelopes and above_thresh exists
    if raw_envelopes.shape[0] >= 2:
        ax_env_diff = ax_env.twinx()
        diff = raw_envelopes[0] - raw_envelopes[1]
        ax_env_diff.plot(
            t, diff, label="Envelope Diff (Ch0-Ch1)", color="C2", alpha=0.8
        )
        ax_env_diff.set_ylabel("Envelope Diff", color="C2")
        ax_env_diff.set_xlim(crop_start_t, crop_end_t)
        ax_env_diff.tick_params(axis="y", labelcolor="C2")
        # Legends for both axes
        lines_env, labels_env = ax_env.get_legend_handles_labels()
        lines_diff, labels_diff = ax_env_diff.get_legend_handles_labels()
        ax_env.legend(
            lines_env + lines_diff, labels_env + labels_diff, loc="upper left"
        )

        # Compute above_thresh here to ensure it is always defined
        threshold = 0.05  # keep consistent with audio subplot
        above_thresh = (diff > threshold).astype(float)
        # Find first time above_thresh is true
        idxs_true = np.where(above_thresh > 0)[0]
        if len(idxs_true) > 0:
            first_idx = idxs_true[0]
            time_arrow = t[first_idx]
            y_fast = raw_envelopes[0][first_idx]
            y_slow = raw_envelopes[1][first_idx]
            # Arrow for fast envelope
            ax_env.annotate(
                "Fast envelop rises quickly",
                xy=(time_arrow, y_fast),
                xytext=(time_arrow - 0.2, y_fast + 0.1),
                arrowprops=dict(
                    facecolor="C0", edgecolor="none", shrink=0.05, width=2, headwidth=8
                ),
                color="C0",
                fontsize=11,
                ha="left",
            )
            # Arrow for slow envelope
            ax_env.annotate(
                "Slow envelop lags",
                xy=(time_arrow, y_slow),
                xytext=(time_arrow + 0.05, y_slow - 0.08),
                arrowprops=dict(
                    facecolor="C1", edgecolor="none", shrink=0.05, width=2, headwidth=8
                ),
                color="C1",
                fontsize=11,
                ha="left",
            )
    else:
        ax_env.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("figures/simple_detector_explainer.png")


def loss_plot(summary_path: Path, summary_name: str):
    """
    Plots the loss history from a results summary file and saves the figure.
    The output filename is figures/loss_{summary_path.stem}.png
    """
    summary = load_summary(summary_path)
    if not hasattr(summary, "loss_history") or summary.loss_history is None:
        print(f"No loss history found in {summary_path}")
        return
    loss = summary.loss_history
    plt.figure(figsize=(8, 4))
    plt.plot(loss, label=f"MSE: {summary_name}", color="C0")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Squared Error Loss")
    # plt.title(f"Loss History: {summary_name}")
    plt.legend()
    plt.tight_layout()
    out_path = Path("figures") / f"loss_{summary_name}.png".replace(" ", "_").replace(',', '').replace('&', '').lower()
    plt.savefig(out_path)
    print(f"Saved loss plot to {out_path}")


def chunk_plot(track_name: str, hyperparameters: Optional[Hyperparameters] = None):
    """
    Plots the audio waveform and labels for a given track name using load_data and filter argument.
    Audio and labels are plotted on the same axis with different scales.
    """
    from main import load_data, FORCE_SAMPLE_RATE
    if hyperparameters is None:
        hyperparameters = Hyperparameters()
    chunks = load_data(hyperparameters, filter={track_name})
    if not chunks:
        print(f"No chunk found for track '{track_name}'")
        return
    chunk = chunks[0]
    audio = np.array(chunk.audio)
    labels = np.array(chunk.labels)
    t = np.arange(len(audio)) / FORCE_SAMPLE_RATE
    fig, ax1 = plt.subplots(figsize=(14, 6))
    color_audio = "C0"
    color_labels = "C1"
    ax1.plot(t, audio, color=color_audio, label="Audio", alpha=0.7)
    ax1.set_ylabel("Audio amplitude", color=color_audio)
    ax1.tick_params(axis="y", labelcolor=color_audio)
    ax2 = ax1.twinx()
    ax2.plot(t, labels, color=color_labels, label="Labels", alpha=0.7)
    ax2.set_ylabel("Labels", color=color_labels)
    ax2.tick_params(axis="y", labelcolor=color_labels)
    ax1.set_xlabel("Time (s)")
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    plt.tight_layout()
    out_path = Path("figures") / f"chunk.png"
    plt.savefig(out_path)
    print(f"Saved chunk plot to {out_path}")

def main():
    jax.config.update("jax_platform_name", "cpu")
    plt.style.use("./style.mplstyle")
    # simple_detector_explainer()
    # loss_plot(Path("data/results/ch2_results.json"), "Simple Model")
    # loss_plot(Path("data/results/ch2_filtcompfix_results.json"), "2 Channels with Filter & Compressor")
    chunk_plot("DarkIllusion_ElecGtr5DI")


if __name__ == "__main__":
    main()
