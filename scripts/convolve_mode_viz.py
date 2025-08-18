import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import os
from pathlib import Path

TRIM_LEN = 0.5


def main():
    # 1) Load the first TRIM_LEN seconds of data/export/DarkIllusion_Kick.wav
    sample_rate, audio = wavfile.read("data/export/DarkIllusion_Kick.wav")
    # Convert to float32 and normalize to +/-1.0
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128) / 128.0
    else:
        audio = audio.astype(np.float32)
    # If stereo, convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Trim to TRIM_LEN seconds
    ten_seconds = int(min(sample_rate * TRIM_LEN, len(audio)))
    audio = audio[:ten_seconds]

    # 2) Generate one-hot vectors (impulses) at beginning, middle, and end
    kernel_size = 4096
    kernels = []

    # Beginning impulse
    kernel_start = np.zeros(kernel_size, dtype=np.float32)
    kernel_start[0:10] = 1.0
    kernels.append(("Start Impulse", kernel_start))

    # Middle impulse
    kernel_middle = np.zeros(kernel_size, dtype=np.float32)
    kernel_middle[kernel_size // 2 : kernel_size // 2 + 10] = 1.0
    kernels.append(("Middle Impulse", kernel_middle))

    # End impulse
    kernel_end = np.zeros(kernel_size, dtype=np.float32)
    kernel_end[-10:] = 1.0
    kernels.append(("End Impulse", kernel_end))

    # 3 & 4) Convolve with each mode
    modes = ["same", "valid", "full"]
    results = {}

    for kernel_name, kernel in kernels:
        kernel_results = {}
        for mode in modes:
            kernel_results[mode] = np.convolve(audio, kernel, mode=mode)
        results[kernel_name] = kernel_results

    # 6) Plot the results in a grid
    fig, axs = plt.subplots(
        len(kernels) + 1, len(modes), figsize=(15, 10), sharex="col"
    )
    fig.suptitle("Convolution Modes Visualization", fontsize=16)

    # Plot original audio in first row
    for i, mode in enumerate(modes):
        axs[0, i].plot(audio)
        axs[0, i].set_title(f"Original Audio ({mode})")
        axs[0, i].set_ylim(-1.1, 1.1)

    # Plot convolution results
    for i, (kernel_name, kernel_results) in enumerate(results.items()):
        for j, mode in enumerate(modes):
            result = kernel_results[mode]
            axs[i + 1, j].plot(result)
            axs[i + 1, j].set_title(f"{kernel_name} ({mode})")
            axs[i + 1, j].set_ylim(-1.1, 1.1)

            # Add vertical lines showing kernel alignment
            if kernel_name == "Start Impulse":
                axs[i + 1, j].axvline(x=0, color="r", linestyle="--", alpha=0.5)
            elif kernel_name == "Middle Impulse":
                axs[i + 1, j].axvline(
                    x=kernel_size // 2, color="r", linestyle="--", alpha=0.5
                )
            elif kernel_name == "End Impulse":
                axs[i + 1, j].axvline(
                    x=kernel_size - 1, color="r", linestyle="--", alpha=0.5
                )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # 7) Save the plot
    os.makedirs("data/plots", exist_ok=True)
    plt.savefig("data/plots/convolve_viz.png", dpi=150)
    plt.close()
    print("Plot saved to data/plots/convolve_viz.png")


if __name__ == "__main__":
    main()
