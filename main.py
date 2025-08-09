from dataclasses import dataclass
import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt

@dataclass
class TransientExample:
    audio: np.ndarray  # 1D numpy array, mono, normalized to [-1.0, 1.0]
    label_array: np.ndarray  # 1D numpy array, 0.0/1.0, same length as audio
    transient_times: list  # List of transient times in seconds
    sample_rate: int  # Sample rate of the audio


def load_transient_example(base_path, window_s=0.01):
    """
    Given a base file path (without extension), load the wav and txt, generate label array, and return a TransientExample.
    Example: base_path='data/export/DarkIllusion_ElecGtr5DI' will load .wav and .txt
    """
    audio_path = base_path + ".wav"
    label_path = base_path + "_Labels.txt"
    audio, sr = load_wav_mono_normalized(audio_path)
    transient_times = load_transient_times(label_path)
    label_array = generate_label_array(
        transient_times, len(audio), sr, window_s=window_s
    )
    return TransientExample(
        audio=audio,
        label_array=label_array,
        transient_times=transient_times,
        sample_rate=sr,
    )


def load_transient_times(txt_path):
    """Load transient times (in seconds) from a label .txt file (first column only) using numpy."""
    times = np.loadtxt(txt_path, usecols=0, comments="#", ndmin=1)
    return times.tolist()


def load_wav_mono_normalized(filepath):
    """Load a wav file, convert to mono if needed, and normalize to [-1.0, 1.0]."""
    sr, data = wavfile.read(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)  # Convert to mono
    # Normalize to [-1.0, 1.0] depending on dtype
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        data = (data.astype(np.float32) - 128) / 128.0
    else:
        data = data.astype(np.float32)
    return data, sr


def generate_label_array(transient_times, audio_length, sr, window_s=0.01):
    """
    Generate a label array (0.0/1.0) for the given transient times.
    Each transient is marked as 1.0 for ±window_s/2 around the transient time.
    """
    label = np.zeros(audio_length, dtype=np.float32)
    half_window = window_s / 2.0
    for t in transient_times:
        start = int(max(0, (t - half_window) * sr))
        end = int(min(audio_length, (t + half_window) * sr))
        label[start:end] = 1.0
    return label


def plot_transient_example(example, out_path, duration_s=1.0):
    """Plot the audio and label array of a TransientExample and save to out_path. Plots up to duration_s seconds."""
    audio = example.audio
    label = example.label_array
    sr = example.sample_rate

    max_samples = min(len(audio), int(duration_s * sr))
    t = np.arange(max_samples) / sr

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(t, audio[:max_samples], label='Audio', color='C0', linewidth=0.8)
    ax1.set_ylabel('Audio')
    ax1.set_xlabel('Time (s)')
    ax2 = ax1.twinx()
    ax2.plot(t, label[:max_samples], label='Label', color='C1', alpha=0.5, linewidth=1.5)
    ax2.set_ylabel('Label (0/1)')
    ax1.set_title(f'Audio and Transient Label (First {duration_s} Second(s))')
    ax1.set_xlim(0, duration_s)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close(fig)

# new functions go right above this line.

def main():
    # Load and plot an example
    base_path = 'data/export/DarkIllusion_ElecGtr5DI'
    example = load_transient_example(base_path)
    plot_path = 'data/plots/input_data.png'
    plot_transient_example(example, plot_path, duration_s=5.0)
    print(f"Plotted and saved to {plot_path}")


if __name__ == "__main__":
    main()
