from dataclasses import dataclass
import numpy as np
from scipy.io import wavfile
import os
import matplotlib.pyplot as plt
import random
from typing import List

@dataclass
class TransientExample:
    audio: np.ndarray  # 1D numpy array, mono, normalized to [-1.0, 1.0]
    label_array: np.ndarray  # 1D numpy array, 0.0/1.0, same length as audio
    transient_times: list  # List of transient times in seconds
    sample_rate: int  # Sample rate of the audio


def load_transient_example(base_path: str, window_s: float = 0.01) -> TransientExample:
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


def load_transient_times(txt_path: str) -> list[float]:
    """Load transient times (in seconds) from a label .txt file (first column only) using numpy."""
    times = np.loadtxt(txt_path, usecols=0, comments="#", ndmin=1)
    return times.tolist()


def load_wav_mono_normalized(filepath: str) -> tuple[np.ndarray, int]:
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


def generate_label_array(
    transient_times: list[float],
    audio_length: int,
    sr: int,
    window_s: float = 0.01
) -> np.ndarray:
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


def plot_transient_example(
    example: TransientExample,
    out_path: str,
    duration_s: float = 1.0
) -> None:
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

def chunkify_examples(
    example: "TransientExample",
    min_length_s: float = 3.0,
    max_length_s: float = 5.0,
    overlap_s: float = 1.0
) -> List["TransientExample"]:
    """
    Split a TransientExample into overlapping chunks of random length.
    Each chunk is a new TransientExample with adjusted transient_times.
    """
    audio = example.audio
    label = example.label_array
    sr = example.sample_rate
    transients = example.transient_times
    total_len = len(audio)
    chunks = []
    start_sample = 0
    while start_sample < total_len:
        chunk_length_s = random.uniform(min_length_s, max_length_s)
        chunk_length = int(chunk_length_s * sr)
        end_sample = min(start_sample + chunk_length, total_len)
        # Slice audio and label
        audio_chunk = audio[start_sample:end_sample]
        label_chunk = label[start_sample:end_sample]
        # Find transients in this chunk, adjust to chunk-relative time
        chunk_start_time = start_sample / sr
        chunk_end_time = end_sample / sr
        chunk_transients = [t - chunk_start_time for t in transients if chunk_start_time <= t < chunk_end_time]
        # Create chunk TransientExample
        chunk = TransientExample(
            audio=audio_chunk,
            label_array=label_chunk,
            transient_times=chunk_transients,
            sample_rate=sr
        )
        chunks.append(chunk)
        # Advance start_sample with overlap
        if end_sample == total_len:
            break
        start_sample = end_sample - int(overlap_s * sr)
        if start_sample < 0:
            start_sample = 0
    return chunks

def plot_chunks(
    chunks: List[TransientExample],
    out_dir: str = "data/plots/chunks"
) -> None:
    """Plot the first 5 chunks using plot_transient_example."""
    os.makedirs(out_dir, exist_ok=True)
    for i, chunk in enumerate(chunks[:5]):
        out_path = os.path.join(out_dir, f"chunk_{i+1}.png")
        plot_transient_example(chunk, out_path)
    print(f"Plotted {min(5, len(chunks))} chunks to {out_dir}")

def main() -> None:
    # Load and plot an example
    base_path = 'data/export/DarkIllusion_ElecGtr5DI'
    example = load_transient_example(base_path)
    chunks = chunkify_examples(example)
    plot_chunks(chunks)


if __name__ == "__main__":
    main()
