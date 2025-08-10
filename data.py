"""
Data loading, chunking, and label generation for transient detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import logging
import os
import random
import numpy as np
from scipy.io import wavfile
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from model import ExperimentHyperparameters  # for type hints only

logger = logging.getLogger(__name__)


@dataclass
class TransientExample:
    audio: np.ndarray  # 1D numpy array, mono, normalized to [-1.0, 1.0]
    label_array: np.ndarray  # 1D numpy array, 0.0/1.0, same length as audio
    transient_times: list  # List of transient times in seconds
    sample_rate: int  # Sample rate of the audio


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
    window_s: Optional[float] = None,
    hyperparams: Optional["ExperimentHyperparameters"] = None,
) -> np.ndarray:
    """
    Generate a label array (0.0/1.0) for the given transient times.
    Each transient is marked as 1.0 for ±window_s/2 around the transient time.
    """
    label = np.zeros(audio_length, dtype=np.float32)
    if window_s is None:
        window_s = hyperparams.window_s if hyperparams is not None else 0.04
    half_window = window_s / 2.0
    for t in transient_times:
        start = int(max(0, (t - half_window) * sr))
        end = int(min(audio_length, (t + half_window) * sr))
        label[start:end] = 1.0
    return label


def load_transient_example(
    base_path: str,
    hyperparams: "ExperimentHyperparameters",
    window_s: Optional[float] = None,
) -> TransientExample:
    """
    Given a base file path (without extension), load the wav and txt, generate label array, and return a TransientExample.
    Example: base_path='data/export/DarkIllusion_ElecGtr5DI' will load .wav and .txt
    """
    audio_path = base_path + ".wav"
    label_path = base_path + "_Labels.txt"
    audio, sr = load_wav_mono_normalized(audio_path)
    transient_times = load_transient_times(label_path)
    if window_s is None:
        window_s = hyperparams.window_s
    label_array = generate_label_array(
        transient_times, len(audio), sr, window_s=window_s, hyperparams=hyperparams
    )
    return TransientExample(
        audio=audio,
        label_array=label_array,
        transient_times=transient_times,
        sample_rate=sr,
    )


def chunkify_examples(
    example: TransientExample,
    hyperparams: "ExperimentHyperparameters",
    min_length_s: Optional[float] = None,
    max_length_s: Optional[float] = None,
    overlap_s: Optional[float] = None,
) -> List[TransientExample]:
    """
    Split a TransientExample into overlapping chunks of random length.
    Each chunk is a new TransientExample with adjusted transient_times.
    """
    audio = example.audio
    label = example.label_array
    sr = example.sample_rate
    transients = example.transient_times
    total_len = len(audio)
    chunks: List[TransientExample] = []
    start_sample = 0
    if min_length_s is None:
        min_length_s = hyperparams.min_length_s
    if max_length_s is None:
        max_length_s = hyperparams.max_length_s
    if overlap_s is None:
        overlap_s = hyperparams.overlap_s
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
        chunk_transients = [
            t - chunk_start_time
            for t in transients
            if chunk_start_time <= t < chunk_end_time
        ]
        # Create chunk TransientExample
        chunk = TransientExample(
            audio=audio_chunk,
            label_array=label_chunk,
            transient_times=chunk_transients,
            sample_rate=sr,
        )
        chunks.append(chunk)
        # Advance start_sample with overlap
        if end_sample == total_len:
            break
        start_sample = end_sample - int(overlap_s * sr)
        if start_sample < 0:
            start_sample = 0
    return chunks


def load_whole_dataset(path: Path, hyperparams: "ExperimentHyperparameters") -> List[TransientExample]:
    """
    Scan a directory for .wav and _Labels.txt pairs, load each as a TransientExample using load_transient_example, and return a list of examples.
    Args:
        path: Path to the folder containing .wav and _Labels.txt files (e.g., Path('data/export'))
        hyperparams: ExperimentHyperparameters instance
    Returns:
        List[TransientExample]
    """
    examples = []
    path = Path(path)
    wav_files = sorted(path.glob("*.wav"))
    for wav_file in wav_files:
        base = wav_file.with_suffix("")
        # Remove trailing extension, e.g. 'DarkIllusion_ElecGtr5DI'
        base_path = str(base)
        label_path = base_path + "_Labels.txt"
        if Path(label_path).exists():
            try:
                example = load_transient_example(base_path, hyperparams)
                examples.append(example)
            except Exception as e:
                logger.warning(f"Failed to load {base_path}: {e}")
        else:
            logger.info(f"No label file for {base_path}, skipping.")
    return examples

def load_dataset(path: Path, hyperparms: ExperimentHyperparameters, split: float = 0.5) -> Tuple[List[TransientExample], List[TransientExample]]:
    """
    Load the dataset from a directory, shuffle, and split into train and validation sets.
    Args:
        path: Path to the folder containing .wav and _Labels.txt files (e.g., Path('data/export'))
        hyperparms: ExperimentHyperparameters instance
        split: Fraction of data to use for training (default 0.5)
    Returns:
        train_set: List[TransientExample]
        val_set: List[TransientExample]
    """
    examples = load_whole_dataset(path, hyperparms)
    if not examples:
        logger.warning(f"No examples found in {path}")
        return [], []
    rng = random.Random(42)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    split_idx = int(len(examples) * split)
    if len(examples) == 1:
        logger.error("Only one example")
        raise ValueError("Not enough examples to split into train and validation sets.")
    else:
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]
        train_set = [examples[i] for i in train_indices]
        val_set = [examples[i] for i in val_indices]
    return train_set, val_set
