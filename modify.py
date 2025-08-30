import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile

from report import load_summary
from main import transient_detector

jax.config.update("jax_platform_name", "cpu")

# Load audio file
wav_path = Path("data/gtr2.wav")
sample_rate, audio_np = wavfile.read(wav_path)
assert sample_rate == 48000, "Sample rate must be 48000 Hz"

# Convert to float32 and normalize to +/-1.0
if audio_np.dtype == np.int16:
    audio_np = audio_np.astype(np.float32) / 32768.0
elif audio_np.dtype == np.int32:
    audio_np = audio_np.astype(np.float32) / 2147483648.0
elif audio_np.dtype == np.uint8:
    audio_np = (audio_np.astype(np.float32) - 128) / 128.0
else:
    audio_np = audio_np.astype(np.float32)
if audio_np.ndim > 1:
    audio_np = np.mean(audio_np, axis=1)
audio = jnp.array(audio_np)

# Load trained model summary
summary_path = Path("data/results/ch5_filtcomp_results.json")
results = load_summary(summary_path)

# Run transient detector
predictions, aux = transient_detector(
    results.hyperparameters,
    results.parameters,
    audio,
    is_training=False,
    return_aux=True,
)

# Plot audio and predictions
time_axis = np.arange(len(audio)) / sample_rate
plt.figure(figsize=(12, 4))
plt.plot(time_axis, np.array(audio), label="Audio", linewidth=0.5)
plt.plot(time_axis, np.array(predictions), label="Prediction", linestyle="dotted")

# Detect transient times
from main import get_predicted_transient_times

threshold = 0.5  # You may adjust this threshold as needed
ignore_window_sec = (
    results.hyperparameters.ignore_window_sec
    if hasattr(results.hyperparameters, "ignore_window_sec")
    else 0.02
)
transient_times = get_predicted_transient_times(
    predictions, threshold, sample_rate, ignore_window_sec
)

# Plot detected transients
for t in np.array(transient_times):
    plt.axvline(
        t,
        color="r",
        linestyle="dashed",
        label="Detected Transient" if t == np.array(transient_times)[0] else None,
    )


# Envelope generator parameters
rise_time = 0.05  # seconds to rise to 1.0
fall_time = 0.1  # seconds to decay to 0.0

# Create envelope array
envelop = np.zeros(len(audio))
last_trigger = -np.inf
for t in np.array(transient_times):
    idx = int(t * sample_rate)
    last_trigger = idx
    # Rise: linear ramp up to 1.0
    rise_samples = int(rise_time * sample_rate)
    if idx + rise_samples < len(envelop):
        envelop[idx : idx + rise_samples] = np.linspace(
            0, 1, rise_samples, endpoint=False
        )
        start_decay = idx + rise_samples
    else:
        envelop[idx:] = np.linspace(0, 1, len(envelop) - idx, endpoint=False)
        start_decay = len(envelop)
    # Fall: exponential decay from 1.0
    fall_samples = int(fall_time * sample_rate)
    if start_decay < len(envelop):
        decay = np.exp(
            -np.arange(fall_samples) / (fall_time * sample_rate / np.log(100))
        )
        decay = decay[: len(envelop) - start_decay]
        envelop[start_decay : start_decay + len(decay)] = decay

# Clip envelope to max 1.0
envelop = np.clip(envelop, 0, 1)

# Plot envelope
plt.plot(time_axis, envelop, label="Envelope CV", color="magenta", linewidth=1.2)


# Formant filter bank with F1=370 Hz and F2=1900 Hz, modulated by envelope
from scipy.signal import iirpeak, lfilter

formant_freqs = np.array([370.0, 1900.0])  # F1 and F2
# Per-formant modulation depths (can be positive or negative)
mod_depths = np.array([-200.0, 400.0])  # F1 decreases, F2 increases
formant_q = 8.0  # Q factor for formant peaks

block_size = 1024
hop_size = block_size // 2
window = np.hanning(block_size)
filtered_audio = np.zeros(len(audio) + block_size)
window_sum = np.zeros(len(audio) + block_size)

for i in range(0, len(audio) - block_size + 1, hop_size):
    block_env = np.mean(envelop[i : i + block_size])
    block = np.array(audio[i : i + block_size])
    block_filtered = np.zeros_like(block)
    for f, base_freq in enumerate(formant_freqs):
        center_freq = base_freq + block_env * mod_depths[f]
        center_freq = np.clip(center_freq, 20, sample_rate / 2 - 100)
        b, a = iirpeak(center_freq / (sample_rate / 2), Q=formant_q)
        block_filtered += lfilter(b, a, block)
    block_filtered /= len(formant_freqs)
    block_windowed = block_filtered * window
    filtered_audio[i : i + block_size] += block_windowed
    window_sum[i : i + block_size] += window

# Normalize by window sum to avoid amplitude modulation
window_sum = np.where(window_sum == 0, 1, window_sum)
filtered_audio = filtered_audio[: len(audio)] / window_sum[: len(audio)]


# Normalize output loudness to match original
def rms(x):
    return np.sqrt(np.mean(np.square(x)))


orig_rms = rms(np.array(audio))
out_rms = rms(filtered_audio)
if out_rms > 0:
    filtered_audio = filtered_audio * (orig_rms / out_rms)

filtered_audio = 0.9 * filtered_audio

# Save filtered audio
from scipy.io.wavfile import write

filtered_int16 = np.int16(np.clip(filtered_audio, -1, 1) * 32767)
write("data/gtr_mod.wav", sample_rate, filtered_int16)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude / Prediction / Envelope")
plt.legend()
plt.title("Transient Detector Output")
plt.tight_layout()

# Plot time-domain and spectrograms of original and modified audio
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

# Time-domain original
axes[0].plot(time_axis, np.array(audio), color="black", linewidth=0.7)
axes[0].set_title("Original Audio (Time Domain)")
axes[0].set_ylabel("Amplitude")

# Time-domain modified
axes[1].plot(time_axis, filtered_audio, color="blue", linewidth=0.7)
axes[1].set_title("Modified Audio (Time Domain)")
axes[1].set_ylabel("Amplitude")

# Spectrogram original
spec0 = axes[2].specgram(
    np.array(audio), NFFT=1024, Fs=sample_rate, noverlap=512, cmap="magma"
)
axes[2].set_title("Original Audio Spectrogram")
axes[2].set_ylabel("Frequency (Hz)")
fig.colorbar(spec0[3], ax=axes[2], label="dB")

# Spectrogram modified
spec1 = axes[3].specgram(
    filtered_audio, NFFT=1024, Fs=sample_rate, noverlap=512, cmap="magma"
)
axes[3].set_title("Modified Audio Spectrogram")
axes[3].set_ylabel("Frequency (Hz)")
axes[3].set_xlabel("Time (s)")
fig.colorbar(spec1[3], ax=axes[3], label="dB")

plt.tight_layout()
plt.savefig("data/gtr_mod_spectrogram.png")
