import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from typing import Literal
from scipy.io import wavfile

def grad_and_conv(audio: jnp.ndarray, mode: Literal["valid", "same", "full"], place: Literal["start", "middle", "end"]):
    kernel_size = 4096

    def do_it(gain):
        if place == "start":
            kernel = jnp.zeros(kernel_size)
            kernel = kernel.at[0].set(gain)
        elif place == "middle":
            kernel = jnp.zeros(kernel_size)
            kernel = kernel.at[kernel_size // 2].set(gain)
        elif place == "end":
            kernel = jnp.zeros(kernel_size)
            kernel = kernel.at[-1].set(gain)
        kernel_scaled = kernel * gain
        return jnp.sum(jnp.convolve(audio, kernel_scaled, mode=mode))

    return jax.value_and_grad(do_it)(1.0)


sample_rate, audio = wavfile.read("data/export/DarkIllusion_Kick.wav")
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
elif audio.dtype == np.int32:
    audio = audio.astype(np.float32) / 2147483648.0
elif audio.dtype == np.uint8:
    audio = (audio.astype(np.float32) - 128) / 128.0
else:
    audio = audio.astype(np.float32)

for place in ["start", "middle", "end"]:
    for mode in ["valid", "same", "full"]:
        v,g = grad_and_conv(audio, mode=mode, place=place)
        print(place, mode, v,g)
