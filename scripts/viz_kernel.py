import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

MAX_KERNEL_SIZE = 512
SIZE_SAMPLES = 100
SHARPNESS = 0.5

idx = jnp.arange(-MAX_KERNEL_SIZE // 2, MAX_KERNEL_SIZE // 2 + 1).astype(jnp.float32)
kernel = jnp.where(idx <= 0, jax.nn.sigmoid(SHARPNESS * (idx + SIZE_SAMPLES)), 0.0)

plt.plot(idx, kernel, label="Kernel")
plt.savefig("data/plots/kernel_viz.png")
