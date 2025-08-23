FROM nvidia/cuda:12.9.0-devel-ubuntu24.04

# System setup
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential \
        curl ca-certificates \
        fonts-roboto && \
    rm -rf /var/lib/apt/lists/*

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set environment variables for CUDA
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
ENV CUDA_HOME="/usr/local/cuda"

# Default command
CMD ["bash"]