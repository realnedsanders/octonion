# Octonion Computation Substrate - Development Container
# ROCm + PyTorch environment for octonionic ML research
#
# Build:  docker compose build
# Run:    docker compose run --rm dev bash

FROM rocm/pytorch:rocm7.2_ubuntu24.04_py3.12_pytorch_release_2.9.1

LABEL maintainer="aescalera"
LABEL description="Development container for octonionic computation substrate with ROCm GPU support"
LABEL rocm.version="7.2"
LABEL pytorch.version="2.9.1"

# Install uv for fast Python dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Ensure uv is on PATH
ENV PATH="/root/.local/bin:${PATH}"

# uv requires copy mode inside containers with mounted volumes
ENV UV_LINK_MODE=copy

WORKDIR /workspace

# Project source is mounted via docker-compose, not copied
# Dependencies are installed at runtime via `uv sync`

CMD ["bash"]
