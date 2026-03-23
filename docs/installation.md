# Installation

The simplest way to install flowMC is through pip:

```bash
pip install flowMC
```

This will install the latest stable release and its dependencies.
flowMC is built on [JAX](https://github.com/google/jax).
By default, this installs the CPU version of JAX.
If you have an NVIDIA GPU, install the CUDA-enabled version:

```bash
pip install flowMC[cuda]
```

If you want to install the latest version of flowMC, you can clone this repo and install it locally:

```bash
git clone https://github.com/GW-JAX-Team/flowMC.git
cd flowMC
pip install -e .
```

We recommend using [uv](https://docs.astral.sh/uv/) to manage your Python environment. After cloning the repository, run `uv sync` to create a virtual environment with all dependencies installed.
