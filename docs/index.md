# flowMC

**A JAX-based normalizing-flow-enhanced MCMC sampler for probabilistic inference**

[![doc](https://badgen.net/badge/Read/the%20doc/blue)](https://flowMC.readthedocs.io/en/latest/) [![license](https://badgen.net/badge/License/MIT/blue)](https://github.com/GW-JAX-Team/flowMC/blob/main/LICENSE) [![coverage](https://badgen.net/coveralls/c/github/GW-JAX-Team/flowMC/main)](https://coveralls.io/github/GW-JAX-Team/flowMC?branch=main) [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/GW-JAX-Team/flowMC/main.svg)](https://results.pre-commit.ci/latest/github/GW-JAX-Team/flowMC/main)

![flowMC_logo](logo.png)

flowMC is a JAX-based package for normalizing-flow-enhanced Markov chain Monte Carlo (MCMC) sampling. By using normalizing flows as a global proposal, flowMC accelerates convergence for multi-modal and high-dimensional posteriors while running natively on GPU with minimal hyperparameter tuning.

!!! warning
    flowMC has not yet reached v1.0.0 and the API may change. Use at your own risk. Consider pinning to a specific version if you need API stability.

## Documentation

- **[Installation](installation.md)** — How to install flowMC
- **[Quick Start](quickstart.md)** — A basic example to get started
- **[Tutorials](tutorials/index.md)** — Step-by-step guides and worked examples
- **[Hyperparameter Reference](hyperparameter_reference.md)** — Guide to all sampler settings
- **[FAQ](FAQ.md)** — Answers to common questions
- **[API Reference](api/flowMC/)** — Full API documentation
