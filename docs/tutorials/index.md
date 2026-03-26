# Tutorials

These tutorials walk you through flowMC from first principles to advanced customization. They are ordered from simplest to most involved — if you are new to flowMC, start with the dual moon example.

---

## [A Step-by-Step Example](dualmoon.ipynb)

The canonical flowMC example. Sample a 5-dimensional multi-modal "dual moon" distribution using the high-level `RQSpline_MALA_Bundle`, which pairs a MALA local kernel with a Masked Coupling RQSpline normalizing flow as a learned global proposal. Covers the full workflow: defining a log-probability function, configuring the sampler, running it, and diagnosing convergence with loss curves, acceptance rates, and R-hat.

---

## [Bundles](bundles.md)

Reference guide for the built-in resource-strategy bundles (`RQSpline_MALA`, `RQSpline_HMC`, `RQSpline_GRW`, and their parallel-tempering variants). Explains what each bundle does, when to choose it, and the structure of the training and production sampling loops.

---

## [Hyperparameter Reference](hyperparameters.md)

Complete reference for every constructor parameter across all bundles, organised into categories (required, local sampler, normalizing flow, training, execution, early stopping, parallel tempering). Each entry includes the default value and guidance on when and how to tune it.

---

## [Parallel Tempering](parallel_tempering.ipynb)

Demonstrates why standard local kernels struggle with well-separated modes and how parallel tempering (replica exchange MCMC) overcomes this. Builds a parallel tempering sampler from scratch using the low-level resource-strategy API, then contrasts it with the convenience `RQSpline_MALA_PT_Bundle`.

---

## [Custom Strategy Composition](custom_strategy.ipynb)

Shows the flexibility of the resource-strategy interface by composing a custom pipeline that prepends an Adam optimization phase to a random-walk MCMC kernel. A useful pattern when chains are initialized far from the high-probability region.

---

## [Training Normalizing Flows](train_normalizing_flow.ipynb)

A standalone introduction to the two normalizing flow architectures available in flowMC — **RealNVP** and **Masked Coupling RQSpline** — trained on a 2D toy dataset. Compares their expressiveness and computational cost, and shows how a trained flow plugs into the resource-strategy system via the `TrainModel` strategy.

---

## [Training Flow Matching Models](train_flow_match.ipynb)

Introduces **flow matching** as an alternative to traditional normalizing flows. Trains a Conditional Optimal Transport model on a 2D distribution using a MLP-parameterized velocity field, then generates samples by integrating the learned ODE and evaluates the inferred density on a grid.
