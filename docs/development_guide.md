# Development Guide

## Anatomy of flowMC

Prior to version 0.4.0, `flowMC` was a package that was designed to execute the algorithm detailed in [this paper](https://arxiv.org/pdf/2105.12603). Since then the community has tried applying `flowMC` to different problems. While there were some successes, there are also limiting factors in terms of performance. One of the biggest issues `flowMC` faced is the fact that the global-local sampling algorithm was baked into the top level `Sampler` API, which means `flowMC` can only use the exact algorithm described in the paper. What if the users want to use a different model? Or run some optimisation steps during the sampling stage? Or apply annealing? These are either impossible or not very intuitive in `flowMC` prior to version 0.4.0.

Seeing this limitation, we redesigned the middle level API of `flowMC` while keeping the top level API as similar as possible. This guide aims to describe the different components of `flowMC` and how they interact with each other, and give users who want to extend `flowMC` to optimise for their specific problems a starting point on what could be useful to change. This also acts as a rule of thumb for users who want to use `flowMC` as a black box and interact with internal components through hyperparameters only.

### Target distribution

The target distribution should be defined as a log-probability density function, which follows the following function signature:

```python
def target_log_prob_fn(x: Array, data: dict) -> Float:
    ...
    return log_prob
```

The `target_log_prob_fn` should take in a 1-D array `x` of length `n_dim` and a dictionary `data` that contains any additional data that the target distribution depends on. The function should return a scalar that is the log-probability density of the target distribution at `x`.

To ensure the target distribution is well-defined and performant, you should also check whether the function is behaving as expected when `jax.jit` and `jax.grad` are applied to it.

### Sampler

On the top level, the `Sampler` class is a thin wrapper on top of the resource-strategy pair (defined below) that provides a couple of extra functionality. The `Sampler` class manages the resources and strategies, as well as run-related parameters such as where would the resources be stored if the user decides to serialise the resources.

```python
nf_sampler = Sampler(
    n_dim,
    n_chains,
    rng_key,
    # you can either supply the resources and strategies directly,
    # which is prioritized over the resource-strategy bundles
    resources=resources,
    strategies=strategies,
    strategy_order=strategy_order,
    # or you can supply the resource-strategy bundles
    resource_strategy_bundles=bundle,
)
```

The main loop of `Sampler` is pretty straightforward after initialisation: given the available resources, it iterates through the list of strategies, each of which takes the resources, performs some actions (such as taking local steps or training a normalising flow), and returns the updated resources. In the current implementation, the `Sampler` simply goes through the list of strategies, but in the future we are planning to make the main loop more flexible (for example, by supporting automatic stopping based on some criteria).

### Resource and Strategy

At the core of the new `flowMC` API are the resource and strategy interfaces. Broadly speaking resources are similar to a data class, and strategies are similar to functions.
**Resources** store some attribute and can be manipulated, but should not have too many methods associated with it. For example, a buffer that stores the sampling results is a resource, a MALA kernel is a resource, and a normalising flow model is a resource. **Strategies** are functions that take in resources and return updated resources. For example, taking a local step requires two kinds of resources: a proposal distribution and the buffer where the samples are stored. Examples of strategies are taking a local step, training a normalising flow, and running an optimisation step.

If you are initialising the resources and strategies directly, you can do something like:

```python
resources = {
    "buffer": Buffer(name, n_chains, n_steps, n_dims),
    "proposal": MALA(step_size),
    "model": NormalizingFlow(model_parameters),
}

strategies = {
    "Strategy 1": Strategy1(),
    "Strategy 2": Strategy2(),
}

strategy_order = ["Strategy 1", "Strategy 2", "Strategy 1", ...]
```

The reason for this separation is to allow users to compose different strategies together. For example, the user may want to update the parameters of a proposal kernel like MALA with the local information from a normalising flow model. Instead of hard coding this functionality to associate with either the MALA kernel or the normalising flow model, the current API allows the user to define a strategy that takes in both the MALA kernel and the normalising flow model, and updates the MALA kernel with the information from the normalising flow model. This separates the concern of intermixing different components of the algorithm and makes experimenting with new strategies more manageable.

Since this API is designed for users who are willing to look into the guts of `flowMC` and experiment with different strategies, the main question to ask is whether a new data structure/functionality should be a resource or a strategy. While there are no hard rules for such implementation other than conforming to the individual base classes, a good rule of thumb is to ask whether the new data structure/functionality is something that should be updated by other strategies. If the answer is yes, then it should be a resource. If the answer is no, then it should be a strategy.

One extra criterion that decides whether an implementation should be a resource or a strategy is whether the implementation is compatible with `jax`'s transformation. Resources should be compatible with `jit`, and strategies are not required to be compatible with `jit`. An example to illustrate the difference is a training loop contains for-looping over a number of epochs and logging the metadata, which is usually not necessary to be jitted, so this should be a strategy. A neural network and its main functions need to run efficiently on GPU no matter in sampling or training, so it should be a resource.

You can find the hyper-parameters of a resource, a strategy, or a resource-strategy bundles in the API docs.

## Guiding principles

### Write the likelihood function in JAX

If your likelihood is fully defined in [JAX](https://github.com/google/jax), there are a couple benefits that compound with each other:

1. JAX allows you to access the gradient of the likelihood function with respect to the parameters of the model through automatic differentiation. Having access to the gradient allows the use of gradient-based local sampler such as Metropolis-adjusted Langevin algorithm (MALA) and Hamiltonian Monte Carlo (HMC). These algorithms allow the sampler to handle high dimensional problems, and is often more efficient than the gradient-free local sampler such as Metropolis-Hastings.
2. JAX uses [XLA](https://www.tensorflow.org/xla) to compile your code not only into machine code but also in a way that is more optimised for accelerators such as GPUs and TPUs. Having multiple MCMC chains helps speed up the training of the normalising flow. Accelerators such as GPUs and TPUs provide parallel computing solutions that are more scalable compared to CPUs.

Since version 0.4.0, we made the design choice of removing support for likelihood functions incompatible with `jax` transformations. The reason is that `flowMC` is designed to leverage GPU acceleration and machine learning methods to solve complex problems. If a developer decides to use `flowMC` to try to solve their problem, it is also a good time to consider rewriting their legacy code base in `jax`, which on its own could provide a significant speedup. Instead of letting people off the hook by allowing non-jax compatible likelihood functions, we decided to enforce the use of `jax` to encourage users to take advantage of its benefits.

### Parallelise whenever you can

One should centre their choice of resource and strategy around leveraging parallelisation. This is reflected by the fact that `n_chains` is a required parameter for the `Sampler` class. The reason for this is `flowMC` is designed to solve problems with complex geometry using adaptive sampling method such as training a normalising flow together with a local proposal, which benefits tremendously from multiple chains running in parallel.
