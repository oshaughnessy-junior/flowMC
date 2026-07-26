# FAQ

**My local sampler is not accepting**

This usually means you are setting the step size of the local sampler to be too big.
Try reducing the step size in your local sampler.

Alternative, this could also mean your sampler is proposing in a region where the target log-PDF is ill-defined (i.e. NaN either in `logpdf` or its derivative if you are using a gradient-based local sampler).
It is worth making sure your `logpdf` is well-defined across the entire region you are sampling.

**In order for my local sampler to accept, I have to choose a very small step size, which makes my chain very correlated.**

This usually indicate some of your parameters are much better measured than others.
Since taking a small step in those directions will already change your `logpdf` value by a lot, the exploration power of the local sampler in other parameters are limited by those which are well measured.
Currently, we support different step size for different parameters, which you can tune to see whether that improves the situation or not.
If you know the scale of each parameter ahead of time, reparameterizing them to maintain roughly equal scale across parameters also helps.

**My global sample's loss is exploding/not decreasing**

This usually means your learning rate used for training the normalizing flow is too large.
Try reducing the learning rate by a factor of ten.

Another reason for a flat loss is your local sampler is not accepting at all.
This is a bit rarer since this means your data used to train the normalizing flow is just your prior, which the normalizing flow should still be able to learn.

**The sampler is stuck a bit until it starts sampling**

If you use the option ``jit`` in constructing the local sampler, the code will compile your code to speed up the execution.
The sampler is not really stuck, but it is compiling the code.
Depending on how you code up your `logpdf` function, the compilation can take a while.
If you don't want to wait, you can set ``jit=False``, which would increase the sampling time.

**The compilation is slow**

If you have a `logpdf` with many lines, JAX will take a long time to compile the code.
JAX is known to be slow in compilation, especially if your computational graph uses some sort of loop that call a function many times.
While we cannot fundamentally get rid of the problem, [using a jax.lax.scan](https://docs.kidger.site/equinox/tricks/#low-overhead-training-loops) is usually how we deal with it.

**I am running out of GPU memory**

flowMC has two independent memory bottlenecks, so there are two knobs to turn.

The first — and usually the bottleneck — is the NF proposal step.
During each global step the target log-PDF (`logpdf`) is evaluated at all `n_chains * n_global_steps` flow proposals at once.
Reduce `n_NFproposal_batch_size` (default `10000`): when `n_global_steps` exceeds it, the proposals are evaluated in smaller `jax.lax.map` chunks rather than a single `vmap`, lowering peak memory.
Try this first.

The second matters when your `logpdf` is expensive to evaluate.
Sometimes the problem is not the `n_chains * n_global_steps` proposals, but that even a single `logpdf` evaluation per chain across all `n_chains` chains will not fit.
In that case also set `chain_batch_size` to a small positive integer (default `0` means all chains are vmapped at once; smaller values use less memory) so the local sampler processes chains in sequential sub-batches.

Both are bundle arguments (e.g. to `RQSpline_MALA_Bundle`); see the [hyperparameters guide](guides/hyperparameters.md) for details.
