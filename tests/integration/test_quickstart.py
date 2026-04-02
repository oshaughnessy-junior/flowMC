import jax
import jax.numpy as jnp
from flowMC.Sampler import Sampler
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle

# Defining the log posterior

def log_posterior(x, data: dict):
    return -0.5 * jnp.sum((x - data["data"]) ** 2)

def test_quickstart():
    # Initializing the strategy bundle
    n_dims = 2
    n_chains = 10
    key_0, key_1, key_2 = jax.random.split(jax.random.key(42), 3)

    bundle = RQSpline_MALA_Bundle(
        rng_key=key_0,
        n_chains=n_chains,
        n_dims=n_dims,
        logpdf=log_posterior,
        n_local_steps=10,
        n_global_steps=10,
        n_training_loops=3,
        n_production_loops=3,
        n_epochs=10,
        rq_spline_hidden_units=[64, 64],
        rq_spline_n_bins=8,
        rq_spline_n_layers=3,
    )

    # Run the sampler

    initial_position = jax.random.normal(key_1, shape=(n_chains, n_dims)) * 1
    sampler = Sampler(
        n_dims,
        n_chains,
        key_2,
        resource_strategy_bundles=bundle,
    )

    sampler.sample(initial_position, {"data": jnp.arange(n_dims).astype(jnp.float32)})
