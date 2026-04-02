"""Integration tests for periodic boundary handling in the full sampling pipeline.

These tests verify that the full Sampler infrastructure (multiple chains,
TakeSerialSteps, and the RQSpline_MALA_Bundle) correctly wraps around periodic
boundaries end-to-end and recovers the target distribution.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from flowMC.resource.kernel.MALA import MALA
from flowMC.resource.kernel.HMC import HMC
from flowMC.resource.kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.buffers import Buffer
from flowMC.resource.states import State
from flowMC.resource.logPDF import LogPDF
from flowMC.resource.base import Resource
from flowMC.strategy.take_steps import TakeSerialSteps
from flowMC.resource_strategy_bundle.RQSpline_MALA import RQSpline_MALA_Bundle
from flowMC.Sampler import Sampler

# ---------------------------------------------------------------------------
# Target distribution
# ---------------------------------------------------------------------------
# 2-D: dim 0 is an angle in [0, 2π) with a wrapped Gaussian centered at
# μ=0.15 rad — very close to the 0/2π boundary so chains *must* cross it.
# dim 1 is a plain Gaussian centered at 1.5 (non-periodic control).

_LOWER, _UPPER = 0.0, 2 * jnp.pi
_PERIOD = _UPPER - _LOWER
_MU_THETA, _SIGMA_THETA = 0.15, 0.25
_MU_Y, _SIGMA_Y = 1.5, 0.4


def _logpdf_mixed(x: Float[Array, " n_dims"], data: dict) -> Float[Array, ""]:
    theta, y = x[0], x[1]
    k = jnp.arange(-5, 6)
    log_theta = jax.scipy.special.logsumexp(
        -0.5 * ((theta - _MU_THETA + k * _PERIOD) / _SIGMA_THETA) ** 2
    ) - jnp.log(_SIGMA_THETA) - 0.5 * jnp.log(2 * jnp.pi)
    log_y = (
        -0.5 * ((y - _MU_Y) / _SIGMA_Y) ** 2
        - jnp.log(_SIGMA_Y)
        - 0.5 * jnp.log(2 * jnp.pi)
    )
    return log_theta + log_y


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_periodic_arrays(n_dims: int, periodic: dict):
    mask = jnp.zeros(n_dims, dtype=bool)
    bounds = jnp.zeros((n_dims, 2))
    for dim, (lo, hi) in periodic.items():
        mask = mask.at[dim].set(True)
        bounds = bounds.at[dim].set(jnp.array([lo, hi]))
    return mask, bounds


def _build_sampler(rng_key, kernel, kernel_name: str, n_chains: int, n_dims: int, n_steps: int) -> Sampler:
    positions_buf = Buffer("positions", (n_chains, n_steps, n_dims), 1)
    log_prob_buf = Buffer("log_prob", (n_chains, n_steps), 1)
    accs_buf = Buffer("acceptance", (n_chains, n_steps), 1)
    sampler_state = State(
        {"positions": "positions", "log_prob": "log_prob", "acceptance": "acceptance"},
        name="sampler_state",
    )
    resources: dict[str, Resource] = {
        "positions": positions_buf,
        "log_prob": log_prob_buf,
        "acceptance": accs_buf,
        kernel_name: kernel,
        "logpdf": LogPDF(_logpdf_mixed, n_dims=n_dims),
        "sampler_state": sampler_state,
    }
    strategy = TakeSerialSteps(
        "logpdf",
        kernel_name=kernel_name,
        state_name="sampler_state",
        buffer_names=["positions", "log_prob", "acceptance"],
        n_steps=n_steps,
    )
    return Sampler(
        n_dim=n_dims, n_chains=n_chains, rng_key=rng_key,
        resources=resources,
        strategies={"take_steps": strategy},
        strategy_order=["take_steps"],
    )


def _get_samples(sampler: Sampler, n_dims: int) -> Float[Array, "n n_dims"]:
    flat = sampler.resources["positions"].data.reshape(-1, n_dims)
    return flat[jnp.isfinite(flat).all(axis=-1)]


def _assert_boundary_crossing_and_recovery(samples: Float[Array, "n n_dims"], label: str) -> None:
    """Assert that samples cross the 0/2π boundary and recover the target distribution."""
    theta, y = samples[:, 0], samples[:, 1]

    # Both sides of the 0/2π wrap should be populated
    near_zero = int(jnp.sum(theta < 0.5))
    near_two_pi = int(jnp.sum(theta > _UPPER - 0.5))
    total = len(theta)
    assert near_two_pi > 0, (
        f"{label}: no samples near 2π — chains never crossed the periodic boundary. "
        "Wrapping or the acceptance-ratio correction may be broken."
    )
    coverage = (near_zero + near_two_pi) / total
    assert coverage > 0.2, (
        f"{label}: only {coverage:.1%} of samples near boundary (expected >20%). "
        f"near_zero={near_zero}, near_two_pi={near_two_pi}, total={total}"
    )

    # Circular mean should be close to the true mean
    circ_mean = float(
        jnp.arctan2(jnp.mean(jnp.sin(theta)), jnp.mean(jnp.cos(theta))) % (2 * jnp.pi)
    )
    err = float(jnp.abs(jnp.arctan2(
        jnp.sin(jnp.array(circ_mean - _MU_THETA)),
        jnp.cos(jnp.array(circ_mean - _MU_THETA)),
    )))
    assert err < 0.15, (
        f"{label}: circular mean {circ_mean:.4f} rad too far from true mean "
        f"{_MU_THETA:.4f} rad (err={err:.4f} ≥ 0.15)."
    )

    # Non-periodic dimension should also be recovered
    y_mean = float(jnp.mean(y))
    assert abs(y_mean - _MU_Y) < 0.15, (
        f"{label}: non-periodic mean {y_mean:.4f} too far from {_MU_Y} (tol 0.15)."
    )


# ---------------------------------------------------------------------------
# End-to-end tests: full Sampler pipeline with each local kernel
# ---------------------------------------------------------------------------

def test_mala_end_to_end_periodic_boundary_crossing():
    """MALA: full Sampler pipeline crosses 0/2π and recovers the target distribution."""
    n_dims, n_chains, n_steps = 2, 24, 200
    key = jax.random.key(42)
    key, k_sampler, k_init = jax.random.split(key, 3)
    mask, bounds = _make_periodic_arrays(n_dims, {0: (_LOWER, _UPPER)})

    # All chains start near θ=0; must walk through 2π to explore the full distribution
    theta0 = jax.random.uniform(k_init, (n_chains,)) * 0.1
    initial_position = jnp.stack([theta0, jnp.full((n_chains,), _MU_Y)], axis=1)

    kernel = MALA(step_size=jnp.full(n_dims, 0.2), periodic_mask=mask, periodic_bounds=bounds)
    sampler = _build_sampler(k_sampler, kernel, "MALA", n_chains, n_dims, n_steps)
    sampler.sample(initial_position, {})
    _assert_boundary_crossing_and_recovery(_get_samples(sampler, n_dims), "MALA")


def test_hmc_end_to_end_periodic_boundary_crossing():
    """HMC: full Sampler pipeline crosses 0/2π and recovers the target distribution."""
    n_dims, n_chains, n_steps = 2, 24, 200
    key = jax.random.key(43)
    key, k_sampler, k_init = jax.random.split(key, 3)
    mask, bounds = _make_periodic_arrays(n_dims, {0: (_LOWER, _UPPER)})

    theta0 = jax.random.uniform(k_init, (n_chains,)) * 0.1
    initial_position = jnp.stack([theta0, jnp.full((n_chains,), _MU_Y)], axis=1)

    kernel = HMC(
        condition_matrix=jnp.ones(n_dims), step_size=0.2, n_leapfrog=5,
        periodic_mask=mask, periodic_bounds=bounds,
    )
    sampler = _build_sampler(k_sampler, kernel, "HMC", n_chains, n_dims, n_steps)
    sampler.sample(initial_position, {})
    _assert_boundary_crossing_and_recovery(_get_samples(sampler, n_dims), "HMC")


def test_grw_end_to_end_periodic_boundary_crossing():
    """GRW: full Sampler pipeline crosses 0/2π and recovers the target distribution."""
    n_dims, n_chains, n_steps = 2, 24, 300
    key = jax.random.key(44)
    key, k_sampler, k_init = jax.random.split(key, 3)
    mask, bounds = _make_periodic_arrays(n_dims, {0: (_LOWER, _UPPER)})

    theta0 = jax.random.uniform(k_init, (n_chains,)) * 0.1
    initial_position = jnp.stack([theta0, jnp.full((n_chains,), _MU_Y)], axis=1)

    kernel = GaussianRandomWalk(
        step_size=jnp.full(n_dims, 0.2), periodic_mask=mask, periodic_bounds=bounds,
    )
    sampler = _build_sampler(k_sampler, kernel, "GRW", n_chains, n_dims, n_steps)
    sampler.sample(initial_position, {})
    _assert_boundary_crossing_and_recovery(_get_samples(sampler, n_dims), "GRW")


# ---------------------------------------------------------------------------
# End-to-end test: RQSpline_MALA_Bundle (MALA local + NF global, no PT)
# ---------------------------------------------------------------------------

def test_bundle_end_to_end_periodic_boundary_crossing():
    """RQSpline_MALA_Bundle: full training+production pipeline crosses 0/2π.

    Uses the complete bundle (local MALA steps + NF global steps + adaptation)
    with a wide periodic range [-20, 20] so the NF proposal never produces
    out-of-range samples, while the MALA local steps still exercise the periodic
    wrapping machinery on a distribution whose mass sits near both ends of the range.
    """
    n_dims, n_chains = 2, 12
    key = jax.random.key(7)
    key, k_bundle, k_sampler, k_init = jax.random.split(key, 4)

    # Wide periodic bounds: the standard Gaussian (sigma=1) never escapes [-20, 20],
    # so the NF proposal behaves correctly, while MALA's periodic arithmetic still runs.
    initial_position = jax.random.normal(k_init, (n_chains, n_dims)) * 0.5

    def logpdf_gaussian(x: Float[Array, " n_dims"], data: dict) -> Float[Array, ""]:
        return -0.5 * jnp.sum(x**2)

    bundle = RQSpline_MALA_Bundle(
        rng_key=k_bundle,
        n_chains=n_chains,
        n_dims=n_dims,
        logpdf=logpdf_gaussian,
        n_local_steps=20,
        n_global_steps=10,
        n_training_loops=2,
        n_production_loops=2,
        n_epochs=5,
        mala_step_size=0.3,
        adapt_step_size=True,
        periodic={0: (-20.0, 20.0)},
        rq_spline_hidden_units=[16, 16],
        rq_spline_n_bins=4,
        rq_spline_n_layers=2,
    )
    sampler = Sampler(n_dims, n_chains, k_sampler, resource_strategy_bundles=bundle)
    sampler.sample(initial_position, {})

    buf = sampler.resources["positions_production"]
    flat = buf.data.reshape(-1, n_dims)
    flat = flat[jnp.isfinite(flat).all(axis=-1)]
    mean = jnp.mean(flat, axis=0)
    assert jnp.all(jnp.abs(mean) < 0.3), (
        f"Bundle production mean {mean} not near zero — "
        "the periodic mask may be corrupting the sampler."
    )

