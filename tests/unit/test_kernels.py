import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree

from flowMC.resource.kernel.Gaussian_random_walk import GaussianRandomWalk
from flowMC.resource.kernel.HMC import HMC
from flowMC.resource.kernel.MALA import MALA
from flowMC.resource.states import State
from flowMC.strategy.take_steps import TakeSerialSteps
from flowMC.resource.buffers import Buffer
from flowMC.resource.logPDF import LogPDF


def log_posterior(x, data=None):
    return -0.5 * jnp.sum(x**2)


n_dims = 2
logpdf = LogPDF(log_posterior, n_dims=2)


class TestHMC:

    def test_repr(self):
        HMC_obj = HMC(condition_matrix=jnp.ones(n_dims), step_size=1, n_leapfrog=5)
        assert repr(HMC_obj) == "HMC with step size 1 and 5 leapfrog steps"

        # Periodic mask/bounds default to zeros when not provided
        assert HMC_obj.periodic_mask.shape == (n_dims,)
        assert not jnp.any(HMC_obj.periodic_mask)
        assert HMC_obj.periodic_bounds.shape == (n_dims, 2)

    def test_print_params(self, caplog):
        import logging
        caplog.set_level(logging.DEBUG)
        HMC_obj = HMC(condition_matrix=jnp.ones(n_dims), step_size=1, n_leapfrog=5)
        HMC_obj.print_parameters()
        assert "HMC parameters:" in caplog.text
        assert "step_size:" in caplog.text
        assert "n_leapfrog:" in caplog.text

    def test_HMC_deterministic(self):
        n_chains = 1
        HMC_obj = HMC(
            condition_matrix=jnp.ones(n_dims),
            step_size=1,
            n_leapfrog=5,
        )

        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_PE = jax.vmap(lambda x, data: -log_posterior(x, data))(
            initial_position, None
        )

        # Test whether the HMC kernel is deterministic

        rng_key, subkey = jax.random.split(rng_key)
        result1 = HMC_obj.kernel(subkey, initial_position[0], initial_PE, logpdf, None)
        result2 = HMC_obj.kernel(subkey, initial_position[0], initial_PE, logpdf, None)

        assert jnp.allclose(result1[0], result2[0])
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]

    def test_leapfrog_reversible(self):
        # Test whether the leapfrog kernel is reversible
        n_chains = 1
        HMC_obj = HMC(
            condition_matrix=jnp.ones(n_dims),
            step_size=1,
            n_leapfrog=5,
        )

        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)
        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_PE = jax.vmap(lambda x, data: -log_posterior(x, data))(
            initial_position, None
        )
        rng_key, subkey = jax.random.split(rng_key)
        key1, key2 = jax.random.split(subkey)

        def potential(x: Float[Array, " n_dims"], data: PyTree) -> Float[Array, "1"]:
            return -log_posterior(x, data)

        def kinetic(
            p: Float[Array, " n_dims"], metric: Float[Array, " n_dims"]
        ) -> Float[Array, "1"]:
            return 0.5 * (p**2 * metric).sum()

        leapfrog_kernel = jax.tree_util.Partial(
            HMC_obj.leapfrog_kernel, kinetic, potential
        )

        initial_momentum = (
            jax.random.normal(key1, shape=initial_position.shape)
            * jnp.ones(n_dims) ** -0.5
        )
        new_position, new_momentum = HMC_obj.leapfrog_step(
            leapfrog_kernel,
            initial_position,
            initial_momentum,
            None,
            jnp.ones(n_dims),
        )
        rev_position, rev_momentum = HMC_obj.leapfrog_step(
            leapfrog_kernel, new_position, -new_momentum, None, jnp.ones(n_dims)
        )

        assert jnp.allclose(rev_position, initial_position)
        assert jnp.allclose(initial_PE, -log_posterior(rev_position, None))

    def test_HMC_acceptance_rate(self):
        # Test acceptance rate goes to one when step size is small

        HMC_obj = HMC(
            condition_matrix=jnp.ones(n_dims),
            step_size=0.0000001,
            n_leapfrog=5,
        )

        n_chains = 100
        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_logp = jax.vmap(log_posterior)(initial_position, None)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(HMC_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )

        assert result[2].all()

    def test_HMC_close_gaussian(self):
        n_chains = 1
        n_local_steps = 30000
        HMC_obj = HMC(
            condition_matrix=jnp.ones(n_dims),
            step_size=0.1,
            n_leapfrog=5,
        )
        positions = Buffer("positions", (n_chains, n_local_steps, n_dims), 1)
        log_prob = Buffer("log_prob", (n_chains, n_local_steps), 1)
        acceptance = Buffer("acceptance", (n_chains, n_local_steps), 1)
        sampler_state = State(
            {
                "positions": "positions",
                "log_prob": "log_prob",
                "acceptance": "acceptance",
            },
            name="sampler_state",
        )
        resource = {
            "positions": positions,
            "log_prob": log_prob,
            "acceptance": acceptance,
            "HMC": HMC_obj,
            "logpdf": logpdf,
            "sampler_state": sampler_state,
        }

        # Defining strategy

        strategy = TakeSerialSteps(
            "logpdf",
            kernel_name="HMC",
            state_name="sampler_state",
            buffer_names=["positions", "log_prob", "acceptance"],
            n_steps=n_local_steps,
        )

        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

        rng_key, subkey = jax.random.split(rng_key)
        result = strategy(subkey, resource, initial_position, {})[1]["positions"]

        assert isinstance(result, Buffer)

        assert jnp.isclose(jnp.mean(result.data), 0, atol=3e-2)
        assert jnp.isclose(jnp.var(result.data), 1, atol=3e-2)

    def test_HMC_adapt_step_size(self):
        # Test that adapt_step_size increases step size when acceptance is high
        HMC_obj = HMC(condition_matrix=jnp.ones(n_dims), step_size=0.1, n_leapfrog=5)
        
        # High acceptance rate should increase step size
        adapted_high = HMC_obj.adapt_step_size(acceptance_rate=0.85, target_rate=0.65)
        assert adapted_high.step_size > HMC_obj.step_size
        
        # Low acceptance rate should decrease step size
        adapted_low = HMC_obj.adapt_step_size(acceptance_rate=0.3, target_rate=0.65)
        assert adapted_low.step_size < HMC_obj.step_size
        
        # Acceptance at target should keep step size approximately same
        adapted_target = HMC_obj.adapt_step_size(acceptance_rate=0.65, target_rate=0.65)
        assert jnp.isclose(adapted_target.step_size, HMC_obj.step_size, atol=1e-6)

    def test_HMC_periodic_samples_in_bounds(self):
        """HMC with periodic boundaries keeps samples within bounds."""
        periodic_mask = jnp.array([True, False])
        periodic_bounds = jnp.array([[0.0, 2 * jnp.pi], [0.0, 0.0]])

        HMC_obj = HMC(
            condition_matrix=jnp.ones(n_dims),
            step_size=0.3,
            n_leapfrog=5,
            periodic_mask=periodic_mask,
            periodic_bounds=periodic_bounds,
        )

        n_steps = 100


        @jax.jit
        def run_chain(rng_key, init_pos, init_lp):
            def step(carry, key):
                pos, lp = carry
                new_pos, new_lp, _acc = HMC_obj.kernel(key, pos, lp, logpdf, None)
                return (new_pos, new_lp), new_pos

            keys = jax.random.split(rng_key, n_steps)
            _, positions = jax.lax.scan(step, (init_pos, init_lp), keys)
            return positions

        init_pos = jnp.array([3.0, 0.5])
        init_lp = log_posterior(init_pos)
        positions = run_chain(jax.random.PRNGKey(0), init_pos, init_lp)

        # Periodic dimension must stay within [0, 2pi)
        assert jnp.all(positions[:, 0] >= 0.0)
        assert jnp.all(positions[:, 0] < 2 * jnp.pi + 1e-6)

    def test_HMC_periodic_acceptance_rate(self):
        """Acceptance rate still goes to 1 with tiny step size and periodic."""
        periodic_mask = jnp.array([True, True])
        periodic_bounds = jnp.array([[0.0, 2 * jnp.pi], [0.0, 2 * jnp.pi]])

        HMC_obj = HMC(
            condition_matrix=jnp.ones(n_dims),
            step_size=0.0000001,
            n_leapfrog=5,
            periodic_mask=periodic_mask,
            periodic_bounds=periodic_bounds,
        )

        n_chains = 20
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)
        initial_position = jax.random.uniform(subkey, shape=(n_chains, n_dims)) * 2 * jnp.pi
        initial_logp = jax.vmap(log_posterior)(initial_position, None)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(HMC_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )
        assert result[2].all()


class TestMALA:

    def test_repr(self):
        MALA_obj = MALA(step_size=jnp.ones(n_dims))
        assert repr(MALA_obj) == "MALA with step size " + str(jnp.ones(n_dims))

        # Periodic mask/bounds default to zeros when not provided
        assert MALA_obj.periodic_mask.shape == (n_dims,)
        assert not jnp.any(MALA_obj.periodic_mask)
        assert MALA_obj.periodic_bounds.shape == (n_dims, 2)

    def test_print_params(self, caplog):
        import logging
        caplog.set_level(logging.DEBUG)
        MALA_obj = MALA(step_size=jnp.ones(n_dims))
        MALA_obj.print_parameters()
        assert "MALA parameters:" in caplog.text
        assert "step_size:" in caplog.text

    def test_MALA_deterministic(self):
        n_chains = 1
        MALA_obj = MALA(step_size=jnp.ones(n_dims))

        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_logp = log_posterior(initial_position, None)

        rng_key, subkey = jax.random.split(rng_key)
        result1 = MALA_obj.kernel(
            subkey, initial_position[0], initial_logp, logpdf, None
        )
        result2 = MALA_obj.kernel(
            subkey, initial_position[0], initial_logp, logpdf, None
        )

        assert jnp.allclose(result1[0], result2[0])
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]

    def test_MALA_acceptance_rate(self):
        # Test acceptance rate goes to one when the step size is small

        MALA_obj = MALA(step_size=jnp.full(n_dims, 0.00001))

        n_chains = 100
        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_logp = jax.vmap(log_posterior)(initial_position, None)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(MALA_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )

        assert result[2].all()

    def test_MALA_close_gaussian(self):
        n_dims = 2
        n_chains = 1
        n_local_steps = 50000
        MALA_Sampler = MALA(step_size=jnp.ones(n_dims))
        positions = Buffer("positions", (n_chains, n_local_steps, n_dims), 1)
        log_prob = Buffer("log_prob", (n_chains, n_local_steps), 1)
        acceptance = Buffer("acceptance", (n_chains, n_local_steps), 1)
        sampler_state = State(
            {
                "positions": "positions",
                "log_prob": "log_prob",
                "acceptance": "acceptance",
            },
            name="sampler_state",
        )

        resource = {
            "positions": positions,
            "log_prob": log_prob,
            "acceptance": acceptance,
            "MALA": MALA_Sampler,
            "logpdf": logpdf,
            "sampler_state": sampler_state,
        }

        # Defining strategy

        strategy = TakeSerialSteps(
            "logpdf",
            kernel_name="MALA",
            state_name="sampler_state",
            buffer_names=["positions", "log_prob", "acceptance"],
            n_steps=n_local_steps,
        )

        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

        rng_key, subkey = jax.random.split(rng_key)
        result = strategy(subkey, resource, initial_position, {})[1]["positions"]

        assert isinstance(result, Buffer)

        assert jnp.isclose(jnp.mean(result.data), 0, atol=3e-2)
        assert jnp.isclose(jnp.var(result.data), 1, atol=3e-2)


    def test_MALA_adapt_step_size(self):
        # Test that adapt_step_size increases step size when acceptance is high
        MALA_obj = MALA(step_size=jnp.full(n_dims, 0.1))
        
        # High acceptance rate should increase step size
        adapted_high = MALA_obj.adapt_step_size(acceptance_rate=0.8, target_rate=0.574)
        assert jnp.all(adapted_high.step_size > MALA_obj.step_size)
        
        # Low acceptance rate should decrease step size
        adapted_low = MALA_obj.adapt_step_size(acceptance_rate=0.2, target_rate=0.574)
        assert jnp.all(adapted_low.step_size < MALA_obj.step_size)
        
        # Acceptance at target should keep step size approximately same
        adapted_target = MALA_obj.adapt_step_size(acceptance_rate=0.574, target_rate=0.574)
        assert jnp.allclose(adapted_target.step_size, MALA_obj.step_size, atol=1e-6)

    def test_MALA_periodic_acceptance_rate(self):
        """Acceptance rate still goes to 1 with tiny step size and periodic."""
        periodic_mask = jnp.array([True, True])
        periodic_bounds = jnp.array([[0.0, 2 * jnp.pi], [0.0, 2 * jnp.pi]])

        MALA_obj = MALA(
            step_size=jnp.full(n_dims, 0.00001),
            periodic_mask=periodic_mask,
            periodic_bounds=periodic_bounds,
        )

        n_chains = 20
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)
        initial_position = jax.random.uniform(subkey, shape=(n_chains, n_dims)) * 2 * jnp.pi
        initial_logp = jax.vmap(log_posterior)(initial_position, None)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(MALA_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )
        assert result[2].all()

    def test_MALA_periodic_explores_full_domain(self):
        """MALA with periodic boundary explores both sides of a wrapped peak.

        Uses log(cos(x) + 1) on [0, 2pi] with a peak at x=0 (and 2pi).
        Starting near x=0.1, the walker should wrap around to explore both
        sides of the peak.
        """

        def periodic_logpdf(x, data):
            return jnp.log(jnp.cos(x[0]) + 1.0 + 1e-8)

        periodic_mala = MALA(
            step_size=jnp.array([0.5]),
            periodic_mask=jnp.array([True]),
            periodic_bounds=jnp.array([[0.0, 2 * jnp.pi]]),
        )

        n_steps = 2000

        @jax.jit
        def run_chain(rng_key, init_pos, init_lp):
            def step(carry, key):
                pos, lp = carry
                new_pos, new_lp, acc = periodic_mala.kernel(
                    key, pos, lp, periodic_logpdf, {}
                )
                return (new_pos, new_lp), new_pos

            keys = jax.random.split(rng_key, n_steps)
            _, positions = jax.lax.scan(step, (init_pos, init_lp), keys)
            return positions

        init_pos = jnp.array([0.1])
        init_lp = periodic_logpdf(init_pos, {})
        positions = run_chain(jax.random.PRNGKey(42), init_pos, init_lp).squeeze()

        # All samples must be within bounds
        assert jnp.all(positions >= 0.0)
        assert jnp.all(positions <= 2 * jnp.pi)

        # Walker should visit both halves of the domain
        in_left = jnp.sum(positions < jnp.pi)
        in_right = jnp.sum(positions >= jnp.pi)
        total = len(positions)
        assert in_right > 0.1 * total, (
            f"Walker did not explore [pi, 2pi]: only {in_right}/{total} samples."
        )
        assert in_left > 0.1 * total, (
            f"Walker did not explore [0, pi): only {in_left}/{total} samples."
        )


class TestGRW:

    def test_repr(self):
        GRW_obj = GaussianRandomWalk(step_size=jnp.ones(n_dims))
        assert repr(GRW_obj) == "Gaussian Random Walk with step size " + str(jnp.ones(n_dims))

        # Periodic mask/bounds default to zeros when not provided
        assert GRW_obj.periodic_mask.shape == (n_dims,)
        assert not jnp.any(GRW_obj.periodic_mask)
        assert GRW_obj.periodic_bounds.shape == (n_dims, 2)

    def test_print_params(self, caplog):
        import logging
        caplog.set_level(logging.DEBUG)
        GRW_obj = GaussianRandomWalk(step_size=jnp.ones(n_dims))
        GRW_obj.print_parameters()
        assert "Gaussian Random Walk parameters:" in caplog.text
        assert "step_size:" in caplog.text

    def test_Gaussian_random_walk_deterministic(self):
        n_chains = 1
        GRW_obj = GaussianRandomWalk(step_size=jnp.ones(n_dims))
        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1
        initial_logp = log_posterior(initial_position)

        rng_key, subkey = jax.random.split(rng_key)
        result1 = GRW_obj.kernel(
            subkey, initial_position[0], initial_logp, logpdf, None
        )
        result2 = GRW_obj.kernel(
            subkey, initial_position[0], initial_logp, logpdf, None
        )

        assert jnp.allclose(result1[0], result2[0])
        assert result1[1] == result2[1]
        assert result1[2] == result2[2]

    def test_Gaussian_random_walk_acceptance_rate(self):
        # Test acceptance rate goes to one when the step size is small

        n_dim = 2
        GRW_obj = GaussianRandomWalk(step_size=jnp.full(n_dim, 0.00001))

        n_chains = 100
        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dim)) * 1
        initial_logp = jax.vmap(log_posterior)(initial_position)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(GRW_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )
        assert result[2].all()

    def test_Gaussian_random_walk_close_gaussian(self):
        n_chains = 1
        n_local_steps = 50000
        GRW_Sampler = GaussianRandomWalk(step_size=jnp.ones(n_dims))

        positions = Buffer("positions", (n_chains, n_local_steps, n_dims), 1)
        log_prob = Buffer("log_prob", (n_chains, n_local_steps), 1)
        acceptance = Buffer("acceptance", (n_chains, n_local_steps), 1)

        sampler_state = State(
            {
                "positions": "positions",
                "log_prob": "log_prob",
                "acceptance": "acceptance",
            },
            name="sampler_state",
        )

        resource = {
            "positions": positions,
            "log_prob": log_prob,
            "acceptance": acceptance,
            "GRW": GRW_Sampler,
            "logpdf": logpdf,
            "sampler_state": sampler_state,
        }

        # Defining strategy

        strategy = TakeSerialSteps(
            "logpdf",
            kernel_name="GRW",
            state_name="sampler_state",
            buffer_names=["positions", "log_prob", "acceptance"],
            n_steps=n_local_steps,
        )

        rng_key = jax.random.key(42)
        rng_key, subkey = jax.random.split(rng_key)

        initial_position = jax.random.normal(subkey, shape=(n_chains, n_dims)) * 1

        rng_key, subkey = jax.random.split(rng_key)
        result = strategy(subkey, resource, initial_position, {})[1]["positions"]

        assert isinstance(result, Buffer)

        assert jnp.isclose(jnp.mean(result.data), 0, atol=3e-2)
        assert jnp.isclose(jnp.var(result.data), 1, atol=3e-2)

    def test_GRW_adapt_step_size(self):
        # Test that adapt_step_size increases step size when acceptance is high
        GRW_obj = GaussianRandomWalk(step_size=jnp.full(n_dims, 0.1))
        
        # High acceptance rate should increase step size
        adapted_high = GRW_obj.adapt_step_size(acceptance_rate=0.5, target_rate=0.234)
        assert jnp.all(adapted_high.step_size > GRW_obj.step_size)
        
        # Low acceptance rate should decrease step size
        adapted_low = GRW_obj.adapt_step_size(acceptance_rate=0.1, target_rate=0.234)
        assert jnp.all(adapted_low.step_size < GRW_obj.step_size)
        
        # Acceptance at target should keep step size approximately same
        adapted_target = GRW_obj.adapt_step_size(acceptance_rate=0.234, target_rate=0.234)
        assert jnp.allclose(adapted_target.step_size, GRW_obj.step_size, atol=1e-6)

    def test_GRW_periodic_acceptance_rate(self):
        """Acceptance rate still goes to 1 with tiny step size and periodic."""
        periodic_mask = jnp.array([True, True])
        periodic_bounds = jnp.array([[0.0, 2 * jnp.pi], [0.0, 2 * jnp.pi]])

        GRW_obj = GaussianRandomWalk(
            step_size=jnp.full(n_dims, 0.00001),
            periodic_mask=periodic_mask,
            periodic_bounds=periodic_bounds,
        )

        n_chains = 20
        rng_key = jax.random.PRNGKey(42)
        rng_key, subkey = jax.random.split(rng_key)
        initial_position = jax.random.uniform(subkey, shape=(n_chains, n_dims)) * 2 * jnp.pi
        initial_logp = jax.vmap(log_posterior)(initial_position)

        rng_key, subkey = jax.random.split(rng_key)
        subkey = jax.random.split(subkey, n_chains)
        result = jax.vmap(GRW_obj.kernel, in_axes=(0, 0, 0, None, None))(
            subkey, initial_position, initial_logp, logpdf, None
        )
        assert result[2].all()

    def test_GRW_periodic_explores_full_domain(self):
        """GRW with periodic boundary explores both sides of a wrapped peak."""

        def periodic_logpdf(x, data):
            return jnp.log(jnp.cos(x[0]) + 1.0 + 1e-8)

        periodic_grw = GaussianRandomWalk(
            step_size=jnp.array([1.0]),
            periodic_mask=jnp.array([True]),
            periodic_bounds=jnp.array([[0.0, 2 * jnp.pi]]),
        )

        n_steps = 2000

        @jax.jit
        def run_chain(rng_key, init_pos, init_lp):
            def step(carry, key):
                pos, lp = carry
                new_pos, new_lp, acc = periodic_grw.kernel(
                    key, pos, lp, periodic_logpdf, {}
                )
                return (new_pos, new_lp), new_pos

            keys = jax.random.split(rng_key, n_steps)
            _, positions = jax.lax.scan(step, (init_pos, init_lp), keys)
            return positions

        init_pos = jnp.array([0.1])
        init_lp = periodic_logpdf(init_pos, {})
        positions = run_chain(jax.random.PRNGKey(42), init_pos, init_lp).squeeze()

        assert jnp.all(positions >= 0.0)
        assert jnp.all(positions <= 2 * jnp.pi)

        in_left = jnp.sum(positions < jnp.pi)
        in_right = jnp.sum(positions >= jnp.pi)
        total = len(positions)
        assert in_right > 0.1 * total
        assert in_left > 0.1 * total
