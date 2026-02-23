from flowMC.resource.base import Resource
from jaxtyping import Array, Float, Key


class SequentialMonteCarlo(Resource):
    def __init__(self):
        raise NotImplementedError

    def __call__(
        self,
        rng_key: Key,
        resources: dict[str, Resource],
        initial_position: Float[Array, "n_chains n_dim"],
        data: dict,
    ) -> tuple[
        Key,
        dict[str, Resource],
        Float[Array, "n_chains n_dim"],
    ]:
        raise NotImplementedError
