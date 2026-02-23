from abc import ABC, abstractmethod

from jaxtyping import Array, Float, Key

from flowMC.resource.base import Resource


class Strategy(ABC):
    """Base class for strategies, which are basically wrapper blocks that modify the
    state of the sampler.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
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
