import logging
from typing import Self

import numpy as np

from flowMC.resource.base import Resource

logger = logging.getLogger(__name__)


class State(Resource):
    """A Resource class that holds the state of the system.
    This is essentially a wrapper around a dictionary such that it can be
    handled by flowMC.

    We restrict the type of the state to be simple types including integers, booleans and strings.
    The main reason for this is State is expected to be used to indiciate stage of individual
    strategies instead of storing parameters to resources.
    I.e. State should hold whehter the sampler is in training phase or production phase.
    But not mass matrix of a kernel per se.
    """

    name: str
    data: dict[str, int | bool | str]

    def __repr__(self):
        return "State " + self.name + " with shape " + str(len(self.data))

    def __init__(self, data: dict[str, int | bool | str], name: str = "State"):
        """Initialize the state.

        Args:
            data (dict): The data to initialize the state with.
            name (str): The name of the state.
        """

        self.name = name
        self.data = data

    def update(self, key: list[str], value: list[int | bool | str]):
        """Update the state with new data.

        This will modify the state in place.

        Args:
            key (str): The key to update.
            value (int | bool | str): The value to update.
        """
        for k, v in zip(key, value):
            self.data[k] = v
            logger.debug(f"Updated state {k} to {v}")

    def print_parameters(self):
        logger.debug(
            f"State: {self.name} with shape {len(self.data)} and data {self.data}"
        )

    def save_resource(self, path: str):
        np.savez(
            path + self.name,
            name=self.name,
            data=np.asarray(self.data, dtype=object),
        )

    def load_resource(self, path: str) -> Self:
        with np.load(path, allow_pickle=True) as loaded:
            name = str(loaded["name"])
            raw_data = loaded["data"].item()

        if not isinstance(raw_data, dict):
            raise TypeError("Saved state data must be a dictionary")

        state_data: dict[str, int | bool | str] = {}
        for key, value in raw_data.items():
            if not isinstance(key, str) or not isinstance(value, (int, bool, str)):
                raise TypeError("Saved state data contains an unsupported key or value")
            state_data[key] = value

        return type(self)(state_data, name=name)
