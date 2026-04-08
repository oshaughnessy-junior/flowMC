from abc import ABC, abstractmethod

from typing import Self


class Resource(ABC):
    """Base class for resources. Resources are objects such as local sampler and neural
    networks.

    This is an abstract template that should not be directly used.
    """

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def print_parameters(self) -> None:
        """Print the tunable parameters of the resource to the debug log."""
        raise NotImplementedError

    @abstractmethod
    def save_resource(self, path: str) -> None:
        """Save the resource to disk.

        Args:
            path (str): File path (without extension) to save the resource.
        """
        raise NotImplementedError

    @abstractmethod
    def load_resource(self, path: str) -> Self:
        """Load a resource from disk.

        Args:
            path (str): File path to load the resource from.

        Returns:
            Self: A new instance populated with the loaded data.
        """
        raise NotImplementedError
