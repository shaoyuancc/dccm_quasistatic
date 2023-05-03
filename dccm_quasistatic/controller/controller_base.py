from abc import ABC, abstractmethod
from typing import List

from pydrake.all import DiagramBuilder, MultibodyPlant, Meshcat


class ControllerBase(ABC):
    """The controller base class."""

    @abstractmethod
    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant, **kwargs) -> None:
        """Setup the controller."""
        raise NotImplementedError
    