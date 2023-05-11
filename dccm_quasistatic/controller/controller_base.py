from abc import ABC, abstractmethod
from typing import List
import numpy as np

from pydrake.all import DiagramBuilder, MultibodyPlant, Meshcat


class ControllerBase(ABC):
    """The controller base class."""

    @abstractmethod
    def control_law(self, xk: np.array, xd: np.array, ud: np.array, t: float = 0):
        """Calculate the control law"""
        raise NotImplementedError
    