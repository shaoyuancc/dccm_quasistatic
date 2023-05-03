from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DCCMParams:
    """
    Parameters for the DCCM controller.
    """
    # Sim  and base position controller parameters
    time_step: float
    pid_gains: Dict[str, float]
    robot_urdf_path: str

    # System parameters
    dim_x: int # Dimension of the state
    dim_u: int # Dimension of the input

    # DCCM Params
    deg: int = 6 # Degree of the polynomial
    beta: float = 0.1 # Convergence rate = 1-beta
    
    # Geodesic calculation parameters
    n_geodesic_segments: int = 5 # Number of segments to discretize the geodesic into