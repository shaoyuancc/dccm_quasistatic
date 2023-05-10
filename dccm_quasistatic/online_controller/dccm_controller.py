import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from dccm_quasistatic.online_controller.controller_base import ControllerBase
from dccm_quasistatic.online_controller.dccm_params import DCCMParams
from dccm_quasistatic.leaf_systems.dccm_system import DCCMSystem
from dccm_quasistatic.leaf_systems.desired_trajectory_source_system import DesiredTrajectorySourceSystem
from dccm_quasistatic.utils.sim_utils import get_parser

from pydrake.all import (MathematicalProgram, Solve, MonomialBasis,
                         DiagramBuilder, Evaluate, LogVectorOutput, Simulator,
                         SymbolicVectorSystem, Variable, ToLatex, Polynomial,
                         VectorSystem, eq, ge, le, Formula, Expression, Evaluate,
                         System, AbstractValue,
    MultibodyPlant,
    InverseDynamicsController,
    StateInterpolatorWithDiscreteDerivative,
    )


class DCCMController(ControllerBase):
    def __init__(self, params: DCCMParams) -> None:
        self._params = params

    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant, **kwargs) -> None:

        robot_model_instance = plant.GetModelInstanceByName("robot")
        robot_controller_plant = MultibodyPlant(time_step=self._params.time_step)
        parser = get_parser(robot_controller_plant)
        parser.AddModelsFromUrl(
            self._params.robot_urdf_path
        )[0]
        robot_controller_plant.set_name("robot_controller_plant")
        robot_controller_plant.Finalize()

        robot_position_controller = builder.AddSystem(
            InverseDynamicsController(
                robot_controller_plant,
                kp=[self._params.pid_gains.kp] * self._params.dim_u,
                ki=[self._params.pid_gains.ki] * self._params.dim_u,
                kd=[self._params.pid_gains.kd] * self._params.dim_u,
                has_reference_acceleration=False,
            )
        )
        robot_position_controller.set_name("robot_position_controller")
        builder.Connect(
            plant.get_state_output_port(robot_model_instance),
            robot_position_controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            robot_position_controller.get_output_port_control(),
            plant.get_actuation_input_port(robot_model_instance),
        )

        # Add discrete derivative to command velocities.
        desired_state_source = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                self._params.dim_u,
                self._params.time_step,
                suppress_initial_transient=True,
            )
        )
        desired_state_source.set_name("desired_state_source")
        builder.Connect(
            desired_state_source.get_output_port(),
            robot_position_controller.get_input_port_desired_state(),
        )

        # DCCM Controller
        dccm_system = builder.AddSystem(DCCMSystem(self._params))

        # Desired trajectory system
        desired_trajectory = builder.AddSystem(DesiredTrajectorySourceSystem(self._params.dim_x, self._params.dim_u))

        builder.Connect(desired_trajectory.GetOutputPort("x_desired"), dccm_system.GetInputPort("x_desired"))
        builder.Connect(desired_trajectory.GetOutputPort("u_desired"), dccm_system.GetInputPort("u_desired"))

        builder.Connect(plant.get_state_output_port(), dccm_system.GetInputPort("state_actual"))
        builder.Connect(dccm_system.GetOutputPort("u_actual"), desired_state_source.get_input_port(0))

        self._dccm_system = dccm_system

    def calc_dccm(self, x_samples, u_samples, x_next_samples, A_samples, B_samples):
        self._dccm_system.calculate_dccm_from_samples(x_samples, u_samples, x_next_samples, A_samples, B_samples)

