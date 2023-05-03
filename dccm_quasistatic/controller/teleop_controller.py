from typing import Dict, Any
from dataclasses import dataclass
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    InverseDynamicsController,
    Multiplexer,
    StateInterpolatorWithDiscreteDerivative,
    System,
)
from underactuated.meshcat_utils import MeshcatSliders

from .controller_base import ControllerBase
from gcs_planar_pushing.utils import get_parser

@dataclass
class TeleopControllerParams:
    time_step: float
    pid_gains: Dict[str, float]
    teleop: Dict[str, Any]
    robot_urdf_path: str

class TeleopController(ControllerBase):
    """An open-loop teleop controller."""

    def __init__(
        self,
        params: TeleopControllerParams,
        meshcat=None,
    ):
        self._params = params
        self._meshcat = meshcat
        self.dim_robot_positions = 2

    def _setup_robot_controller(
        self, builder: DiagramBuilder, plant: MultibodyPlant
    ) -> System:
        robot_model_instance = plant.GetModelInstanceByName("robot")
        robot_controller_plant = MultibodyPlant(time_step=self._params.time_step)
        parser = get_parser(robot_controller_plant)
        parser.AddModelsFromUrl(
            self._params.robot_urdf_path
        )[0]
        robot_controller_plant.set_name("robot_controller_plant")
        robot_controller_plant.Finalize()

        robot_controller = builder.AddSystem(
            InverseDynamicsController(
                robot_controller_plant,
                kp=[self._params.pid_gains.kp] * self.dim_robot_positions,
                ki=[self._params.pid_gains.ki] * self.dim_robot_positions,
                kd=[self._params.pid_gains.kd] * self.dim_robot_positions,
                has_reference_acceleration=False,
            )
        )
        robot_controller.set_name("robot_controller")
        builder.Connect(
            plant.get_state_output_port(robot_model_instance),
            robot_controller.get_input_port_estimated_state(),
        )
        builder.Connect(
            robot_controller.get_output_port_control(),
            plant.get_actuation_input_port(robot_model_instance),
        )
        return robot_controller

    def _setup_sphere_teleop(self, builder: DiagramBuilder) -> System:
        self._sim_duration = 5.0

        input_limit = self._params.teleop.input_limit
        step = self._params.teleop.step_size
        robot_starting_translation = self._params.teleop.start_translation
        self._meshcat.AddSlider(
            "x",
            min=-input_limit,
            max=input_limit,
            step=step,
            value=robot_starting_translation[0],
        )
        self._meshcat.AddSlider(
            "y",
            min=-input_limit,
            max=input_limit,
            step=step,
            value=robot_starting_translation[1],
        )
        force_system = builder.AddSystem(MeshcatSliders(self._meshcat, ["x", "y"]))
        mux = builder.AddNamedSystem("teleop_mux", Multiplexer(2))
        builder.Connect(force_system.get_output_port(0), mux.get_input_port(0))
        builder.Connect(force_system.get_output_port(1), mux.get_input_port(1))

        # Add discrete derivative to command velocities.
        desired_state_source = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                self.dim_robot_positions,
                self._params.time_step,
                suppress_initial_transient=True,
            )
        )
        desired_state_source.set_name("teleop_desired_state_source")
        builder.Connect(mux.get_output_port(), desired_state_source.get_input_port())

        self.desired_pos_source = mux

        return desired_state_source

    def setup(self, builder: DiagramBuilder, plant: MultibodyPlant, **kwargs) -> None:
        self._meshcat = kwargs.get("meshcat", None)
        if self._meshcat is None:
            raise RuntimeError(
                "Need to pass meshcat to teleop controller."
            )

        robot_controller = self._setup_robot_controller(builder, plant)
        teleop_state_source = self._setup_sphere_teleop(builder)
        builder.Connect(
            teleop_state_source.get_output_port(),
            robot_controller.get_input_port_desired_state(),
        )
