import os
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    MultibodyPlant,
    PlanarJoint,
    RigidTransform,
    RotationMatrix,
    MeshcatVisualizer,
    Simulator,
    MeshcatVisualizerParams,
    Role,
    LogVectorOutput,
    Rgba,
    Box,
    Parser,
)
from manipulation.utils import AddPackagePaths

from dccm_quasistatic.controller.controller_base import ControllerBase

@dataclass
class EnvParams:
    time_step: float
    planar_joint_damping: float
    scene_directive_path: str
    initial_robot_position: List[float]
    initial_object_position: List[float]

class PlanarPushingEnvironment():
    def __init__(self, params: EnvParams, controller: ControllerBase, meshcat=None):
        self._params = params
        self._controller = controller

        if meshcat is None:
            self._meshcat = StartMeshcat()
        else:
            self._meshcat = meshcat
    
    def setup(self) -> None:

        # Setup environment
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(
            builder, time_step=self._params.time_step
        )
        parser = get_parser(plant)
        parser.AddAllModelsFromFile(self._params.scene_directive_path)

        plant.AddJoint(
            PlanarJoint(
                "object_joint",
                plant.world_frame(),
                plant.GetFrameByName("object"),
                damping=[self._params.planar_joint_damping, self._params.planar_joint_damping, self._params.planar_joint_damping],
            )
        )
        plant.Finalize()

        # Setup controller
        self._controller.setup(builder, plant, meshcat=self._meshcat)

        visualizer_params = MeshcatVisualizerParams()
        visualizer_params.role = Role.kIllustration
        self._visualizer = MeshcatVisualizer.AddToBuilder(
            builder,
            scene_graph,
            self._meshcat,
            visualizer_params,
        )

        diagram = builder.Build()

        self._simulator = Simulator(diagram)

        # Set initial object and robot position
        context = self._simulator.get_mutable_context()
        plant_context = plant.GetMyMutableContextFromRoot(context)
        object = plant.GetBodyByName("object")
        object_model_instance = object.model_instance()

        plant.SetPositions(
            plant_context, object_model_instance, self._params.initial_object_position
        )

        robot = plant.GetBodyByName("robot")
        robot_model_instance = robot.model_instance()
        plant.SetPositions(
            plant_context, robot_model_instance, self._params.initial_robot_position
        )
    
    def simulate(self, duration=None) -> None:

        if duration is None:
            print("Press 'Stop Simulation' in MeshCat to continue.")
            self._simulator.set_target_realtime_rate(1.0)
            self._meshcat.AddButton("Stop Simulation")
            while self._meshcat.GetButtonClicks("Stop Simulation") < 1:
                self._simulator.AdvanceTo(self._simulator.get_context().get_time() + 1.0)

            self._meshcat.DeleteAddedControls()
            self._meshcat.Delete()
        
        else:
            self._visualizer.StartRecording()

            print(f"Meshcat URL: {self._meshcat.web_url()}")

            for t in np.arange(0.0, duration, self._params.time_step):
                self._simulator.AdvanceTo(t)

            self._visualizer.StopRecording()
            self._visualizer.PublishRecording()
    

def get_parser(plant: MultibodyPlant) -> Parser:
    """Creates a parser for a plant and adds package paths to it."""
    parser = Parser(plant)
    AddPackagePaths(parser)
    parser.package_map().AddPackageXml(os.path.abspath("package.xml"))
    return parser
