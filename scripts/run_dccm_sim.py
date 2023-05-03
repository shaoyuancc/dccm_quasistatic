from time import sleep
import hydra
from hydra.utils import instantiate, call, get_original_cwd
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict
# import wandb
from tqdm import tqdm
import os
import numpy as np
from datetime import datetime
from dccm_quasistatic.controller.controller_base import ControllerBase
from dccm_quasistatic.controller.dccm_controller import DCCMController
from dccm_quasistatic.environment.planar_pushing_environment import PlanarPushingEnvironment

from pydrake.all import Box, Sphere

from qsim.parser import (
    QuasistaticParser,
    QuasistaticSystemBackend,
    GradientMode,
)

from qsim.simulator import ForwardDynamicsMode, InternalVisualizationType
from qsim.model_paths import models_dir


@hydra.main(version_base=None, config_path="../config", config_name="basic")
def main(cfg: OmegaConf) -> None:
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Add log dir to config
    hydra_config = HydraConfig.get()
    full_log_dir = hydra_config.runtime.output_dir
    with open_dict(cfg):
        cfg.log_dir = os.path.relpath(full_log_dir, get_original_cwd() + "/outputs")
    
    q_model_path = os.path.join(models_dir, "q_sys", "box_pushing.yml")
    q_parser = QuasistaticParser(q_model_path)
    q_sim = q_parser.make_simulator_cpp()
    q_sim_py = q_parser.make_simulator_py(InternalVisualizationType.Cpp)
    sample_generator = instantiate(cfg.sample_generator, q_sim=q_sim, q_sim_py=q_sim_py, parser=q_parser)
    samples = sample_generator.generate_samples()
    
    controller: ControllerBase = instantiate(cfg.controller)
    
    environment: PlanarPushingEnvironment = instantiate(cfg.environment, controller=controller)
    environment.setup()

    if isinstance(controller, DCCMController):
        controller.calc_dccm(*samples)
        print(f"calc DCCM completed!")
    
    environment.simulate(duration=10)
    
    print("Done!")

if __name__ == "__main__":
    main()