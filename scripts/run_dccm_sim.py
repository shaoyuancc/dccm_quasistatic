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
    test = instantiate(cfg.sample_generator_params)
    # sample_generator = instantiate(cfg.sample_generator, q_sim=q_sim, q_sim_py=q_sim_py, parser=q_parser)
    print(test.params.n_samples)
    print("Done!")

if __name__ == "__main__":
    main()