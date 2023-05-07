import os, copy
from dataclasses import dataclass
from time import sleep

import numpy as np
from tqdm import tqdm
from typing import List, Tuple

# import torch

from qsim.simulator import QuasistaticSimulator, InternalVisualizationType
from qsim_cpp import QuasistaticSimulatorCpp, ForwardDynamicsMode
from qsim.parser import (
    QuasistaticParser,
    GradientMode,
)

@dataclass
class SampleGeneratorParams:
    log_barrier_weight: float = 100
    n_samples: int = 100 # Number of samples to use for creating the DCCM
    workspace_radius: float = 5
    actuated_collision_geomtery_names: List = None



class SampleGenerator():

    def __init__(
        self,
        params: SampleGeneratorParams,
        q_sim: QuasistaticSimulatorCpp,
        q_sim_py: QuasistaticSimulator,
        parser: QuasistaticParser,
    ):
        self.params = params
        self.q_sim = q_sim
        self.q_sim_py = q_sim_py
        # Indices into x_all of the object
        self.idx_qu = self.q_sim.get_q_u_indices_into_q()
        # Indices into x_all of the robot
        self.idx_qa = self.q_sim.get_q_a_indices_into_q()

        # Collision geometry names required for checking if q is admissible
        # TODO: Come up with cleaner implementation
        assert (
            self.params.actuated_collision_geomtery_names is not None
        ), "Must provide collision geometry names"

        self.dim_x = len(self.idx_qu) + len(self.idx_qa)
        self.dim_u = self.q_sim.num_actuated_dofs()

        self.sim_p = copy.deepcopy(q_sim.get_sim_params())
        self.sim_p.h = 0.1
        self.sim_p.unactuated_mass_scale = 10
        self.q_sim_batch = parser.make_batch_simulator()
        
        self.enable_analytical_smoothing(True, self.params.log_barrier_weight)

    def enable_analytical_smoothing(
        self, enable: bool, log_barrier_weight: float = None
    ) -> None:
        """
        Enable or disable analytical smoothing of the dynamics.
        Args:
            enable (bool): enable or disable analytical smoothing
            log_barrier_weight (float): weight of the log barrier term (only required for enabling)"""
        self.analytical_smoothing_enabled = enable
        if enable:
            assert (
                log_barrier_weight is not None
            ), "Must provide log barrier weight when enabling analytical smoothing"
            self.sim_p.gradient_mode = GradientMode.kAB
            self.sim_p.log_barrier_weight = log_barrier_weight
            self.sim_p.forward_mode = ForwardDynamicsMode.kLogIcecream
        else:
            self.sim_p.gradient_mode = GradientMode.kNone
            self.sim_p.forward_mode = ForwardDynamicsMode.kQpMp


    def generate_samples(self):
        # Hardcoded robot and object dimensions:
        x_bounds = [
            [-self.params.workspace_radius, self.params.workspace_radius],  #  object pos_x bounds
            [-self.params.workspace_radius, self.params.workspace_radius],  #  object pos_y bounds
            [-np.pi, np.pi], # object orientation bounds
            [-self.params.workspace_radius, self.params.workspace_radius],  #  robot pos_x bounds
            [-self.params.workspace_radius, self.params.workspace_radius],  #  robot pos_y bounds
        ]

        u_bounds = [
            [-self.params.workspace_radius, self.params.workspace_radius],  #  robot pos_x bounds
            [-self.params.workspace_radius, self.params.workspace_radius],  #  robot pos_y bounds
        ]

        x_samples = np.zeros((self.params.n_samples, self.dim_x))
        u_samples = np.zeros((self.params.n_samples, self.dim_u))

        count = 0
        while count != self.params.n_samples:
            for i in range(self.dim_x):
                x_samples[count, i] = np.random.uniform(
                    x_bounds[i][0], x_bounds[i][1]
                )
            for i in range(self.dim_u):
                u_samples[count, i] = np.random.uniform(
                    u_bounds[i][0], u_bounds[i][1]
                )

            if self.is_x_admissible(x_samples[count]):
                count += 1
            else:
                print(
                    f"Rejected sample {count}, x_sample: {x_samples[count]}"
                )
        
        (
            x_next_samples,
            A_samples,
            B_samples,
            is_valid_batch,
        ) = self.q_sim_batch.calc_dynamics_parallel(
            x_samples, u_samples, self.sim_p
        )
        assert np.all(is_valid_batch), "Dynamics batch had invalid results"

        return(
            x_samples,
            u_samples,
            x_next_samples,
            A_samples,
            B_samples,
        )
    
    def generate_samples_submanifold(self, visualize=False):
        robot_radius = 0.1
        object_radius = 0.5
        u_rel_lim = 0.5

        object_buffer = robot_radius*2 + object_radius
        
        x_bounds = [
            [-self.params.workspace_radius + object_buffer, self.params.workspace_radius - object_buffer],  #  object pos_x bounds
            [-self.params.workspace_radius + object_buffer, self.params.workspace_radius - object_buffer],  #  object pos_y bounds
            [-np.pi, np.pi], # object orientation bounds
            [-self.params.workspace_radius + robot_radius, self.params.workspace_radius - robot_radius],  #  robot pos_x bounds
            [-self.params.workspace_radius + robot_radius, self.params.workspace_radius - robot_radius],  #  robot pos_y bounds
        ]

        x_samples = []
        u_samples = []
        x_next_samples = []
        A_samples = []
        B_samples = []

        workspace_bounds = [-self.params.workspace_radius*2,0]
        robot_radius = 0.1
        object_radius = 0.5
        u_rel_lim = 0.3

        count = 0
        idx_qu = self.q_sim.get_q_u_indices_into_q()
        while count != self.params.n_samples:
            q = [0, 0, 0, 0, 0]
            u = [0, 0]
            # Randomly pick object_x
            for i in range(self.dim_x):
                q[i] = np.random.uniform(
                    x_bounds[i][0], x_bounds[i][1]
                )
            if self.is_x_admissible(q):
                count += 1
            else:
                continue

            for i in range(self.dim_u):
                u[i] = np.random.uniform(
                    q[i + 3] - u_rel_lim, q[i + 3] + u_rel_lim # HARDCODED relative indexes between x and u
                )            
            
            
            if visualize:
                sleep(1.0)
                print(f"q: {q}, u: {u}")
                self.q_sim_py.update_mbp_positions_from_vector(q)
                self.q_sim_py.draw_current_configuration()
            q_next = self.q_sim.calc_dynamics(q, u, self.sim_p)
            A = self.q_sim.get_Dq_nextDq()
            B = self.q_sim.get_Dq_nextDqa_cmd()

            if visualize:
                sleep(0.5)
                self.q_sim_py.update_mbp_positions_from_vector(q_next)
                self.q_sim_py.draw_current_configuration()
            
            x_samples.append(q[idx_qu])
            u_samples.append(u)
            x_next_samples.append(q_next[idx_qu])
            A_samples.append(A)
            B_samples.append(B)

            
        
        return(
            np.array(x_samples),
            np.array(u_samples),
            np.array(x_next_samples),
            np.array(A_samples),
            np.array(B_samples),
        )
    def generate_two_spheres_samples(self, visualize=False):
        robot_radius = 0.1
        object_radius = 0.1
        u_rel_lim = 0.3

        x_samples = []
        u_samples = []
        x_next_samples = []
        A_samples = []
        B_samples = []

        # workspace_bounds = [-self.params.workspace_radius,0]
        workspace_bounds = [-self.params.workspace_radius,self.params.workspace_radius]

        count = 0
        while count != self.params.n_samples:
            # Randomly pick object_x
            x = np.random.uniform(workspace_bounds[0], workspace_bounds[1], size=(2,))
            
            # Randomly pick valid u_x
            u_x = np.random.uniform(workspace_bounds[0], workspace_bounds[1])

            q = x
            u = [u_x]
            if visualize:
                sleep(1.0)
                print(f"q: {q}, u: {u}")
                self.q_sim_py.update_mbp_positions_from_vector(q)
                self.q_sim_py.draw_current_configuration()

            q_next = self.q_sim.calc_dynamics(q, u, self.sim_p)
            A = self.q_sim.get_Dq_nextDq()
            B = self.q_sim.get_Dq_nextDqa_cmd()

            if visualize:
                sleep(0.5)
                self.q_sim_py.update_mbp_positions_from_vector(q_next)
                self.q_sim_py.draw_current_configuration()

            # # Randomly pick object_x
            # object_x = np.random.uniform(workspace_bounds[0]+ robot_radius*2 + object_radius, workspace_bounds[1] - object_radius)
            # # Randomly pick robot_x +  < object_x - object_rradius
            # robot_x = np.random.uniform(workspace_bounds[0]+ robot_radius, object_x - object_radius - robot_radius)
            # # Randomly pick valid u_x
            # u_x = np.random.uniform(robot_x - u_rel_lim, robot_x + u_rel_lim)

            # q = [object_x, robot_x]
            # u = [u_x]
            # if visualize:
            #     sleep(1.0)
            #     print(f"q: {q}, u: {u}")
            #     self.q_sim_py.update_mbp_positions_from_vector(q)
            #     self.q_sim_py.draw_current_configuration()

            # q_next = self.q_sim.calc_dynamics(q, u, self.sim_p)
            # A = self.q_sim.get_Dq_nextDq()
            # B = self.q_sim.get_Dq_nextDqa_cmd()

            # if visualize:
            #     sleep(0.5)
            #     self.q_sim_py.update_mbp_positions_from_vector(q_next)
            #     self.q_sim_py.draw_current_configuration()
            
            x_samples.append(q)
            u_samples.append(u)
            x_next_samples.append(q_next)
            A_samples.append(A)
            B_samples.append(B)

            count += 1
        
        return(
            x_samples,
            u_samples,
            x_next_samples,
            A_samples,
            B_samples,
        )
    
    def generate_robot_only_samples(self, visualize=False):
        x_bounds = [
            [-self.params.workspace_radius, self.params.workspace_radius],  #  robot pos_x bounds
            [-self.params.workspace_radius , self.params.workspace_radius],  #  robot pos_y bounds
        ]
        x_samples = []
        u_samples = []
        x_next_samples = []
        A_samples = []
        B_samples = []

        u_rel_lim = 0.3

        count = 0
        while count != self.params.n_samples:
            q = np.zeros((self.dim_x,))
            u = np.zeros((self.dim_u,))
            # Randomly pick object_x
            for i in range(self.dim_x):
                q[i] = np.random.uniform(
                    x_bounds[i][0], x_bounds[i][1]
                )
            
            count += 1

            for i in range(self.dim_u):
                u[i] = np.random.uniform(
                    q[i] - u_rel_lim, q[i] + u_rel_lim
                )            
            
            if visualize:
                sleep(1.0)
                print(f"q: {q}, u: {u}")
                self.q_sim_py.update_mbp_positions_from_vector(q)
                self.q_sim_py.draw_current_configuration()
            q_next = self.q_sim.calc_dynamics(q, u, self.sim_p)
            A = self.q_sim.get_Dq_nextDq()
            B = self.q_sim.get_Dq_nextDqa_cmd()

            if visualize:
                sleep(0.5)
                self.q_sim_py.update_mbp_positions_from_vector(q_next)
                self.q_sim_py.draw_current_configuration()
            
            x_samples.append(q)
            u_samples.append(u)
            x_next_samples.append(q_next)
            A_samples.append(A)
            B_samples.append(B)

        return(
            np.array(x_samples),
            np.array(u_samples),
            np.array(x_next_samples),
            np.array(A_samples),
            np.array(B_samples),
        )
    
    def generate_samples_close_u(self, visualize=False):
        robot_radius = 0.1
        object_radius = 0.5
        u_rel_lim = 0.5

        object_buffer = robot_radius*2 + object_radius
        
        x_bounds = [
            [-self.params.workspace_radius + object_buffer, self.params.workspace_radius - object_buffer],  #  object pos_x bounds
            [-self.params.workspace_radius + object_buffer, self.params.workspace_radius - object_buffer],  #  object pos_y bounds
            [-np.pi, np.pi], # object orientation bounds
            [-self.params.workspace_radius + robot_radius, self.params.workspace_radius - robot_radius],  #  robot pos_x bounds
            [-self.params.workspace_radius + robot_radius, self.params.workspace_radius - robot_radius],  #  robot pos_y bounds
        ]

        x_samples = []
        u_samples = []
        x_next_samples = []
        A_samples = []
        B_samples = []

        workspace_bounds = [-self.params.workspace_radius*2,0]
        robot_radius = 0.1
        object_radius = 0.5
        u_rel_lim = 0.3

        count = 0
        while count != self.params.n_samples:
            q = [0, 0, 0, 0, 0]
            u = [0, 0]
            # Randomly pick object_x
            for i in range(self.dim_x):
                q[i] = np.random.uniform(
                    x_bounds[i][0], x_bounds[i][1]
                )
            if self.is_x_admissible(q):
                count += 1
            else:
                continue

            for i in range(self.dim_u):
                u[i] = np.random.uniform(
                    q[i + 3] - u_rel_lim, q[i + 3] + u_rel_lim # HARDCODED relative indexes between x and u
                )            
            
            
            if visualize:
                sleep(1.0)
                print(f"q: {q}, u: {u}")
                self.q_sim_py.update_mbp_positions_from_vector(q)
                self.q_sim_py.draw_current_configuration()
            q_next = self.q_sim.calc_dynamics(q, u, self.sim_p)
            A = self.q_sim.get_Dq_nextDq()
            B = self.q_sim.get_Dq_nextDqa_cmd()

            if visualize:
                sleep(0.5)
                self.q_sim_py.update_mbp_positions_from_vector(q_next)
                self.q_sim_py.draw_current_configuration()
            
            x_samples.append(q)
            u_samples.append(u)
            x_next_samples.append(q_next)
            A_samples.append(A)
            B_samples.append(B)

            
        
        return(
            np.array(x_samples),
            np.array(u_samples),
            np.array(x_next_samples),
            np.array(A_samples),
            np.array(B_samples),
        )
    
    def generate_1d_pushing_samples(self, visualize = False):
        robot_radius = 0.1
        object_radius = 0.5
        u_rel_lim = 0.3

        x_samples = []
        u_samples = []
        x_next_samples = []
        A_samples = []
        B_samples = []

        workspace_bounds = [-self.params.workspace_radius,0]

        count = 0
        while count != self.params.n_samples:
            # Randomly pick object_x
            object_x = np.random.uniform(workspace_bounds[0]+ robot_radius*2 + object_radius, workspace_bounds[1] - object_radius)
            # Randomly pick robot_x +  < object_x - object_rradius
            robot_x = np.random.uniform(workspace_bounds[0]+ robot_radius, object_x - object_radius - robot_radius)
            # Randomly pick valid u_x
            u_x = np.random.uniform(robot_x - u_rel_lim, robot_x + u_rel_lim)

            q = [object_x, 0, 0, robot_x, 0]
            u = [u_x, 0]
            if visualize:
                sleep(1.0)
                print(f"q: {q}, u: {u}")
                self.q_sim_py.update_mbp_positions_from_vector(q)
                self.q_sim_py.draw_current_configuration()

            q_next = self.q_sim.calc_dynamics(q, u, self.sim_p)
            A = self.q_sim.get_Dq_nextDq()
            B = self.q_sim.get_Dq_nextDqa_cmd()

            if visualize:
                sleep(0.5)
                self.q_sim_py.update_mbp_positions_from_vector(q_next)
                self.q_sim_py.draw_current_configuration()
            
            x_samples.append(q)
            u_samples.append(u)
            x_next_samples.append(q_next)
            A_samples.append(A)
            B_samples.append(B)

            count += 1
        
        return(
            x_samples,
            u_samples,
            x_next_samples,
            A_samples,
            B_samples,
        )

    def is_x_admissible(self, x: np.array) -> bool:
        """Check if a given q is admissible.
        Args:
            x (n,)
            q (q,)
        Returns:
            is_admissible (bool)
        """
        x_all_dict_original = self.q_sim.get_mbp_positions()

        x_all_dict = self.q_sim.get_q_dict_from_vec(x)
        self.q_sim.update_mbp_positions(x_all_dict)

        query_object = self.q_sim.get_query_object()
        inspector = query_object.inspector()
        point_pairs = self.q_sim.get_query_object().ComputePointPairPenetration()
        for pair in point_pairs:
            if np.isin(
                self.params.actuated_collision_geomtery_names,
                test_elements=[
                    inspector.GetName(pair.id_A),
                    inspector.GetName(pair.id_B),
                ],
            ).any():
                self.q_sim.update_mbp_positions(x_all_dict_original)
                return False

        self.q_sim.update_mbp_positions(x_all_dict_original)
        return True
