
from dccm_quasistatic.controller.controller_base import ControllerBase
from pydrake.all import (MathematicalProgram, Solve, MonomialBasis,
                         DiagramBuilder, Evaluate, LogVectorOutput, Simulator,
                         SymbolicVectorSystem, Variable, ToLatex, Polynomial,
                         VectorSystem, eq, ge, le, Formula, Expression, Evaluate,
                         LeafSystem, AbstractValue,
                         )

from pydrake.all import (PiecewisePolynomial, ModelInstanceIndex,
    RotationMatrix, RigidTransform, Rgba, Box, Sphere, BaseField,
    Evaluate, Fields, PointCloud, MeshcatAnimation)

from IPython.display import clear_output

import os
import copy
import time
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from dataclasses import dataclass
from dccm_quasistatic.utils.math_utils import (create_square_symmetric_matrix_from_lower_tri_array,
                                               get_n_lower_tri_from_matrix_dim,
                                               matrix_inverse)
from dccm_quasistatic.utils.sample_generator import (SampleGenerator, SampleGeneratorParams)
from dccm_quasistatic.controller.dccm_params import DCCMParams


class DCCMController(ControllerBase):
    def __init__(self, params: DCCMParams, wijc, lijc) -> None:
        self._params = params
        self._wijc = wijc
        self._lijc = lijc


    def control_law(self, xk: np.array, xd: np.array, ud: np.array, t: float = 0):
        
        assert (self._wijc is not None) and (self._lijc is not None), "DCCM has not been calculated"
        print(f"Calculating geodesic at time: {t}, xk = {xk}, xd = {xd}, ud = {ud}")
        start_time = time.time()
        succeeded, xi, delta_xs, delta_s, geodesic = self.calculate_geodesic(xd, xk)
        if not succeeded:
            print(f"Geodesic calculation failed at time: {t}, u = {ud}")
            return ud, geodesic, succeeded
    
        x = [Variable(f"x_{i}") for i in range(self._params.dim_x)]
        v = [monomial.ToExpression() for monomial in MonomialBasis(x, self._params.deg)] # might need to wrap x in Variables()

        # Probably need to set this u* to something else!
        u = ud
        for i in range(self._params.n_geodesic_segments):
            # Create mapping of variables to values
            env = dict(zip(x, xi[i]))
            # Substitute xi into v(xi)
            v_xi = Evaluate(v, env).flatten()
            # Construct L(xi)
            Li_elements = self._lijc.dot(v_xi)
            Li = Li_elements.reshape(self._params.dim_u, self._params.dim_x)
            # Construct W(xi)
            Wi_lower_tri = self._wijc.dot(v_xi)
            # print(f"Wk_lower_tri.shape: {Wk_lower_tri.shape}")

            Wi = create_square_symmetric_matrix_from_lower_tri_array(Wi_lower_tri)
            # Get M(xi) by inverting W(xi)
            Mi = np.linalg.inv(Wi)
            # Add marginal control input to u
            u = u + delta_s[i] * Li @ Mi @ delta_xs[i]
        
        print(f"Geodesic calculation succeeded at time: {t}, u = {u}, calculation took {time.time() - start_time} seconds")

        return u, geodesic, succeeded

    def calculate_geodesic(self, x0, x1):
        """
        Calculate the geodesic from x0 to x1.
        Based on optimization (27)
        Args:
            x0: (dim_x,): initial state, will correspond to x_k
            x1: (dim_x,): final state, will correspond to x*_k
        """
        print("calculate_geodesic initialize")
        start_time = time.time()
        prog = MathematicalProgram()
        
        # Numerical state evaluation along the geodesic
        x = prog.NewContinuousVariables(self._params.n_geodesic_segments + 1, self._params.dim_x, 'x')

        # For getting around inverting W(x_i)
        m = prog.NewContinuousVariables(self._params.n_geodesic_segments, self._params.dim_x * self._params.dim_x, 'm')

        # For optimizing over the epigraph instead of the original objective
        y = prog.NewContinuousVariables(self._params.n_geodesic_segments, 'y')

        # Displacement vector discretized wrt s parameter
        delta_xs = prog.NewContinuousVariables(self._params.n_geodesic_segments, self._params.dim_x, '\delta x_s')
        
        # Small positive scaler value
        delta_s = prog.NewContinuousVariables(self._params.n_geodesic_segments, 's')

        # Add constraint: make sure delta_s's are positive
        si_positive = prog.AddLinearConstraint(ge(delta_s, np.ones_like(delta_s) * 1e-6))

        # Add constraints
        # Constraint 1
        si_sum_to_one = prog.AddLinearConstraint(sum(delta_s) == 1)
        discrete_distances_sum = x0
        # Constraint: Initial state matches x0
        prog.AddConstraint(eq(x[0], x0))
        for i in range(self._params.n_geodesic_segments):
            discrete_distances_sum = discrete_distances_sum + delta_s[i] * delta_xs[i]
            # Constraint 2: Intermediate state matches sum of deltas

            prog.AddConstraint(eq(x[i+1], discrete_distances_sum))
        # Constraint 3
        total_distances_match = prog.AddConstraint(eq(discrete_distances_sum, x1))
        # Sum cost over all segments
        prog.AddCost(np.sum(y))
        # Constraints for the values of y
        
        for i in range(self._params.n_geodesic_segments):
            v = [monomial.ToExpression() for monomial in MonomialBasis(x[i], self._params.deg)]
            # Construct W(x_i)
            Wk_lower_tri = self._wijc.dot(v)
            Wi = create_square_symmetric_matrix_from_lower_tri_array(Wk_lower_tri)

            # Mi = matrix_inverse(Wi) # <= because of the division, this is not a polynomial anymore.
            Mi = m[i].reshape(self._params.dim_x, self._params.dim_x) 
            MiWi = Mi @ Wi
            # WiMi = Wi @ Mi
            for j in range(self._params.dim_x):
                for k in range(self._params.dim_x):
                    if j == k:
                        prog.AddConstraint(MiWi[j, k] == 1)
                        # prog.AddConstraint(WiMi[j, k] == 1)
                    else:
                        prog.AddConstraint(MiWi[j, k] == 0)
                        # prog.AddConstraint(WiMi[j, k] == 0)
            metric_dist = delta_s[i] * delta_xs[i].T @ Mi @ delta_xs[i]
            # print(f"metric_dist: {metric_dist}")
            # print(f"metric_dist.is_polynomial(): {metric_dist.is_polynomial()}")
            # print(f"metric_dist type: {type(metric_dist)}")
            y_constraint = prog.AddConstraint(metric_dist <= y[i])
            # prog.AddConstraint(metric_dist >= 0)
            
            y_constraint.evaluator().set_description(f"y_constraint_{i}")
            
        
        # Try to keep delta_s small
        prog.AddCost(np.sum(delta_s**2))

        # Seed initial guess as all 1's so that determinant will not be 0 and cause a failure
        prog.SetInitialGuessForAllVariables(np.ones(prog.num_vars()))
        prog.SetInitialGuess(delta_s, np.ones_like(delta_s) * 1 / self._params.n_geodesic_segments)
        evenly_spaced_delta = (x1 - x0) / self._params.n_geodesic_segments
        for i in range (self._params.n_geodesic_segments):
            prog.SetInitialGuess(delta_xs[i],  i * evenly_spaced_delta)
            prog.SetInitialGuess(x[i], x0 + i * evenly_spaced_delta)
        print("Start solving geodesic, time taken to setup: ", time.time() - start_time, " seconds")
        start_time = time.time()
        result = Solve(prog)
        print("Solver succeeded: ", result.is_success(), " in ", time.time() - start_time, " seconds")

        # infeasible_constraints = result.GetInfeasibleConstraints(prog)
        # for c in infeasible_constraints:
        #     print(f"infeasible constraint: {c}")

        geodesic_length = np.sum(result.GetSolution(y))
        return result.is_success(), result.GetSolution(x), result.GetSolution(delta_xs), result.GetSolution(delta_s), geodesic_length

