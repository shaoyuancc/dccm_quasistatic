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

class DCCMSynthesizer():
    def __init__(self, params: DCCMParams):
        self._params = params
        self._wijc = None
        self._lijc = None
    
    def calculate_dccm_from_samples(self, x_samples, u_samples, x_next_samples, A_samples, B_samples, wijc_initial_guess = None, lijc_initial_guess = None) -> None:
        n_dccm_samples = len(x_samples)
        start_time = time.time()
        print(f"Calculating DCCM from {n_dccm_samples} samples")
        prog = MathematicalProgram()
        # Indeterminates
        x = prog.NewIndeterminates(self._params.dim_x, 'x{k}')
        u = prog.NewIndeterminates(self._params.dim_u, 'u{k}')
        w = prog.NewIndeterminates(self._params.dim_x * 2, 'w')
        w = np.array(w).reshape(-1, 1)

        # Monomial basis
        v = [monomial.ToExpression() for monomial in MonomialBasis(x, self._params.deg)]

        dim_v = len(v)
        # print(f"dim_v: {dim_v}")
        # for vi in v:
        #     display_expression("v_i", vi)
        
        n_lower_tri = get_n_lower_tri_from_matrix_dim(self._params.dim_x)
        wijc = prog.NewContinuousVariables(rows=n_lower_tri, cols=dim_v, name='wijc')
        # print(f"n_lower_tri: {n_lower_tri}")
        # print("wijc: ", wijc.shape)

        lijc = prog.NewContinuousVariables(rows=self._params.dim_x * self._params.dim_u, cols=dim_v, name='lijc')
        # print("lijc: ", lijc.shape)

        r = prog.NewContinuousVariables(1, 'r')

        for i in range(n_dccm_samples):
            xi = x_samples[i]
            ui = u_samples[i]
            # A and B matrices
            Ak = A_samples[i]
            Bk = B_samples[i]

            # Create mapping of variables to values
            env = dict(zip(x, xi))
            # Substitute xi into v(xi)
            v_xi = Evaluate(v, env).flatten()

            xi_next = x_next_samples[i]
            # Create mapping of variables to values
            env = dict(zip(x, xi_next))
            # Substitute xi_next into v(xi_next)
            v_xi_next = Evaluate(v, env).flatten()
            # print(f"v_xi.shape: {v_xi.shape}")

            Wk_lower_tri = wijc.dot(v_xi)
            # print(f"Wk_lower_tri.shape: {Wk_lower_tri.shape}")

            Wk = create_square_symmetric_matrix_from_lower_tri_array(Wk_lower_tri)
            # Wk has shape (dim_x, dim_x)

            Wk_next_lower_tri = wijc.dot(v_xi_next)
            Wk_next = create_square_symmetric_matrix_from_lower_tri_array(Wk_next_lower_tri)


            Lk_elements = lijc.dot(v_xi)
            Lk = Lk_elements.reshape(self._params.dim_u, self._params.dim_x)

            # print("Wk: ", Wk.shape)
            # print("Wk_next: ", Wk_next.shape)
            # print("Ak: ", Ak.shape)
            # print("Bk: ", Bk.shape)
            # print("Lk: ", Lk.shape)

            print("Adding constraint for sample ", i)
            clear_output(wait=True)
            cross_diag = Ak @ Wk + Bk @ Lk
            omega = np.block([[Wk_next, cross_diag],
                            [cross_diag.T, (1-self._params.beta)*Wk]])
            # print("omega: ", omega.shape)
            # Note: w is an additional indeterminate that enforces that omega is PSD

            prog.AddSosConstraint((w.T @ omega @ w - r[0] * w.T @  w).flatten()[0])
            

        prog.AddLinearCost(r[0])
        prog.AddLinearConstraint(r[0] >= 0.01)
        # print(f"num_vars: {prog.num_vars()}")
        # print(f"num_indeterminates: {prog.num_indeterminates()}")

        if wijc_initial_guess is not None:
            prog.SetInitialGuess(wijc, wijc_initial_guess)
        if lijc_initial_guess is not None:
            prog.SetInitialGuess(lijc, lijc_initial_guess)

        print("Start solving DCCM")
        result = Solve(prog)
        print("Solver succeeded: ", result.is_success(), " in ", time.time() - start_time, " seconds")

        infeasible_constraints = result.GetInfeasibleConstraints(prog)
        for c in infeasible_constraints:
            print(f"infeasible constraint: {c}")

        # print('solver is: ', result.get_solver_id().name())

        # Extract the solution
        self._wijc = result.GetSolution(wijc)
        self._lijc = result.GetSolution(lijc)
        print("wijc:\n", self._wijc)
        print("\nlijc:\n", self._lijc)
        print("r:\n", result.GetSolution(r))
        return result.is_success(), result.GetSolution(wijc), result.GetSolution(lijc)