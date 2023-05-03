import time
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from dataclasses import dataclass
from dccm_quasistatic.controller.controller_base import ControllerBase
from dccm_quasistatic.controller.dccm_params import DCCMParams
from dccm_quasistatic.utils.math_utils import (create_square_symmetric_matrix_from_lower_tri_array,
                                               get_n_lower_tri_from_matrix_dim,
                                               matrix_inverse)

from pydrake.all import (MathematicalProgram, Solve, MonomialBasis,
                         DiagramBuilder, Evaluate, LogVectorOutput, Simulator,
                         SymbolicVectorSystem, Variable, ToLatex, Polynomial,
                         VectorSystem, eq, ge, le, Formula, Expression, Evaluate,
                         LeafSystem, AbstractValue,
                         )

scaling_factor = 1e15

class DCCMSystem(LeafSystem):
    def __init__(self, params: DCCMParams):
        LeafSystem.__init__(self)
        self._params = params
        self._wijc = None
        self._lijc = None

        self._geodesic_index = self.DeclareAbstractState(
            AbstractValue.Make(0.0)
        )

        # Inputs
        self._state_actual_index = self.DeclareVectorInputPort("state_actual", self._params.dim_x * 2).get_index()
        self._xd_index = self.DeclareVectorInputPort("x_desired", self._params.dim_x).get_index()
        self._ud_index = self.DeclareVectorInputPort("u_desired", self._params.dim_u).get_index()

        # Outputs
        self.DeclareVectorOutputPort("u_actual", self._params.dim_u, self.DoCalcOutput)
        self.DeclareVectorOutputPort("geodesic_actual", 1, self.RetrieveGeodesicLength)
    
    def RetrieveGeodesicLength(self, context, output):
        geodesic = context.get_abstract_state(int(self._geodesic_index)).get_value()
        output.SetFromVector([geodesic])
    
    def DoCalcOutput(self, context, output):
        # unpack inputs
        state = self.get_input_port(self._state_actual_index).Eval(context)
        xk = state[:self._params.dim_x]
        xd = self.get_input_port(self._xd_index).Eval(context)
        # Should the base desired command be the goal position or the actual position of the robot?
        # Goal position of the robot
        # ud = self.get_input_port(self._ud_index).Eval(context)
        # Actual position of the robot
        ud = xk[:self._params.dim_u] # I think this makes more sense, because the nominal control is "do nothing"

        t = context.get_time()
        u, geodesic = self.control_law(xk, xd, ud, t)
        context.get_mutable_abstract_state(int(self._geodesic_index)).set_value(geodesic)
        output.SetFromVector(u)
        
    def control_law(self, xk: np.array, xd: np.array, ud: np.array, t: float = 0) -> np.array:
        assert (self._wijc is not None) and (self._lijc is not None), "DCCM has not been calculated"
        print(f"Calculating geodesic at time: {t}, xk = {xk}, xd = {xd}, ud = {ud}")
        start_time = time.time()
        succeeded, xi, delta_xs, delta_s, geodesic = self.calculate_geodesic(xk, xd)
        if not succeeded:
            print(f"Geodesic calculation failed at time: {t}, u = {ud}")
            return ud, geodesic
    
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
            u = u - delta_s[i] * Li @ Mi @ delta_xs[i]
        
        print(f"Geodesic calculation succeeded at time: {t}, u = {u}, calculation took {time.time() - start_time} seconds")

        return u, geodesic

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

            Mi = matrix_inverse(Wi) # <= because of the division, this is not a polynomial anymore.
            # print(f"Mi.shape: {Mi.shape}")
            # Rational Polynomial Expression
            metric_dist = delta_s[i] * delta_xs[i].T @ Mi @ delta_xs[i]
            # print(f"metric_dist: {metric_dist}")
            # print(f"metric_dist.is_polynomial(): {metric_dist.is_polynomial()}")
            # print(f"metric_dist type: {type(metric_dist)}")
            y_constraint = prog.AddConstraint(metric_dist <= y[i])
            y_constraint.evaluator().set_description(f"y_constraint_{i}")
        
        # Try to keep delta_s small
        prog.AddCost(np.sum(delta_s**2))

        # Seed initial guess as all 1's so that determinant will not be 0 and cause a failure
        prog.SetInitialGuessForAllVariables(np.ones(prog.num_vars()))
        print("Start solving geodesic, time taken to setup: ", time.time() - start_time, " seconds")
        start_time = time.time()
        result = Solve(prog)
        print("Solver succeeded: ", result.is_success(), " in ", time.time() - start_time, " seconds")

        # infeasible_constraints = result.GetInfeasibleConstraints(prog)
        # for c in infeasible_constraints:
        #     print(f"infeasible constraint: {c}")

        geodesic_length = np.sum(result.GetSolution(y))
        return result.is_success(), result.GetSolution(x), result.GetSolution(delta_xs), result.GetSolution(delta_s), geodesic_length
    
    def calculate_dccm_from_samples(self, x_samples, u_samples, x_next_samples, A_samples, B_samples) -> None:
        n_dccm_samples = len(x_samples)
        start_time = time.time()
        print(f"Calculating DCCM from {n_dccm_samples} samples")
        prog = MathematicalProgram()
        # Indeterminates
        x = prog.NewIndeterminates(self._params.dim_x, 'x_{k}')
        u = prog.NewIndeterminates(self._params.dim_u, 'u_{k}')
        w = prog.NewIndeterminates(self._params.dim_x * 2, 'w')
        w = np.array(w).reshape(-1, 1)

        # Monomial basis
        v = [monomial.ToExpression() for monomial in MonomialBasis(x, self._params.deg)]
        dim_v = len(v)
        # print(f"dim_v: {dim_v}")
        n_lower_tri = get_n_lower_tri_from_matrix_dim(self._params.dim_x)
        wijc = prog.NewContinuousVariables(rows=n_lower_tri, cols=dim_v, name='wijc')
        
        # print("wijc: ", wijc.shape)

        lijc = prog.NewContinuousVariables(rows=self._params.dim_x * self._params.dim_u, cols=dim_v, name='lijc')

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
            cross_diag = Ak @ Wk + Bk @ Lk
            omega = np.block([[Wk_next, cross_diag],
                            [cross_diag.T, (1-self._params.beta)*Wk]])
            # print("omega: ", omega.shape)
            # Note: w is an additional indeterminate that enforces that omega is PSD

            prog.AddSosConstraint((w.T @ omega @ w - r[0]).flatten()[0])
            

        prog.AddLinearCost(r[0])
        prog.AddLinearConstraint(r[0] >= 0)

        print("Start solving DCCM")
        result = Solve(prog)
        print("Solver succeeded: ", result.is_success(), " in ", time.time() - start_time, " seconds")

        infeasible_constraints = result.GetInfeasibleConstraints(prog)
        for c in infeasible_constraints:
            print(f"infeasible constraint: {c}")

        # Extract the solution
        

        self._wijc = result.GetSolution(wijc) * scaling_factor
        self._lijc = result.GetSolution(lijc) * scaling_factor
        print("wijc:\n", self._wijc)
        print("\nlijc:\n", self._lijc)