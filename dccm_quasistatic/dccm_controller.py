import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from dataclasses import dataclass

from pydrake.all import (MathematicalProgram, Solve, MonomialBasis,
                         DiagramBuilder, Evaluate, LogVectorOutput, Simulator,
                         SymbolicVectorSystem, Variable, ToLatex, Polynomial,
                         VectorSystem, eq, ge, le, Formula, Expression, Evaluate,
                         LeafSystem, AbstractValue)

@dataclass
class DCCMParams:
    """
    Parameters for the DCCM controller.
    """
    # System parameters
    dim_x: int = 2 # Dimension of the state
    dim_u: int = 1 # Dimension of the input

    # DCCM Params
    deg: int = 6 # Degree of the polynomial
    beta: float = 0.1 # Convergence rate = 1-beta
    

    # Geodesic calculation parameters
    n_geodesic_segments: int = 5 # Number of segments to discretize the geodesic into

class DCCMOnlineController(LeafSystem):
    def __init__(self, params: DCCMParams):
        LeafSystem.__init__(self)
        self.params = params

        self._geodesic_index = self.DeclareAbstractState(
            AbstractValue.Make(0.0)
        )

        # Inputs
        self._xk_index = self.DeclareVectorInputPort("x_current", self.params.dim_x).get_index()
        self._xd_index = self.DeclareVectorInputPort("x_desired", self.params.dim_x).get_index()
        self._ud_index = self.DeclareVectorInputPort("u_desired", self.params.dim_u).get_index()

        # Outputs
        self.DeclareVectorOutputPort("u_current", self.params.dim_u, self.DoCalcOutput)
        self.DeclareVectorOutputPort("geodesic_current", 1, self.RetrieveGeodesicLength)
    
    def RetrieveGeodesicLength(self, context, output):
        geodesic = context.get_abstract_state(int(self._geodesic_index)).get_value()
        output.SetFromVector([geodesic])
    
    def DoCalcOutput(self, context, output):

        # unpack inputs
        xk = self.get_input_port(self._xk_index).Eval(context)
        xd = self.get_input_port(self._xd_index).Eval(context)
        ud = self.get_input_port(self._ud_index).Eval(context)

        t = context.get_time()
        u, geodesic = self.control_law(xk, xd, ud, t)
        context.get_mutable_abstract_state(int(self._geodesic_index)).set_value(geodesic)
        output.SetFromVector(u)
        
    def control_law(self, xk: np.array, xd: np.array, ud: np.array, t: float = 0) -> np.array:
        # return [1]
        # print(f"t: {t}, xk = {xk}, xd = {xd}, ud = {ud}")
        succeeded, xi, delta_xs, delta_s, geodesic = self.calculate_geodesic(xk, xd)
        if not succeeded:
            print(f"Geodesic calculation failed at time: {t}, u = {ud}")
            return ud, geodesic
    
        x = [Variable(f"x_{i}") for i in range(self.params.dim_x)]
        v = [monomial.ToExpression() for monomial in MonomialBasis(x, self.params.deg)] # might need to wrap x in Variables()

        # Probably need to set this u* to something else!
        u = ud
        for i in range(self.params.n_geodesic_segments):
            # Create mapping of variables to values
            env = dict(zip(x, xi[i]))
            # Substitute xi into v(xi)
            v_xi = Evaluate(v, env).flatten()
            # Construct L(xi)
            Li = np.array([[self.params.l1c.dot(v_xi), self.params.l2c.dot(v_xi)]])
            # Construct W(xi)
            W11i = self.params.w11c.dot(v_xi)
            W12i = self.params.w12c.dot(v_xi)
            W22i = self.params.w22c.dot(v_xi)
            Wi = np.array([[W11i, W12i], [W12i, W22i]])
            # Get M(xi) by inverting W(xi)
            Mi = np.linalg.inv(Wi)
            # Add marginal control input to u
            u = u - delta_s[i] * Li @ Mi @ delta_xs[i]
        
        print(f"Geodesic calculation succeeded at time: {t}, u = {u}")

        return u, geodesic

    def calculate_geodesic(self, x0, x1):
        """
        Calculate the geodesic from x0 to x1.
        Based on optimization (27)
        Args:
            x0: (dim_x,): initial state, will correspond to x_k
            x1: (dim_x,): final state, will correspond to x*_k
        """
        prog = MathematicalProgram()
        
        # Numerical state evaluation along the geodesic
        x = prog.NewContinuousVariables(self.params.n_geodesic_segments + 1, self.params.dim_x, 'x')

        # For optimizing over the epigraph instead of the original objective
        y = prog.NewContinuousVariables(self.params.n_geodesic_segments, 'y')

        # Displacement vector discretized wrt s parameter
        delta_xs = prog.NewContinuousVariables(self.params.n_geodesic_segments, self.params.dim_x, '\delta x_s')
        
        # Small positive scaler value
        delta_s = prog.NewContinuousVariables(self.params.n_geodesic_segments, 's')

        # Add constraint: make sure delta_s's are positive
        si_positive = prog.AddLinearConstraint(ge(delta_s, np.ones_like(delta_s) * 1e-6))

        # Add constraints
        # Constraint 1
        si_sum_to_one = prog.AddLinearConstraint(sum(delta_s) == 1)

        discrete_distances_sum = x0
        # Constraint: Initial state matches x0
        prog.AddConstraint(eq(x[0], x0))
        for i in range(self.params.n_geodesic_segments):
            discrete_distances_sum = discrete_distances_sum + delta_s[i] * delta_xs[i]
            # Constraint 2: Intermediate state matches sum of deltas

            prog.AddConstraint(eq(x[i+1], discrete_distances_sum))
        # Constraint 3
        total_distances_match = prog.AddConstraint(eq(discrete_distances_sum, x1))
    
        # Sum cost over all segments
        prog.AddCost(np.sum(y))
        # Constraints for the values of y
        for i in range(self.params.n_geodesic_segments):
            v = [monomial.ToExpression() for monomial in MonomialBasis(x[i], self.params.deg)]
            # Construct W(x_i)
            W11i = self.params.w11c.dot(v)
            W12i = self.params.w12c.dot(v)
            W22i = self.params.w22c.dot(v)
            Wi = np.array([[W11i, W12i], [W12i, W22i]])
            # Get M(x_i) by inverting W(x_i)
            Mi = self.get_2x2_inverse(Wi) # <= because of the division, this is not a polynomial anymore.
            
            # Rational Polynomial Expression
            metric_dist = delta_s[i] * delta_xs[i].T @ Mi @ delta_xs[i]
            # print(f"metric_dist: {metric_dist}")
            # print(f"metric_dist.is_polynomial(): {metric_dist.is_polynomial()}")
            # print(f"metric_dist type: {type(metric_dist)}")
            prog.AddConstraint(metric_dist <= y[i])
        
        # Try to keep delta_s small
        prog.AddCost(np.sum(delta_s**2))

        # Seed initial guess as all 1's so that determinant will not be 0 and cause a failure
        prog.SetInitialGuessForAllVariables(np.ones(prog.num_vars()))

        result = Solve(prog)
        geodesic_length = np.sum(result.GetSolution(y))
        return result.is_success(), result.GetSolution(x), result.GetSolution(delta_xs), result.GetSolution(delta_s), geodesic_length
    
    def get_2x2_inverse(self, A: np.array) -> np.array:
        # This doesn't work: np.linalg.inv(A)
        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        return np.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det
    
    def calculate_dccm_from_samples(self, x_samples, u_samples, x_next_samples, A_samples, B_samples) -> None:
        prog = MathematicalProgram()
        # Indeterminates
        x = prog.NewIndeterminates(self.params.dim_x, 'x_{k}')
        u = prog.NewIndeterminates(self.params.dim_u, 'u_{k}')
        w = prog.NewIndeterminates(4, 'w')
        w = np.array(w).reshape(-1, 1)

        # Monomial basis
        v = [monomial.ToExpression() for monomial in MonomialBasis(x, self.params.deg)]
        dim_v = len(v)
        w11c = prog.NewContinuousVariables(dim_v, 'w11c')
        w12c = prog.NewContinuousVariables(dim_v, 'w12c')
        w22c = prog.NewContinuousVariables(dim_v, 'w22c')
        # print("w11c: ", w11c)

        l1c = prog.NewContinuousVariables(dim_v, 'l1c')
        l2c = prog.NewContinuousVariables(dim_v, 'l2c')

        r = prog.NewContinuousVariables(1, 'r')

        x_samples = np.hstack(np.random.uniform(-self.params.workspace_radius, self.params.workspace_radius, (self.params.n_dccm_samples, 2)),
                              np.random.uniform(-np.pi, np.pi, (self.params.n_dccm_samples, 1)))
        u_samples = np.random.uniform(-self.params.workspace_radius, self.params.workspace_radius, (self.params.n_dccm_samples, self.params.dim_u))

        for i in range(self.params.n_dccm_samples):
            xi = x_samples[i, :]
            ui = u_samples[i, :]
            # A and B matrices
            Ak = np.array([[1.1-0.1*xi[1],   0],
                        [0.1         ,   0.9]])
            Bk = np.array([1, 0])[:, np.newaxis]

            # Create mapping of variables to values
            env = dict(zip(x, xi))
            # Substitute xi into v(xi)
            v_xi = Evaluate(v, env).flatten()

            xi_next = [1.1*xi[0] - 0.1*xi[0]*xi[1] + ui[0],
                    0.9*xi[1] + 0.9*xi[0]]
            # Create mapping of variables to values
            env = dict(zip(x, xi_next))
            # Substitute xi_next into v(xi_next)
            v_xi_next = Evaluate(v, env).flatten()

            W11k = w11c.dot(v_xi)
            W12k = w12c.dot(v_xi)
            W22k = w22c.dot(v_xi)
            Wk = np.array([[W11k, W12k], [W12k, W22k]])
            # print("W11k: ", W11k)

            W11k_next = w11c.dot(v_xi_next)
            W12k_next = w12c.dot(v_xi_next)
            W22k_next = w22c.dot(v_xi_next)
            Wk_next = np.array([[W11k_next, W12k_next], [W12k_next, W22k_next]])


            L1k = l1c.dot(v)
            L2k = l2c.dot(v)
            Lk = np.array([[L1k, L2k]])

            # print("Wk: ", Wk.shape)
            # print("Ak: ", Ak.shape)
            # print("Bk: ", Bk.shape)
            # print("Lk: ", Lk.shape)


            cross_diag = Ak @ Wk + Bk @ Lk
            omega = np.block([[Wk_next, cross_diag],
                            [cross_diag.T, (1-beta)*Wk]])
            # print("omega: ", omega.shape)
            # Note: w is an additional indeterminate that enforces that omega is PSD

            prog.AddSosConstraint((w.T @ omega @ w - r[0]).flatten()[0])
            

        prog.AddLinearCost(r[0])
        prog.AddLinearConstraint(r[0] >= 0)



        # Verify that the solution is meets constraints:
        # prog.SetInitialGuess(w11c, w11c_ans)
        # prog.SetInitialGuess(w12c, w12c_ans)
        # prog.SetInitialGuess(w22c, w22c_ans)
        # prog.SetInitialGuess(l1c, l1c_ans)
        # prog.SetInitialGuess(l2c, l2c_ans)

        result = Solve(prog)
        print("Solver succeeded: ", result.is_success())

        infeasible_constraints = result.GetInfeasibleConstraints(prog)
        for c in infeasible_constraints:
            print(f"infeasible constraint: {c}")

        # Extract the solution
        print("w11c:\n", result.GetSolution(w11c))
        print("\nw12c:\n", result.GetSolution(w12c))
        print("\nw22c:\n", result.GetSolution(w22c))
        print("\nl1c:\n", result.GetSolution(l1c))
        print("\nl2c:\n", result.GetSolution(l2c))

        w11c_ans = result.GetSolution(w11c)
        w12c_ans = result.GetSolution(w12c)
        w22c_ans = result.GetSolution(w22c)
        l1c_ans = result.GetSolution(l1c)
        l2c_ans = result.GetSolution(l2c)
         
