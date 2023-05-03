import numpy as np

from pydrake.all import (LeafSystem)


class DesiredTrajectorySourceSystem(LeafSystem):
    def __init__(self, dim_x, dim_u):
        LeafSystem.__init__(self)
        self.dim_x = dim_x
        self.dim_u = dim_u
        self.DeclareVectorOutputPort("x_desired", dim_x, self.CalcPosDesired)
        self.DeclareVectorOutputPort("u_desired", dim_u, self.CalcCommandDesired)
    
    def CalcPosDesired(self, context, output):
        # t = context.get_time()
        # if t < 3.3:
        #     xd = 0
        # elif t < 6.6:
        #     xd = 1
        # else:
        #     xd = 0.5
        x_desired = np.array([0, -1, 0, 0, 0])
        output.SetFromVector(x_desired)

    def CalcCommandDesired(self, context, output):
        u_desired = np.array([0, -1])
        output.SetFromVector(u_desired)
        # Should the base desired command be the goal position or the actual position of the robot?