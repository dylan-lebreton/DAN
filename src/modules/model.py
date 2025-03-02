import torch
from torch import nn

from src.modules.gaussian import Gaussian


class Observation(nn.Module):

    def __init__(self, gauss_dim=2, scale_vec_init = torch.tensor([-2.3026])):
        nn.Module.__init__(self)
        self.gauss_dim = gauss_dim
        self.loc = Id()
        self.scale_vec = Cst(init=scale_vec_init)

    def forward(self, *args):
        lc = self.loc(*args)
        sc = self.scale_vec(*args)
        return Gaussian(self.gauss_dim, torch.cat((lc, sc), dim=1))


class Model(nn.Module):

    def __init__(self, x_dim=2, N=1, gauss_dim=2, scale_vec_init = torch.tensor([-4.6052])):
        nn.Module.__init__(self)
        self.gauss_dim = gauss_dim
        self.loc = Lin2d(x_dim=x_dim, N=N)
        self.scale_vec = Cst(init=scale_vec_init)

    def forward(self, *args):
        lc = self.loc(*args)
        sc = self.scale_vec(*args)
        return Gaussian(self.gauss_dim, torch.cat((lc, sc), dim=1))


class Id(nn.Module):
    """ A simple id function
    """
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x):
        """ trivial
        """
        return x


class Cst(nn.Module):
    """ A constant scale_vec
    """
    def __init__(self, init):
        nn.Module.__init__(self)
        if isinstance(init, torch.Tensor):
            self.c = init.unsqueeze(0)
        else:
            raise NameError("Cst init unknown")

    def forward(self, x):
        return self.c.expand(x.size(0), self.c.size(0))


class Lin2d(nn.Module):
    # rotation dynamics
    def __init__(self, x_dim, N):
        import numpy as np
        assert (x_dim == 2)
        nn.Module.__init__(self)
        theta = np.pi / 100
        linM = torch.Tensor([[np.cos(theta), np.sin(theta)],
                             [-np.sin(theta), np.cos(theta)]])
        self.linM = linM
        self.linMt = linM.t()
        self.x_dim = x_dim
        self.N = N

    def forward(self, x):
        x_ = torch.clone(x)
        for _ in range(self.N):
            x_ = torch.matmul(x_, self.linMt)
        return x_
