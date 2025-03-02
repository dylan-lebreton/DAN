import torch
from torch import nn


class FcZero(nn.Module):
    """Fully connected neural network with ReZero trick
    """

    def __init__(self, dim, deep, activation_classname):
        """
        layers: the list of the layers dimensions
        """
        nn.Module.__init__(self)
        layers = (deep + 1) * [dim]
        self.lins = nn.ModuleList([nn.Linear(d0, d1) for d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList([eval(activation_classname)() for _ in range(deep)])
        self.alphas = [nn.Parameter(torch.Tensor([0.])) for _ in range(deep)]

    def forward(self, h):
        for lin, act, alpha in zip(self.lins, self.acts, self.alphas):
            h = h + alpha * act(lin(h))
        return h


class FcZeroLin(nn.Module):
    """
    FcZero network ending with linear layer
    """

    def __init__(self, in_dim, out_dim, deep, activation_classname):
        """
        layers: the list of the layers dimensions
        """
        nn.Module.__init__(self)
        # TODO init
        self.lins = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(deep - 1)] + [nn.Linear(in_dim, out_dim)])
        self.acts = nn.ModuleList([eval(activation_classname)() for _ in range(deep - 1)])
        self.alphas = [nn.Parameter(torch.Tensor([0.])) for _ in range(deep - 1)]

    def forward(self, h):
        # TODO rewrite output
        for lin, act, alpha in zip(self.lins[:-1], self.acts, self.alphas):
            h = h + alpha * act(lin(h))
        return self.lins[-1](h)


class FullyConnected(nn.Module):
    """Fully connected NN ending with a linear layer
    """

    def __init__(self, layers, activation_classname):
        nn.Module.__init__(self)
        n = len(layers)
        self.lins = nn.ModuleList([nn.Linear(d0, d1) for d0, d1 in zip(layers[:-1], layers[1:])])
        self.acts = nn.ModuleList([eval(activation_classname)() for _ in range(n - 2)])

    def forward(self, h):
        for lin, act in zip(self.lins[:-1], self.acts):
            h = act(lin(h))
        return self.lins[-1](h)
