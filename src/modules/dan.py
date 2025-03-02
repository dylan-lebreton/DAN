import torch
from torch import nn

from src.modules.gaussian import Gaussian
from src.modules.network import FcZeroLin, FcZero, FullyConnected


class Analyzer(nn.Module):
  def __init__(self, in_dim=6, out_dim=4, deep=1, activation_classname='nn.LeakyReLU'):
        nn.Module.__init__(self)
        self.loc = FcZeroLin(in_dim=in_dim, out_dim=out_dim, deep=deep, activation_classname=activation_classname)

  def forward(self, *args):
      lc = self.loc(*args)
      return lc


class Propagator(nn.Module):
 def __init__(self, dim=4, deep=1, activation_classname='nn.LeakyReLU'):
  nn.Module.__init__(self)
  self.loc = FcZero(dim=dim, deep=deep, activation_classname=activation_classname)

 def forward(self, *args):
  lc = self.loc(*args)
  return lc


class Procoder(nn.Module):

 def __init__(self, layers = None, activation_classname = 'nn.LeakyReLU', gauss_dim=2):
  nn.Module.__init__(self)
  self.gauss_dim = gauss_dim
  self.loc = FullyConnected(layers=[4, 5] if layers is None else layers, activation_classname=activation_classname)

 def forward(self, *args):
  lc = self.loc(*args)
  if self.gauss_dim is not None:
    return Gaussian(self.gauss_dim, lc)
  else:
   return lc


class DAN(nn.Module):
    """
    A Data Assimilation Network class
    """
    def __init__(self, analyzer, propagator, procoder):

        nn.Module.__init__(self)
        self.a = analyzer
        self.b = propagator
        self.c = procoder
        self.scores = {
            "RMSE_b": [],
            "RMSE_a": [],
            "LOGPDF_b": [],
            "LOGPDF_a": [],
            "LOSS": []}

    def forward(self, ha, x, y):
        """
        forward pass in the DAN
        """
        # propagate past mem into prior mem
        hb = self.b.forward(ha)
        # translate prior mem into prior pdf
        pdf_b = self.c.forward(hb)
        # analyze prior mem
        ha = self.a.forward(torch.cat([hb, y], dim=1))
        # translate post mem into post pdf
        pdf_a = self.c.forward(ha)

        logpdf_a = -torch.mean(pdf_a.log_prob(x))
        logpdf_b = -torch.mean(pdf_b.log_prob(x))

        loss = logpdf_a + logpdf_b

        # Compute scores
        with torch.no_grad():
            if logpdf_a is not None:
                self.scores["RMSE_b"].append(torch.mean(torch.norm(
                    pdf_b.mean - x, dim=1)*x.size(1)**-.5).item())
                self.scores["RMSE_a"].append(torch.mean(torch.norm(
                    pdf_a.mean - x, dim=1)*x.size(1)**-.5).item())
                self.scores["LOGPDF_b"].append(logpdf_b.item())
                self.scores["LOGPDF_a"].append(logpdf_a.item())
                self.scores["LOSS"].append(loss.item())

        return loss, ha

    def clear_scores(self):
        """ clear the score lists
        """
        for v in self.scores.values():
            v.clear()
