import torch
from torch.distributions import MultivariateNormal as Mvn


class Gaussian(Mvn):
    """
    Return a pytorch Gaussian pdf from args
    args is either a (loc, scale_tril) or a (x_dim, vec)
    """
    def __init__(self, *args):
        self.minexp = torch.Tensor([-8.0])
        self.maxexp = torch.Tensor([8.0])
        if isinstance(args[0], int):
            """args is a (x_dim, vec)
            loc is the first x_dim coeff of vec
            if the rest is one coeff c then
                scale_tril = e^c*I
            else
                scale_tril is filled diagonal by diagonal
                starting by the main one
                (which is exponentiated to ensure strict positivity)
            """
            x_dim, vec = args
            vec_dim = vec.size(-1)
            if vec_dim == x_dim + 1:
                loc = vec[:, :x_dim]
                scale_tril = torch.eye(x_dim).unsqueeze(0).expand(vec.size(0), -1, -1)
                scale_tril = torch.exp(vec[:, x_dim]).view(vec.size(0), 1, 1)*scale_tril
            else:
                inds = self.vec_to_inds(x_dim, vec_dim)
                loc = vec[:, :x_dim]
                diaga = vec[:, x_dim:2*x_dim]
                diaga = torch.max(self.minexp.expand_as(diaga), diaga)
                diaga = torch.min(self.maxexp.expand_as(diaga), diaga)
                lbda = torch.cat((torch.exp(diaga), vec[:, 2*x_dim:]), 1)
                scale_tril = torch.zeros(vec.size(0), x_dim, x_dim)
                scale_tril[:, inds[0], inds[1]] = lbda

            Mvn.__init__(self, loc=loc, scale_tril=scale_tril)

        else:
            """args is a loc, scale_tril
            """
            print('Init Mvn by full arg')
            Mvn.__init__(self, loc=args[0], scale_tril=args[1])

    def vec_to_inds(self, x_dim, vec_dim):
        """Computes the indices of scale_tril coeffs,
        scale_tril is filled main diagonal first

        x_dim: dimension of the random variable
        vec_dim: dimension of the vector containing
                 the coeffs of loc and scale_tril
        """
        ldiag, d, c = x_dim, 0, 0  # diag length, diag index, column index
        inds = [[], []]  # list of line and column indexes
        for i in range(vec_dim - x_dim):  # loop over the non-mean coeff
            inds[0].append(c+d)  # line index
            inds[1].append(c)  # column index
            if c == ldiag-1:  # the current diag end is reached
                ldiag += -1  # the diag length is decremented
                c = 0  # the column index is reinitialized
                d += 1  # the diag index is incremented
            else:  # otherwize, only the column index is incremented
                c += 1
        return inds
