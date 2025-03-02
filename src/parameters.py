import torch


def generate_x0(b_size, x_dim, sigma):
    return 3 * torch.ones(b_size, x_dim) + sigma * torch.randn(b_size, x_dim)


def generate_ha0(b_size, h_dim):
    v0 = torch.zeros(1, h_dim)
    ha0 = torch.zeros(b_size, h_dim)
    for b in range(b_size):
        ha0[b, :] = v0
    return ha0
