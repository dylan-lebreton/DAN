import torch


@torch.no_grad()
def test(net, T, prop, obs, x0, ha0):
    x = x0
    ha = ha0
    xt = []
    yt = []
    posterior_densities = [net.c.forward(ha0)]

    for t in range(1, T + 1):
        # on the fly data generation
        x = prop(x).sample(sample_shape=torch.Size([1])).squeeze(0)
        y = obs(x).sample(sample_shape=torch.Size([1])).squeeze(0)
        xt.append(x)
        yt.append(y)

        # Evaluates the loss
        _, ha = net(ha, x, y)

        posterior_densities.append(net.c.forward(ha))

    return xt, yt, posterior_densities
