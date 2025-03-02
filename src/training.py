# -*- coding: utf-8 -*-
# training.py
"""
File defining the routines to train the DAN.
"""
import copy

import torch

from src.log import logger


def train_full(net, T, prop, obs, x0, ha0):
    """
    Train over full time 0..T with BPTT
    # learn the parameters in net.a, net.b, net.c using t=0..T
    # by minimizing the total loss
    """

    # generate training data seq for t=0..T
    xt = []
    yt = []
    x = x0

    for t in range(1, T + 1):
        x = prop(x).sample(sample_shape=torch.Size([1])).squeeze(0)
        y = obs(x).sample(sample_shape=torch.Size([1])).squeeze(0)
        xt.append(x)
        yt.append(y)

    # train net using xt and yt, t = 1 .. T and x0
    optimizer = torch.optim.LBFGS(net.parameters(), max_iter=1000, max_eval=2000, line_search_fn='strong_wolfe',
                                  tolerance_grad=1e-14, tolerance_change=1e-14, history_size=20)
    ite = 0

    def closure():
        nonlocal ite
        ha = ha0
        optimizer.zero_grad()
        loss0 = -torch.mean(net.c.forward(ha).log_prob(x0))
        total_loss = torch.zeros_like(loss0)
        for t in range(T):
            loss, ha = net.forward(ha, xt[t], yt[t])
            total_loss += loss
        total_loss = total_loss / T + loss0
        logger.info(f"iteration {ite}: loss = {total_loss}")
        total_loss.backward()
        ite += 1
        return total_loss

    # run the optimizer
    optimizer.step(closure)

    # get the posterior densities over t at the end of the training
    # to compare their means with model and obs data
    posterior_densities = [net.c.forward(ha0)]
    ha = ha0
    for t in range(T):
        _, ha = net.forward(ha, xt[t], yt[t])
        posterior_densities.append(net.c.forward(ha))

    return xt, yt, posterior_densities


def pre_train_full(net, x0, ha0):
    """
    Pre-train c at t=0
    # learn the parameters in net.c using ha0 and x0
    # by minimizing the L_0(q_0^a) loss
    """

    # empirical mean of x0
    logger.info(f"empirical mean of x0: {x0.mean().item()}")

    # create an optimizer optimizer0 for the parameters in c
    optimizer0 = torch.optim.LBFGS(net.c.parameters(), max_iter=1000, max_eval=2000, line_search_fn='strong_wolfe',
                                   tolerance_grad=1e-14, tolerance_change=1e-14, history_size=20)

    # TODO minimize L_0(q_0^a), check how small is the loss
    ite = 0
    logger.info(f"initial loss: {-torch.mean(net.c.forward(ha0).log_prob(x0))}")
    loss = []

    # Use closure0 to compute the loss and gradients
    def closure0():
        # TODO first use optimizer0 to set all the gradients to zero
        optimizer0.zero_grad()
        # then compute the loss logpdf_a0 = L_0(q_0^a), by using x0, h0, and c
        logpdf_a0 = -torch.mean(net.c.forward(ha0).log_prob(x0))
        # perform backpropagation of the loss
        logpdf_a0.backward()
        loss.append(logpdf_a0.item())
        # a counter of number of evaluations
        nonlocal ite
        logger.info(f"iteration {ite}: loss = {logpdf_a0}")
        ite = ite + 1
        return logpdf_a0

    # TODO run optimizer
    optimizer0.step(closure0)

    # print out the final mean and covariance of q_0^a
    pdf_a0 = net.c(ha0)
    logger.info(f"a0 mean: {pdf_a0.mean[0, :].detach().numpy()}")
    logger.info(f"a0 var: {pdf_a0.variance[0, :].detach().numpy()}")
    logger.info(f"a0 covar: {pdf_a0.covariance_matrix[0, :, :].detach().numpy().tolist()}")

    return loss


