import torch

from src.log import logger
from src.parameters import generate_x0, generate_ha0
from src.modules.dan import Analyzer, Propagator, Procoder, DAN
from src.modules.model import Observation, Model
from src.plot import plot_pretrain_loss, plot_model_obs, plot_final_mean_densities, plot_scores
from src.testing import test
from src.training import train_full, pre_train_full

############
# Parameters
############
b_size = 256
h_dim = 4
x_dim = 2
T = 50

###############
# configuration
###############

torch.autograd.set_detect_anomaly(True)
torch.set_default_tensor_type(torch.DoubleTensor)

def experiment(deep=1, plot: bool = True, sigma0 = 0.01):
    logger.info(f"deep = {deep} - plot = {plot} - sigma0 = {sigma0}")

    ###############
    # configuration
    ###############

    analyzer = Analyzer(deep=deep)
    propagator = Propagator(deep=deep)
    procoder = Procoder()
    net = DAN(analyzer=analyzer, propagator=propagator, procoder=procoder)
    prop = Model()
    obs = Observation()

    torch.manual_seed(100)
    x0 = generate_x0(b_size, x_dim, sigma0)
    ha0 = generate_ha0(b_size, h_dim)

    ###################
    # train = full mode
    ###################

    # pre-training of procoder at t=0
    #################################

    pre_train_loss = pre_train_full(net, x0, ha0)
    if plot:
        plot_pretrain_loss(pre_train_loss)

    # training of DAN
    #################

    xt, yt, posterior_densities = train_full(net, T, prop, obs, x0, ha0)
    if plot:
        plot_model_obs(xt, yt, T, 0)
        plot_model_obs(xt, yt, T, 10)
        plot_final_mean_densities(xt, yt, posterior_densities, T, 0)
        plot_final_mean_densities(xt, yt, posterior_densities, T, 10)
        plot_scores(net, T)

    # print of first and last scores
    for key, val in net.scores.items():
        if len(val) > 0:
            logger.info(f"train - {key} at iteration 0 = {val[0]}")
            logger.info(f"train - {key} at last iteration = {val[-1]}")

    # test of DAN
    #############

    # clear scores
    net.clear_scores()

    # new first hidden state and prior
    torch.manual_seed(86)
    x0 = generate_x0(b_size, x_dim, sigma0)
    ha0 = generate_ha0(b_size, h_dim)

    xt, yt, posterior_densities = test(net, 2 * T, prop, obs, x0, ha0)
    if plot:
        plot_model_obs(xt, yt, 2*T, 0)
        plot_model_obs(xt, yt, 2*T, 10)
        plot_final_mean_densities(xt, yt, posterior_densities, 2*T, 0)
        plot_final_mean_densities(xt, yt, posterior_densities, 2*T, 10)

    # print of first and last scores
    for key, val in net.scores.items():
        if len(val) > 0:
            logger.info(f"test - {key} at iteration 0 = {val[0]}")
            logger.info(f"test - {key} at last iteration = {val[-1]}")


experiment(deep=1, plot=True, sigma0=0.01)
experiment(deep=1, plot=True, sigma0=0.1)
experiment(deep=5, plot=True, sigma0=0.01)
experiment(deep=5, plot=True, sigma0=0.1)
experiment(deep=10, plot=False, sigma0=0.01)
experiment(deep=20, plot=False, sigma0=0.01)
experiment(deep=100, plot=False, sigma0=0.01)