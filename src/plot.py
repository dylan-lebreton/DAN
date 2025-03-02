from typing import Union, List, Optional

import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.use('Qt5Agg')

def plot_pretrain_loss(loss):
    fig, axs = plt.subplots(1, 2)
    axs[0].plot(loss)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel(r"$\mathcal{L}_0\left(q_0^a\right)$")
    axs[0].set_title("Log scale")
    axs[1].plot(loss)
    axs[1].set_xlabel('Iteration')
    axs[1].set_title("Linear scale")
    plt.suptitle(r"Evolution of $\mathcal{L}_0\left(q_0^a\right)$ loss for procoder pre-train")
    plt.tight_layout()
    plt.show()


def plot_model_obs(xt, yt, T, batch_index: int = 0):
    # create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    t = []
    x_model = []
    y_model = []
    x_obs = []
    y_obs = []

    for i in range(T):
        t.append(i)

        model_tensor = xt[i].detach().numpy()[batch_index, :]
        x_model.append(model_tensor[0])
        y_model.append(model_tensor[1])

        obs_tensor = yt[i].detach().numpy()[batch_index, :]
        x_obs.append(obs_tensor[0])
        y_obs.append(obs_tensor[1])

    # plot the model and observations
    ax.plot(t, x_model, y_model, color="red", label='Model')
    ax.plot(t, x_obs, y_obs, color="blue", label='Observations')

    # set the labels for the axes
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('x')
    ax.set_zlabel('y')

    # add a legend
    ax.legend()

    # add title
    plt.title(f"Model and observations for batch example n°{batch_index}")

    # display the plot
    plt.tight_layout()
    plt.show()

def plot_final_mean_densities(xt, yt, densities, T, batch_index):
    # create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    t = []
    x_model = []
    y_model = []
    x_obs = []
    y_obs = []
    x_mean_density = []
    y_mean_density = []

    for i in range(T):
        t.append(i)

        model_tensor = xt[i].detach().numpy()[batch_index, :]
        x_model.append(model_tensor[0])
        y_model.append(model_tensor[1])

        obs_tensor = yt[i].detach().numpy()[batch_index, :]
        x_obs.append(obs_tensor[0])
        y_obs.append(obs_tensor[1])

        loc = densities[i].loc.detach().numpy()
        x_mean_density.append(loc[batch_index, 0])
        y_mean_density.append(loc[batch_index, 1])

    # plot the model and observations
    ax.plot(t, x_model, y_model, color="red", label='Model')
    ax.plot(t, x_obs, y_obs, color="blue", label='Observations')
    ax.plot(t, x_mean_density, y_mean_density, color="orange", label=r"mean of $q_t^a$")

    # set the labels for the axes
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('x')
    ax.set_zlabel('y')

    # add a legend
    ax.legend()

    # add title
    plt.title(f"Model, observations and posteriors means for batch example n°{batch_index}")

    # display the plot
    plt.tight_layout()
    plt.show()


def plot_score_with_log(score, T, score_label):
    # Create a figure with two subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    max_iter = int(len(score) / T)
    iter = np.repeat(np.arange(max_iter), T)
    t = np.tile(np.arange(T), max_iter)

    # plot the model and observations
    ax1.plot(iter, t, score)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Time (t)')
    ax1.set_zlabel(score_label)
    ax1.set_title("3D representation")

    ax2.plot(score)
    ax2.set_xlabel("Iteration x time")
    ax2.set_ylabel(score_label)
    ax2.set_title("2D representation (linear scale)")

    ax3.plot(score)
    ax3.set_xlabel("Iteration x time")
    ax3.set_ylabel(score_label)
    ax3.set_title("2D representation (log scale)")
    ax3.set_yscale("log")

    plt.suptitle(f"Evolution of {score_label} over time and iterations")

    # display the plot
    plt.tight_layout()
    plt.show()

def plot_score(score, T, score_label):
    # Create a figure with two subplots
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    max_iter = int(len(score) / T)
    iter = np.repeat(np.arange(max_iter), T)
    t = np.tile(np.arange(T), max_iter)

    # plot the model and observations
    ax1.plot(iter, t, score)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Time (t)')
    ax1.set_zlabel(score_label)
    ax1.set_title("3D representation")

    ax2.plot(score)
    ax2.set_xlabel("Iteration x time")
    ax2.set_ylabel(score_label)
    ax2.set_title("2D representation")

    plt.suptitle(f"Evolution of {score_label} over time and iterations")

    # display the plot
    plt.tight_layout()
    plt.show()

def plot_scores(net, T):
    RMSE_b = net.scores['RMSE_b']
    RMSE_a = net.scores['RMSE_a']
    LOGPDF_b = net.scores['LOGPDF_b']
    LOGPDF_a = net.scores['LOGPDF_a']
    LOSS = net.scores['LOSS']

    plot_score_with_log(LOSS, T, r"LOSS")
    plot_score_with_log(LOGPDF_a, T, r"LOGPDF_a")
    plot_score_with_log(LOGPDF_b, T, r"LOGPDF_b")
    plot_score(RMSE_a, T, r"RMSE a")
    plot_score(RMSE_b, T, r"RMSE b")



