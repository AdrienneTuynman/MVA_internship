"""Analyzes and displays the results of the runs"""
import pickle
import time
from random import shuffle
import numpy as np
from matplotlib.pyplot import figure, boxplot, savefig, legend, title, plot, axhline
from matplotlib.colors import XKCD_COLORS, TABLEAU_COLORS
from experiments.utils import get_project_root_dir


ROOT = get_project_root_dir() + "/experiments/"


def computestoptimes(
    names, dump_cumulativerewards_, timehorizon, envname="", verbose=False, limsup=None
):
    """plots the stop times of all algorithms and, if verbose, the mean of I(t) for each algorithm

    Args:
        names (list of str): names of the algorithms
        dump_cumulativerewards_ (list of list of str): locations of the dumps
        timehorizon (int): maximal time the algorithm can run
        envname (str, optional): Defaults to "".
        verbose (bool, optional): Whether to plot I(t). Defaults to False.

    Returns:
        mean (list of float): mean of stop times for each algorithm
        policies (list of dict of tuple): for each algorithm, policies of the runs that stopped
        badpols (dict of tuples): for each algorithm, policies of the runs that didn't stop
    """

    mean = []

    nbalgs = len(dump_cumulativerewards_)

    colors = list(TABLEAU_COLORS.keys())
    if nbalgs > 10:
        colors = list(XKCD_COLORS.keys())
        shuffle(colors)

    figure()

    data, policies, badpols, names_allfinished = [], [], [], []
    # policies = []
    # badpols = []
    for j in range(nbalgs):
        allfinished = True
        epsemprs_j, data_j = [], []
        # data_j = []
        badpols_j, policies_j = {}, {}
        # policies_j = {}
        if verbose:
            plot([], [], color=colors[j], label=names[j])
        for i in range(32):
            with open(dump_cumulativerewards_[j][i], "rb") as file:
                cum_rewards_ij = pickle.load(file)
                data_j.append(cum_rewards_ij[0])
                pol = cum_rewards_ij[1]
                if cum_rewards_ij[0] < timehorizon - 1:
                    try:
                        policies_j[pol] += 1
                    except KeyError:
                        policies_j[pol] = 1
                else:
                    allfinished = False
                    try:
                        badpols_j[pol] += 1
                    except KeyError:
                        badpols_j[pol] = 1

                if verbose:
                    epsemprs_j.append(cum_rewards_ij[2][:, 0])
                    times = cum_rewards_ij[2][:, 1]
                # file_oracle.close()
            # file.close()
        if verbose:
            epsemprs_j = np.array(epsemprs_j)
            # print(epsemprs_j)
            epsemprs_j = np.mean(epsemprs_j, 0)
            indmin = np.sum(epsemprs_j == np.inf)
            plot(epsemprs_j[indmin:], times[indmin:], color=colors[j])
        if allfinished:
            names_allfinished.append(names[j])
            data.append(data_j)
        policies.append(policies_j)
        badpols.append(badpols_j)
        mean.append(np.mean(data_j, axis=0))
    if verbose:
        legend()
        title(envname + "_I")
        savefig(f"{ROOT}results/im{time.time()}_I.png")
    if len(data) > 0:
        figure()
        # print(len(data), len(names_allfinished))
        if not limsup is None:
            axhline(y=limsup, color="r", linestyle="dashed")
        boxplot(data, labels=names_allfinished)
        title(envname)
        savefig(f"{ROOT}results/im{time.time()}.png")
    return mean, policies, badpols
