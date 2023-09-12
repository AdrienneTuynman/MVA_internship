"""to run full experiments, with all agents, on one environment"""
import time

import experiments.analyzeRuns as aR
import experiments.oneRun_BPI as oR
import experiments.parallelRuns as pR
from numpy import array, log
from scipy.optimize import bisect
import os
import pickle as pickle

# import experiments.plotResults as plR
import learners.discreteMDPs.OptimalControl as opt
from experiments.utils import get_project_root_dir

ROOT = get_project_root_dir() + "/experiments/"


def runLargeMulticoreExperiment(
    env,
    agents,
    delta,
    epsilon,
    timehorizon=1000,
    nbreplicates=100,
    testevery=1000,
    printevery=None,
    verbose=False,
    compare=False,
):
    """runs the experiment several times for all agents

    Args:
        env (gym environment): environment
        agents (Agent): list of agents
        timehorizon (int, optional): number of timesteps. Defaults to 1000.
        nbreplicates (int, optional): number of times the exp is repeated. \
            Defaults to 100.
        testevery (int, optional): delay between two tests
        printevery (int,optional): delay between two prints
        verbose (boolean, optional): if we plot the stopping criterion curve
    """
    if printevery is None:
        printevery = timehorizon + 1
    envfullname = env.name
    numst = env.observation_space.n
    numa = env.action_space.n
    opti_learner = opt.build_opti(
        envfullname,
        env.env,
        numst,
        numa,
        epsilon=0.001 * epsilon,
    )
    learners = [x[0](**x[1]) for x in agents]

    print("*********************************************")
    dump_ts = []
    names = []
    meanelapsedtimes = []

    for learner in learners:
        names.append(learner.name())
        dump_t, meanelapsedtime = pR.multicoreRuns(
            envfullname,
            learner,
            nbreplicates,
            timehorizon,
            delta,
            epsilon,
            testevery,
            printevery,
            verbose,
            oR.onexp,
        )
        dump_ts.append(dump_t)
        meanelapsedtimes.append(meanelapsedtime)
        # dump_t = []
        # for file in os.listdir(ROOT + "results/"):
        #     if file.startswith(f"cumMeans_{env.name}_{learner.name()}_{timehorizon}"):
        #         dump_t.append(ROOT + "results/" + file)
        # dump_ts.append(dump_t)
        # print(dump_t)
    # print('************** ANALYSIS **************')
    limsup = None
    if compare:
        filename = ROOT + "results/envs/" + envfullname
        try:
            # raise FileNotFoundError
            with open(filename, "rb") as file:
                # print(file)
                (d1, A, B, b) = pickle.load(file)
                print(d1, A, B, b)
        except FileNotFoundError:
            d1, A, B, b = opti_learner.computes_environment_data()
            file = open(filename, "wb")
            file.truncate(0)  # empties
            pickle.dump((d1, A, B, b), file)
            file.close()

        D = A * (B + d1 / 2)
        # print(f"{A=}, {B=}, {D=}")
        C = min(
            d1**2 / (8 * D**2),
            1 / (12 * D),
            epsilon / (18 * (b + 2.5 * D) ** 2),
        )
        # limsup = (
        #     numst
        #     * (numst - 1)
        #     * numa
        #     * (1 + log(1 / delta) / (numst - 1) + 1 / C - log(1 + 1 / C))
        #     * (1 + C)
        #     / C
        # )
        y = 1 + log(1 / delta) / (numst - 1)
        yb = y + C - log(C)
        limsup = yb * (1 + yb / (yb - 1)) / C - 1
        print(f"{C=}, {d1=}")
        print(f"{limsup=}")
        limsup = bisect(
            lambda x: C * x - log(1 + x) - (1 + log(1 / delta) / (numst - 1)),
            1,
            limsup,
            maxiter=1000,
        )
        print(f"{limsup=}")
        raise ValueError
    timestamp = str(time.time())
    logfilename = ROOT + "results/logfile_" + env.name + "_" + timestamp + ".txt"
    logfile = open(logfilename, "w")
    logfile.write("Environment " + env.name + "\n")
    thresh = opti_learner.gain - epsilon
    logfile.write(
        f"Optimal gain is {opti_learner.gain}, minimum tolerance is {thresh}\n"
    )
    logfile.write("Learners " + str([learner.name() for learner in learners]) + "\n")
    logfile.write(
        "Time horizon is "
        + str(timehorizon)
        + ", nb of replicates is "
        + str(nbreplicates)
        + "\n \n"
    )
    mean, policies, badpolicies = aR.computestoptimes(
        names,
        dump_ts,
        timehorizon,
        envfullname + "_te=" + str(testevery),
        verbose,
        limsup,
    )
    for i in range(len(names)):
        pols = policies[i]
        badpols = badpolicies[i]
        gains = 0
        bgains = 0
        nbpols = 0
        accurate = 0
        for pol, size in pols.items():
            gain = opti_learner.compute_pol(array(pol))[0][0]
            # print(gain, size, pol)
            # print("\n")
            gains += gain * size
            nbpols += size
            if gain > thresh:
                accurate += size
        for pol, size in badpols.items():
            gain = opti_learner.compute_pol(array(pol))[0][0]
            bgains += size * gain
        logfile.write(str(names[i]))
        if nbpols == 0:
            logfile.write(" Nothing finished \n")
        else:
            logfile.write(
                f" {100*accurate/nbpols} % of {nbpols} finished policies were optimal and the average gain was {gains/nbpols} \n"
            )
        logfile.write(
            f"Average gain is {(bgains+gains)/nbreplicates}, runtime is {i}, nb of steps is {mean[i]} \n \n"
        )

    # oR.clear_auxiliaryfiles(env)
    print("\n[INFO] A log-file has been generated in ", logfilename)
