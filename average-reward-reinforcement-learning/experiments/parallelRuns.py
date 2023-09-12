"""parallelizes the experiments"""
import copy
import multiprocessing
import time
from tqdm import tqdm
from joblib import Parallel, delayed

import environments.RegisterEnvironments as bW


## Parallelization
def multicoreRuns(
    envregistrname,
    learner,
    nbreplicates,
    timehorizon,
    delta,
    epsilon,
    testevery,
    printevery,
    verbose,
    onerunfunction,
):
    """parallelizes the execution of the experiment

    Args:
        envregistrname (str): name under which the environment is registered
        learner (Agent): algorithm
        nbreplicates (int): how many times to repeat each experience
        timehorizon (int)
        delta (float): confidence level
        epsilon (float): optimality of the policy we want
        testevery (int)
        printevery (int)
        verbose (bool)
        onerunfunction (function): function that implements one run of the algorithm

    Returns:
        dumpnames (list of str): list of the dump file names
        (float): time elapsed during the total run
    """
    num_cores = multiprocessing.cpu_count()
    envs = []
    learners = []
    timehorizons = []
    deltas = []
    epsilons = []
    printeverys = []
    testeverys = []
    verboses = []
    for i in range(nbreplicates):
        envs.append(bW.makeWorld(envregistrname))
        learners.append(copy.deepcopy(learner))
        timehorizons.append(copy.deepcopy(timehorizon))
        deltas.append(copy.deepcopy(delta))
        epsilons.append(copy.deepcopy(epsilon))
        printeverys.append(copy.deepcopy(printevery))
        testeverys.append(copy.deepcopy(testevery))
        verboses.append(verbose)

    tbeginning = time.time()

    dumpnames = Parallel(n_jobs=num_cores)(
        delayed(onerunfunction)(*i)
        for i in tqdm(
            list(
                zip(
                    envs,
                    learners,
                    epsilons,
                    deltas,
                    timehorizons,
                    testeverys,
                    printeverys,
                    verboses,
                )
            )
        )
    )
    elapsed = time.time() - tbeginning
    return dumpnames, elapsed / nbreplicates
