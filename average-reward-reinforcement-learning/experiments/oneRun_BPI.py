"""for one run of one agent in the environment"""
import pickle
import time
import os
from numpy import array, inf

from experiments.utils import get_project_root_dir


ROOT = get_project_root_dir() + "/experiments/"


def onexp(env, learner, epsilon, delta, timehorizon, testevery, printevery, verbose):
    """performs one run until timehorizon and stores the results in a file

    Args:
        env (_type_): _description_
        learner (Agent): _description_
        epsilon (float): level of precision of the policy we want
        delta (float): confidence level
        timehorizon (int): longest time up to which run
        testevery (int): how often to test the stopping rule
        printevery (int): how often to print the timestep (in lieu of progress bar)
        verbose (bool): whether to go on until the end to plot I(timestep)

    Returns:
        str: name of the file in which the results are stored
    """
    observation = env.reset()
    learner.reset(observation)
    deltab = delta / (2)  # * learner.nbr_states * learner.nbr_actions)
    # cumreward = 0.0
    # cumrewards = []
    # cummean = 0.0
    # cummeans = []
    # print(
    #     "[Info] New initialization of ", learner.name(), " for environment ", env.name
    # )

    timestop = None
    flag = False
    epsemprs = []
    for timestep in range(timehorizon):
        if timestep % printevery == 0 and timestep > 0:
            print(timestep)
            # try:
            #     print(f"time save={learner.ts[1]/(learner.ts[0]+learner.ts[1])}")
            # except AttributeError:
            #     ()
            # print(learner.state_action_pulls)
            # except AttributeError:
            #     print(learner.Nk + learner.vk)
        state = observation
        action = learner.play(state)  # Get action
        observation, reward, _, _ = env.step(action)  # done, info
        learner.update(state, action, reward, observation)  # Update learners
        # print("info:",info, "reward:", reward)
        # cumreward += reward
        # try:
        #     cummean += info
        # except TypeError:
        #     cummean += reward
        # cumrewards.append(cumreward)
        # cummeans.append(cummean)
        if timestep % testevery == 0 and timestep > 0:
            flag, epsempr = learner.stop_test(epsilon, deltab)
            if verbose and epsempr is not None:
                epsemprs.append([timestep, epsempr])
            elif verbose:
                epsemprs.append([timestep, inf])

        # if done:
        #     print("Episode finished after {} timesteps".format(timestep + 1))
        #     observation = (
        #         env.reset()
        # )  # converts an episodic MDP into an infinite time horizon MDP
        # break
        if flag and timestop is None:
            timestop = timestep
            if not verbose:
                break

    if timestop is None:
        timestop = timehorizon
    filename = (
        ROOT
        + "results/cumMeans_"
        + env.name
        + "_"
        + learner.name()
        + "_"
        + str(timehorizon)
        + "_"
        + str(time.time())
    )
    file = open(filename, "wb")
    file.truncate(0)  # empties
    epsemprs = array(epsemprs)
    pickle.dump([timestop, learner.pol_final(), array(epsemprs)], file)
    file.close()
    return filename


def clear_auxiliaryfiles(env):
    """removes the auxiliary files created during the run"""
    print("non")
    # for file in os.listdir(ROOT + "results/"):
    #     if file.startswith("cumMeans_" + env.name):
    #         os.remove(ROOT + "results/" + file)
