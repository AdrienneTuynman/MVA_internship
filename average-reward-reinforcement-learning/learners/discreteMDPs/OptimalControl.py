"""used to compute the optimal gain AND the gain of any policy"""
from learners.discreteMDPs.utils import categorical_sample
from learners.discreteMDPs.AgentInterface import Agent

import numpy as np


def build_opti(name, env, nbr_states, nbr_actions, epsilon):
    """Looks outdated and unused"""
    # if ("2-room" in name):
    #    return  Opti_911_2room(env)
    # elif ("4-room" in name):
    #    return  Opti_77_4room(env)
    # elif ("RiverSwim" in name):
    #    return Opti_swimmer(env)
    # else:
    return OptiController(env, nbr_states, nbr_actions, epsilon)


class OptiController(Agent):
    """an agent who performs the optimal policy"""

    def __init__(
        self, env, nbr_states, nbr_actions, epsilon=0.001, max_iter=100000, name="Opti"
    ):
        """

        :param env:
        :param nbr_states:
        :param nbr_actions:
        :param epsilon: precision of VI stopping criterion
        :param max_iter: maximum iterations of VI
        """
        Agent.__init__(self, nbr_states, nbr_actions, name=name)
        self.env = env
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.phi = np.zeros(self.nbr_states)
        self.gain = 0
        self.epsilon = epsilon
        self.max_iter = max_iter

        self.not_converged = True
        self.transitions = np.zeros(
            (self.nbr_states, self.nbr_actions, self.nbr_states)
        )
        self.rewards = np.zeros((self.nbr_states, self.nbr_actions))
        self.policy = np.zeros((self.nbr_states, self.nbr_actions))

        try:
            for state in range(self.nbr_states):
                for action in range(self.nbr_actions):
                    self.transitions[state, action] = self.env.getTransition(
                        state, action
                    )
                    self.rewards[state, action] = self.env.getMeanReward(state, action)
                    self.policy[state, action] = 1.0 / self.nbr_actions

        except AttributeError:
            for state in range(self.nbr_states):
                for action in range(self.nbr_actions):
                    (
                        self.transitions[state, action],
                        self.rewards[state, action],
                    ) = self.extractRewardsAndTransitions(state, action)
                    self.policy[state, action] = 1.0 / self.nbr_actions
        self.value_iteration()
        # self.gain = self.phi
        max_values = -np.inf
        for action in self.skeleton[0]:
            psa = self.transitions[0, action]
            rsa = self.rewards[0, action]
            max_values = max(max_values, rsa + psa @ self.phi)
        self.gain = max_values - self.phi[0]

    def extractRewardsAndTransitions(self, state, action):
        """gets the transition probabilities and mean rewards from the environment

        Args:
            s (_type_): state
            a (_type_): action

        Returns:
            _type_: transition and mean reward
        """
        transition = self.env.getTransition(state, action)
        reward = self.env.getMeanReward(state, action)
        return transition, reward

    def name(self):
        """returns the name of the agent"""
        return "Optimal_controller"

    # def reset(self, inistate):
    #     ()

    def play(self, state):
        """selects an action to play in state according to the policy

        Args:
            state (_type_): _description_

        Returns:
            _type_: action
        """
        action = categorical_sample(
            [self.policy[state, a] for a in range(self.nbr_actions)], np.random
        )
        return action

    # def update(self, state, action, reward, observation):
    #     ()

    def compute_pol(self, pol):
        """computes the gain of a deterministic policy pol with precision epsilon

        Args:
            pol (nparray): deterministic policy to evaluate
            epsilon (float, optional): _description_. Defaults to 0.01.

        Returns:
            _type_: _description_
        """
        numst = self.nbr_states
        imp = np.eye(numst)
        meanr = np.zeros((numst))
        imp[:, 0] = np.ones(numst)
        # print(pol)
        # print(self.rewards)
        for state in range(numst):
            # print(state, pol[state])
            imp[state, 1:] = imp[state, 1:] - self.transitions[state, pol[state]][1:]
            meanr[state] = self.rewards[state, pol[state]]
        # gaba = np.linalg.lstsq(imp, meanr, rcond=None)
        # print(gaba[1])
        try:
            impm1 = np.linalg.inv(imp)
        except np.linalg.LinAlgError:
            print(imp)
            raise ValueError
        # print(imp, meanr, np.matmul(impm1, meanr))
        # raise ValueError
        return (np.matmul(impm1, meanr), impm1)

    def computes_environment_data(self):
        numst = self.nbr_states
        numa = self.nbr_actions
        pol = np.zeros(numst, dtype=int)
        A, B = -np.inf, -np.inf
        # A, B = [], []
        gmax = -np.inf
        gains = []
        compt = 0
        while True:
            compt += 1
            if compt % 10000 == 0:
                print(compt / (numa**numst))
            (gb, impm1) = self.compute_pol(pol)
            B = max(B, np.max(gb))
            # B.append(np.max(gb))
            if gb[0] > gmax:
                gmax = gb[0]
                b = np.max(gb[1:])
            A = max(A, np.linalg.norm(impm1, np.inf))
            # A.append(np.linalg.norm(impm1, np.inf))
            gains.append(gb[0])
            i = 0
            while pol[i] == numa - 1:
                if i == numst - 1:
                    i = None
                    break
                pol[i] = 0
                i += 1
            if i is None:
                break
            pol[i] += 1
        sort = np.array(gains)
        sort = np.sort(sort)
        g = sort[-1]
        sort = sort - g
        significant = np.sum(sort < -1e-8 * g)
        d1 = -sort[significant - 1]
        print(f"{d1=}")
        # e = max(d1 / 2, -sort - d1 / 2)
        # m = -sort - d1 / 2 >= d1 / 2
        # e = np.array(m * (-sort - d1 / 2) + (1 - m) * d1 / 2)
        # A = np.array(A)
        # print(
        #     np.shape(e),
        #     np.shape(A),
        #     np.shape(B),
        #     np.shape(B + e),
        #     np.shape(A * numst * (B + e)),
        # )
        # xi = np.min(e / (A * numst * (B + e)))
        # print(xi**2 / 2)
        # raise ValueError

        # gains.sort()
        # gains = np.array(gains)
        # gainsdiff = gains - gains[-1]
        # gainsdiff = np.sort(gainsdiff)
        # print(gainsdiff)
        # print(gains)
        # print(gainsdiff < 1e-8 * gains[-1])
        # significant = np.sum(gainsdiff < -1e-8 * gains[-1])
        # print(significant, gainsdiff[significant - 1], gainsdiff[significant + 1])
        print(f"{b=}")
        # d1 = -gainsdiff[significant - 1]
        return (d1, A, B, b)


# class Opti_swimmer:
#     def __init__(self, env):
#         self.env = env
#         self.policy = np.zeros(self.env.nbr_states)

#     def name(self):
#         return "Opti_swimmer"

#     def reset(self, inistate):
#         ()

#     def play(self, state):
#         return 0

#     def update(self, state, action, reward, observation):
#         ()


# class Opti_77_4room:
#     def __init__(self, env):
#         self.env = env
#         pol = [
#             [0, 0, 0, 0, 0, 0, 0],
#             [0, 1, 3, 3, 1, 1, 0],
#             [0, 1, 2, 0, 1, 2, 0],
#             [0, 1, 0, 0, 1, 0, 0],
#             [0, 3, 3, 3, 3, 1, 0],
#             [0, 3, 0, 0, 3, 1, 0],
#             [0, 0, 0, 0, 0, 0, 0],
#         ]
#         self.policy = np.zeros(49)
#         for x in range(7):
#             for y in range(7):
#                 self.policy[x * 7 + y] = pol[x][y]
#         self.mapping = env.mapping

#     def name(self):
#         return "Opti_77_4room"

#     def reset(self, inistate):
#         ()

#     def play(self, state):
#         s = self.mapping[state]
#         return self.policy[s]

#     def update(self, state, action, reward, observation):
#         ()


# class Opti_911_2room:
#     def __init__(self, env):
#         self.env = env
#         pol = [
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#             [0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 0],
#             [0, 3, 3, 3, 1, 1, 1, 2, 2, 2, 0],
#             [0, 3, 3, 3, 3, 1, 2, 2, 2, 2, 0],
#             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#             [0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0],
#             [0, 3, 3, 3, 3, 3, 3, 3, 1, 1, 0],
#             [0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0],
#             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#         ]
#         self.policy = np.zeros(9 * 11)
#         for x in range(9):
#             for y in range(11):
#                 self.policy[x * 11 + y] = pol[x][y]
#         self.mapping = env.mapping

#     def name(self):
#         return "Opti_911_2room"

#     def reset(self, inistate):
#         ()

#     def play(self, state):
#         s = self.mapping[state]
#         return self.policy[s]

#     def update(self, state, action, reward, observation):
#         ()
