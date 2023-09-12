"""constructs the general class for BPI agents"""
import numpy as np


import learners.discreteMDPs.stopping_rule as stru
from learners.discreteMDPs.utils import randamax


class Agent:
    """BPI agent"""

    def __init__(self, nbr_states, nbr_actions, name="Agent"):
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.agentname = name
        self.all_selected = np.zeros(self.nbr_states, dtype=bool)
        self.phi = np.zeros(self.nbr_states)
        self.rewards_distributions = {
            s: {a: {} for a in range(self.nbr_actions)} for s in range(self.nbr_states)
        }
        self.transitions = (
            np.ones((self.nbr_states, self.nbr_actions, self.nbr_states))
            / self.nbr_states
        )
        self.state_action_pulls = np.zeros(
            (self.nbr_states, self.nbr_actions), dtype=int
        )
        self.rewards = self.rewards = np.zeros((self.nbr_states, self.nbr_actions))
        self.skeleton = {
            s: np.arange(self.nbr_actions, dtype=int) for s in range(self.nbr_states)
        }
        self.max_iteration = 3000
        self.epsilon = 1e-3

    def name(self):
        return self.agentname

    def reset(self, state):
        ()

    def play(self, state):
        """selects the action to play"""
        return np.random.randint(self.nbr_actions)

    def update(self, state, action, reward, observation):
        ()

    def value_iteration(self):
        """computes the bias with precision self.epsilon and stores it in self.phi"""
        ctr = 0
        stop = False
        phi = np.copy(self.phi)
        phip = np.copy(self.phi)
        while not stop:
            ctr += 1
            for state in range(self.nbr_states):
                max_value = -np.inf
                for action in self.skeleton[state]:
                    psa = self.transitions[state, action]
                    rsa = self.rewards[state, action]
                    max_value = max(max_value, rsa + psa @ phi)
                phip[state] = max_value
            phip = phip - np.min(phip)
            delta = np.max(np.abs(phi - phip))
            phi = np.copy(phip)
            stop = (delta < self.epsilon) or (ctr >= self.max_iteration)
        self.phi = np.copy(phi)

    def stop_test(self, epsilon, delta):
        """stopping rule

        Args:
            epsilon (float): level of optimality desired
            delta (float): confidence level for each state action pair

        Returns:
            (bool): whether the stopping condition is met
            (float,optional): the value of I(t) if not infinite
        """
        for state in range(self.nbr_states):
            if not self.all_selected[state]:
                return (False, None)
        self.value_iteration()
        bias = self.phi
        max_s = -np.inf
        min_s = np.inf
        max_s_debug = -np.inf
        min_s_debug = np.inf
        for state in range(self.nbr_states):
            max_debug = -np.inf
            max_au = -np.inf
            max_al = -np.inf
            for action in range(self.nbr_actions):
                r_d = self.rewards_distributions[state][action]
                valr = np.fromiter(r_d.keys(), dtype=float)  # values reward
                probr = np.fromiter(r_d.values(), dtype=float)  # probas reward
                probr = probr / probr.sum()
                rsa = np.sum(probr * valr)  # mean reward
                probp = self.transitions[state][action]
                max_debug = max(max_debug, rsa + probp.dot(bias))
                nsamples = self.state_action_pulls[state, action]
                usa = stru.upper_sa(
                    delta, probp, bias, nsamples, self.nbr_states, epsilon
                )
                lsa = stru.lower_sa(
                    delta, probp, bias, nsamples, self.nbr_states, epsilon
                )
                max_au = max(max_au, usa + rsa)
                if max_au == np.inf:
                    # print("max=infty")
                    return (False, None)
                max_al = max(max_al, lsa + rsa)
            max_s = max(max_s, max_au - bias[state])
            min_s = min(min_s, max_al - bias[state])
            max_s_debug = max(max_s_debug, max_debug - bias[state])
            min_s_debug = min(min_s_debug, max_debug - bias[state])
            if min_s == -np.inf:
                print("min=-infty")
                return (False, None)
        if max_s == -np.inf or min_s == np.inf:  # possible? yes
            return (False, None)
        # print(
        # f"epsempr={np.abs(max_s - min_s)}, real is {np.abs(max_s_debug-min_s_debug)}"
        # )
        return (np.abs(max_s - min_s) <= epsilon, np.abs(max_s - min_s))

    def pol_final(self):
        """gives the final deterministic policy maximizing rewards"""
        for state in range(self.nbr_states):
            if not self.all_selected[state]:
                return tuple([0] * self.nbr_states)
        self.value_iteration()
        pol = []
        for state in range(self.nbr_states):
            pol.append(
                randamax(self.rewards[state] + self.transitions[state] @ self.phi)
            )
        return tuple(pol)
