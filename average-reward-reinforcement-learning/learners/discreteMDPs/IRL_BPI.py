"""Adding a stopping rule to the Imed-RL agent"""
import numpy as np
from scipy.optimize import minimize_scalar

from learners.discreteMDPs.AgentInterface import Agent

from learners.discreteMDPs.utils import randamin

# import learners.discreteMDPs.stopping_rule as stru


class IRL_BPI(Agent):
    """an agent that performs BPI with the IMED-RL sampling rule"""

    def __init__(
        self,
        nbr_states,
        nbr_actions,
        name="IMED-RL-BPI",
        max_iter=3000,
        epsilon=1e-3,
        max_reward=1,
    ):
        Agent.__init__(self, nbr_states, nbr_actions, name=name)
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.dirac = np.eye(self.nbr_states, dtype=int)
        self.actions = np.arange(self.nbr_actions, dtype=int)
        self.max_iteration = max_iter  # max nb of iteration in VI
        self.epsilon = epsilon
        self.max_reward = max_reward

        # Empirical estimates
        self.state_action_pulls = np.zeros(
            (self.nbr_states, self.nbr_actions), dtype=int
        )
        self.state_visits = np.zeros(self.nbr_states, dtype=int)
        self.rewards = np.zeros((self.nbr_states, self.nbr_actions)) + 0.5
        self.transitions = (
            np.ones((self.nbr_states, self.nbr_actions, self.nbr_states))
            / self.nbr_states
        )
        self.all_selected = np.zeros(self.nbr_states, dtype=bool)
        self.phi = np.zeros(self.nbr_states)
        self.skeleton = {
            s: np.arange(self.nbr_actions, dtype=int) for s in range(self.nbr_states)
        }
        self.index = np.zeros(self.nbr_actions)
        self.rewards_distributions = {
            s: {a: {} for a in range(self.nbr_actions)} for s in range(self.nbr_states)
        }

        self.s = None

    def reset(self, state):
        """resets the agent's knowledge and goes to state s"""
        self.state_action_pulls = np.zeros(
            (self.nbr_states, self.nbr_actions), dtype=int
        )
        self.state_visits = np.zeros(self.nbr_states, dtype=int)
        self.rewards = np.zeros((self.nbr_states, self.nbr_actions)) + 0.5
        self.transitions = (
            np.ones((self.nbr_states, self.nbr_actions, self.nbr_states))
            / self.nbr_states
        )
        self.all_selected = np.zeros(self.nbr_states, dtype=bool)
        self.phi = np.zeros(self.nbr_states)
        self.skeleton = {
            s: np.arange(self.nbr_actions, dtype=int) for s in range(self.nbr_states)
        }
        self.index = np.zeros(self.nbr_actions)
        self.rewards_distributions = {
            s: {a: {1: 0, 0.5: 1} for a in range(self.nbr_actions)}
            for s in range(self.nbr_states)
        }

        self.s = state

    def update(self, state, action, reward, observation):
        """updates the agent self after a step

        Args:
            state (_type_): the initial state
            action (_type_): the action taken
            reward (_type_): the observed reward
            observation (_type_): the state after the step
        """
        pulls_sa = self.state_action_pulls[state, action]
        pulls_s = self.state_visits[state]
        mean_reward = self.rewards[state, action]
        trans_probas = self.transitions[state, action]

        self.state_action_pulls[state, action] = pulls_sa + 1
        self.state_visits[state] = pulls_s + 1
        self.rewards[state, action] = ((pulls_sa + 1) * mean_reward + reward) / (
            pulls_sa + 2
        )
        self.transitions[state, action] = (
            (pulls_sa + 1) * trans_probas + self.dirac[observation]
        ) / (pulls_sa + 2)

        if reward in self.rewards_distributions[state][action].keys():
            self.rewards_distributions[state][action][reward] += 1
        else:
            self.rewards_distributions[state][action][reward] = 1

        max_na = np.max(self.state_action_pulls[state])
        mask = self.state_action_pulls[state] >= np.log(max_na) ** 2
        self.skeleton[state] = self.actions[mask]

        self.s = observation

        if not self.all_selected[state]:
            self.all_selected[state] = np.all(self.state_action_pulls[state] > 0)

    def multinomial_imed(self, state):
        """Generates the index H_sa for s=state and for every action, and stores it in self.index

        Args:
            state (_type_): the current state, in which the index is calculated
        """
        upper_bound = self.max_reward + np.max(self.phi)
        values_sa = self.rewards[state] + self.transitions[state] @ self.phi
        # the varphi for each state
        gamma = np.max(values_sa)  # gamma
        upper_nu = upper_bound / (upper_bound - gamma) - 1e-2  # upper bound for nu

        for action in range(self.nbr_actions):
            if values_sa[action] >= gamma:  # if action in argmax
                self.index[action] = np.log(self.state_action_pulls[state, action])
            else:
                r_d = self.rewards_distributions[state][action]
                reward_vals = np.fromiter(r_d.keys(), dtype=float)  # values reward
                reward_probs = np.fromiter(r_d.values(), dtype=float)  # probas reward
                reward_probs = reward_probs / reward_probs.sum()

                pt = self.transitions[state][action]

                trans_probas = np.zeros(
                    len(reward_probs) * self.nbr_states
                )  # nR x nbr_states
                v = np.zeros(len(reward_probs) * self.nbr_states)
                k = 0
                for i in range(self.nbr_states):
                    for j in range(len(reward_probs)):
                        trans_probas[k] = pt[i] * reward_probs[j]
                        v[k] = self.phi[i] + reward_vals[j]
                        k += 1

                delta = v - gamma

                res = minimize_scalar(
                    lambda x, trans_probas, delta: -np.sum(
                        trans_probas * np.log(upper_bound - delta * x)
                    ),
                    bounds=(0, upper_nu),
                    method="bounded",
                    args=(trans_probas, delta),
                )
                k_inf = -res.fun  # L_max
                n_pulls_sa = self.state_action_pulls[state, action]
                self.index[action] = n_pulls_sa * k_inf + np.log(n_pulls_sa)

    def play(self, state):
        """selects the action to take in a given state

        Args:
            state (_type_): initial state

        Returns:
            _type_: action to run
        """
        if not self.all_selected[state]:  # taking the log of 0 is not optimal
            action = randamin(self.state_action_pulls[state])
        else:
            self.value_iteration()
            self.multinomial_imed(state)
            action = randamin(self.index)
        return action
