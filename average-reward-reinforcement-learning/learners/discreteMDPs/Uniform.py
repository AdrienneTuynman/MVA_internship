"""Uniform sampling"""

import numpy as np

# from scipy.optimize import minimize_scalar

from learners.discreteMDPs.AgentInterface import Agent

# from learners.discreteMDPs.utils import randamax
# import learners.discreteMDPs.stopping_rule as stru


class Uniform(Agent):
    """Uniform selection of actions"""

    def __init__(
        self,
        nbr_states,
        nbr_actions,
        name="Uniform",
        max_iter=3000,
        epsilon=1e-3,
        max_reward=1,
        beta=0.5,
    ):
        """using the top two algorithm EB-TCI for approximate optimality

        Args:
            nbr_states (int):
            nbr_actions (int):
            name (str, optional): Defaults to "EBTCI".
            max_iter (int, optional): maximum number of iterations for...? Defaults to 3000.
            epsilon (float, optional): _description_. Defaults to 1e-3.
            max_reward (int, optional): _description_. Defaults to 1.
            beta (float, optional): proportion of times to choose the leader. Defaults to 0.5.
        """
        Agent.__init__(self, nbr_states, nbr_actions, name=name)
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.dirac = np.eye(self.nbr_states, dtype=int)
        self.actions = np.arange(self.nbr_actions, dtype=int)
        self.max_iteration = max_iter  # max nb of iteration in VI
        self.epsilon = epsilon
        self.max_reward = max_reward
        self.beta = beta

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
        na = self.state_action_pulls[state, action]
        ns = self.state_visits[state]
        mean_reward = self.rewards[state, action]
        p = self.transitions[state, action]

        self.state_action_pulls[state, action] = na + 1
        self.state_visits[state] = ns + 1
        self.rewards[state, action] = ((na + 1) * mean_reward + reward) / (na + 2)
        self.transitions[state, action] = ((na + 1) * p + self.dirac[observation]) / (
            na + 2
        )

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

    def play(self, state):
        """selects the action to take in a given state

        Args:
            state (_type_): initial state

        Returns:
            int: action to run
        """
        return np.random.randint(0, self.nbr_actions)
