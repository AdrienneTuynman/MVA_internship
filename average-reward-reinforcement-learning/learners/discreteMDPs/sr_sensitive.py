"""An algorithm that aims to reach the stopping rule asap"""

import numpy as np

from learners.discreteMDPs.AgentInterface import Agent
from learners.discreteMDPs.utils import randamax, randamin
import learners.discreteMDPs.stopping_rule as stru


class Srs1(Agent):
    """aims to reach the stopping rule asap"""

    def __init__(
        self,
        nbr_states,
        nbr_actions,
        name="Srs",
        max_iter=3000,
        epsilon=1e-3,
        delta=0.05,
        epsstop=1,
    ):
        Agent.__init__(self, nbr_states, nbr_actions, name)
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.dirac = np.eye(self.nbr_states, dtype=int)
        self.epsilon = epsstop * 1e-2
        self.max_iteration = max_iter
        self.delta = delta / (2)  # * nbr_states * nbr_actions)
        self.epsstop = epsstop
        self.actions = np.arange(self.nbr_actions, dtype=int)

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
        self.phi = np.zeros(self.nbr_states)  # bias
        self.skeleton = {
            s: np.arange(self.nbr_actions, dtype=int) for s in range(self.nbr_states)
        }
        self.index = np.inf * np.ones((2, self.nbr_states, self.nbr_actions))
        self.index[1] = -self.index[1]
        self.s = None

    def reset(self, state):
        ()

    # def value_iteration(self):
    #     """computes the bias and stores it in self.phi"""
    #     ctr = 0
    #     stop = False
    #     phi = np.copy(self.phi)
    #     phip = np.copy(self.phi)
    #     while not stop:
    #         ctr += 1
    #         for state in range(self.nbr_states):
    #             u = -np.inf
    #             for action in self.skeleton[state]:
    #                 psa = self.transitions[state, action]
    #                 rsa = self.rewards[state, action]
    #                 u = max(u, rsa + psa @ phi)
    #             phip[state] = u
    #         phip = phip - np.min(phip)
    #         delta = np.max(np.abs(phi - phip))
    #         phi = np.copy(phip)
    #         stop = (delta < self.epsilon) or (ctr >= self.max_iteration)
    #     self.phi = np.copy(phi)

    def update(self, state, action, reward, observation):
        """updates the agent after one step

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            observation (_type_): _description_
        """
        na = self.state_action_pulls[state, action]
        ns = self.state_visits[state]
        r = self.rewards[state, action]
        p = self.transitions[state, action]

        phi = np.copy(self.phi)
        u = -np.inf
        amax = -1
        for act in self.skeleton[state]:
            psa = self.transitions[state, act]
            rsa = self.rewards[state, act]
            u2 = rsa + psa @ phi
            if u2 > u:
                u = u2
                amax = act

        self.state_action_pulls[state, action] = na + 1
        self.state_visits[state] = ns + 1
        self.rewards[state, action] = ((na + 1) * r + reward) / (na + 2)
        self.transitions[state, action] = ((na + 1) * p + self.dirac[observation]) / (
            na + 2
        )
        max_na = np.max(self.state_action_pulls[state])
        mask = self.state_action_pulls[state] >= np.log(max_na) ** 2
        self.skeleton[state] = self.actions[mask]
        psa = self.transitions[state, action]
        rsa = self.rewards[state, action]
        un = max(u, rsa + psa @ phi)
        if (  # the bias changed
            action == amax
            or (self.state_action_pulls[state, amax] < np.log(max_na) ** 2)
            or ((na + 1 >= np.log(max_na) ** 2) and un > u)
        ):
            self.value_iteration()
        if not self.all_selected[state]:
            self.all_selected[state] = np.all(self.state_action_pulls[state] > 0)

        self.s = observation

    def update_index(self, state, action):
        """Updates the index in one state and action pair

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """
        bias = self.phi
        rsa = self.rewards[state, action]
        probp = self.transitions[state][action]
        nsamples = self.state_action_pulls[state, action]
        usa = stru.upper_sa(
            self.delta, probp, bias, nsamples, self.nbr_states, self.epsstop
        )
        lsa = stru.lower_sa(
            self.delta, probp, bias, nsamples, self.nbr_states, self.epsstop
        )
        self.index[0, state, action] = rsa + usa - bias[state]
        self.index[1, state, action] = rsa + lsa - bias[state]

    def play(self, state):
        if not self.all_selected[state]:
            return randamin(self.state_action_pulls[state])
        for action in range(self.nbr_actions):
            self.update_index(state, action)
        # if np.max(self.state_action_pulls[state]) >= 100:
        #     print(self.index[0, state])
        #     raise ValueError
        unif = np.random.random()
        if unif > 0.5:
            return randamax(self.index[0, state])
        return randamax(self.index[1, state])

    # def pol_final(self):
    #     """gives the final deterministic policy maximizing rewards"""
    #     self.value_iteration()
    #     pol = []
    #     for state in range(self.nbr_states):
    #         pol.append(
    #             randamax(self.rewards[state] + self.transitions[state] @ self.phi)
    #         )
    #     return tuple(pol)

    def stop_test(self, epsilon, delta):
        """determines whether to stop learning

        Args:
            epsilon (_type_): _description_
            delta (_type_): _description_
        """
        if epsilon != self.epsstop or delta != self.delta:
            print(f"{delta=}, {self.delta=},{self.epsstop=},{epsilon=}")
            raise ValueError
        for state in range(self.nbr_states):
            if not self.all_selected[state]:
                return (False, None)
        max_s = -np.inf
        min_s = np.inf
        for state in range(self.nbr_states):
            for action in range(self.nbr_actions):
                self.update_index(state, action)
            max_au = self.index[0, state, randamax(self.index[0, state])]
            if max_au == np.inf:
                return (False, None)
            max_al = self.index[1, state, randamax(self.index[1, state])]
            max_s = max(max_s, max_au)
            min_s = min(min_s, max_al)
            if min_s == -np.inf:
                return (False, None)
        # print(f"epsempr={np.abs(max_s-min_s)}")
        return (np.abs(max_s - min_s) <= epsilon, np.abs(max_s - min_s))


class Srs2(Agent):
    """aims to reach the stopping rule asap"""

    def __init__(
        self,
        nbr_states,
        nbr_actions,
        name="Srs",
        max_iter=3000,
        epsilon=1e-3,
        delta=0.05,
        epsstop=1,
    ):
        Agent.__init__(self, nbr_states, nbr_actions, name)
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.dirac = np.eye(self.nbr_states, dtype=int)
        self.epsilon = epsstop * 1e-2
        self.max_iteration = max_iter
        self.delta = delta / (2)  # * nbr_states * nbr_actions)
        self.epsstop = epsstop
        self.actions = np.arange(self.nbr_actions, dtype=int)

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
        self.phi = np.zeros(self.nbr_states)  # bias
        self.skeleton = {
            s: np.arange(self.nbr_actions, dtype=int) for s in range(self.nbr_states)
        }
        self.index = np.inf * np.ones((2, self.nbr_states, self.nbr_actions))
        self.index[1] = -self.index[1]
        self.s = None

    def reset(self, state):
        ()

    # def value_iteration(self):
    #     """computes the bias and stores it in self.phi"""
    #     ctr = 0
    #     stop = False
    #     phi = np.copy(self.phi)
    #     phip = np.copy(self.phi)
    #     while not stop:
    #         ctr += 1
    #         for state in range(self.nbr_states):
    #             u = -np.inf
    #             for action in self.skeleton[state]:
    #                 psa = self.transitions[state, action]
    #                 rsa = self.rewards[state, action]
    #                 u = max(u, rsa + psa @ phi)
    #             phip[state] = u
    #         phip = phip - np.min(phip)
    #         delta = np.max(np.abs(phi - phip))
    #         phi = np.copy(phip)
    #         stop = (delta < self.epsilon) or (ctr >= self.max_iteration)
    #     self.phi = np.copy(phi)

    def update(self, state, action, reward, observation):
        """updates the agent after one step

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            observation (_type_): _description_
        """
        na = self.state_action_pulls[state, action]
        ns = self.state_visits[state]
        r = self.rewards[state, action]
        p = self.transitions[state, action]

        phi = np.copy(self.phi)
        u = -np.inf
        amax = -1
        for act in self.skeleton[state]:
            psa = self.transitions[state, act]
            rsa = self.rewards[state, act]
            u2 = rsa + psa @ phi
            if u2 > u:
                u = u2
                amax = act

        self.state_action_pulls[state, action] = na + 1
        self.state_visits[state] = ns + 1
        self.rewards[state, action] = ((na + 1) * r + reward) / (na + 2)
        self.transitions[state, action] = ((na + 1) * p + self.dirac[observation]) / (
            na + 2
        )
        max_na = np.max(self.state_action_pulls[state])
        mask = self.state_action_pulls[state] >= np.log(max_na) ** 2
        self.skeleton[state] = self.actions[mask]
        psa = self.transitions[state, action]
        rsa = self.rewards[state, action]
        un = max(u, rsa + psa @ phi)
        if (  # the bias changed
            action == amax
            or (self.state_action_pulls[state, amax] < np.log(max_na) ** 2)
            or ((na + 1 >= np.log(max_na) ** 2) and un > u)
        ):
            self.value_iteration()
        if not self.all_selected[state]:
            self.all_selected[state] = np.all(self.state_action_pulls[state] > 0)

        self.s = observation

    def update_index(self, state, action):
        """Updates the index in one state and action pair

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """
        bias = self.phi
        rsa = self.rewards[state, action]
        probp = self.transitions[state][action]
        nsamples = self.state_action_pulls[state, action]
        usa = stru.upper_sa(
            self.delta, probp, bias, nsamples, self.nbr_states, self.epsstop
        )
        lsa = stru.lower_sa(
            self.delta, probp, bias, nsamples, self.nbr_states, self.epsstop
        )
        self.index[0, state, action] = rsa + usa - bias[state]
        self.index[1, state, action] = rsa + lsa - bias[state]

    def play(self, state):
        if not self.all_selected[state]:
            return randamin(self.state_action_pulls[state])
        for action in range(self.nbr_actions):
            self.update_index(state, action)
        # if np.max(self.state_action_pulls[state]) >= 100:
        #     print(self.index[0, state])
        #     raise ValueError
        return randamax(self.index[0, state])

    # def pol_final(self):
    #     """gives the final deterministic policy maximizing rewards"""
    #     self.value_iteration()
    #     pol = []
    #     for state in range(self.nbr_states):
    #         pol.append(
    #             randamax(self.rewards[state] + self.transitions[state] @ self.phi)
    #         )
    #     return tuple(pol)

    def stop_test(self, epsilon, delta):
        """determines whether to stop learning

        Args:
            epsilon (_type_): _description_
            delta (_type_): _description_
        """
        if epsilon != self.epsstop or delta != self.delta:
            print(f"{delta=}, {self.delta=},{self.epsstop=},{epsilon=}")
            raise ValueError
        for state in range(self.nbr_states):
            if not self.all_selected[state]:
                return (False, None)
        max_s = -np.inf
        min_s = np.inf
        for state in range(self.nbr_states):
            for action in range(self.nbr_actions):
                self.update_index(state, action)
            max_au = self.index[0, state, randamax(self.index[0, state])]
            if max_au == np.inf:
                return (False, None)
            max_al = self.index[1, state, randamax(self.index[1, state])]
            max_s = max(max_s, max_au)
            min_s = min(min_s, max_al)
            if min_s == -np.inf:
                return (False, None)
        # print(f"epsempr={np.abs(max_s-min_s)}")
        return (np.abs(max_s - min_s) <= epsilon, np.abs(max_s - min_s))


###########################################


class Srs2tt(Agent):
    """aims to reach the stopping rule asap, top two version"""

    def __init__(
        self,
        nbr_states,
        nbr_actions,
        name="Srstt",
        max_iter=3000,
        epsilon=1e-3,
        delta=0.05,
        epsstop=1,
    ):
        Agent.__init__(self, nbr_states, nbr_actions, name)
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.dirac = np.eye(self.nbr_states, dtype=int)
        self.epsilon = epsstop * 1e-2
        self.max_iteration = max_iter
        self.delta = delta / (2)  # * nbr_states * nbr_actions)
        self.epsstop = epsstop
        self.actions = np.arange(self.nbr_actions, dtype=int)

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
        self.phi = np.zeros(self.nbr_states)  # bias
        self.skeleton = {
            s: np.arange(self.nbr_actions, dtype=int) for s in range(self.nbr_states)
        }
        self.index = np.inf * np.ones((2, self.nbr_states, self.nbr_actions))
        self.index[1] = -self.index[1]
        self.s = None

    def reset(self, state):
        ()

    def update(self, state, action, reward, observation):
        """updates the agent after one step

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            observation (_type_): _description_
        """
        na = self.state_action_pulls[state, action]
        ns = self.state_visits[state]
        r = self.rewards[state, action]
        p = self.transitions[state, action]

        phi = np.copy(self.phi)
        u = -np.inf
        amax = -1
        for act in self.skeleton[state]:
            psa = self.transitions[state, act]
            rsa = self.rewards[state, act]
            u2 = rsa + psa @ phi
            if u2 > u:
                u = u2
                amax = act

        self.state_action_pulls[state, action] = na + 1
        self.state_visits[state] = ns + 1
        self.rewards[state, action] = ((na + 1) * r + reward) / (na + 2)
        self.transitions[state, action] = ((na + 1) * p + self.dirac[observation]) / (
            na + 2
        )
        max_na = np.max(self.state_action_pulls[state])
        mask = self.state_action_pulls[state] >= np.log(max_na) ** 2
        self.skeleton[state] = self.actions[mask]
        psa = self.transitions[state, action]
        rsa = self.rewards[state, action]
        un = max(u, rsa + psa @ phi)
        if (  # the bias changed
            action == amax
            or (self.state_action_pulls[state, amax] < np.log(max_na) ** 2)
            or ((na + 1 >= np.log(max_na) ** 2) and un > u)
        ):
            self.value_iteration()
        if not self.all_selected[state]:
            self.all_selected[state] = np.all(self.state_action_pulls[state] > 0)

        self.s = observation

    def update_index(self, state, action):
        """Updates the index in one state and action pair

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """
        bias = self.phi
        rsa = self.rewards[state, action]
        probp = self.transitions[state][action]
        nsamples = self.state_action_pulls[state, action]
        usa = stru.upper_sa(
            self.delta, probp, bias, nsamples, self.nbr_states, self.epsstop
        )
        lsa = stru.lower_sa(
            self.delta, probp, bias, nsamples, self.nbr_states, self.epsstop
        )
        self.index[0, state, action] = rsa + usa - bias[state]
        self.index[1, state, action] = rsa + lsa - bias[state]

    def play(self, state):
        """plays the optimal action with probability 0.5, \
            and the action with highest upper index otherwise"""
        if not self.all_selected[state]:
            return randamin(self.state_action_pulls[state])
        unif = np.random.random()
        act = randamax(self.rewards[state] + self.transitions[state] @ self.phi)
        if unif > 0.5:  # replace with beta if necessary
            return act

        for action in range(self.nbr_actions):
            self.update_index(state, action)
        ind = np.copy(self.index[0, state])
        if unif > 0.75:
            ind = np.copy(self.index[1, state])
        ind[act] = -np.inf
        return randamax(ind)

    # def pol_final(self):
    #     """gives the final deterministic policy maximizing rewards"""
    #     self.value_iteration()
    #     pol = []
    #     for state in range(self.nbr_states):
    #         pol.append(
    #             randamax(self.rewards[state] + self.transitions[state] @ self.phi)
    #         )
    #     return tuple(pol)

    def stop_test(self, epsilon, delta):
        """determines whether to stop learning

        Args:
            epsilon (_type_): _description_
            delta (_type_): _description_
        """
        if epsilon != self.epsstop or delta != self.delta:
            print(f"{delta=}, {self.delta=},{self.epsstop=},{epsilon=}")
            raise ValueError
        for state in range(self.nbr_states):
            if not self.all_selected[state]:
                return (False, None)
        max_s = -np.inf
        min_s = np.inf
        for state in range(self.nbr_states):
            for action in range(self.nbr_actions):
                self.update_index(state, action)
            max_au = self.index[0, state, randamax(self.index[0, state])]
            if max_au == np.inf:
                return (False, None)
            max_al = self.index[1, state, randamax(self.index[1, state])]
            max_s = max(max_s, max_au)
            min_s = min(min_s, max_al)
            if min_s == -np.inf:
                return (False, None)
        # print(f"epsempr={np.abs(max_s-min_s)}")
        return (np.abs(max_s - min_s) <= epsilon, np.abs(max_s - min_s))


################################################


class Armdpgape(Agent):
    """aims to reach the stopping rule asap, mimicking the mdpgape method"""

    def __init__(
        self,
        nbr_states,
        nbr_actions,
        name="ARMDPGapE",
        max_iter=3000,
        epsilon=1e-3,
        delta=0.05,
        epsstop=1,
    ):
        Agent.__init__(self, nbr_states, nbr_actions, name)
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.dirac = np.eye(self.nbr_states, dtype=int)
        self.epsilon = epsstop * 1e-2
        self.max_iteration = max_iter
        self.delta = delta / (2)  # * nbr_states * nbr_actions)
        self.epsstop = epsstop
        self.actions = np.arange(self.nbr_actions, dtype=int)

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
        self.phi = np.zeros(self.nbr_states)  # bias
        self.skeleton = {
            s: np.arange(self.nbr_actions, dtype=int) for s in range(self.nbr_states)
        }
        self.index = np.inf * np.ones((2, self.nbr_states, self.nbr_actions))
        self.index[1] = -self.index[1]
        self.s = None

    # def reset(self, state):
    #     ()

    # def value_iteration(self):
    #     """computes the bias and stores it in self.phi"""
    #     ctr = 0
    #     stop = False
    #     phi = np.copy(self.phi)
    #     phip = np.copy(self.phi)
    #     while not stop:
    #         ctr += 1
    #         for state in range(self.nbr_states):
    #             u = -np.inf
    #             for action in self.skeleton[state]:
    #                 psa = self.transitions[state, action]
    #                 rsa = self.rewards[state, action]
    #                 u = max(u, rsa + psa @ phi)
    #             phip[state] = u
    #         phip = phip - np.min(phip)
    #         delta = np.max(np.abs(phi - phip))
    #         phi = np.copy(phip)
    #         stop = (delta < self.epsilon) or (ctr >= self.max_iteration)
    #     self.phi = np.copy(phi)

    def update(self, state, action, reward, observation):
        """updates the agent after one step

        Args:
            state (_type_): _description_
            action (_type_): _description_
            reward (_type_): _description_
            observation (_type_): _description_
        """
        na = self.state_action_pulls[state, action]
        ns = self.state_visits[state]
        r = self.rewards[state, action]
        p = self.transitions[state, action]

        phi = np.copy(self.phi)
        max_values_s = -np.inf
        amax = -1
        for act in self.skeleton[state]:
            psa = self.transitions[state, act]
            rsa = self.rewards[state, act]
            u2 = rsa + psa @ phi
            if u2 > max_values_s:
                max_values_s = u2
                amax = act

        self.state_action_pulls[state, action] = na + 1
        self.state_visits[state] = ns + 1
        self.rewards[state, action] = ((na + 1) * r + reward) / (na + 2)
        self.transitions[state, action] = ((na + 1) * p + self.dirac[observation]) / (
            na + 2
        )
        max_na = np.max(self.state_action_pulls[state])
        mask = self.state_action_pulls[state] >= np.log(max_na) ** 2
        self.skeleton[state] = self.actions[mask]
        psa = self.transitions[state, action]
        rsa = self.rewards[state, action]
        new_max_values_s = max(max_values_s, rsa + psa @ phi)
        if (  # the bias changed
            action == amax
            or (self.state_action_pulls[state, amax] < np.log(max_na) ** 2)
            or ((na + 1 >= np.log(max_na) ** 2) and new_max_values_s > max_values_s)
        ):
            self.value_iteration()
        if not self.all_selected[state]:
            self.all_selected[state] = np.all(self.state_action_pulls[state] > 0)

        self.s = observation

    def update_index(self, state, action):
        """Updates the index in one state and action pair

        Args:
            state (_type_): _description_
            action (_type_): _description_
        """
        bias = self.phi
        rsa = self.rewards[state, action]
        probp = self.transitions[state][action]
        nsamples = self.state_action_pulls[state, action]
        usa = stru.upper_sa(
            self.delta, probp, bias, nsamples, self.nbr_states, self.epsstop
        )
        lsa = stru.lower_sa(
            self.delta, probp, bias, nsamples, self.nbr_states, self.epsstop
        )
        self.index[0, state, action] = rsa + usa - bias[state]
        self.index[1, state, action] = rsa + lsa - bias[state]

    def play(self, state):
        """mimicks the mdpgape method"""
        if not self.all_selected[state]:
            return randamin(self.state_action_pulls[state])
        for action in range(self.nbr_actions):
            self.update_index(state, action)
        ind = np.copy(self.index[:, state])
        a = np.argmax(ind[0])
        maxs = ind[0, a] * np.ones(self.nbr_actions)
        ind[0, a] = -np.inf
        maxs[a] = np.max(ind[0])
        leader = np.argmin(maxs - ind[1])
        ind = np.copy(self.index[0, state])
        ind[leader] = -np.inf
        challenger = np.argmax(ind)
        if (
            self.index[0, state, leader] - self.index[1, state, leader]
            > self.index[0, state, challenger] - self.index[1, state, challenger]
        ):
            return leader
        return challenger

    # def pol_final(self):
    #     """gives the final deterministic policy maximizing rewards"""
    #     self.value_iteration()
    #     pol = []
    #     for state in range(self.nbr_states):
    #         pol.append(
    #             randamax(self.rewards[state] + self.transitions[state] @ self.phi)
    #         )
    #     return tuple(pol)

    def stop_test(self, epsilon, delta):
        """determines whether to stop learning

        Args:
            epsilon (_type_): _description_
            delta (_type_): _description_
        """
        if epsilon != self.epsstop or delta != self.delta:
            print(f"{delta=}, {self.delta=},{self.epsstop=},{epsilon=}")
            raise ValueError
        for state in range(self.nbr_states):
            if not self.all_selected[state]:
                return (False, None)
        max_s = -np.inf
        min_s = np.inf
        for state in range(self.nbr_states):
            for action in range(self.nbr_actions):
                self.update_index(state, action)
            max_au = self.index[0, state, randamax(self.index[0, state])]
            if max_au == np.inf:
                return (False, None)
            max_al = self.index[1, state, randamax(self.index[1, state])]
            max_s = max(max_s, max_au)
            min_s = min(min_s, max_al)
            if min_s == -np.inf:
                return (False, None)
        # print(f"epsempr={np.abs(max_s-min_s)}")
        return (np.abs(max_s - min_s) <= epsilon, np.abs(max_s - min_s))
