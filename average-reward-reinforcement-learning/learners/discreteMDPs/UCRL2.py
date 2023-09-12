"""Implementation(s) of UCRL2"""
import copy as cp
import numpy as np
from learners.discreteMDPs.AgentInterface import Agent
from learners.discreteMDPs.utils import allmax, categorical_sample, randamax

import learners.discreteMDPs.stopping_rule as stru


class UCRL2(Agent):
    def __init__(self, nbr_states, nbr_actions, delta, epsilon=1e-3, max_iter=3000):
        Agent.__init__(self, nbr_states, nbr_actions, name="UCRL2")
        """Vanilla UCRL2 based on "Jaksch, Thomas, Ronald Ortner, and Peter Auer. \
            "Near-optimal regret bounds for reinforcement learning." \
                Journal of Machine Learning Research 11.Apr (2010): 1563-1600."
        :param nbr_states: the number of states
        :param nbr_actions: the number of actions
        :param delta:  confidence level in (0,1)
        """
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions
        self.dirac = np.eye(self.nbr_states, dtype=int)
        self.t = 1
        self.delta = delta
        self.phi = np.zeros(self.nbr_states)
        self.epsilon = epsilon
        self.max_iteration = max_iter
        self.all_selected = np.zeros(self.nbr_states, dtype=bool)

        self.observations = [
            [],
            [],
            [],
        ]  # list of the observed (states, actions, rewards) ordered by time
        self.vk = np.zeros(
            (self.nbr_states, self.nbr_actions)
        )  # the state-action count for the current episode k
        self.Nk = np.zeros(
            (self.nbr_states, self.nbr_actions)
        )  # the state-action count prior to episode k

        self.rmeans = np.zeros((self.nbr_states, self.nbr_actions))
        self.tmeans = (
            np.ones((self.nbr_states, self.nbr_actions, self.nbr_states))
            / self.nbr_states
        )
        self.r_distances = np.zeros((self.nbr_states, self.nbr_actions))
        self.p_distances = np.zeros((self.nbr_states, self.nbr_actions))
        self.Pk = np.zeros(
            (self.nbr_states, self.nbr_actions, self.nbr_states)
        )  # transition count
        self.Rk = np.zeros((self.nbr_states, self.nbr_actions))  # reward count

        self.u = np.zeros(self.nbr_states)
        self.span = []
        self.policy = np.zeros(
            (self.nbr_states, self.nbr_actions)
        )  # policy, seen as a stochastic policy here.
        for s in range(self.nbr_states):
            for a in range(self.nbr_actions):
                self.policy[s, a] = 1.0 / self.nbr_actions

    #   def name(self):
    #       return "UCRL2"

    def updaten(self):
        """Auxiliary function to update N the current state-action count."""
        for s in range(self.nbr_states):
            for a in range(self.nbr_actions):
                self.Nk[s, a] += self.vk[s, a]

    def updater(self):
        """Auxiliary function to update R the accumulated reward."""
        self.Rk[
            self.observations[0][-2], self.observations[1][-1]
        ] += self.observations[2][-1]

    def updatep(self):
        """Auxiliary function to update P the transitions count."""
        self.Pk[
            self.observations[0][-2], self.observations[1][-1], self.observations[0][-1]
        ] += 1

    #
    def distances(self):
        """Auxiliary function updating the values of r_distances and p_distances\
            (i.e. the confidence bounds used to build the set of plausible MDPs)."""
        for s in range(self.nbr_states):
            for a in range(self.nbr_actions):
                self.r_distances[s, a] = np.sqrt(
                    (
                        7
                        * np.log(
                            2 * self.nbr_states * self.nbr_actions * self.t / self.delta
                        )
                    )
                    / (2 * max([1, self.Nk[s, a]]))
                )
                self.p_distances[s, a] = np.sqrt(
                    (
                        14
                        * self.nbr_states
                        * np.log(2 * self.nbr_actions * self.t / self.delta)
                    )
                    / (max([1, self.Nk[s, a]]))
                )

    def max_proba(self, p_estimate, sorted_indices, s, a):
        """Computing the maximum proba in the Extended Value Iteration for given state s\
            and action a."""
        min1 = min(
            [1, p_estimate[s, a, sorted_indices[-1]] + (self.p_distances[s, a] / 2)]
        )
        max_p = np.zeros(self.nbr_states)
        if min1 == 1:
            max_p[sorted_indices[-1]] = 1
        else:
            max_p = cp.deepcopy(p_estimate[s, a])
            max_p[sorted_indices[-1]] += self.p_distances[s, a] / 2
            l = 0
            while sum(max_p) > 1:
                max_p[sorted_indices[l]] = max(
                    [0, 1 - sum(max_p) + max_p[sorted_indices[l]]]
                )  # Error?
                l += 1
        return max_p

    def evi(self, r_estimate, p_estimate, epsilon=0.01, max_iter=1000):
        """The Extend Value Iteration algorithm (approximated with precision epsilon),\
            in parallel policy updated with the greedy one."""
        u0 = self.u - min(
            self.u
        )  # sligthly boost the computation and doesn't seems to change the results
        u1 = np.zeros(self.nbr_states)
        sorted_indices = np.arange(self.nbr_states)
        niter = 0
        while True:
            niter += 1
            for s in range(self.nbr_states):
                temp = np.zeros(self.nbr_actions)
                for a in range(self.nbr_actions):
                    max_p = self.max_proba(p_estimate, sorted_indices, s, a)
                    temp[a] = min((1, r_estimate[s, a] + self.r_distances[s, a])) + sum(
                        [u * p for (u, p) in zip(u0, max_p)]
                    )
                # This implements a tie-breaking rule by choosing:  Uniform(Argmmin(Nk))
                (u1[s], arg) = allmax(temp)
                nn = [-self.Nk[s, a] for a in arg]
                (_, arg2) = allmax(nn)
                choice = [arg[a] for a in arg2]
                self.policy[s] = [
                    1.0 / len(choice) if x in choice else 0
                    for x in range(self.nbr_actions)
                ]

            diff = [abs(x - y) for (x, y) in zip(u1, u0)]
            if (max(diff) - min(diff)) < epsilon:
                self.u = u1 - min(u1)
                break
            else:
                u0 = u1 - min(u1)
                u1 = np.zeros(self.nbr_states)
                sorted_indices = np.argsort(u0)
            if niter > max_iter:
                self.u = u1 - min(u1)
                print("No convergence in evi")
                break

    def new_episode(self):
        """To start a new episode (init var, computes estmates and run evi)."""
        self.updaten()
        self.vk = np.zeros((self.nbr_states, self.nbr_actions))
        r_estimate = np.zeros((self.nbr_states, self.nbr_actions))
        p_estimate = np.zeros((self.nbr_states, self.nbr_actions, self.nbr_states))
        for s in range(self.nbr_states):
            for a in range(self.nbr_actions):
                div = max([1, self.Nk[s, a]])
                r_estimate[s, a] = self.Rk[s, a] / div
                for next_s in range(self.nbr_states):
                    p_estimate[s, a, next_s] = self.Pk[s, a, next_s] / div
        self.distances()
        self.evi(r_estimate, p_estimate, epsilon=1.0 / max(1, self.t))

    # To reinitialize the learner with a given initial state inistate.
    def reset(self, inistate):
        self.t = 1
        self.observations = [[inistate], [], []]
        self.vk = np.zeros((self.nbr_states, self.nbr_actions))
        self.Nk = np.zeros((self.nbr_states, self.nbr_actions))
        self.u = np.zeros(self.nbr_states)
        self.Pk = np.zeros((self.nbr_states, self.nbr_actions, self.nbr_states))
        self.Rk = np.zeros((self.nbr_states, self.nbr_actions))
        self.span = [0]
        for s in range(self.nbr_states):
            for a in range(self.nbr_actions):
                self.policy[s, a] = 1.0 / self.nbr_actions
        self.new_episode()

    def play(self, state):
        """To choose an action for a given state (and start a new episode if necessary \
            -> stopping criterion defined here)."""
        action = categorical_sample(
            [self.policy[state, a] for a in range(self.nbr_actions)], np.random
        )
        if self.vk[state, action] >= max(
            [1, self.Nk[state, action]]
        ):  # Stopping criterion
            self.new_episode()
            action = categorical_sample(
                [self.policy[state, a] for a in range(self.nbr_actions)], np.random
            )
        return action

    # To update the learner after one step of the current policy.
    def update(self, state, action, reward, observation):
        self.vk[state, action] += 1
        self.observations[0].append(observation)
        self.observations[1].append(action)
        self.observations[2].append(reward)
        na = self.vk[state, action] + self.Nk[state, action]
        self.rmeans[state, action] = (self.rmeans[state, action] * na + reward) / (
            na + 1
        )
        self.tmeans[state, action] = (
            self.tmeans[state, action] * na + self.dirac[observation]
        ) / (na + 1)
        self.updatep()
        self.updater()
        self.t += 1
        if not self.all_selected[state]:
            self.all_selected[state] = np.all(self.Nk[state] + self.vk[state] > 0)

    def pol_final(self):
        """gives the final deterministic policy maximizing rewards"""
        # print(f"the Nk is {self.Nk}")
        for state in range(self.nbr_states):
            if not self.all_selected[state]:
                # print("yay")
                return tuple([0] * self.nbr_states)
        self.value_iteration()
        pol = []
        for state in range(self.nbr_states):
            pol.append(
                randamax(
                    self.Rk[state] / (self.vk[state] + self.Nk[state])
                    + (self.Pk[state] @ self.phi) / (self.Nk[state] + self.vk[state])
                )
            )
        return tuple(pol)

    def value_iteration(self):
        """computes the bias and stores it in self.phi"""
        ctr = 0
        stop = False
        phi = np.copy(self.phi)
        phip = np.copy(self.phi)
        # from warnings import filterwarnings
        # filterwarnings("error")
        while not stop:
            ctr += 1
            for state in range(self.nbr_states):
                max_na = np.max(self.Nk[state])
                mask = (self.Nk[state] + self.vk[state]) >= np.log(max_na) ** 2
                skeleton = np.arange(self.nbr_actions, dtype=int)[mask]
                u = -np.inf
                for action in skeleton:
                    psa = self.Pk[state, action] + 1
                    psa = psa / np.sum(psa)
                    # if np.sum(psa) != 1:
                    #     print(
                    #         np.sum(psa),
                    #     )
                    # raise ValueError
                    rsa = self.Rk[state, action] / (
                        self.Nk[state, action] + self.vk[state, action]
                    )
                    # print(f"{rsa=}, {psa=}, {phi=}")
                    # try:
                    u = max(u, rsa + psa @ phi)
                    # except ValueError:
                    #     print(f"{rsa=}, {psa=}, {phi=}")
                phip[state] = u
            phip = phip - np.min(phip)
            delta = np.max(np.abs(phi - phip))
            phi = np.copy(phip)
            stop = (delta < self.epsilon) or (ctr >= self.max_iteration)
        self.phi = np.copy(phi)

    def stop_test(self, epsilon, delta):
        """determines whether to stop learning

        Args:
            epsilon (float): how optimal we want our policy to be
        """

        for state in range(self.nbr_states):
            if not self.all_selected[state]:
                # print("yay")
                return (False, None)
        self.value_iteration()
        bias = self.phi
        max_s = -np.inf
        min_s = np.inf
        for state in range(self.nbr_states):
            max_au = -np.inf
            max_al = -np.inf
            for action in range(self.nbr_actions):
                # r_d = self.rewards_distributions[state][action]
                # valr = np.fromiter(r_d.keys(), dtype=float)  # values reward
                # probr = np.fromiter(r_d.values(), dtype=float)  # probas reward
                # probr = probr / probr.sum()
                nsamples = self.Nk[state, action] + self.vk[state, action]
                rsa = self.Rk[state, action] / nsamples  # mean reward
                probp = self.Pk[state][action] / nsamples
                usa = stru.upper_sa(
                    delta, probp, bias, nsamples, self.nbr_states, epsilon
                )
                lsa = stru.lower_sa(
                    delta, probp, bias, nsamples, self.nbr_states, epsilon
                )
                # print(f"{usa=},{lsa=}")
                max_au = max(max_au, usa + rsa)
                if max_au == np.inf:
                    # print("max=infty")
                    return (False, None)
                max_al = max(max_al, lsa + rsa)
            max_s = max(max_s, max_au - bias[state])
            min_s = min(min_s, max_al - bias[state])
            if min_s == -np.inf:
                print("min=-infty")
                return (False, None)
        if max_s == -np.inf or min_s == np.inf:
            return (False, None)
        # print(f"epsempr={np.abs(max_s - min_s)}")
        return (np.abs(max_s - min_s) <= epsilon, np.abs(max_s - min_s))


# # This "UCRL2" algorithm is a slight modfication of UCRL2. the idea is to add some
# # forced exploration trying all the unknown action in every state befor starting
# # the optimism phase,
# # and to run a random policy in unknown states.
# class UCRL2_bis(UCRL2):
#     def __init__(self, nbr_states, nbr_actions, delta):
#         UCRL2.__init__(self, nbr_states, nbr_actions, delta)
#         self.nbr_states = nbr_states
#         self.nbr_actions = nbr_actions
#         self.agentname = "UCRL2_bis"
#         self.t = 1
#         self.delta = delta
#         self.observations = [[], [], []]
#         self.vk = np.zeros((self.nbr_states, self.nbr_actions))
#         self.Nk = np.zeros((self.nbr_states, self.nbr_actions))
#         self.policy = np.zeros((self.nbr_states,), dtype=int)
#         self.r_distances = np.zeros((self.nbr_states, self.nbr_actions))
#         self.p_distances = np.zeros((self.nbr_states, self.nbr_actions))
#         self.Pk = np.zeros((self.nbr_states, self.nbr_actions, self.nbr_states))
#         self.Rk = np.zeros((self.nbr_states, self.nbr_actions))
#         self.u = np.zeros(self.nbr_states)
#         self.span = []
#         self.visited = np.zeros(
#             (self.nbr_states, self.nbr_actions + 1)
#         )  # +1 to register that the state is known, the rest to make sure
# # that every action had been tried

#     # at least one time

#     # To reinitialize the learner with a given initial state inistate.
#     def reset(self, inistate):
#         self.t = 1
#         self.visited = np.zeros((self.nbr_states, self.nbr_actions + 1))
#         self.observations = [[inistate], [], []]
#         self.vk = np.zeros((self.nbr_states, self.nbr_actions))
#         self.Nk = np.zeros((self.nbr_states, self.nbr_actions))
#         self.new_episode()
#         self.u = np.zeros(self.nbr_states)
#         self.Pk = np.zeros((self.nbr_states, self.nbr_actions, self.nbr_states))
#         self.Rk = np.zeros((self.nbr_states, self.nbr_actions))
#         self.span = [0]

#     # To chose an action for a given state (and start a new episode if necessary ->
#       # stopping criterion defined here).
#     def play(self, state):
#         if self.visited[state, -1] == 0:
#             self.visited[state, -1] = 1
#             self.visited[state, 0] = 1
#             return 0
#         else:
#             for a in range(self.nbr_actions):
#                 if self.visited[state, a] == 0:
#                     self.visited[state, a] = 1
#                     return a
#         action = self.policy[state]
#         if self.vk[state, action] >= max(
#             [1, self.Nk[state, action]]
#         ):  # Stoppping criterion
#             self.new_episode()
#         action = self.policy[state]
#         return action

#     # To update the learner after one step of the current policy.
#     def update(self, state, action, reward, observation):
#         self.vk[state, action] += 1
#         self.observations[0].append(observation)
#         self.observations[1].append(action)
#         self.observations[2].append(reward)
#         self.updatep()
#         self.updater()
#         self.t += 1
