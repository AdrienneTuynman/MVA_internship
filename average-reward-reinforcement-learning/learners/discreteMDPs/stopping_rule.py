"""Everything needed to implement the epsilon-optimality stopping rule"""

from scipy.optimize import minimize_scalar, bisect
from numpy import log, inf as npinf, sum as npsum, min as npmin, max as npmax, sqrt


def x(n, delta, ns):
    """Computes an approximation of x(n,delta,m) for the stopping rule

    Args:
        n (int): number of samples
        d (float): delta admissible
        ns (int): size of the distribution
    """
    return log(1 / delta) + log(n)  # (ns - 1) * log(e * (1 + (n / (ns - 1))))


def k_inf_p(gamma, bias, prob, epsilon, threshold):
    """computes inf_{p'} KL(p,p') : p'b > gamma

    Args:
        bias (float nparray): normalized bias function (min = 0, max = 1)
        gamma (float): threshold
        prob (float nparray): probability vector to be compared with
    """
    if gamma > 1:
        return npinf
    if gamma < 0:
        return -threshold
    u = 1 / (1 - gamma)
    res = minimize_scalar(
        lambda x: -npsum(prob * log(1 - (bias - gamma) * x)),
        bounds=(0, u),
        method="bounded",
        options={"xatol": 0.1 * epsilon},
    )
    resultat = -res.fun - threshold
    return resultat


def k_inf_m(gamma, bias, prob, epsilon, threshold):
    """computes inf_{p'} KL(p,p') : p'b < gamma

    Args:
        bias (float nparray): normalized bias function (min = 0, max = 1)
        gamma (float): threshold
        prob (float nparray): probability vector to be compared with
    """
    return k_inf_p(1 - gamma, 1 - bias, prob, epsilon, threshold)


# def upper_sa(delta, prob, bias, nsamples, nstates, epsilon):
#     """computes the upper bound for pb

#     Args:
#         delta (float): _description_
#         prob (float nparray): _description_
#         bias (float nparray): (non normalized) bias
#         nsamples (int): N_sa^t
#         ns (int): size of the distribution, |S|
#     """
#     if nsamples == 0:
#         print("nsamples=0")
#         return npinf
#     thr = x(nsamples, delta, nstates) / nsamples
#     minb = npmin(bias)
#     maxb = npmax(bias)
#     span = maxb - minb
#     if span < 1e-07:
#         return npinf
#     bias_n = (bias - minb) / span
#     # risk if the threshold is unattainable..? Shouldn't be the case
#     func = lambda gamma: k_inf_p(gamma, bias_n, prob, epsilon, thr)
#     try:
#         usa = bisect(
#             func,
#             1e-8,
#             1 - 1e-8,
#         )
#     except ValueError:
#         usa = 1
#     return minb + span * usa


def upper_sa(delta, prob, bias, nsamples, nstates, epsilon):
    """computes the upper bound for pb

    Args:
        delta (float): _description_
        prob (float nparray): _description_
        bias (float nparray): (non normalized) bias
        nsamples (int): N_sa^t
        ns (int): size of the distribution, |S|
    """
    if nsamples == 0:
        print("nsamples=0")
        return npinf
    thr = x(nsamples, delta, nstates) / nsamples
    minb = npmin(bias)
    maxb = npmax(bias)
    span = maxb - minb
    if span < 1e-07 * epsilon:
        return npinf
    return maxb * sqrt(2 * thr) + prob.dot(bias)


# import numpy as np
# nsamples = 10
# delta = 0.1
# nstates = 4
# bias = np.array([0, 0.5, 1])
# prob = np.array([0.3, 0.3, 0.4])
# epsilon = 0.01
# import learners.discreteMDPs.stopping_rule as stru
# usa = stru.upper_sa(delta, prob, bias, nsamples, nstates, epsilon)
# usa = upper_sa(delta, prob, bias, nsamples, nstates, epsilon)
# print(f"{usa=}")


# def lower_sa(delta, prob, bias, nsamples, nstates, epsilon):
#     """computes the lower bound for pb

#     Args:
#         delta (float): _description_
#         prob (float nparray): _description_
#         bias (float nparray): (non normalized) bias
#         nsamples (int): N_sa^t
#         ns (int): size of the distribution, |S|
#     """
#     if nsamples == 0:
#         return -npinf
#     threshold = x(nsamples, delta, nstates) / nsamples
#     minb = npmin(bias)
#     maxb = npmax(bias)
#     span = maxb - minb
#     if span == 0:
#         return -npinf
#     bias_n = (bias - minb) / span
#     # risk if the threshold is unattainable..? Shouldn't be the case
#     try:
#         res = minb + span * bisect(
#             lambda gamma: k_inf_m(gamma, bias_n, prob, epsilon, threshold),
#             1e-8,
#             1 - 1e-8,
#         )
#     except ValueError:
#         # print("errorl")
#         # print(func(1e-8))
#         # print(func(1 - 1e-8))
#         res = minb
#     return res


def lower_sa(delta, prob, bias, nsamples, nstates, epsilon):
    """computes the lower bound for pb

    Args:
        delta (float): _description_
        prob (float nparray): _description_
        bias (float nparray): (non normalized) bias
        nsamples (int): N_sa^t
        ns (int): size of the distribution, |S|
    """
    if nsamples == 0:
        return -npinf
    thr = x(nsamples, delta, nstates) / nsamples
    minb = npmin(bias)
    maxb = npmax(bias)
    span = maxb - minb
    if span < 1e-07 * epsilon:
        return -npinf
    return -maxb * sqrt(2 * thr) + prob.dot(bias)
