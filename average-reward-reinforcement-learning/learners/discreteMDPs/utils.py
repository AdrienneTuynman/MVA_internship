import numpy as np

########################################
#         Argument selectors           #
########################################


def randamax(V, T=None, I=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax


    """
    if I is None:
        idxs = np.where(V == np.amax(V))[0]
        if T is None:
            idx = np.random.choice(idxs)
        else:
            assert len(V) == len(
                T
            ), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(V[I] == np.amax(V[I]))[0]
        if T is None:
            idx = I[np.random.choice(idxs)]
        else:
            assert len(V) == len(
                T
            ), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t = T[I]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = I[idxs[t_idxs]]
    return idx


def randamin(V, T=None, I=None):
    """
    V: array of values
    T: array used to break ties
    I: array of indices from which we should return an amax
    """
    if I is None:
        idxs = np.where(V == np.amin(V))[0]
        if T is None:
            idx = np.random.choice(idxs)
        else:
            assert len(V) == len(
                T
            ), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t_idxs = np.where(T[idxs] == np.amin(T[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = idxs[t_idxs]
    else:
        idxs = np.where(V[I] == np.amin(V[I]))[0]
        if T is None:
            idx = I[np.random.choice(idxs)]
        else:
            assert len(V) == len(
                T
            ), f"Lengths should match: len(V)={len(V)} - len(T)={len(T)}"
            t = T[I]
            t_idxs = np.where(t[idxs] == np.amin(t[idxs]))[0]
            t_idxs = np.random.choice(t_idxs)
            idx = I[idxs[t_idxs]]
    return idx


def allmax(a):
    """returns the max and argmax of list a"""
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return (max_, all_)


def allmin(a):
    """returns the min and argmin of list a"""
    if len(a) == 0:
        return []
    all_ = [0]
    min_ = a[0]
    for i in range(1, len(a)):
        if a[i] < min_:
            all_ = [i]
            min_ = a[i]
        elif a[i] == min_:
            all_.append(i)
    return (min_, all_)


def categorical_sample(prob_n, np_random):
    """
    Sample from categorical distribution
    Each row specifies class probabilities
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.rand()).argmax()


def kl(x, y):
    """returns the KL divergence between B(x) and B(y)

    Args:
        x (float): _description_
        y (float): _description_

    Returns:
        float: _description_
    """
    if x == 0:
        if y == 1.0:
            return np.infty
        return np.log(1.0 / (1.0 - y))
    if x == 1:
        if y == 0.0:
            return np.infty
        return np.log(1.0 / y)
    if (y == 0) or (y == 1):
        return np.infty
    return x * np.log(x / y) + (1.0 - x) * np.log((1.0 - x) / (1.0 - y))


def search_up(f, up, down, epsilon=0.0001):
    """Dichotomic search for...? between down and up, with precision epsilon

    Args:
        f (_type_): function?
        up (float): upper bound of the interval to look into
        down (float): lower bound of the interval to look into
        epsilon (float, optional): precision. Defaults to 0.0001.

    Returns:
        float: _description_
    """
    mid = (up + down) / 2
    if up - down > epsilon:
        if f(mid):
            return search_up(f, up, mid)
        else:
            return search_up(f, mid, down)
    else:
        if f(up):
            return up
        return down


def search_down(f, up, down, epsilon=0.0001):
    """Dichotomic search for...? between down and up, with precision epsilon

    Args:
        f (_type_): function?
        up (float): upper bound of the interval to look into
        down (float): lower bound of the interval to look into
        epsilon (float, optional): precision. Defaults to 0.0001.

    Returns:
        float: _description_
    """
    mid = (up + down) / 2
    if up - down > epsilon:
        if f(mid):
            return search_down(f, mid, down)
        else:
            return search_down(f, up, mid)
    else:
        if f(down):
            return down
        return up
