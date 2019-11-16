import numpy as np

def k_means(categoria, actions, limites):
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    n_lim = np.size(limites, 0)  # number of limits

    return 0

def get_ordered_centroids(categoria, actions, limites, n_acc, n_lim, n_cri):
    """
    :param categoria: array that contains categoria
    :param actions: array with the acciones
    :param limites: array with the limits
    :param n_acc: number of acciones
    :param n_lim: number of limits
    :param n_cri: number of criteria
    :return: limites
    """
    suma = np.zeros((n_lim, n_cri))
    freq_categoria = np.zeros(n_lim)

    for i in range(0, n_acc):
        for j in range(1, n_lim - 1):
            if categoria[i][j] == 1:
                freq_categoria[j] = freq_categoria[j] + 1
                for h in range(0, n_cri):
                    suma[j][h] = suma[j][h] + actions[i][h]

    for j in range(1, n_lim - 1):
        if freq_categoria[j] > 0:
            for h in range(0, n_cri):
                limites[j][h] = 1.0 * suma[j][h] / freq_categoria[j]

    return limites
