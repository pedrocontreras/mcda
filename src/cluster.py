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


def minimo_p_accion(categoria,n_lim,cati,n_acc,i,sigma_D_a,sigma_I_a):
    minimo=1
    jmin=0
    for j in range(0,n_acc):
        if categoria[j][cati]==1:
            if minimo>min(sigma_D_a[i][j],sigma_I_a[j][i]):
                minimo=min(sigma_D_a[i][j],sigma_I_a[j][i])
                jmin=j
    return jmin


def get_new_centroids(categoria,n_lim,n_acc,sigma_D_a,sigma_I_a,yleast,izero,beta,maximo,n_cri,limites,actions):
    for j in range(1, n_lim-1):
        for i in range(0, n_acc):
            if categoria[i][j] == 1:
                yleast[i] = minimo_p_accion(categoria, n_lim, n_acc, i, sigma_D_a, sigma_I_a)
                izero[i] = min(sigma_D_a[i][yleast[i]], sigma_I_a[yleast[i]][i])

        for i in range(0, n_acc):
            if izero[i] <= beta:
                if izero[i] > izero[maximo[j]]:
                    maximo[j] = i
        for h in range(0, n_cri):
            limites[j][h] = actions[maximo[j]][h]

    return limites