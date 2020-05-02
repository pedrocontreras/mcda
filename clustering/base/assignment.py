
def regla_desc(categoria, belonging, n_acc, n_lim, sigma_I, sigma_D, lam):
    """
    implemena la regla descendente
    :param n_acc:  number of acciones
    :param n_lim: number of limits
    :param sigma_I:
    :param sigma_D:
    :param lam:
    :return: categoria
    """
    for i in range(0, n_acc):
        j = n_lim - 1
        while True:
            if j == 0 :
                categoria[i][1] = 1
                belonging[i]=1
                break
            else:
                if sigma_I[i][j] >= lam:
                    if j == n_lim-1 :
                        categoria[i][n_lim - 2]=1
                        belonging[i]=n_lim - 2
                    else:
                        if min(sigma_I[i][j], sigma_D[j][i]) > min(sigma_I[i][j + 1], sigma_D[j + 1][i]):
                            categoria[i][j] = 1
                            belonging[i]=j
                        else:
                            if j == (n_lim - 2):
                                categoria[i][n_lim - 2] = 1
                                belonging[i]=n_lim - 2
                            else:
                                categoria[i][j + 1] = 1
                                belonging[i]=j+1
                    break
            j = j-1

    return categoria
