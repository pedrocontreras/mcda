import pandas as pd
from pandas import DataFrame as df
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
import openpyxl as px
import numpy as np
import code.clustering_base.outranking_clustering


# from plot_clusters import *
from pandas.tests.groupby.test_value_counts import df

import code.clustering_base.cluster


def init_data(excel_file):
    """
    initialises data
    :param excel_file:
    :return: actions, centroids, limites
    """
    work_book  = px.load_workbook(excel_file)
    work_sheet = work_book['data']
    df_data    = pd.DataFrame(work_sheet.values)

    # slice data to get data frames for actions, centroids, min and max
    actions    = df.to_numpy(df_data.iloc[0:189])
    centroids  = df.to_numpy(df_data.iloc[190:194])
    limites    = df.to_numpy(df_data.iloc[189:195])

    return actions, centroids, limites


def get_weights():
    """
    get criteria weights
    :return: w
    """
    w = [0.333, 0.334, 0.333]  # pesos de los criterios
    return w


def get_umbrales():
    """
    umbrales de preferencia directos e inversos de cada criterio
    :return: p_dir, q_dir, p_inv, q_inv
    """

    # umbrales HDI
    # p_dir = [0.21,0.07,0.1]
    # q_dir = [0.105,0.035,0.05]
    # p_inv = [0.21,0.07,0.1]
    # q_inv = [0.105,0.035,0.05]

    # umbrales SSI
    p_dir = [0.51,0.58,0.43]
    q_dir = [0.25,0.29,0.22]
    p_inv = [0.51,0.58,0.43]
    q_inv = [0.25,0.29,0.22]

    return p_dir, q_dir, p_inv, q_inv


def get_metrics(actions, limites):
    """
    Get lengths of input data
    :param actions:
    :param limites:
    :return: n_acc, n_cri, n_limites
    """
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    n_lim = np.size(limites, 0)  # number of limits
    return n_acc, n_cri, n_lim

def conc_p_directa_actions(actions, p_dir, q_dir):
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    cpda  = np.zeros((n_acc, n_acc, n_cri))

    # calcula indice de concordancia parcial directo
    for h in range(0, n_cri):
        # mueve j en las filas del arreglo de perfiles de categorias
        for j in range(0, n_acc):
            # mueve i en las filas del arreglo de acciones
            for i in range(0, n_acc):
                if actions[i][h] - actions[j][h] > p_dir[h]:
                    cpda[j][i][h] = 0
                else:
                    if actions[i][h] - actions[j][h] <= q_dir[h]:
                        cpda[j][i][h] = 1
                    else:
                        cpda[j][i][h] = 1.0 * (actions[j][h] - actions[i][h] + p_dir[h]) / (p_dir[h] - q_dir[h])
    return cpda


def conc_p_inversa_actions(actions, p_inv, q_inv):
    """
    calcula la concordancia parcial inversa
    :param actions:
    :param p_inv:
    :param q_inv:
    :return:
    """
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria

    cpia  = np.zeros((n_acc, n_acc, n_cri))  # cpi array to store values

    for h in range(0, n_cri):
        # calcula indice de concordancia parcial inverso
        #mueve i en las filas del arreglo de acciones
        for i in range(0, n_acc):
            #mueve j en las filas del arreglo de perfiles de categorias
            for j in range(0, n_acc):
                if actions[j][h] - actions[i][h] > p_inv[h]:
                    cpia[i][j][h] = 0
                else:
                    if actions[j][h] - actions[i][h] <= q_inv[h]:
                        cpia[i][j][h] = 1
                    else:
                        cpia[i][j][h] = 1.0 * (actions[i][h] - actions[j][h] + p_inv[h]) / (p_inv[h] - q_inv[h])
    return cpia


def conc_p_directa(actions, limites, p_dir, q_dir):
    """
    calcula la concordancia parcial directa
    :param actions:
    :param limites:
    :param p_dir:
    :param q_dir:
    :return: cod
    """
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    n_lim = np.size(limites, 0)  # number of limits
    cpd = np.zeros((n_lim, n_acc, n_cri))

    # calcula indice de concordancia parcial directo
    for h in range(0, n_cri):
        # mueve j en las filas del arreglo de perfiles de categorias
        for j in range(0, n_lim):
            # mueve i en las filas del arreglo de acciones
            for i in range(0, n_acc):
                if actions[i][h] - limites[j][h] > p_dir[h]:
                    cpd[j][i][h] = 0
                else:
                    if actions[i][h] - limites[j][h] <= q_dir[h]:
                        cpd[j][i][h] = 1
                    else:
                        cpd[j][i][h] = 1.0 * (limites[j][h] - actions[i][h] + p_dir[h]) / (p_dir[h] - q_dir[h])
    return cpd

def conc_p_inversa(actions, limites, p_inv, q_inv):
    """
    calcula la concordancia parcial inversa
    :param actions:
    :param limites:
    :param p_inv:
    :param q_inv:
    :return:
    """
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    n_lim = np.size(limites, 0)  # number of limits

    cpi   = np.zeros((n_acc, n_lim, n_cri))  # cpi array to store values

    for h in range(0, n_cri):
        # calcula indice de concordancia parcial inverso
        #mueve i en las filas del arreglo de acciones
        for i in range(0, n_acc):
            #mueve j en las filas del arreglo de perfiles de categorias
            for j in range(0, n_lim):
                if limites[j][h] - actions[i][h] > p_inv[h]:
                    cpi[i][j][h] = 0
                else:
                    if limites[j][h] - actions[i][h] <= q_inv[h]:
                        cpi[i][j][h] = 1
                    else:
                        cpi[i][j][h] = 1.0 * (actions[i][h] - limites[j][h] + p_inv[h]) / (p_inv[h] - q_inv[h])
    return cpi

def concordancia_I_actions(cpia, n_acc, n_cri, w):
    """
    calcula la concordancia inversa
    :param cpi:
    :param n_acc:
    :param n_cri:
    :param w:
    :return: sigma_I
    """
    sigma_I = np.zeros((n_acc, n_acc))
    for i in range(0, n_acc):
        for j in range(0, n_acc):
            x = 0.0
            for h in range(0, n_cri):
                x = x + w[h] * cpia[i][j][h]
            sigma_I[i][j] = x
    return sigma_I


def concordancia_D_actions(cpda, n_acc,  n_cri, w):
    """
    calcula la concordancia directa
    :param cpd:
    :param n_acc:
    :param n_cri:
    :param w:
    :return: sigma_D
    """
    sigma_D = np.zeros((n_acc, n_acc))
    for i in range(0, n_acc):
        for j in range(0, n_acc):
            x = 0.0
            for h in range(0, n_cri ):
                x = x + w[h] * cpda[i][j][h]
            sigma_D[i][j] = x
    return sigma_D


def concordancia_I(cpi, n_acc, n_lim, n_cri, w):
    """
    calcula la concordancia inversa
    :param cpi:
    :param n_acc:
    :param n_lim:
    :param n_cri:
    :param w:
    :return: sigma_I
    """
    sigma_I = np.zeros((n_acc, n_lim))
    for i in range(0, n_acc):
        for j in range(0, n_lim):
            x = 0.0
            for h in range(0, n_cri):
                x = x + w[h] * cpi[i][j][h]
            sigma_I[i][j] = x
    return sigma_I


def concordancia_D(cpd, n_acc, n_lim, n_cri, w):
    """
    calcula la concordancia directa
    :param cpd:
    :param n_acc:
    :param n_lim:
    :param n_cri:
    :param w:
    :return: sigma_D
    """
    sigma_D = np.zeros((n_lim, n_acc))
    for i in range(0, n_lim):
        for j in range(0, n_acc):
            x = 0.0
            for h in range(0, n_cri ):
                x = x + w[h] * cpd[i][j][h]
            sigma_D[i][j] = x
    return sigma_D


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


def perform_outranking(actions, limites, lam, beta, iter):
    w = get_weights()
    p_dir, q_dir, p_inv, q_inv = get_umbrales()
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    n_lim = np.size(limites, 0)  # number of limits
    # -------------------------------------------------------
    for k in range(0, iter):
        # calcula concordancia parcial directa e inversa (formulas (1) y (2)
        from code.clustering_base.concordancia import conc_p_directa
        cpd = conc_p_directa(actions, limites, p_dir, q_dir)
        from code.clustering_base.concordancia import conc_p_inversa
        cpi = conc_p_inversa(actions, limites, p_inv, q_inv)

        # calcula concordancia global directa e inversa (formulas (3) y (4)
        from code.clustering_base.concordancia import concordancia_D
        sigma_D = concordancia_D(cpd, n_acc, n_lim, n_cri, w)
        from code.clustering_base.concordancia import concordancia_I
        sigma_I = concordancia_I(cpi, n_acc, n_lim, n_cri, w)


        # determina categoria de cada accion, usando regla descendente
        categoria = np.zeros((n_acc, n_lim), dtype=int)
        belonging = np.zeros((n_acc), dtype=int)

        categoria = regla_desc(categoria,belonging, n_acc, n_lim, sigma_I, sigma_D, lam)

        # actualiza centroides por el metodo de los promedios
        #limites = get_ordered_centroids(categoria, actions, limites, p_dir,p_inv,n_acc, n_lim, n_cri)

        # actualiza centroides por el metodo de los promedios limitados a acciones dentro de trapezoides
        #limites = get_ordered_centroids_2(categoria, actions, limites, p_dir,p_inv,n_acc, n_lim, n_cri)

        #determina los nuevos centroides de categoria, usando técnica de Fernandez et al. (2010)
        #limites=get_ordered_centroids_3(categoria,n_lim,n_acc,sigma_D_a,sigma_I_a,beta,n_cri,limites,actions)

        #determina los nuevos centroides de categoria, usando técnica cuchufletina

        #limitesold=limites
        limites=code.clustering_base.cluster.get_ordered_centroids_4(belonging, limites, n_lim, n_cri, p_dir, p_inv, actions)


        print('--------------- ITERACION: {} -------------'.format(k+1))
        print('<CATEGORIAS>')
        #print(categoria)
        # for j in range (1,n_lim-1):
        #     for i in range (0,n_acc):
        #         if categoria[i][j]==1:
        #             print (j,"-", i,":",actions[i][0],actions[i][1],actions[i][2],actions[i][3],actions[i][4])

        # CALL PLOTTING HERE
        print('<CENTROIDES>')
        np.set_printoptions(precision=2)
        print (n_lim)
        print(limites[:])
        for i in range(0,n_acc):
            print (categoria[i][0],categoria[i][1],categoria[i][2],categoria[i][3],categoria[i][4])
    # ############################################
    # plot_centroids(actions, limites, k) # experiments with plotting centroids
    # print('<ACTION>')
    # print(actions[:,0])
    # print(limites[:,2:3])
    return 0


#########  MAIN ###############
def main():
    actions, centroids, limites = init_data('SSI.xlsx')
    lam  = 0.5
    beta=0.1
    iter_stochastic=1
    iter = 50

    perform_outranking(actions, limites,lam,beta, iter)


if __name__ == '__main__':
    main()



