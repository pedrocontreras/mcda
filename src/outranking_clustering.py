import pandas as pd
from pandas import DataFrame as df
import openpyxl as px
import numpy as np
# from plot_clusters import *
from cluster import *


def init_data(excel_file):
    """
    initialises data
    :param excel_file:
    :return: actions, centroids, limites
    """
    work_book  = px.load_workbook(excel_file)
    work_sheet = work_book['matriz']
    df_data    = pd.DataFrame(work_sheet.values)

    # slice data to get data frames for actions, centroids, min and max
    actions    = df.to_numpy(df_data.iloc[0:49])
    centroids  = df.to_numpy(df_data.iloc[50:54])
    limites    = df.to_numpy(df_data.iloc[49:55])

    return actions, centroids, limites


def get_weights():
    """
    get criteria weights
    :return: w
    """
    w = [0.20, 0.20, 0.20, 0.30, 0.10]  # pesos de los criterios
    return w


def get_umbrales():
    """
    umbrales de preferencia directos e inversos de cada criterio
    :return: p_dir, q_dir, p_inv, q_inv
    """
    p_dir = [20,20,20,20,20]
    q_dir = [10,10,10,10,10]
    p_inv = [20,20,20,20,20]
    q_inv = [10,10,10,10,10]

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

def conc_p_dircta_actions(actions, p_dir, q_dir):
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


def regla_desc(n_acc, n_lim, sigma_I, sigma_D, lam):
    """
    implemena la regla descendente
    :param n_acc:  number of acciones
    :param n_lim: number of limits
    :param sigma_I:
    :param sigma_D:
    :param lam:
    :return: categoria
    """
    categoria = np.zeros((n_acc, n_lim))
    for i in range(0, n_acc):
        j = n_lim - 1
        while True:
            if j == 0 :
                categoria[i][1] = 1
                break
            else:
                if sigma_I[i][j] >= lam:
                    if j == n_lim-1 :
                        categoria[i][n_lim - 2]=1
                    else:
                        if min(sigma_I[i][j], sigma_D[j][i]) > min(sigma_I[i][j + 1], sigma_D[j + 1][i]):
                            categoria[i][j] = 1
                        else:
                            if j == (n_lim - 2):
                                categoria[i][n_lim - 2] = 1
                            else:
                                categoria[i][j + 1] = 1
                    break
            j = j-1
    return categoria


def perform_outranking(actions, limites, lam, iter):
    w = get_weights()
    p_dir, q_dir, p_inv, q_inv = get_umbrales()
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    n_lim = np.size(limites, 0)  # number of limits

    for k in range(0, iter):
        # inicializa la matriz de pertenencia de clases, en cada round de simulacion
        categoria = np.zeros((n_acc, n_lim))

        # calcula concordancia parcial directa e inversa (formulas (1) y (2)
        cpd = conc_p_directa(actions, limites, p_dir, q_dir)
        cpi = conc_p_inversa(actions, limites, p_inv, q_inv)

        # calcula concordancia global directa e inversa (formulas (3) y (4)
        sigma_D = concordancia_D(cpd, n_acc, n_lim, n_cri, w)
        sigma_I = concordancia_I(cpi, n_acc, n_lim, n_cri, w)

        # determina categoria de cada accion, usando regla descendente
        categoria = regla_desc(n_acc, n_lim, sigma_I, sigma_D, lam)

        print('--------------- ITERACION: {} -------------'.format(k+1))
        # print('<CATEGORIAS>')
        # print(categoria)
        # actualiza centroides
        limites = get_ordered_centroids(categoria, actions, limites, n_acc, n_lim, n_cri)

        # CALL PLOTTING HERE
        print('<CENTROIDES>')
        print(limites[:])
       #############################################
        # plot_centroids(actions, limites, k) # experiments with plotting centroids
        # print('<ACTION>')
        # print(actions[:,0])
        # print(limites[:,2:3])
    return 0

#########  MAIN ###############
def main():
    actions, centroids, limites = init_data('matriz-valores.xlsx')
    lam  = 0.8
    iter = 10
    perform_outranking(actions, limites,lam, iter)


if __name__ == '__main__':
    main()




