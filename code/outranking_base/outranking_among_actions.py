import pandas as pd
from pandas import DataFrame as df
import openpyxl as px
import numpy as np
#from plot_clusters import *
from cluster import *

from code.clustering_base.cluster import get_ordered_centroids_4


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
    # Acciones de HDI
    # actions    = df.to_numpy(df_data.iloc[0:189])
    # centroids  = df.to_numpy(df_data.iloc[190:194])
    # limites    = df.to_numpy(df_data.iloc[189:195])

    #Acciones de SSI
    actions    = df.to_numpy(df_data.iloc[0:159])
    centroids  = df.to_numpy(df_data.iloc[155:158])
    limites    = df.to_numpy(df_data.iloc[154:159])

    return actions, centroids, limites

def random_thresholds(excel_file):
    """
     read random thresholds (externally generated)
     :param excel_file:
     :return: p_dir,q_dir,p_inv,q_inv
     """
    work_book = px.load_workbook(excel_file)
    work_sheet_p_dir = work_book['p_dir']
    work_sheet_q_dir = work_book['q_dir']
    work_sheet_p_inv = work_book['p_inv']
    work_sheet_q_inv = work_book['q_inv']
    df_data_p_dir = pd.DataFrame(work_sheet_p_dir.values)
    df_data_q_dir = pd.DataFrame(work_sheet_q_dir.values)
    df_data_p_inv = pd.DataFrame(work_sheet_p_inv.values)
    df_data_q_inv = pd.DataFrame(work_sheet_q_inv.values)

    # slice data to get data frames for thresholds
    pdir = df.to_numpy(df_data_p_dir.iloc[0:3000])
    qdir = df.to_numpy(df_data_q_dir.iloc[0:3000])
    pinv = df.to_numpy(df_data_p_inv.iloc[0:3000])
    qinv = df.to_numpy(df_data_q_inv.iloc[0:3000])

    return pdir,qdir,pinv,qinv


def get_weights():
    """
    get criteria weights
    :return: w
    """
    w = [0.333, 0.333, 0.334]  # pesos de los criterios
    return w


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




def perform_outranking(actions, limites, lam, p_dir, q_dir, p_inv, q_inv,iter_stochastic):
    w = get_weights()
    #p_dir, q_dir, p_inv, q_inv = get_umbrales()
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number of criteria
    n_lim = np.size(limites, 0)  # number of limits
    # -------------------------------------------------------
    # Calcula concordancia global directa e inversa entre cada par de acciones
    # cpda = conc_p_directa_actions(actions, p_dir[iter_stochastic], q_dir[iter_stochastic])
    # cpia = conc_p_inversa_actions(actions, p_inv[iter_stochastic], q_inv[iter_stochastic])
    # sigma_D_a = concordancia_D_actions(cpda, n_acc, n_cri, w)
    # sigma_I_a = concordancia_I_actions(cpia, n_acc, n_cri, w)

    freq_sigma_D_a = np.zeros((n_acc, n_acc))
    freq_sigma_I_a = np.zeros((n_acc, n_acc))
    for l in range (0,iter_stochastic):
        print (l)
            # Calcula concordancia global directa e inversa entre cada par de acciones
        cpda = conc_p_directa_actions(actions, p_dir[l], q_dir[l])
        cpia = conc_p_inversa_actions(actions, p_inv[l], q_inv[l])
        sigma_D_a = concordancia_D_actions(cpda, n_acc, n_cri, w)
        sigma_I_a = concordancia_I_actions(cpia, n_acc, n_cri, w)

        for i in range(0, n_acc):
            for j in range(0, n_acc):
                # if i==0 and j==2:
                    # print(i + 1, j + 1, ": ",
                    #     sigma_D_a[i][j],
                    #     sigma_I_a[j][i])
                if sigma_D_a[i][j] > lam:
                    freq_sigma_D_a[i][j] = freq_sigma_D_a[i][j] + 1
                if sigma_I_a[j][i] > lam:
                    freq_sigma_I_a[j][i] = freq_sigma_I_a[j][i] + 1
    for i in range(0, n_acc):
        for j in range(0, n_acc):
            print(i+1, j+1, ": ",
                  sigma_D_a[i][j], freq_sigma_D_a[i][j]/(iter_stochastic),
                  sigma_I_a[j][i], freq_sigma_I_a[j][i]/(iter_stochastic))
        print(" ")

    return 0


#########  MAIN ###############
def main():
    actions, centroids, limites = init_data('code.data.SSI.xlsx')
    p_dir, q_dir, p_inv, q_inv=random_thresholds('code.data.random_umbrales_SSI.xlsx')
    lam  = 0.5

    iter_stochastic=2

    perform_outranking(actions, limites,lam,  p_dir, q_dir, p_inv, q_inv,iter_stochastic)
if __name__ == '__main__':
    main()




