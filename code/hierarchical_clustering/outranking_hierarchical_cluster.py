import pandas as pd
from pandas import DataFrame as df
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
import openpyxl as px
import numpy as np
from src.outranking_clustering import get_weights, get_umbrales, conc_p_directa_actions, conc_p_inversa_actions, \
    concordancia_D_actions, concordancia_I_actions


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


def sigma_global(sigma_D_a,sigma_I_a,n_acc,lam):
    """
    computes the indifference indices between each pair of actions, does not consider any central action
    :param sigma_D:
    :param sigma_I:
    :param n_acc:
    :return: sigma_min,sigma_min_inverse
    """

    sigma_min = np.zeros((n_acc, n_acc))  # matrix of indifference,
    sigma_min_inverse = np.zeros((n_acc, n_acc)) #inverse matrix of indifference
    for i in range(0,n_acc):
        for j in range(0,n_acc):
            if i==j:
                sigma_min[i][j]=0
                sigma_min_inverse[i][j]=0
            else:
                if min(sigma_I_a[i][j],sigma_D_a[j][i])>=lam:
                    sigma_min[i][j]=min(sigma_I_a[i][j],sigma_D_a[j][i])
                else:
                    sigma_min[i][j]=0
                sigma_min_inverse[i][j]=1-sigma_min[i][j]


    return sigma_min,sigma_min_inverse


def perform_clustering(actions, lam):
    w = get_weights()
    p_dir, q_dir, p_inv, q_inv = get_umbrales()
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    sigma_min=np.size((n_acc,n_acc)) #matrix of indifference
    sigma_min_inverse = np.size((n_acc, n_acc)) #inverse matrix of indifference

    #computes direct concordance, on each criterion
    cpda = conc_p_directa_actions(actions, p_dir, q_dir)

    #computes inverse concordance, on each criterion
    cpia = conc_p_inversa_actions(actions, p_inv, q_inv)

    #computes global direct concordance
    sigma_D_a = concordancia_D_actions(cpda, n_acc, n_cri, w)
    #computes global inverse concordance
    sigma_I_a = concordancia_I_actions(cpia, n_acc, n_cri, w)

    #computes the indifferences and inverse-difference matrices
    sigma_min,sigma_min_inverse=sigma_global(sigma_D_a,sigma_I_a,n_acc,lam)

    for i in range(0, n_acc):
        for j in range(i, n_acc):
            print (sigma_min_inverse[i][j])
    #builds the hierarchy clustering
    Z=squareform(sigma_min_inverse)
    Z = hierarchy.linkage(Z, 'average')
    labels = hierarchy.fcluster(Z, t=5, criterion='maxclust')
    for i in range(0, n_acc):
        print (labels[i])
    fig = plt.figure(figsize=(10, 5))
    dn = hierarchy.dendrogram(Z)
    plt.show()

    return 0


#########  MAIN ###############
def main():
    actions, centroids, limites = init_data('SSI.xlsx')
    lam  = 0.6
    perform_clustering(actions,lam)


if __name__ == '__main__':
    main()




