from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
import numpy as np
from clustering_base.init_data_file import folder, init_data, get_metrics
from outranking_base.outranking_among_actions import conc_p_directa_actions
from outranking_base.outranking_among_actions import conc_p_inversa_actions
from outranking_base.outranking_among_actions import concordancia_D_actions
from outranking_base.outranking_among_actions import concordancia_I_actions
from parameters.init_parameter import lambda_parameter, get_umbrales, get_weights, beta_parameter, stochastic_iter, \
    km_iter, parameter_running, parameter_outranking


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


def perform_clustering(actions, limites,n_acc, n_cri, n_lim,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w):
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
    iter, iter_stochastic = parameter_running(50,1)
    lam,beta = parameter_outranking(0.5,0.1)
    actions, centroids, limites = init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'SSI.xlsx',0,159,155,158,154,159)
    n_acc, n_cri, n_lim=get_metrics(actions, limites)
    p_dir, q_dir, p_inv, q_inv = get_umbrales([0.51,0.58,0.43],[0.25,0.29,0.22],[0.51,0.58,0.43],[0.25,0.29,0.22])
    w = get_weights([0.333, 0.333, 0.334])
    perform_clustering(actions, limites,n_acc, n_cri, n_lim,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)
if __name__ == '__main__':
    main()




