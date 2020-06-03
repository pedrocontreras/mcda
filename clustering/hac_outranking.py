from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
from outranking.actions import *


def perform_clustering(actions, ext_centroids, n_acc, n_cri, n_lim, lam, beta, iter, p_dir, q_dir, p_inv, q_inv, iter_stochastic, w):
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
    actions, centroids, ext_centroids = init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'SSI.xlsx',0,159,155,158,154,159)
    n_acc, n_cri, n_lim=get_metrics(actions, ext_centroids)
    p_dir, q_dir, p_inv, q_inv = get_umbrales([0.51,0.58,0.43],[0.25,0.29,0.22],[0.51,0.58,0.43],[0.25,0.29,0.22])
    w = get_weights([0.333, 0.333, 0.334])
    perform_clustering(actions, ext_centroids,n_acc, n_cri, n_lim,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)
if __name__ == '__main__':
    main()




