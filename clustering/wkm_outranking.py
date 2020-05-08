from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
from outranking.actions import *

def boundaries(n_lim,n_acc,sigma_min):
    b=np.zeros(n_lim)
    L = np.zeros(n_lim)
    L[1]=0
    for i in range (2,n_acc+1):
        L[i]=L[i-1]+sigma_min[i][i-1]
    lsegment=1.0*L[n_acc]/n_lim
    i=1
    for j in range(1,n_lim+1):
        i=i+1
        b[j]=i
    return b

def wkm_algorithm(actions,n_lim):
    b=np.zeros(n_lim)
    c=np.zeros(n_lim)

    for j in range (1,n_lim+1):
        # Computar c_j,n_j,J
        c=0
    return b,c

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

    #Performing ELECTRE III to obtain a complete pre-order of actions

    #Defining initial clusters' bounds
    b=boundaries(n_lim,n_acc,sigma_min)

    #Iterating process to compute clusters (WKM algorithm)

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




