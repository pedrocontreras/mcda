import numpy as np
from clustering.simple_outranking import perform_outranking
from init.data_file import init_data, folder, get_metrics, random_thresholds
from init.param import parameter_running, parameter_outranking, get_weights, get_umbrales
from outranking.concordance import conc_p_directa_actions
from outranking.concordance import conc_p_inversa_actions
from outranking.concordance import concordancia_D_actions
from outranking.concordance import concordancia_I_actions

def outcoming_flow(sigma_D_a,u,n_acc):
    sum=0.0
    for j in range(0,n_acc):
        if u!=j:
            sum=sum+sigma_D_a[u][j]
    return sum/(n_acc-1)

def pos_flow(sigma_D_a,n_acc):
    pflow  = np.zeros(n_acc)
    for i in range(0,n_acc):
        pflow[i]=outcoming_flow(sigma_D_a,i,n_acc)
    return pflow

def incoming_flow(sigma_D_a,u,n_acc):
    sum=0.0
    for j in range(0,n_acc):
        if u!=j:
            sum=sum+sigma_D_a[j][u]
    return sum/(n_acc-1)

def neg_flow(sigma_D_a,n_acc):
    ngflow  = np.zeros(n_acc)
    for i in range(0,n_acc):
        ngflow[i]=incoming_flow(sigma_D_a,i,n_acc)
    return ngflow

def netflow(pos,neg,n_acc):
    nflow  = np.zeros(n_acc)
    for i in range(0,n_acc):
        nflow[i]=pos[i]-neg[i]
    return nflow

def perform_outranking(actions, ext_centroids, n_acc, n_cri, n_lim, n_cent, lam, beta, iter, p_dir, q_dir, p_inv, q_inv, iter_stochastic, w):
    # computes direct concordance, on each criterion
    cpda = conc_p_directa_actions(actions, p_dir, q_dir)

    # computes global direct concordance
    sigma_D_a = concordancia_D_actions(cpda, n_acc, n_cri, w)

    #computes the positive outranking flow
    pflow=pos_flow(sigma_D_a,n_acc)
    #computes the negative outranking flow
    nflow=neg_flow(sigma_D_a,n_acc)

    #computes the net outranking flow defined in PROMETHEE II
    Phi=netflow(pflow,nflow,n_acc)
    return Phi



#########  MAIN ###############
def main():
    iter, iter_stochastic = parameter_running(50,1)
    lam,beta = parameter_outranking(0.0,0.1)
    actions, centroids, ext_centroids= init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'SSI.xlsx',154,3)
    n_acc, n_cri, n_lim, n_cent=get_metrics(actions, ext_centroids)
    p_dir, q_dir, p_inv, q_inv = get_umbrales([0.51,0.58,0.43],[0.25,0.29,0.22],[0.51,0.58,0.43],[0.25,0.29,0.22])
    w = get_weights([0.333, 0.333, 0.334])
    Phi=perform_outranking(actions, ext_centroids,n_acc, n_cri, n_lim,n_cent,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)
    print (Phi)
if __name__ == '__main__':
    main()
