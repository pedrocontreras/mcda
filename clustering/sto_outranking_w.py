import numpy as np
from clustering.base import *
from clustering.base.assignment import regla_desc
from clustering.base.cluster import get_ordered_centroids_4
from outranking.concordance import *
from init.param import *
from init.data_file import *


def sumAscendente(freq_acceptability,categoria,n_lim,n_acc):
    for i in range(0,n_acc):
        for j in range(0,n_lim):
            freq_acceptability[i][j]=freq_acceptability[i][j]+categoria[i][j]
    return freq_acceptability

def aceptabilidadDescendente(iter_stochastic,freq_acceptability,n_lim,n_acc):
    for i in range(0,n_acc):
        for j in range(0,n_lim):
            freq_acceptability[i][j]=freq_acceptability[i][j]/float(iter_stochastic)
            print(str("%.2f" %  round(freq_acceptability[i][j],2)),"\t", end="")
        print (" ")
    print ("")

def check_separability(limites,n_lim,n_cri):
    no_check='False'
    for i in range(2,n_lim):
        for j in range(0,n_cri):
            if limites[i][j]<limites[i-1][j]:
                no_check='True'
    return no_check


def perform_outranking(actions, limites,n_acc, n_cri, n_lim,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w):
    freq_no_check=0.0
    freq_acceptability = np.zeros((n_acc, n_lim))
    # -------------------------------------------------------

    for l in range (0,iter_stochastic):
        print (l)
        for k in range(0, iter):
            # calcula concordancia parcial directa e inversa (formulas (1) y (2)
            cpd = conc_p_directa(actions, limites, p_dir, q_dir)
            cpi = conc_p_inversa(actions, limites, p_inv, q_inv)

            # calcula concordancia global directa e inversa (formulas (3) y (4)
            sigma_D = concordancia_D(cpd, n_acc, n_lim, n_cri, w[l])
            sigma_I = concordancia_I(cpi, n_acc, n_lim, n_cri, w[l])


            # determina categoria de cada accion, usando regla descendente
            categoria = np.zeros((n_acc, n_lim), dtype=int)
            belonging = np.zeros((n_acc), dtype=int)

            categoria = regla_desc(categoria,belonging, n_acc, n_lim, sigma_I, sigma_D, lam)

            limites=get_ordered_centroids_4(belonging,limites,n_lim,n_cri,p_dir,p_inv,actions)


        freq_acceptability=sumAscendente(freq_acceptability, categoria, n_lim, n_acc)

        #counts how many times the weak separability condition is not satisfied
        #print(limites)
        if check_separability(limites, n_lim, n_cri) == 'True':
            freq_no_check = freq_no_check + 1

    aceptabilidadDescendente(iter_stochastic,freq_acceptability,n_lim,n_acc)
    print (freq_no_check/iter_stochastic)
    return 0


#########  MAIN ###############
def main():
    iter, iter_stochastic = parameter_running(50,1)
    lam,beta = parameter_outranking(0.5,0.1)
    actions, centroids, limites = init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'HDI.xlsx',0,189,190,194,189,195)
    n_acc, n_cri, n_lim=get_metrics(actions, limites)
    w=random_weights(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'Weights.xlsx',0,1000)
    p_dir, q_dir, p_inv, q_inv=get_umbrales([0.19,0.14,0.10],[0.09,0.07,0.05],[0.19,0.14,0.10],[0.09,0.07,0.05])
    perform_outranking(actions, limites,n_acc, n_cri, n_lim,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)

    print (centroids)
if __name__ == '__main__':
    main()




