import numpy as np
from clustering.base.cluster import get_ordered_centroids_4
from clustering.base.assignment import regla_desc
from outranking.concordance import *
from init.data_file import folder, init_data, get_metrics
from init.param import *
from outranking.concordance import conc_p_directa


def perform_outranking(actions, ext_centroids, n_acc, n_cri, n_lim, lam, beta, iter, p_dir, q_dir, p_inv, q_inv, iter_stochastic, w):
    # -------------------------------------------------------
    for k in range(0, iter):
        # calcula concordancia parcial directa e inversa (formulas (1) y (2))
        cpd = conc_p_directa(actions, ext_centroids, p_dir, q_dir)
        cpi = conc_p_inversa(actions, ext_centroids, p_inv, q_inv)

        # calcula concordancia global directa e inversa (formulas (3) y (4))
        sigma_D = concordancia_D(cpd, n_acc, n_lim, n_cri, w)
        sigma_I = concordancia_I(cpi, n_acc, n_lim, n_cri, w)


        # determina categoria de cada accion, usando regla descendente
        categoria = np.zeros((n_acc, n_lim), dtype=int)
        belonging = np.zeros((n_acc), dtype=int)

        categoria = regla_desc(categoria, belonging, n_acc, n_lim, sigma_I, sigma_D, lam)

        ext_centroids= get_ordered_centroids_4(belonging, ext_centroids, n_lim, n_cri, p_dir, p_inv, actions)


        print('--------------- ITERACION: {} -------------'.format(k+1))
        print('<CATEGORIAS>')
        # CALL PLOTTING HERE
        print('<CENTROIDES>')
        np.set_printoptions(precision=2)
        print (n_lim)
        print(ext_centroids[:])
        for i in range(0,n_acc):
            print (categoria[i][0],categoria[i][1],categoria[i][2],categoria[i][3],categoria[i][4])
    # ############################################
    return 0

#########  MAIN ###############
def main():
    iter, iter_stochastic = parameter_running(50,1)
    lam,beta = parameter_outranking(0.5,0.1)
    actions, centroids, ext_centroids = init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'SSI.xlsx',0,159,155,158,154,159)
    n_acc, n_cri, n_lim=get_metrics(actions, ext_centroids)
    p_dir, q_dir, p_inv, q_inv = get_umbrales([0.51,0.58,0.43],[0.25,0.29,0.22],[0.51,0.58,0.43],[0.25,0.29,0.22])
    w = get_weights([0.333, 0.333, 0.334])
    perform_outranking(actions, ext_centroids,n_acc, n_cri, n_lim,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)
if __name__ == '__main__':
    main()




