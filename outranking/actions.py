import numpy as np
from init.param import *
from outranking.concordance import *
from init.data_file import *


def perform_outranking(actions, limites,n_acc, n_cri, n_lim,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w):

    freq_sigma_D_a = np.zeros((n_acc, n_acc))
    freq_sigma_I_a = np.zeros((n_acc, n_acc))
    for l in range (0,iter_stochastic):
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
    iter, iter_stochastic = parameter_running(50,1)
    lam,beta = parameter_outranking(0.5,0.1)
    actions, centroids, limites = init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'SSI.xlsx',0,159,155,158,154,159)
    n_acc, n_cri, n_lim=get_metrics(actions, limites)
    p_dir, q_dir, p_inv, q_inv=random_thresholds(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'random_umbrales_SSI.xlsx',0,3000)
    w = get_weights([0.333, 0.333, 0.334])
    perform_outranking(actions, limites,n_acc, n_cri, n_lim,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)
if __name__ == '__main__':
    main()




