from clustering.base.cluster import get_ordered_centroids_4
from clustering.base.assignment import regla_desc
from init.param import parameter_running, parameter_outranking, get_weights
from init.data_file import folder, init_data, random_thresholds, get_metrics
from outranking.concordance import *



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



def perform_outranking(actions, ext_centroids, n_acc, n_cri, n_lim, n_cent, lam, beta, iter, p_dir, q_dir, p_inv, q_inv, iter_stochastic, w):
    freq_no_check=0.0
    freq_acceptability = np.zeros((n_acc, n_lim))
    # -------------------------------------------------------

    for l in range (0,iter_stochastic):
        print (l)
        for k in range(0, iter):
            # calcula concordancia parcial directa e inversa (formulas (1) y (2)
            cpd = conc_p_directa(actions, ext_centroids, p_dir[l], q_dir[l])
            cpi = conc_p_inversa(actions, ext_centroids, p_inv[l], q_inv[l])

            # calcula concordancia global directa e inversa (formulas (3) y (4)
            sigma_D = concordancia_D(cpd, n_acc, n_lim, n_cri, w)
            sigma_I = concordancia_I(cpi, n_acc, n_lim, n_cri, w)

            # determina categoria de cada accion, usando regla descendente
            categoria = np.zeros((n_acc, n_lim), dtype=int)
            belonging = np.zeros((n_acc), dtype=int)

            categoria = regla_desc(categoria,belonging, n_acc, n_lim, sigma_I, sigma_D, lam)

            ext_centroids=get_ordered_centroids_4(belonging,ext_centroids,n_lim,n_cri,p_dir[l],p_inv[l],actions)


        freq_acceptability=sumAscendente(freq_acceptability, categoria, n_lim, n_acc)

        #counts how many times the weak separability condition is not satisfied
        #print(ext_centroids)
        if check_separability(ext_centroids, n_lim, n_cri) == 'True':
            freq_no_check = freq_no_check + 1


    aceptabilidadDescendente(iter_stochastic,freq_acceptability,n_lim,n_acc)
    print (freq_no_check/iter_stochastic)
    return ext_centroids


#########  MAIN ###############
def main():
    iter, iter_stochastic = parameter_running(50,1)
    lam,beta = parameter_outranking(0.5,0.1)
    actions, centroids, ext_centroids= init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'SSI.xlsx',154,3)
    n_acc, n_cri, n_lim, n_cent=get_metrics(actions, ext_centroids)
    p_dir, q_dir, p_inv, q_inv=random_thresholds(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'random_umbrales_SSI.xlsx',0,3000)
    w = get_weights([0.333, 0.333, 0.334])
    ext_centroids=perform_outranking(actions, ext_centroids,n_acc, n_cri, n_lim,n_cent,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)
    print (ext_centroids)
if __name__ == '__main__':
    main()




