import math
from math import floor
from outranking.actions import *
from Promethee.Promethee_II import *

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def order_actions(actions):
    return actions

def boundaries(k,n_acc,sigma_min):
    b=np.zeros(k, dtype=int)
    L = np.zeros(n_acc)
    L[0]=0
    for i in range (1,n_acc):
        L[i]=L[i-1]+sigma_min[i][i-1]
    lsegment=1.0*L[n_acc-1]/k
    i=0
    for j in range(0,k):
        while lsegment*(j)>L[i]:
            i=i+1
            b[j]=i
    return b

def globalJ(sigma_min,n_acc, n_lim):
    """
    calcula la similaridad global J
    :param sigma_min: similaridad entre accion y cluster
    :param n: arreglo de número de elementos por cluster
    :param i: número de cluster
    :param j: número de acción en el array de acciones
    :return dJ: diferencia de similaridad global J
    """

    J=0.0
    for i in range(0, n_lim):
        for j in range(0, n_acc):
            J=J+sigma_min[i][j]
    return J

def difJminus(sigma_min,n,i,j):
    """
    calcula la diferencia de similaridad global J, cuando se mueve elemento de cluster inferior a superior
    :param sigma_min: similaridad entre accion y cluster
    :param n: arreglo de número de elementos por cluster
    :param i: número de cluster
    :param j: número de acción en el array de acciones
    :return dJ: diferencia de similaridad global J
    """
    dJ= (n[j]/(n[j]+1))*sigma_min[j][i]-(n[j-1]/(n[j-1]-1))*sigma_min[j-1][i]
    return dJ

def difJplus(sigma_min,n,i,j):
    """
    calcula la diferencia de similaridad global J, cuando se mueve elemento de cluster superior a inferior
    :param sigma_min: similaridad entre accion y cluster
    :param n: arreglo de número de elementos por cluster
    :param i: número de cluster
    :param j: número de acción en el array de acciones
    :return dJ: diferencia de similaridad global J
    """
    dJ= (n[j]/(n[j]+1))*sigma_min[j][i]-(n[j+1]/(n[j+1]-1))*sigma_min[j+1][i]
    return dJ


def Update(dJ,J):
    J=J+dJ
    return J

def n_clusters(k,b,n_ac):
    n=np.zeros(k)
    for j in range(1,k):
        n[j-1]=b[j]-b[j-1]
    n[k-1]=n_ac-b[k-1]
    return n


def centered_column(actions,h,first_action,last_action):
    sum = 0
    for i in range(first_action, last_action):
        sum=sum+actions[i][h]
    col=sum/(last_action-first_action)
    return col

def centroids(actions,k,n_cri,b,last_action):
    mu=np.zeros((k,n_cri))
    for j in range(0,k-1):
        for h in range(0,n_cri):
            mu[j][h]=centered_column(actions,h,b[j],b[j+1])
    for h in range(0,n_cri):
        mu[k-1][h]=centered_column(actions,h,b[k-1],last_action)
    return mu

def wkm_algorithm_old(actions,indices_sort,sigma_D,sigma_I,n_acc,n_cri,k,J,mu,n,b):
    """
    computa los clusters usando la estrategia del algoritmo Warped K-means y la regla ascendente de ELECTRE TRI-C
    :param actions: acciones ordenadas según outranking global
    :param sigma_min: similaridad entre accion y cluster
    :param n_acc: número de acciones
    :param n_cri: número de criterios
    :param k: número de clusters
    :param J: similaridad global J
    :param mu: centroides
    :param n: número de acciones en cada cluster
    :param b: fronteras de clusters
    :return b,mu
    """

    delta=0.0
    categoria=np.zeros((n_acc,k))

    #aplica proceso  según número de iteraciones transfers
    transfers=1
    while transfers<=30:
        for j in range(0,k):
            if j>0:
                 first=b[j]
                 last=first+floor(1.0*(1-delta)*(n[j]/2))
                 print("fl",first,last)
                 for i in range(first,last+1):
                    if n[j]>1 and min(sigma_I[i][j+1-1],sigma_D[j+1-1][i])>min(sigma_I[i][j+1],sigma_D[j+1][i]):
                        print("j>0")
                        b[j]=b[j]+1
                        n[j]=n[j]-1
                        n[j-1]=n[j-1]+1
                        mu=centroids(actions,k,n_cri,b,n_acc)
                        J=Update(min(sigma_I[i][j+1-1],sigma_D[j+1-1][i])-min(sigma_I[i][j+1],sigma_D[j+1][i]),J)

            # if j == 0:
            #      last = b[j+1]-1
            #      first = last - floor(1.0 * (1 - delta) * n[j] / 2)
            #      #print("fl",first,last)
            #      #last=first+floor(1.0*(1-delta)*(n[j]/2))
            #      for i in range(last, first-1,-1):
            #         if n[j] > 1 and min(sigma_I[i][j+1],sigma_D[j+1][i])<min(sigma_I[i][j+1+1],sigma_D[j+1+1][i]):#dJ>0:
            #             print("j=0")
            #             b[j+1] = b[j+1] - 1
            #             n[j] = n[j] - 1
            #             n[j + 1] = n[j + 1] + 1
            #             mu=centroids(actions,k,n_cri,b,n_acc)
            #             J=Update(min(sigma_I[i][j+1],sigma_D[j+1][i])-min(sigma_I[i][j+1+1],sigma_D[j+1+1][i]),J)
        transfers=transfers+1
        print (n)
    for j in range(0,k):
        if j<k-1:
            for i in range(b[j],b[j+1]):
                categoria[indices_sort[i]][j]=1
        else:
            for i in range(b[j], n_acc):
                categoria[indices_sort[i]][j] = 1
    # for i in range(0,n_acc):
    #     print(i,categoria[i])

    return b,mu,n,categoria


def wkm_algorithm(actions,indices_sort,sigma_D,sigma_I,n_acc,n_cri,k,J,mu,n,b):
    """
    computa los clusters usando la estrategia del algoritmo Warped K-means y la regla ascendente de ELECTRE TRI-C
    :param actions: acciones ordenadas según outranking global
    :param sigma_min: similaridad entre accion y cluster
    :param n_acc: número de acciones
    :param n_cri: número de criterios
    :param k: número de clusters
    :param J: similaridad global J
    :param mu: centroides
    :param n: número de acciones en cada cluster
    :param b: fronteras de clusters
    :return b,mu
    """

    delta=0.0
    categoria=np.zeros((n_acc,k))

    #aplica proceso  según número de iteraciones transfers
    transfers=1
    while transfers<=30:
        for j in range(0,k):
            if j>0:
                 first=b[j]
                 last=first+floor(1.0*(1-delta)*(n[j]/2))
                 for i in range(first,last+1):
                    if n[j]>1 and min(sigma_I[i][j+1-1],sigma_D[j+1-1][i])>min(sigma_I[i][j+1],sigma_D[j+1][i]):
                        b[j]=b[j]+1
                        n[j]=n[j]-1
                        n[j-1]=n[j-1]+1
                        mu=centroids(actions,k,n_cri,b,n_acc)
                        J=Update(min(sigma_I[i][j+1-1],sigma_D[j+1-1][i])-min(sigma_I[i][j+1],sigma_D[j+1][i]),J)

            for j in range(k-1, 0,-1):
                if j < k-1:
                    last = b[j+1]-1
                    first = last - floor(1.0 * (1 - delta) * (n[j] / 2))
                    #print("fl", first, last)
                    for i in range(last, first - 1, -1):
                        if n[j] > 1 and min(sigma_I[i][j+1],sigma_D[j+1][i])<min(sigma_I[i][j+1+1],sigma_D[j+1+1][i]):#dJ>0:
                            b[j+1] = b[j+1] - 1
                            n[j] = n[j] - 1
                            n[j + 1] = n[j + 1] + 1
                            mu=centroids(actions,k,n_cri,b,n_acc)
                            J=Update(min(sigma_I[i][j+1],sigma_D[j+1][i])-min(sigma_I[i][j+1+1],sigma_D[j+1+1][i]),J)

            # if j == 0:
            #      last = b[j+1]-1
            #      first = last - floor(1.0 * (1 - delta) * n[j] / 2)
            #      #print("fl",first,last)
            #      #last=first+floor(1.0*(1-delta)*(n[j]/2))
            #      for i in range(last, first-1,-1):
            #         if n[j] > 1 and min(sigma_I[i][j+1],sigma_D[j+1][i])<min(sigma_I[i][j+1+1],sigma_D[j+1+1][i]):#dJ>0:
            #             print("j=0")
            #             b[j+1] = b[j+1] - 1
            #             n[j] = n[j] - 1
            #             n[j + 1] = n[j + 1] + 1
            #             mu=centroids(actions,k,n_cri,b,n_acc)
            #             J=Update(min(sigma_I[i][j+1],sigma_D[j+1][i])-min(sigma_I[i][j+1+1],sigma_D[j+1+1][i]),J)
        transfers=transfers+1
        #print (n)
    for j in range(0,k):
        if j<k-1:
            for i in range(b[j],b[j+1]):
                categoria[indices_sort[i]][j]=1
        else:
            for i in range(b[j], n_acc):
                categoria[indices_sort[i]][j] = 1
    # for i in range(0,n_acc):
    #     print(i,categoria[i])

    return b,mu,n,categoria

def sumAscendente(freq_acceptability,categoria,indices_sort,k,n_acc):
    for i in range(0,n_acc):
        for j in range(0,k):
            freq_acceptability[indices_sort[i]][j]=freq_acceptability[indices_sort[i]][j]+categoria[indices_sort[i]][j]
    return freq_acceptability

def aceptabilidad(iter_stochastic,freq_acceptability,k,n_acc):
    acceptability=np.zeros((n_acc, k))
    for i in range(0,n_acc):
        for j in range(0,k):
            acceptability[i][j]=freq_acceptability[i][j]/float(iter_stochastic)

    return acceptability

def perform_clustering(actions, ext_centroids, n_acc, n_cri, n_lim, n_cent, lam, beta, iter, p_dir, q_dir, p_inv, q_inv, iter_stochastic, w):

    freq_no_check=0.0
    freq_acceptability = np.zeros((n_acc, n_cent))
    # -------------------------------------------------------

    for l in range (0,iter_stochastic):
        print (l)

        #computes the Phi netflow values among actions, using the Promethee II method
        Phi,sigma_D_a,actions,indices_sort=promethee_method(actions, n_acc, n_cri, p_dir[l], q_dir[l],w)

        #print(actions)
        #Defining initial clusters' boundaries, following the WKM algorithm
        b=boundaries(n_cent,n_acc,sigma_D_a)
        #print("b", b)

        #Defining the initial number of actions inside each cluster
        n=n_clusters(n_cent,b,n_acc)
        #print("n", n)

        #Defining initial centroids and number of actions inside each cluster
        mu=centroids(actions,n_cent, n_cri,b,n_acc)
        #print ("mu", mu)


        for i in range(0,n_cent):
            ext_centroids[i+1]=mu[i]

            #computa la concordancia entre acciones y centroides
        cpd=conc_p_directa(actions, ext_centroids, p_dir[l], q_dir[l])
        cpi=conc_p_inversa(actions, ext_centroids, p_inv[l], q_inv[l])

        sigma_D=concordancia_D(cpd, n_acc, n_lim, n_cri, w)
        sigma_I=concordancia_I(cpi, n_acc, n_lim, n_cri, w)

        #computes the initial global similarity function, as defined in K-means, but adapted to outranking models
        J=globalJ(sigma_D,n_acc, n_cent)

        #Iterating process to compute clusters (Warped-KM algorithm)
        b,mu,n,categoria=wkm_algorithm(actions,indices_sort,sigma_D,sigma_I,n_acc,n_cri,n_cent,J,mu,n,b)

        freq_acceptability = sumAscendente(freq_acceptability, categoria, indices_sort, n_cent, n_acc)

        # for i in range(0,n_acc):
        #     print(i,freq_acceptability[i])
    # counts how many times the weak separability condition is not satisfied
    # print(ext_centroids)

    acceptability=aceptabilidad(iter_stochastic, freq_acceptability, n_cent, n_acc)

    for i in range(0,n_acc):
        for j in range(0,n_cent):
            print(str("%.2f" %  round(acceptability[i][j],2)),"\t", end="")
        print (" ")
    print ("")

    return b,mu,n


#########  MAIN ###############
def main():
    iter, iter_stochastic = parameter_running(50,1000)
    lam,beta = parameter_outranking(0.5,0.1)
    actions, centroids, ext_centroids = init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'HDI.xlsx',189,4)
    n_acc, n_cri, n_lim,n_cent=get_metrics(actions, ext_centroids)
    #p_dir, q_dir, p_inv, q_inv = get_umbrales([0.19,0.14,0.10],[0.1,0.07,0.05],[0.19,0.14,0.10],[0.1,0.07,0.05])
    p_dir, q_dir, p_inv, q_inv=random_thresholds(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'random_umbrales.xlsx',0,3000)
    w = get_weights([0.333, 0.333, 0.334])
    #w=random_weights(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'Weights.xlsx',0,1000)
    b,mu,n=perform_clustering(actions, ext_centroids,n_acc, n_cri, n_lim,n_cent,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)
    #verificando los centroides finales
    print ("")
    print ("b",b)
    print ("n",n)
    print ("mu",mu)

if __name__ == '__main__':
    main()




