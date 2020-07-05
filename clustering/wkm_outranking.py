import math
from math import floor
from outranking.actions import *
from Promethee.Promethee_II import *


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
    #print ("j", j, n[j],n[j-1])
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
    #print ("j,i",j,i,sigma_min[j][i],sigma_min[j+1][i])
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

def wkm_algorithm(actions,sigma_min,n_acc,n_cri,k,J,mu,n,b):
    delta=0.0
    transfers=1
    while transfers<=1:
        #transfers=0
        for j in range(0,k):
            #print("aqui")
            if j>0:
                #print("aqui 1")
                first=b[j]
                last=first+floor(1.0*(1-delta)*(n[j]/2))
            # else:
            #     #print("aqui 2")
            #     first=0
            #     last=floor(1.0*(1-delta)*(n[0]/2))
                for i in range(first,last+1):
                    dJ=difJminus(sigma_min,n,i,j)
                    if n[j]>1 and dJ>0:
                        #transfers=1
                        b[j]=b[j]+1
                        n[j]=n[j]-1
                        n[j-1]=n[j-1]+1
                        mu=centroids(actions,k,n_cri,b,n_acc)
                        J=Update(dJ,J)
                    #else:
                    #    break
            if j < k-1:
                last = b[j+1]-1
                first = last - floor(1.0 * (1 - delta) * n[j] / 2)
            # else:
            #     last=n_acc-1
            #     first=last-floor(1.0 * (1 - delta) * n[k-1] / 2)
                for i in range(last, first -1, -1):
                    dJ = difJplus(sigma_min, n, i, j)
                    #print (j)
                    if n[j] > 1 and dJ>0:
                        #transfers = 1
                        b[j+1] = b[j+1] - 1
                        n[j] = n[j] - 1
                        n[j + 1] = n[j + 1] + 1
                        mu=centroids(actions,k,n_cri,b,n_acc)
                        J=Update(dJ,J)
                    #else:
                    #    break
        transfers=transfers+1
        #print (j)
        #verificando si la similitud global se reduce
        # print (J)
        # print ("")

    return b,mu

def perform_clustering(actions, ext_centroids, n_acc, n_cri, n_lim, n_cent, lam, beta, iter, p_dir, q_dir, p_inv, q_inv, iter_stochastic, w):

    #computes the Phi netflow values among actions, using the Promethee II method
    Phi,sigma_D_a,actions=promethee_method(actions, n_acc, n_cri, p_dir, q_dir, w)

    print(actions)
    #Defining initial clusters' boundaries, following the WKM algorithm
    b=boundaries(n_cent,n_acc,sigma_D_a)
    print("b", b)

    #Defining the initial number of actions inside each cluster
    n=n_clusters(n_cent,b,n_acc)
    print("n", n)

    #Defining initial centroids and number of actions inside each cluster
    mu=centroids(actions,n_cent, n_cri,b,n_acc)
    print ("mu", mu)

    #computes the initial global similarity function, as defined in K-means, but adapted to outranking models
    J=globalJ(sigma_D_a,n_acc, n_cent)

    #Iterating process to compute clusters (Warped-KM algorithm)
    b,mu=wkm_algorithm(actions,sigma_D_a,n_acc,n_cri,n_cent,J,mu,n,b)

    #verificando los centroides finales
    print (b)

    return mu


#########  MAIN ###############
def main():
    iter, iter_stochastic = parameter_running(50,1)
    lam,beta = parameter_outranking(0.5,0.1)
    actions, centroids, ext_centroids = init_data(str(folder("/Users/jpereirar/Documents/GitHub/mcda/data"))+'/'+'SSI.xlsx',154,3)
    n_acc, n_cri, n_lim,n_cent=get_metrics(actions, ext_centroids)
    p_dir, q_dir, p_inv, q_inv = get_umbrales([0.51,0.58,0.43],[0.25,0.29,0.22],[0.51,0.58,0.43],[0.25,0.29,0.22])
    w = get_weights([0.333, 0.333, 0.334])
    mu=perform_clustering(actions, ext_centroids,n_acc, n_cri, n_lim,n_cent,lam,beta, iter, p_dir, q_dir, p_inv, q_inv,iter_stochastic,w)
    print ("")
    print (mu)

if __name__ == '__main__':
    main()




