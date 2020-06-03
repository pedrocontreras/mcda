import numpy as np

def k_means(categoria, actions, limites):
    n_acc = np.size(actions, 0)  # number of acciones
    n_cri = np.size(actions, 1)  # number if criteria
    n_lim = np.size(limites, 0)  # number of limits

    return 0

def get_ordered_centroids(categoria, actions, limites, n_acc, n_lim, n_cri):
    """
    :param categoria: array that contains categoria
    :param actions: array with the acciones
    :param limites: array with the limits
    :param n_acc: number of acciones
    :param n_lim: number of limits
    :param n_cri: number of criteria
    :return: limites
    """
    suma = np.zeros((n_lim, n_cri))
    freq_categoria = np.zeros(n_lim)

    for i in range(0, n_acc):
        for j in range(1, n_lim - 1):
            if categoria[i][j] == 1:
                freq_categoria[j] = freq_categoria[j] + 1
                for h in range(0, n_cri):
                    suma[j][h] = suma[j][h] + actions[i][h]

    for j in range(1, n_lim - 1):
        if freq_categoria[j] > 0:
            for h in range(0, n_cri):
                limites[j][h] = 1.0 * suma[j][h] / freq_categoria[j]

    return limites


def get_ordered_centroids_2(categoria, actions, limites, p_dir,p_inv,n_acc, n_lim, n_cri):
    """
    :param categoria: array that contains categoria
    :param actions: array with the acciones
    :param limites: array with the limits
    :param n_acc: number of acciones
    :param n_lim: number of limits
    :param n_cri: number of criteria
    :return: limites
    """
    suma = np.zeros((n_lim, n_cri))
    freq_categoria = np.zeros(n_lim)

    for i in range(0, n_acc):
        for j in range(1, n_lim - 1):
            if categoria[i][j] == 1:
                freq_categoria[j] = freq_categoria[j] + 1
                concordante = 'true'
                for h in range(n_cri-1, 0,-1):
                    if (actions[i][h]-limites[j][h]<=p_dir[h]) and (limites[j][h]-actions[i][h]<=p_inv[h]):
                        concordante='false'
                if concordante == 'true':
                    for h in range(0, n_cri):
                        suma[j][h] = suma[j][h] + actions[i][h]
                else:
                    freq_categoria[j]=freq_categoria[j]-1

    for j in range(1, n_lim - 1):
        if freq_categoria[j] > 0:
            for h in range(0, n_cri):
                limites[j][h] = 1.0 * suma[j][h] / freq_categoria[j]

    return limites


def minimo_p_accion(n_acc,categoria,cati,i,sigma_D_a,sigma_I_a):
    minimo=1
    jmin=0

    for k in range(0,n_acc):
        if categoria[k][cati]==1:
            if minimo>min(sigma_D_a[i][k],sigma_I_a[k][i]):
                minimo=min(sigma_D_a[i][k],sigma_I_a[k][i])
                jmin=k

    return jmin


def get_ordered_centroids_3(categoria,n_lim,n_acc,sigma_D_a,sigma_I_a,beta,n_cri,limites,actions):
    #print ("new centroids")
    maximo = np.zeros((n_lim,), dtype=int)
    yleast = np.zeros((n_acc,), dtype=int)  # yleast alternative for each alternative in a category
    # inicializa la matriz de pertenencia de clases, en cada round de simulacion
    izero = np.zeros((n_acc))  # minimum indifference for a given alternative belonging to a category
    for j in range(1, n_lim-1):
        hay='false'
        for i in range(0, n_acc):
            #print("categoria, alternativa mas lejana para accion i en categoria j: ")
            if categoria[i][j] == 1:
                yleast[i] = minimo_p_accion(n_acc,categoria, j, i, sigma_D_a, sigma_I_a)
                izero[i] = min(sigma_D_a[i][yleast[i]], sigma_I_a[yleast[i]][i])
                #print ("i", i, "yleast[i]", yleast[i], "izero[i]", izero[i])
                if izero[i]<=beta:
                    maximo[j]=yleast[i]
                    hay = 'true'
                    #print (j, i, yleast[i], izero[i],maximo[j])
        #print (hay)
        for i in range(0, n_acc):
            if categoria[i][j] == 1 and izero[i] <= beta:
                #print ("izero: ",izero[i],izero[maximo[j]])
                if izero[i] > izero[maximo[j]]:
                    maximo[j] = i
        if hay=='true':
            #print ("centroide de categoria ",j, ":", maximo[j])
            for h in range(0, n_cri):
                limites[j][h] = actions[maximo[j]][h]
        #else:
                #print ("categoria ", j, "sin acciones")
        hay = 'false'

    return limites

def get_inner_actions(belonging,limites,j,h,p_dir,p_inv,actions):
    c=np.where(belonging==j)
    a=[]
    for i in range(0,np.size(c)):
        if actions[c[0][i]][h] >= limites[j][h]-p_inv[h] and actions[c[0][i]][h] <= limites[j][h]+p_dir[h]:
            a.append(actions[c[0][i]][h])
    a.sort()
    return a

def get_ordered_centroids_4(belonging,limites,n_lim,n_cri,p_dir,p_inv,actions):
    for j in range(0,n_lim):
        for h in range(0,n_cri):
            a=get_inner_actions(belonging, limites, j, h, p_dir, p_inv, actions)
            if not a or j==0 or j==n_lim-1:
                limites[j][h]=limites[j][h]
            else:
                if limites[j-1][h]+p_dir[h]<=np.mean(a)-p_inv[h] and np.mean(a)+p_dir[h]<=limites[j+1][h]-p_inv[h]:
                    limites[j][h]=np.mean(a)
    return limites


