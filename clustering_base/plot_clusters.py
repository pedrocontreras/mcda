import numpy as np
import matplotlib.pyplot as plt


def plot_centroids(actions, limites,k):
    fig = plt.figure(figsize=(6, 6))

    x = actions[:, 3:4]
    # plt.scatter(x, np.zeros_like(x))
    plt.plot(x, np.zeros_like(x), 'o', markersize=3, color='blue')
    y = limites[:, 3:4]
    #plt.scatter(y, np.zeros_like(y),  color='red')
    plt.plot(y, np.zeros_like(y), 'o', markersize=8, color='red')

    fig.savefig('{}.png'.format(k), dpi=fig.dpi)


#from apng import APNG
#APNG.from_files(["0.png", "1.png", "2.png", "3.png", "4.png", "5.png", "6.png", "7.png", "8.png", "9.png"], delay=100).save("result_3.png")