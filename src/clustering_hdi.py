import pandas as pd
from pandas import DataFrame as df
import openpyxl as px
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    work_book = px.load_workbook(file_path)
    work_sheet = work_book['data']
    df_data = pd.DataFrame(work_sheet.values)
    #print(df_data)
    return df_data

def get_kmeans(df_data):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=3, init='random'  )
    y_km = km.fit(df_data)
    return y_km


def kmeans_elbow(df_data):
    from sklearn.cluster import KMeans
    distortions = []
    for i in range(1, 11):
        km = KMeans(
            n_clusters=i, init='random',
            n_init=10, max_iter=300,
            tol=1e-04, random_state=0
        )
        km.fit(df_data)
        distortions.append(km.inertia_)

    # plot
    plt.plot(range(1, 11), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('Elbow Method For Optimal k')
    plt.show()


def plot_3d(df_data):
    ax = plt.axes(projection='3d')
    x = df_data.iloc[:, 0]
    y = df_data.iloc[:, 1]
    z = df_data.iloc[:, 2]
    ax.scatter3D(x, y, z, 'gray')
    return 0


def get_hierarchical_ward(df_data):

    return 0

def get_near_neibor(df_data):

    return 0


def plot_scatter(df_data):
    return 0


def plot_scatter3D(file_path):

    df = load_data(file_path)
    y_km = get_kmeans(df)
    cnt = y_km.cluster_centers_
    print(cnt)
    clusters = y_km.labels_
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(projection='3d')
    ax.scatter3D(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], c=clusters)
    ax.scatter3D(cnt[:, 0], cnt[:, 1], cnt[:, 2], s=150, c='r', marker='*', label='Centroid')

    ax.set_title("HDI Clustering")
    ax.set_xlabel("Life Expectancy")
    ax.set_ylabel("Expected years of schooling")
    ax.set_zlabel("GNI")
    plt.show()
    return 0


def main():
    file_path = 'HDI.xlsx'
    plot_scatter3D(file_path)

if __name__ == '__main__':
    main()