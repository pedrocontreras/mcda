import pandas as pd
from pandas import DataFrame as df
import openpyxl as px
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    work_book = px.load_workbook(file_path)
    work_sheet = work_book['data']
    df_data = pd.DataFrame(work_sheet.values)
    print(df_data)
    return df_data

def get_kmeans(df_data):
    from sklearn.cluster import KMeans
    km = KMeans(
        n_clusters=7, init='random',
        n_init=10, max_iter=300,
        tol=1e-04, random_state=0
    )
    y_km = km.fit_predict(df_data)
    print(y_km)
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
    plt.show()


def get_hierarchical_ward(df_data):

    return 0

def get_near_neibor(df_data):

    return 0


def plot_scatter(df_data):
    plt.scatter()
    plt.show()


def main():
    file_path = 'HDI.xlsx'
    df = load_data(file_path)
    #get_kmeans(df)

    kmeans_elbow(df) # method to determine the number of clusters


if __name__ == '__main__':
    main()