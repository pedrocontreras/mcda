
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


# #############################################################################
work_book = px.load_workbook('HDI.xlsx')
work_sheet = work_book['data']
df_data = pd.DataFrame(work_sheet.values)


print (df_data)

df_data = StandardScaler().fit_transform(df_data)


#neigh = NearestNeighbors(n_neighbors=2)
#nbrs = neigh.fit(df_data)
#distances, indices = nbrs.kneighbors(df_data)
#distances = np.sort(distances, axis=0)
#distances = distances[:,1]
#plt.plot(distances)
#plt.show()

# #############################################################################
# Compute DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples = 10)
dbscan.fit(df_data)
clusters=dbscan.labels_
colors = ['royalblue', 'maroon', 'forestgreen', 'mediumorchid', 'tan', 'deeppink', 'olive', 'goldenrod', 'lightcyan', 'navy']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])
plt.scatter(df_data[:,0], df_data[:,1], c=vectorizer(clusters))
plt.title("DBSCAN")
plt.show()
