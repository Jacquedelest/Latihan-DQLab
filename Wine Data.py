# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:42:34 2021

@author: Jacque de l'est
"""

import numpy as numpy
import pandas as pd
from sklearn.datasets import load_wine

data = load_wine()
wine = pd.DataFrame(data.data, columns=data.feature_names)
print(wine.shape)
print(wine.columns)
print(wine)
print(wine.iloc[:,:3].describe())


"Plotting data"
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

scatter_matrix(wine.iloc[:,[0,5]])
plt.savefig("plot.png")
plt.show()


"Standardization"
X = wine[['alcohol', 'total_phenols']] 

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
scale.fit(X)
print(scale.mean_)
print(scale.scale_)

X_scaled = scale.transform(X)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))


"Elbow Method"
from sklearn.cluster import KMeans
# calculate distortion for a range of number of cluster
inertia = []
for i in numpy.arange(1, 11):
    km = KMeans(
        n_clusters=i
    )
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# plot
plt.plot(numpy.arange(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.savefig("plot.png")
plt.show()
#Inertia decreases, k increases

"K-means modelling"
from sklearn.cluster import KMeans
# instantiate the model
kmeans = KMeans(n_clusters=3)
# fit the model
kmeans.fit(X_scaled)
# make predictions
y_pred = kmeans.predict(X_scaled)
print(y_pred)
print(kmeans.cluster_centers_)


"Example of New Data"
#alchohol 13 and total phenols 2.5
X_new = numpy.array([[13, 2.5]])
# Standardize new data
X_new_scaled = scale.transform(X_new)

print(kmeans.predict(X_new_scaled))


"Visualization Model"
import matplotlib.pyplot as plt
# plot the scaled data
plt.scatter(X_scaled[:,0],
X_scaled[:,1],
c= y_pred)
# identify the centroids
plt.scatter(kmeans.cluster_centers_[:, 0],
kmeans.cluster_centers_[:, 1],
marker="*",
s = 250,
c = [0,1,2],
edgecolors='k')
plt.xlabel('Alcohol'); plt.ylabel('Total Phenols')
plt.title('K-Means (k=3)')
plt.savefig("plot.png")
plt.show()
