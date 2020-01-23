'''
Created on 10 Apr 2018
This is a first implementation of the DBSCAN algorithm on a random set of data.
@author: Fabio C.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from Clustering.my_points import sample_set

# X, label = make_moons(n_samples=20, noise=0.1, random_state=19)
# print(X)
# print(X[:5,])

data = np.loadtxt('density.dat')
 
  
#X = sample_set(data)

X = np.array([[0, 0.2], [0, 0.3], [0, 0.5], [0, 0.8], [0, 0.85], [0, 1.1],\
              [0, 1.3], [0, 1.5], [0, 1.6], [0, 2], [0.5, 0.5], [0.6, 0.6],\
              [0.7, 0.7], [0.8, 0.8], [0.85, 0.85], [1, 1], [1.2, 1.2],\
              [1.3, 1.3], [1.4, 1.4], [1.5, 1.5], [0.4, 0], [1, 0.], [1.3, 0],\
              [1.4, 0], [1.6, 0.2], [1.7, 0], [1.9, 0], [2, 0.2], [2.2, 0.1],\
              [2.6, 0]])
model = DBSCAN(eps=0.4, min_samples=2).fit(X)
print(model)

'''The following command gives back the label for each point to say to each 
cluster each point belongs. For example if there are two clusters it will give a
sequence of 0 and 1, representing the two clusters.
If there are '-1' is because those are outliers.'''
labels = model.labels_
print(labels)
 
'''The following command is to know which points are CORE points.'''
core_points = model.core_sample_indices_
print(core_points)
 
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(n_clusters)
 
 
core_samples_mask = np.zeros_like(model.labels_, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True
unique_labels = set(labels)
colors = [plt.get_cmap('Spectral')(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
         
    class_member_mask = (labels == k)
     
    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], -xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=10)
     
    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], -xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)
 
plt.title('Estimated number of clusters: %d' % n_clusters)
plt.show()