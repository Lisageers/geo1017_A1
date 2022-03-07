import Functions
import Algorithms
import KDistance
# import sklearn
# from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np

"""
Features:
option 1:  x-range, y-range, z-range, the height (maximum of z);
option 2: x-range, y-range, z-range, and the density (the number of points/the volume of bBox);
option 3: the height (maximum of z), percentage/ratio of the number of tier 1 (divided by height), the density (the number of points/the area of projection).
"""
option = 3

"""
Distance type:
'euclidean': squared difference
'manhattan': pairwise absolute difference
"""
dtype = "manhattan"

"""
Algorithm variables:
k = number of clusters (given k = 5)
linkage = "single", "complete or "average"
distance_threshold = maximal distance of two clusters (20, 30, 50)
eps = search radius
min_Pts = minimum points required to form a cluster (4 or 8)
"""
k = 5
linkage = "complete"
distance_threshold = 50
min_Pts = 4
eps = 2.75

feature_list = []
for i in range(500):
    feature_list.append(Functions.getFeatures(i, option))
feature_list = np.array(feature_list)

"""
plots Dendogram in order to find the distance threshold for the cutoff
"""
# if dtype == "manhattan":
#     Z = hierarchy.linkage(feature_list, linkage, 'cityblock')
#     plt.figure()
#     dn = hierarchy.dendrogram(Z)
#     plt.ylabel("Distance")
#     plt.title("Dendrogram linkage")
#     plt.show()
# else:
#     Z = hierarchy.linkage(feature_list, linkage)
#     plt.figure()
#     dn = hierarchy.dendrogram(Z)
#     plt.ylabel("Distance")
#     plt.title("Dendrogram linkage")
#     plt.show()

"""
plots K-distance graph in order to find the 'elbow', value for eps
"""
# KDistance.plotKDistance(feature_list, dtype, min_Pts)

"""
gives the spread of kmeans
"""
# Functions.accuracySpread(Algorithms.K_Means(feature_list, k, dtype))

"""
gives the spread of DBSCAN
"""
Functions.accuracySpread(Algorithms.DBSCAN(feature_list, eps, min_Pts, dtype))

# dict_k = Functions.accuracySpread(Algorithms.K_Means(feature_list, k, dtype), mode="dict")
# dict_h = Functions.accuracySpread(Algorithms.Hierarchical(feature_list, linkage, dtype, distance_threshold), mode="dict")
# dict_d = Functions.accuracySpread(Algorithms.DBSCAN(feature_list, eps, min_Pts, dtype), mode="dict")

# Functions.plotAccuracy(dict_k, dict_h, dict_d)




# Functions.plotAccuracy(option, dtype)

# KDistance.plotKDistance(feature_list, dtype)
# for i in range(len(feature_list)):
#     print(feature_list[i])

# print(Algorithms.K_Means(feature_list, 5))
# Functions.Accuracy(Algorithms.K_Means(feature_list, 5), Functions.ground_truth())
# Functions.accuracySpread(Algorithms.DBSCAN(feature_list, 2.25, 4))
# Functions.accuracySpread(Algorithms.K_Means(feature_list, 5))
# Functions.accuracySpread(Algorithms.Hierarchical(feature_list, "complete"))
# Functions.accuracySpread(Algorithms.DBSCAN(feature_list, 2.25, 4))

# Functions.accuracySpread(Algorithms.Hierarchical(feature_list, "average"))
# Functions.accuracySpread(Algorithms.DBSCAN(feature_list, 2.25, 4))

# Functions.accuracySpread(Algorithms.K_Means(feature_list, 5))

# print(Algorithms.Hierarchical(feature_list, "complete"))
# Functions.Accuracy(Algorithms.Hierarchical(feature_list, "single"), Functions.ground_truth())

# print(Algorithms.DBSCAN(feature_list, 2.5, 5))
# Functions.Accuracy(Algorithms.DBSCAN(feature_list, 2.5, 5), Functions.ground_truth())

# clustering = DBSCAN(eps=2.5, min_samples=5).fit(feature_list)
# print(clustering.labels_)

# Z = hierarchy.linkage(feature_list, linkage)
# plt.figure()
# dn = hierarchy.dendrogram(Z)
# plt.ylabel("Distance")
# plt.title("Dendrogram linkage")
# plt.show()