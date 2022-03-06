import Functions
import Algorithms
import KDistance
# import sklearn
# from sklearn.cluster import DBSCAN
import numpy as np

"""
Features:
option 1: the area of projection, the height (maximum of z), the number of points
option 2: x-range, y-range, z-range, and the number of points
option 3: the maximum x or y, the height (maximum of z), the density (the number of points/the area of projection)
option 4: percentage/ratio of the number of tier 1, 2 and 3 (divided by height)
option 5: x-range, y-range, z-range, and the density (the number of points/the volume of bBox)
"""
option = 3

"""
Distance type:
'euclidian': squared difference
'manhattan': pairwise absolute difference
"""
dtype = "manhattan"

"""
Algorithm variables:
k = number of clusters (given k = 5)
linkage = "average" or "complete"
eps = search radius
min_Pts = minimum points required to form a cluster
"""
k = 5
linkage = "complete"
eps = 2.25
min_Pts = 4

feature_list = []
for i in range(500):
    feature_list.append(Functions.getFeatures(i, option))
feature_list = np.array(feature_list)

dict_k = Functions.accuracySpread(Algorithms.K_Means(feature_list, k, dtype), mode="dict")
dict_h = Functions.accuracySpread(Algorithms.Hierarchical(feature_list, linkage, dtype), mode="dict")
dict_d = Functions.accuracySpread(Algorithms.DBSCAN(feature_list, eps, min_Pts, dtype), mode="dict")

Functions.plotAccuracy(dict_k, dict_h, dict_d)

# Functions.plotAccuracy(option, dtype)

# KDistance.plotKDistance(feature_list)
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