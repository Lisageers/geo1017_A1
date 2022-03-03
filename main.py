import Functions
import Algorithms
# import sklearn
# from sklearn.cluster import DBSCAN
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np

feature_list = []
for i in range(500):
    feature_list.append(Functions.getFeatures(i))
feature_list = np.array(feature_list)
# for i in range(len(feature_list)):
#     print(feature_list[i])

# print(Algorithms.K_Means(feature_list, 5))
# Functions.Accuracy(Algorithms.K_Means(feature_list, 5), Functions.ground_truth())

# print(Algorithms.Hierarchical(feature_list, "complete"))
Functions.Accuracy(Algorithms.Hierarchical(feature_list, "complete"), Functions.ground_truth())

# print(Algorithms.DBSCAN(feature_list, 2.5, 5))
# Functions.Accuracy(Algorithms.DBSCAN(feature_list, 2.5, 5), Functions.ground_truth())

# clustering = DBSCAN(eps=2.5, min_samples=5).fit(feature_list)
# print(clustering.labels_)

# Z = hierarchy.linkage(feature_list, 'average')
# plt.figure()
# dn = hierarchy.dendrogram(Z)
# plt.ylabel("Distance")
# plt.title("Dendogram average linkage")
# plt.show()