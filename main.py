import Functions
import Algorithms
# import sklearn
# from sklearn.cluster import DBSCAN
import numpy as np

feature_list = []
for i in range(500):
    feature_list.append(Functions.getFeatures(i))
feature_list = np.array(feature_list)
# for i in range(len(feature_list)):
#     print(feature_list[i])
ground_truth_dict = Functions.ground_truth()

# labels_dbscan = Algorithms.DBSCAN(feature_list, 2.5, 5)
# Functions.Accuracy(labels_dbscan, ground_truth_dict)
labels_hierarchical = Algorithms.Hierarchical(feature_list, "single")
# print(labels_hierarchical)
Functions.Accuracy(labels_hierarchical, ground_truth_dict)

# clustering = DBSCAN(eps=2.5, min_samples=5).fit(feature_list)
# print(clustering.labels_)