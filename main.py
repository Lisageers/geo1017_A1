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

# print(Algorithms.K_Means(feature_list, 5))
# Functions.Accuracy(Algorithms.K_Means(feature_list, 5), Functions.ground_truth())
# Functions.accuracySpread(Algorithms.DBSCAN(feature_list, 2.25, 4))
Functions.accuracySpread(Algorithms.DBSCAN(feature_list, 2.25, 4))

# Functions.accuracySpread(Algorithms.K_Means(feature_list, 5))

# print(Algorithms.Hierarchical(feature_list, "single"))
# Functions.Accuracy(Algorithms.Hierarchical(feature_list, "single"), Functions.ground_truth())

# print(Algorithms.DBSCAN(feature_list, 2.5, 5))
# Functions.Accuracy(Algorithms.DBSCAN(feature_list, 2.5, 5), Functions.ground_truth())

# clustering = DBSCAN(eps=2.5, min_samples=5).fit(feature_list)
# print(clustering.labels_)