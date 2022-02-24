import Functions
import Algorithms
import numpy as np

feature_list = []
for i in range(500):
    feature_list.append(Functions.getFeatures(i))

print(Algorithms.DBSCAN(feature_list, 200, 5))