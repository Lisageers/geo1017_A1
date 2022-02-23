import Functions
import Algorithms

dict_features = {}
for i in range(500):
    dict_features[i] = Functions.getFeatures(i)

print(dict_features)