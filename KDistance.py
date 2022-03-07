import numpy as np
import matplotlib.pyplot as plt

import Functions

def plotKDistance(feature_list, k=4):
    k_dists = []
    for i in range(len(feature_list)):
        dists_ = []
        for j in range (len(feature_list)):
            dists_.append(Functions.getDist(feature_list[j], feature_list[i], 'euclidean'))
        dists_ = np.array(dists_)
        dists_ = np.sort(dists_)
        k_dists.append(dists_[k])
    k_dists = np.array(k_dists)
    k_dists = np.sort(k_dists)
    x_ = np.linspace(0, 500, 500, endpoint=False)
    z = np.polyfit(x_, k_dists, 3)
    f = np.poly1d(z)
    y_ = f(x_)

    fig = plt.subplots(figsize=(12, 10))
    plt.plot(x_, y_)
    plt.yticks(np.arange(0, max(y_), 0.5))
    plt.grid(axis='y', linestyle='-')
    plt.title("k-distance")
    plt.xlabel("Sorted points number")
    plt.ylabel(str(k)+"th nearest neighbour distance")
    plt.show()
    return

if __name__ == "__main__":
    feature_list = []
    option = 1
    k = 4
    for i in range(500):
        feature_list.append(Functions.getFeatures(i, option))
    feature_list = np.array(feature_list)
    plotKDistance(feature_list, k)