from numpy import dtype
import Functions
import numpy as np
import pandas as pd
import copy
import random
import math

def K_Means(dataset, k):

    # randomly pick k amount of feature points -> centroids of k clusters
    c_centroids = np.array(random.choices(dataset, k=k))
    # pick again if any of the centroids are identical
    while len(np.unique(c_centroids, axis=0)) != k:
        c_centroids = np.array(random.choices(dataset, k=k))

    label_list = [0 for i in range(len(dataset))]
    
    centroid_move = True

    while centroid_move:
        for i in range(len(dataset)):
            min_distance = float('inf')
            for j in range(len(c_centroids)):
                distance = Functions.getDist(dataset[i], c_centroids[j], 0, 'euclidean')
                if min_distance > distance:
                    min_distance = distance
                    label_list[i] = j
        # calculate the new centroids by taking the mean of each feature of each cluster
        new_centroids = pd.DataFrame(dataset).groupby(by=label_list).mean().values
        
        # if the new centroids are the same as the 'old' centroids -> centroids are not moving anymore
        if np.count_nonzero(c_centroids-new_centroids) == 0:
            centroid_move = False

        # else new centroids will be used
        else:
            c_centroids = new_centroids
    return label_list

def Hierarchical(dataset, linkage):

    # put each feature in a separate cluster
    label_list = [i for i in range(len(dataset))]
    label_dict = {}
    for i in range(len(dataset)):
        label_dict[(i,)] = dataset[i]

    distance_dict = {}
    for i in range(len(dataset)):
        for j in range(len(dataset)):
                distance = Functions.getDist(dataset[i], dataset[j], 1, 'minkowski')
                if i != j:
                    sorted_indices = sorted([i, j])
                    distance_dict[((sorted_indices[0],), (sorted_indices[1],))] = distance              

    while len(label_dict.keys()) > 5:
        # find closest points
        indexes_closest = min(distance_dict, key=distance_dict.get)
        
        # put distance treshold at 50 
        if distance_dict[indexes_closest] > 50:
            break

        # update label dict
        average_c1 = label_dict[indexes_closest[0]]
        average_c2 = label_dict[indexes_closest[1]]
        coor_list = []
        index_list = [element for tupl in indexes_closest for element in tupl]
        for index in index_list:
            coor_list.append(dataset[index])

        average_combined = np.mean(np.array(coor_list), axis=0)
        label_dict[tuple(index_list)] = average_combined
        
        label_dict.pop(indexes_closest[0])
        label_dict.pop(indexes_closest[1])

        # update distance dict
        distance_dict.pop(indexes_closest)
        for key in list(distance_dict):
            if key[0] in indexes_closest:
                distance_dict.pop(key)
                if linkage == "average":
                    distance_dict[(tuple(index_list), key[1])] = Functions.getDist(average_combined, label_dict[key[1]], 1, 'minkowski')
                
                elif linkage == "complete":
                    distance_dict[(tuple(index_list), key[1])] = Functions.completeLinkageDist(dataset, key[1], index_list)

           
            elif key[1] in indexes_closest:
                distance_dict.pop(key)
                if linkage == "average":
                    distance_dict[(tuple(index_list), key[0])] = Functions.getDist(average_combined, label_dict[key[0]], 1, 'minkowski')
                
                elif linkage == "complete":
                    distance_dict[(tuple(index_list), key[0])] = Functions.completeLinkageDist(dataset, key[0], index_list)

    # convert output to list
    count = 1
    for key in label_dict.keys():
        for index in key:
            label_list[index] = count
        count +=1
    return label_list


def findNeighbor(vec_A, dataset, eps):
    neighbor_list = []
    for i in range(len(dataset)):
        distance = Functions.getDist(vec_A, dataset[i], 0, 'euclidean')
        if distance <= eps:
            neighbor_list.append(i)
    return neighbor_list

def DBSCAN(dataset, eps, min_Pts):
    # Initialize the list for unvisited vectors, set all as unvisited:
    unvisited_list = [i for i in range(len(dataset))]
    # Initialize an empty list for the visited ones:
    visited_list = []
    # Set all labels as -1 (outliers)
    label_list = [-1 for i in range(len(dataset))]
    # the label/cluster
    k = -1

    while len(unvisited_list) > 0:
        # pick a vector randomly for the unvisited list
        p = random.choice(unvisited_list)
        unvisited_list.remove(p)
        visited_list.append(p)
        # the epsilon-neighborhood:
        N = findNeighbor(dataset[p], dataset, eps)
        # if the number of objects in N is bigger than MinPts, then p is a core point
        if len(N) >= min_Pts:
            k = k+1
            label_list[p] = k

            for n in N:
                # mark items in N as visited if it is not visited
                if n in unvisited_list:
                    unvisited_list.remove(n)
                    visited_list.append(n)
                # computer the epsilon-neighborhood for n:
                N_ = findNeighbor(dataset[n], dataset, eps)
                # if n is a core point, add its epsilon-neighborhood into N
                if len(N_)  >= min_Pts:
                    for n_ in N_:
                        if n_ not in N:
                            N.append(n_)
                # the label for n is k if it is -1 (not signed before or not a core point)
                if label_list[n] == -1:
                    label_list[n] = k
        # else label it as -1
        else:
            label_list[p] = -1

    return label_list
