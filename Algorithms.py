import Functions
import numpy as np
import copy
import random

def K_Means():
    return

def Hierarchical():
    return


def findNeighbor(vec_A, dataset, eps):
    neighbor_list = []
    for i in range(len(dataset)):
        distance = Functions.getDist(vec_A, dataset[i])
        if distance < eps:
            neighbor_list.append(i)
    return set(neighbor_list)


def DBSCAN(dataset, eps, min_Pts):
    k = -1
    neighbor_list = []
    core_list = []
    unvisited_set = set([i for i in range(len(dataset))])
    label_list = [-1 for i in range(len(dataset))]
    for i in range(len(dataset)):
        neighbor_list.append(findNeighbor(dataset[i], dataset, eps))
        if len(neighbor_list[-1]) > min_Pts:
            core_list.append(i)
    core_set = set(core_list)
    while len(core_set) > 0:
        k = k+1
        old_set = copy.deepcopy(unvisited_set)
        core = random.choice(list(core_set))
        cluster = []
        cluster.append(core)
        unvisited_set.remove(core)
        while len(cluster) > 0:
            core_ = cluster.pop(0)
            if len(neighbor_list[core_]) >= min_Pts:
                delta = neighbor_list[core_] & unvisited_set
                delta_list = list(delta)
                for i in range(len(delta)):
                    cluster.append(delta_list[i])
                unvisited_set = unvisited_set - delta
        cluster_set = old_set-unvisited_set
        core_set = core_set - cluster_set
        cluster_list = list(cluster_set)
        for i in range(len(cluster_list)):
            label_list[cluster_list[i]] = k
    return label_list