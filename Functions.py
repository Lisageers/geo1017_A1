import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math
import Algorithms

file_path = 'data/pointclouds/'

def getDist(vec_A, vec_B, dtype):
    if dtype == "euclidean":
        return np.sqrt(np.sum(np.square(vec_A-vec_B)))
    if dtype == "manhattan":
        return sum(abs(v1-v2) for v1, v2 in zip(vec_A,vec_B))
    else:
        print('"' + str(dtype) + '" distance type does not exist. Select distance type "euclidean" or "manhattan".')
        exit()
    return   

def completeLinkageDist(dataset, key, index_list, dtype):
    largest_distance = -math.inf
    for index1 in key:
        for index2 in index_list:
            distance = getDist(dataset[index1], dataset[index2], dtype)
            if distance > largest_distance:
                largest_distance = distance
    return largest_distance

def singleLinkageDist(dataset, key, index_list, dtype):
    smallest_distance = math.inf
    for index1 in key:
        for index2 in index_list:
            distance = getDist(dataset[index1], dataset[index2], dtype)
            if distance < smallest_distance:
                smallest_distance = distance
    return smallest_distance


def getFeatures (file_num, option, mode="pub"):
    """
    Features:
    option 1:  x-range, y-range, z-range, the height (maximum of z);
    option 2: x-range, y-range, z-range, and the density (the number of points/the volume of bBox);
    option 3: the height (maximum of z), percentage/ratio of the number of tier 1 (divided by height), the density (the number of points/the area of projection).
    """

    # check the path of the file
    try:
        file_name = str(file_num).zfill(3)+'.xyz'
        point_clouds = open(file_path+file_name, 'r')
    except:
        print("No such a file")
        return None
    # compute the BBox of the point cloud
    x_min = y_min = z_min = sys.float_info.max
    x_max = y_max = z_max = sys.float_info.min
    count = 0
    point_list = []
    for line in point_clouds.readlines():
        count += 1
        point = [float(line.split()[0]), float(line.split()[1]), float(line.split()[2])]
        point_list.append(point)
    for point in point_list:
        if point[0] < x_min:
            x_min = point[0]
        if point[1] < y_min:
            y_min = point[1]
        if point[2] < z_min:
            z_min = point[2]
        if point[0] > x_max:
            x_max = point[0]
        if point[1] > y_max:
            y_max = point[1]
        if point[2] > z_max:
            z_max = point[2]

    #  The points in Tier 1 are those with the top 1/3 heights, and in Tier 3 are those with the bottom 1/3 heights.
    #  The rest are in Tier 2.
    count_t1 = count_t2 = count_t3 = 0
    boundary = [z_min, (z_min+z_max)/3, 2*(z_min+z_max)/3, z_max]
    for point in point_list:
        if point[2] > boundary[2]:
            count_t1 += 1
        elif point[2] > boundary[1]:
            count_t2 += 1
        else:
            count_t3 += 1
    # the dev mode is used to create the features table
    if mode == "dev":
        return np.array([abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min), z_max, count,
                         100*count_t1/count, 100*count_t2/count, 100*count_t3/count,
                         count/(abs(x_max-x_min)*abs(y_max-y_min)*abs(z_max-z_min))])
    # options:
    if option == 1:
        return np.array([abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min), z_max])
    if option == 2:
        return np.array([abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min), count/(abs(x_max-x_min)*abs(y_max-y_min)*abs(z_max-z_min))])
    if option == 3:
        return np.array([z_max, 100*count_t1/count, count/(abs(x_max-x_min)*abs(y_max-y_min)*abs(z_max-z_min))])
    # if option == 4:
    #     return np.array([np.maximum(abs(x_max-x_min), abs(y_max-y_min)), z_max, count/(abs(x_max-x_min)*abs(y_max-y_min))])
    # if option == 5:
    #     return np.array([100*count_t1/count, 100*count_t2/count, 100*count_t3/count])
    else:
        print('Option "' + str(option) + '" does not exist. Select option 1-5.')
        exit()
    return

def ground_truth():
    ground_truth_dict = {}
    label = -1
    for i in range(500):
        if (i % 100) == 0:
            label += 1
            ground_truth_dict[label] = [i]
        else:
            ground_truth_dict[label].append(i)
    return ground_truth_dict

def Accuracy(label_list, ground_truth_dict):
    # convert list to dictionary
    cluster_dict = {}
    for i in range(len(label_list)):
        # skip outliers
        if label_list[i] != -1: 
            # add indices as values to cluster dict, key is label
            if not label_list[i] in cluster_dict:
                cluster_dict[label_list[i]] = [i]
            else:
                cluster_dict[label_list[i]].append(i)

    # check which cluster resembles the ground truth the most for each ground truth group
    for key1, value1 in ground_truth_dict.items():
        max_similar = 0
        for key2, value2 in cluster_dict.items():
            # calculate how many indices of the ground truth label are in this cluster
            similar_count = len([c for c in value2 if c in value1])
            if similar_count > max_similar:
                max_similar = similar_count
                max_label = key2
        # accuracy is the most similar cluster
        print(f'Accuracy of ground truth label {key1} is {max_similar}%')
        # delete most similar cluster
        if max_similar != 0:
            cluster_dict.pop(max_label)
    return

def accuracySpread(cluster_list, mode="void"):
    # convert list to dictionary
    cluster_dict = {}
    for i in range(len(cluster_list)):
        # skip outliers
        if cluster_list[i] != -1:
            # add a numpy array with 5 zeros into the dict, the key is the label
            if not cluster_list[i] in cluster_dict:
                cluster_dict[cluster_list[i]] = np.zeros(5)
                # count the times a label shows in each group of 100
                cluster_dict[cluster_list[i]][i//100] = cluster_dict[cluster_list[i]][i//100] + 1
            else:
                cluster_dict[cluster_list[i]][i // 100] = cluster_dict[cluster_list[i]][i // 100] + 1
    label_dict = {}
    labels = ['building', 'car', 'fence', 'pole', 'tree']
    #  the cluster that has a maximum number in the group wins the label of objects
    for i in range(5):
        max_label = -1
        max_num = -99
        max_array = np.zeros(5)
        for key1, value1 in cluster_dict.items():
            if value1[i] > max_num:
                max_label = key1
                max_num = value1[i]
                max_array = value1
        label_dict[labels[i]] = max_array
        cluster_dict.pop(max_label)

    if mode == "dict":
        return label_dict
    else:
        for key1, value1 in label_dict.items():
            print('{0}: {1}'.format(key1, value1))
        return

def plotAccuracy(dict_k, dict_h, dict_d):
# def plotAccuracy(option, dtype, k=5, linkage="complete", eps=2.25, min_Pts=4):
#     feature_list = []
#     for i in range(500):
#         feature_list.append(getFeatures(i, option))
#     feature_list = np.array(feature_list)
#     dict_k = accuracySpread(Algorithms.K_Means(feature_list, k, dtype), mode="dict")
#     dict_h = accuracySpread(Algorithms.Hierarchical(feature_list, linkage, dtype), mode="dict")
#     dict_d = accuracySpread(Algorithms.DBSCAN(feature_list, eps, min_Pts, dtype), mode="dict")
    x_ = []
    y_k = []
    y_h = []
    y_d = []
    index = 0
    print("K-means: ")
    for key1, value1 in dict_k.items():
        print('{0}: {1}'.format(key1, value1))
        x_.append(key1)
        y_k.append(value1[index])
        index = index+1
    index = 0
    print("Hierarchical: ")
    for key1, value1 in dict_h.items():
        print('{0}: {1}'.format(key1, value1))
        y_h.append(value1[index])
        index = index+1
    index = 0
    print("DBSCAN: ")
    for key1, value1 in dict_d.items():
        print('{0}: {1}'.format(key1, value1))
        y_d.append(value1[index])
        index = index+1

    plt.figure(figsize=(10, 8))
    plt.plot(x_, y_k, label='K-means')
    plt.plot(x_, y_h, label='Hierarchical')
    plt.plot(x_, y_d, label='DBSCAN')
    plt.xlabel('Labels')
    plt.ylabel('Accuracy')
    plt.title('The accuracy of different label using different algorithms')
    plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    # output the means for all the features (dev mode)
    features_dict = {}
    for i in range(5):
        label = "features_"+str(i)
        features_dict[label] = np.zeros(9)
    for i in range(5):
        for j in range (i*100, (i+1)*100):
            temp = getFeatures(j, "dev")
            label_ = "features_"+str(i)
            features_dict[label_] = features_dict[label_] + temp
    for i in range(5):
        label = "features_" + str(i)
        features_dict[label] = features_dict[label]/100
        np.set_printoptions(suppress=True, precision=3)
        print(features_dict[label])