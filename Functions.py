import sys
import numpy as np
import math

file_path = 'data/pointclouds/'

def getDist(vec_A, vec_B, p, dtype):
    if dtype == "euclidean":
        return np.sqrt(np.sum(np.square(vec_A-vec_B)))
    elif dtype == "manhattan":
        return sum(abs(v1-v2) for v1, v2 in zip(vec_A,vec_B))
    elif dtype == "minkowski":
        return sum(pow(abs(v1-v2), p) for v1, v2 in zip(vec_A, vec_B)) ** (1/p)
    return   

def singleLinkageDist(dataset, key, index_list):
    smallest_distance = math.inf
    for index1 in key:
        for index2 in index_list:
            distance = getDist(dataset[index1], dataset[index2], 2, 'minkowski')
            if distance < smallest_distance:
                smallest_distance = distance
    return smallest_distance


def getFeatures (file_num, mode="pub"):
    """
    Features:
    option 1: the area of projection, the height (maximum of z), the number of points
    option 2: x-range, y-range, z-range, and the number of points
    option 3: the maximum x or y, the height (maximum of z), the density (the number of points/the area of projection)
    option 4: percentage/ratio of the number of tier 1, 2 and 3 (divided by height)
    option 2: x-range, y-range, z-range, and the density (the number of points/the volume of bBox)
    """

    try:
        file_name = str(file_num).zfill(3)+'.xyz'
        point_clouds = open(file_path+file_name, 'r')
    except:
        print("No such a file")
        return None
    x_min = y_min = z_min = sys.float_info.max
    x_max = y_max = z_max = sys.float_info.min
    count = 0
    point_list = []
#     lines = point_clouds.readlines()
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

    count_t1 = count_t2 = count_t3 = 0
    boundary = [z_min, (z_min+z_max)/3, 2*(z_min+z_max)/3, z_max]
    for point in point_list:
        if point[2] > boundary[2]:
            count_t1 += 1
        elif point[2] > boundary[1]:
            count_t2 += 1
        else:
            count_t3 += 1

    if mode == "dev":
        return np.array([abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min), z_max, count,
                         100*count_t1/count, 100*count_t2/count, 100*count_t3/count,
                         count/(abs(x_max-x_min)*abs(y_max-y_min)*abs(z_max-z_min))])
    # return np.array([abs(x_max-x_min)*abs(y_max-y_min), z_max, count])
    # return np.array([abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min)])
    return np.array([abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min), count/(abs(x_max-x_min)*abs(y_max-y_min)*abs(z_max-z_min))])
    # return np.array([np.maximum(abs(x_max-x_min), abs(y_max-y_min)), z_max, count/(abs(x_max-x_min)*abs(y_max-y_min))])
    # return np.array([100*count_t1/count, 100*count_t2/count, 100*count_t3/count])

def ground_truth():
    ground_truth_dict = {}
    label = 0
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

if __name__ == "__main__":
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