import sys
import numpy as np

file_path = 'data/pointclouds/'

def getDist(vec_A, vec_B):
    return np.sqrt(np.sum(np.square(vec_A-vec_B)))

def getFeatures (file_num):
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

    # return np.array([abs(x_max-x_min)*abs(y_max-y_min), z_max, count])
    # return np.array([abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min)])
    return np.array([abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min), count/(abs(x_max-x_min)*abs(y_max-y_min)*abs(z_max-z_min))])
    # return np.array([np.maximum(abs(x_max-x_min), abs(y_max-y_min)), z_max, count/(abs(x_max-x_min)*abs(y_max-y_min))])
    # return np.array([100*count_t1/count, 100*count_t2/count, 100*count_t3/count])


def Accuracy():
    return