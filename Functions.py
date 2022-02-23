import sys

file_path = 'data/pointclouds/'

def getFeatures (file_num):
# Features: the area of projection, the height (maximum of z), the number of points
    try:
        file_name = str(file_num).zfill(3)+'.xyz'
        point_clouds = open(file_path+file_name, 'r')
    except:
        print("No such a file")
        return None
    x_min = y_min = z_min = sys.float_info.max
    x_max = y_max = z_max = sys.float_info.min
    count = 0
#     lines = point_clouds.readlines()
    for line in point_clouds.readlines():
        count += 1
#         print(file_num, count)
        x_line = float(line.split()[0])
        y_line = float(line.split()[1])
        z_line = float(line.split()[2])
        if x_line < x_min:
            x_min = x_line
        if y_line < y_min:
            y_min = y_line
        if z_line < z_min:
            z_min = z_line
        if x_line > x_max:
            x_max = x_line
        if y_line > y_max:
            y_max = y_line
        if z_line > z_max:
            z_max = z_line
    return (abs(x_max-x_min)*abs(y_max-y_min), z_max, count)

def Accuracy():
    return