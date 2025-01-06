import math
import sys
import numpy as np
from datetime import datetime,timedelta
import os
import csv
from tqdm import tqdm
import re
from some_func import util
from math import atan, pi, log, tan, exp, radians
import pandas as pd
import random
import ast

def wgs84_to_mercator(lon, lat):
    x = lon * 20037508.342789 / 180
    y = log(tan((90 + lat) * pi / 360)) / (pi / 180)
    y = y * 20037508.34789 / 180
    return [x, y]

def mercator_to_wgs84(x, y):
    lon = x / 20037508.34 * 180
    lat = y / 20037508.34 * 180
    lat = 180 / pi * (2 * atan(exp(lat * pi / 180)) - pi / 2)
    return [lon, lat]

def manhattan_distance(coord1, coord2):
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def load_data():# data:key->taxi_id,value->[[time],[coordinates(not mercator)],[status]],
    data = {}
    lon_min = 180
    lon_max = 0
    lat_min = 90
    lat_max = 0

    file_need_remove = []
    file_need_stay = []
    traj_folder = os.path.join(os.getcwd(), "traj")
    for file_name in tqdm(os.listdir(traj_folder)) :
        if file_name.startswith("traj_") and file_name.endswith(".csv"):
            match = re.search(r"traj_(\d+)\.csv", file_name)
            taxi_id = match.group(1)
            file_path = os.path.join(traj_folder, file_name)
            # data_illegal_count = 0

            time_list = []
            coordinates = []
            status_list = []

            if_use_this_data  = True

            with open(file_path, mode='r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    ID = int(row['ID'])
                    time_str = row['Time']
                    time = datetime.strptime(time_str, "%H:%M:%S").time()  # 转换为 time 对象
                    longitude = float(row['Longitude'])
                    latitude = float(row['Latitude'])
                    status = int(row['Status'])

                    # 检测到非法数据就弃用该出租车的数据
                    if not (113.75 < longitude < 114.63):
                        if_use_this_data  = False
                        break
                    if not (22.45 < latitude < 22.85):
                        if_use_this_data  = False
                        break
                    if status != 1 and status != 0 :
                        if_use_this_data  = False
                        break


                    time_list.append(time)
                    coordinates.append([longitude,latitude])
                    status_list.append(status)


                    # lon_min = min(lon_min,longitude)
                    # lon_max = max(lon_max,longitude)
                    # lat_min = min(lat_min,latitude)
                    # lat_max = max(lat_max,latitude)

                    # data.setdefault(ID, [[], [], []])
                    # data[ID][0].append(time)
                    # data[ID][1].append([longitude,latitude])
                    # data[ID][1].append(util.mercator_to_wgs84(longitude, latitude))
                    # data[ID][2].append(status)
                    # print(ID)
                    # print(time,type(time))
                    # print([longitude, latitude])
                    # print(status,type(status))
                    # print("----------------------")

            if if_use_this_data:
                data[taxi_id] = [time_list,coordinates,status_list]
                lon_min = min(lon_min,min(coord[0] for coord in coordinates))
                lon_max = max(lon_max,max(coord[0] for coord in coordinates))
                lat_min = min(lat_min,min(coord[1] for coord in coordinates))
                lat_max = max(lat_max,max(coord[1] for coord in coordinates))
                file_need_stay.append(file_name)
            else:
                file_need_remove.append(file_name)


                # if data_illegal_count > 2:
                #     file_need_remove.append(file_name)
    outputfile = "file_need_stay.txt"
    print(len(file_need_stay))
    with open(outputfile, mode='w', encoding='utf-8') as f:
        for filename in file_need_stay:
            f.write(filename + "\n")  # 每个文件名写入一行
    outputfile = "file_need_remove.txt"
    with open(outputfile, mode='w', encoding='utf-8') as f:
        for filename in file_need_remove:
            f.write(filename + "\n")  # 每个文件名写入一行
    return data,[lon_min,lon_max,lat_min,lat_max]

def get_train_data_1(traj_data: dict) -> pd.DataFrame:
    """
    从轨迹数据中，对于每条轨迹，找到某个上下车点a，假设其时间为t，选定时间t之前的某个车辆轨迹点，
    选定的时间范围为5分钟到30分钟，比如现在找到了一个点，为上车点，时间为中午12点，我们找到11：30-11：55，这个时间段中的随机一个点，设为接单时的出租车位置点b，
    我们在点a的以1000m为半径的圆内，随机选择一个点，设为接单时的乘客位置c，在这一步选择点时，在mercator坐标系中进行，最后得到c时，将其转化为wgs84
    这样有了一个三个元素的列表[b,a,c]，
    b为taxi_location,
    a:pick_up_point,
    c:user_location，
    坐标都为wgs84
    将所有的列表组织成一个dataframe，保存输出为train_data.csv

    Args:
        traj_data: 轨迹数据,data:key->taxi_id,value->[[timestamp],[coordinates(wgs84)],[status]],status中，1代表上车点，0代表下车点
    timestamp提取时的代码为time = datetime.strptime(time_str, "%H:%M:%S").time()  # 转换为 time 对象
    请注意time的格式，如果选定的上下车点的时间早于00:30，就不用这个点

    Returns: dataframe

    """
    train_data = []
    count_ID = 1

    for key, value in tqdm(traj_data.items()):
        timestamps = value[0]
        coordinates = value[1]
        status_list = value[2]

        for i in range(1, len(coordinates)):
            # 检查是否是上下车点
            if status_list[i] != status_list[i - 1]:
                ######################################################################################
                # 提取上下车点 a 的信息
                pick_up_time = timestamps[i]  # 直接使用传入的 time 对象
                if pick_up_time < datetime.strptime("00:30:00", "%H:%M:%S").time():
                    continue

                a = {
                    "taxi_location": coordinates[i],
                    "pick_up_point": [coordinates[i][0], coordinates[i][1]],
                    "pick_up_time": pick_up_time
                }
                ######################################################################################

                # 选定时间范围 5 分钟到 30 分钟之前的随机一个点 b
                start_time = datetime.combine(datetime.today(), pick_up_time) - timedelta(minutes=30)
                end_time = datetime.combine(datetime.today(), pick_up_time) - timedelta(minutes=5)
                b_candidates = [
                    j for j in range(len(timestamps))
                    if start_time.time() <= timestamps[j] <= end_time.time()
                ]

                if not b_candidates:
                    continue

                b_index = random.choice(b_candidates)
                b = coordinates[b_index]
                #####################################################################################

                # 以 500 米为半径的圆内随机选择一个点 c
                mercator_a = wgs84_to_mercator(a["pick_up_point"][0], a["pick_up_point"][1])
                while True:
                    random_offset_x = random.uniform(-500, 500)
                    random_offset_y = random.uniform(-500, 500)
                    if random_offset_x ** 2 + random_offset_y ** 2 <= 500 ** 2:
                        break
                mercator_c = [mercator_a[0] + random_offset_x, mercator_a[1] + random_offset_y]
                c = mercator_to_wgs84(mercator_c[0], mercator_c[1])

                # 组织数据 [b, a, c]
                train_data.append({
                    "taxi_location": b,
                    "user_location": c,
                    "pick_up_point": a["pick_up_point"]
                })

                count_ID += 1

    # 转换为 DataFrame 并保存
    train_df = pd.DataFrame(train_data)
    train_df.to_csv("train_data.csv", index=False)
    print(count_ID)
    return train_df

# 根据经纬度最大小值以及网格大小确定区域被分为rows*cols数量的网格
def get_grids_i_j(lon_min, lat_min, lon_max, lat_max, grid_size):

    min_mercator = util.wgs84_to_mercator(lon_min, lat_min)
    max_mercator = util.wgs84_to_mercator(lon_max, lat_max)

    x_length = max_mercator[0] - min_mercator[0]
    y_length = max_mercator[1] - min_mercator[1]

    cols = math.ceil(x_length / grid_size)
    rows = math.ceil(y_length / grid_size)

    return cols, rows

# 获得一个上下车点在区域中所在网格的索引
def get_i_j_of_pickup_point(x_min_y_min_mercator,x_max_y_max_mercator,grid_size,pickup_point):
    pickup_point_mercator = util.wgs84_to_mercator(pickup_point[0],pickup_point[1])
    x_dis = pickup_point_mercator[0] - x_min_y_min_mercator[0]
    y_dis = pickup_point_mercator[1] - x_min_y_min_mercator[1]
    j = math.floor(x_dis / grid_size)
    i = math.floor(y_dis / grid_size)
    # 检查边界是否超出区域
    # if i < 0 or j < 0 or pickup_point_mercator[0] > x_max_y_max_mercator[0] or pickup_point_mercator[1] > x_max_y_max_mercator[1]:
    #     raise ValueError("Pickup point is outside the grid region.")
    return i,j

# traj_data,district = load_data()
# print(district)
# train_data = get_train_data_1(traj_data)


# # 读取CSV文件
# file_path = "train_data.csv"
# df = pd.read_csv(file_path)
#
# # 转换数据
# new_data = []
#
# for _, row in df.iterrows():
#     # 解析每个坐标列为列表
#     taxi_location = ast.literal_eval(row['taxi_location'])
#     user_location = ast.literal_eval(row['user_location'])
#     pick_up_point = ast.literal_eval(row['pick_up_point'])
#
#     # 将 WGS84 坐标系转换为 Mercator 坐标系
#     taxi_location_mercator = wgs84_to_mercator(taxi_location[0], taxi_location[1])
#     user_location_mercator = wgs84_to_mercator(user_location[0], user_location[1])
#     pick_up_point_mercator = wgs84_to_mercator(pick_up_point[0], pick_up_point[1])
#
#     # 计算曼哈顿距离
#     taxi2pickup = manhattan_distance(taxi_location, pick_up_point)
#     user2pickup = manhattan_distance(user_location, pick_up_point)
#
#     # 将结果保存到新数据列表
#     new_data.append({
#         "taxi_location (Mercator)": taxi_location_mercator,
#         "user_location (Mercator)": user_location_mercator,
#         "taxi2pickup": taxi2pickup,
#         "user2pickup": user2pickup,
#         "pick_up_point (Mercator)": pick_up_point_mercator
#     })
#
# # 创建新的 DataFrame
# new_df = pd.DataFrame(new_data)
#
# # 保存为新的 CSV 文件
# output_file = "train_data_mercator.csv"
# new_df.to_csv(output_file, index=False)
#
# print(f"转换完成，新的数据已保存到 {output_file}")

# [lon_min,lon_max,lat_min,lat_max]为
district = [113.752022, 114.567619, 22.456734, 22.8493]

# cols,rows = get_grids_i_j(district[0],district[2],district[1],district[3],grid_size=50)

x_min_y_min = wgs84_to_mercator(district[0],district[2])
x_max_y_max = wgs84_to_mercator(district[1],district[3])

# 读取CSV文件
file_path = "train_data.csv"
df = pd.read_csv(file_path)

# 转换数据
new_data = []

for _, row in df.iterrows():
    # 解析每个坐标列为列表
    taxi_location = ast.literal_eval(row['taxi_location'])
    user_location = ast.literal_eval(row['user_location'])
    pick_up_point = ast.literal_eval(row['pick_up_point'])

    i,j = get_i_j_of_pickup_point(x_min_y_min,x_max_y_max,50,taxi_location)
    taxi_location_rowcol = [i,j]

    i, j = get_i_j_of_pickup_point(x_min_y_min, x_max_y_max, 50, user_location)
    user_location_rowcol = [i, j]

    i, j = get_i_j_of_pickup_point(x_min_y_min, x_max_y_max, 50, pick_up_point)
    pick_up_location_rowcol = [i, j]

    # # 将 WGS84 坐标系转换为 Mercator 坐标系
    # taxi_location_mercator = wgs84_to_mercator(taxi_location[0], taxi_location[1])
    # user_location_mercator = wgs84_to_mercator(user_location[0], user_location[1])
    # pick_up_point_mercator = wgs84_to_mercator(pick_up_point[0], pick_up_point[1])

    # # 计算曼哈顿距离
    # taxi2pickup = manhattan_distance(taxi_location, pick_up_point)
    # user2pickup = manhattan_distance(user_location, pick_up_point)

    # 将结果保存到新数据列表
    new_data.append({
        "taxi_location_rowcol": taxi_location_rowcol,
        "user_location_rowcol": user_location_rowcol,
        "pick_up_location_rowcol": pick_up_location_rowcol
    })

# 创建新的 DataFrame
new_df = pd.DataFrame(new_data)

# 保存为新的 CSV 文件
output_file = "train_data_rowcol.csv"
new_df.to_csv(output_file, index=False)

print(f"转换完成，新的数据已保存到 {output_file}")
