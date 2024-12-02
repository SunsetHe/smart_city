import math
import sys
import numpy as np
from datetime import datetime
import os
import csv
from tqdm import tqdm
import re
from some_func import util
from math import atan, pi, log, tan, exp

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



def get_pickup_point(traj_data: dict) -> list:
    """

    Args:
        traj_data: 轨迹数据,data:key->taxi_id,value->[[timestamp],[coordinates(not mercator)],[status]]


    Returns: 上下车点的list[ [count_ID, up/down, timestamp, lon, lat] ] ,up/down中，1代表上车点，0代表下车点

    """
    pickup_point_list = []
    count_ID = 1
    for key,value in tqdm(traj_data.items()):
        timestamps = value[0]
        coordinates = value[1]
        status_list = value[2]
        for i in range(1,len(coordinates)):
            if status_list[i] != status_list[i-1]:
                pickup_point = [count_ID, status_list[i], timestamps[i], coordinates[i][0], coordinates[i][1]]
                pickup_point_list.append(pickup_point)
                count_ID += 1
                print(pickup_point)


    return pickup_point_list




traj_data,district = load_data()
pickup_points = get_pickup_point(traj_data)