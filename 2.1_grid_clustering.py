import math
import sys
import time

import numpy as np
from datetime import datetime
import os
import csv
from tqdm import tqdm
import re
from some_func import util
from math import atan, pi, log, tan, exp
import pandas as pd
import json


class Grid:
    def __init__(self):
        self.pickup_points = []  # 位于当前网格的定位点
        self.locating_of_grid = [] # 网格的位置，里面所有点的经纬度平均



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
    if i < 0 or j < 0 or pickup_point_mercator[0] > x_max_y_max_mercator[0] or pickup_point_mercator[1] > x_max_y_max_mercator[1]:
        raise ValueError("Pickup point is outside the grid region.")
    return i,j


def Distance_of_two_points(pointa,pointb):
    pointa_mercator = util.wgs84_to_mercator(pointa[0],pointa[1])
    pointb_mercator = util.wgs84_to_mercator(pointb[0],pointb[1])
    distance = ((pointa_mercator[0] - pointb_mercator[0]) ** 2 + (pointa_mercator[1] - pointb_mercator[1]) ** 2) ** 0.5
    return distance

def calculate_distance(g, h):
    """
    计算两个点之间的欧几里得距离。
    """
    g = util.wgs84_to_mercator(g[0],g[1])
    h = util.wgs84_to_mercator(h[0],h[1])
    return ((g[0] - h[0]) ** 2 + (g[1] - h[1]) ** 2) ** 0.5


# 获得每个网格的平均位置
def get_Grid_locating_point(grid:Grid):
    lon_sum = 0
    lat_sum = 0
    points = grid.pickup_points
    for point in points:
        lon_sum += point[0]
        lat_sum += point[1]
    return [lon_sum/len(points),lat_sum/len(points)]


def process_coordinates(coords_dict, delta_k):
    """
    根据伪代码逻辑处理 coords_dict。
    """
    G = list(coords_dict.values())  # 提取所有坐标点作为初始集合 G
    O = []  # 最终结果集合
    i = 0   # 初始化 i

    while G:  # 当 G 非空时
        # print(len(G))
        HA_i = []  # 初始化集合 HA_i
        G1 = G.pop(0)  # 将集合 G 中的第一个元素取出
        HA_i.append(G1)  # 插入 HA_i

        for g in G[:]:  # 遍历 G 中的每个 g
            for h in HA_i:  # 遍历 HA_i 中的每个 h
                if calculate_distance(g, h) < delta_k:  # 如果距离小于
                    HA_i.append(g)  # 将 g 插入 HA_i
                    G.remove(g)  # 将 g 从 G 中删除
                    break  # 删除后跳出内部循环，继续处理下一个 g

        O.append(HA_i)  # 将 HA_i 插入集合 O
        i += 1  # i 自增 1

    return O  # 返回结果集合 O


def export_to_geojson(clusters, output_file="clusters.geojson"):
    # 构造 GeoJSON 数据
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for cluster_id, cluster in enumerate(clusters):
        for lon, lat in cluster:
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lon, lat]
                },
                "properties": {
                    "cluster_id": cluster_id  # 每个点所属的聚类编号
                }
            }
            geojson["features"].append(feature)

    # 导出为 GeoJSON 文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=4)

    print(f"GeoJSON 文件已保存到 {output_file}")


########################################################



k = 100# 网格的大小，单位为米
lameda = 50 # 网格密度阈值
fai = 1.3 # 定义9里面那个参数
limit = fai*k
delta_k = fai*k
######################################################



# 读取CSV文件
df = pd.read_csv('pickup_points.csv')
pickup_points = df[['lon', 'lat']].values.tolist()

# 计算 lon 和 lat 的最大最小值
lon_min = df['lon'].min()
lon_max = df['lon'].max()
lat_min = df['lat'].min()
lat_max = df['lat'].max()

x_min_y_min = util.wgs84_to_mercator(lon_min,lat_min)
x_max_y_max = util.wgs84_to_mercator(lon_max,lat_max)

cols, rows = get_grids_i_j(lon_min, lat_min, lon_max, lat_max, k)
print(f"Grid dimensions: cols={cols}, rows={rows}")



# 创建grids
grids = [[Grid() for _ in range(cols)] for _ in range(rows)]

print(len(grids))

# 遍历上下车点，将其映射到grid，添加进grid
for pickup_point in tqdm(pickup_points) :
    i, j = get_i_j_of_pickup_point(x_min_y_min, x_max_y_max, k, pickup_point)
    grids[i][j].pickup_points.append(pickup_point)

# 计算每个grid的点数量，超过lameda就计算平均位置，记录index
G = {}
count_hot_point = 0
for row in range(rows):
    for col in range(cols) :
        if len(grids[row][col].pickup_points) > lameda:
            # 计算平均位置
            grids[row][col].locating_of_grid = get_Grid_locating_point(grids[row][col])

            G[count_hot_point] = grids[row][col].locating_of_grid
            count_hot_point += 1



time1 = time.time()
clustering_result = process_coordinates(G,delta_k)
time2 = time.time()
print(time2-time1)
# print(clustering_result)

export_to_geojson(clustering_result)