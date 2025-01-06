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
import pandas as pd
import json
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import time
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 选择SimHei字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class Grid:
    def __init__(self):
        self.pickup_points = []  # 位于当前网格的定位点
        self.locating_of_grid = []  # 网格的位置，里面所有点的经纬度平均

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
def get_i_j_of_pickup_point(x_min_y_min_mercator, x_max_y_max_mercator, grid_size, pickup_point):
    pickup_point_mercator = util.wgs84_to_mercator(pickup_point[0], pickup_point[1])
    x_dis = pickup_point_mercator[0] - x_min_y_min_mercator[0]
    y_dis = pickup_point_mercator[1] - x_min_y_min_mercator[1]
    j = math.floor(x_dis / grid_size)
    i = math.floor(y_dis / grid_size)
    if i < 0 or j < 0 or pickup_point_mercator[0] > x_max_y_max_mercator[0] or pickup_point_mercator[1] > x_max_y_max_mercator[1]:
        raise ValueError("Pickup point is outside the grid region.")
    return i, j

def Distance_of_two_points(pointa, pointb):
    pointa_mercator = util.wgs84_to_mercator(pointa[0], pointa[1])
    pointb_mercator = util.wgs84_to_mercator(pointb[0], pointb[1])
    distance = ((pointa_mercator[0] - pointb_mercator[0]) ** 2 + (pointa_mercator[1] - pointb_mercator[1]) ** 2) ** 0.5
    return distance

def calculate_distance(g, h):
    g = util.wgs84_to_mercator(g[0], g[1])
    h = util.wgs84_to_mercator(h[0], h[1])
    return ((g[0] - h[0]) ** 2 + (g[1] - h[1]) ** 2) ** 0.5

# 获得每个网格的平均位置
def get_Grid_locating_point(grid: Grid):
    lon_sum = 0
    lat_sum = 0
    points = grid.pickup_points
    for point in points:
        lon_sum += point[0]
        lat_sum += point[1]
    return [lon_sum / len(points), lat_sum / len(points)]

def dbscan_clustering(points, eps, min_samples):
    start_time = time.time()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 计算轮廓系数
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(points, labels)
    else:
        silhouette_avg = -1  # 如果只有一个簇，轮廓系数设为 -1
    
    return labels, silhouette_avg, execution_time

def agglomerative_clustering(points, n_clusters):
    start_time = time.time()
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    labels = agglomerative.fit_predict(points)
    end_time = time.time()
    execution_time = end_time - start_time
    
    # 计算轮廓系数
    if len(set(labels)) > 1:
        silhouette_avg = silhouette_score(points, labels)
    else:
        silhouette_avg = -1  # 如果只有一个簇，轮廓系数设为 -1
    
    return labels, silhouette_avg, execution_time

def process_coordinates(coords_dict, delta_k):
    G = list(coords_dict.values())
    O = []
    i = 0
    while G:
        HA_i = []
        G1 = G.pop(0)
        HA_i.append(G1)
        for g in G[:]:
            for h in HA_i:
                if calculate_distance(g, h) < delta_k:
                    HA_i.append(g)
                    G.remove(g)
                    break
        O.append(HA_i)
        i += 1
    return O

def export_to_geojson(clusters, output_file="clusters.geojson"):
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
                    "cluster_id": cluster_id
                }
            }
            geojson["features"].append(feature)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson, f, ensure_ascii=False, indent=4)
    print(f"GeoJSON 文件已保存到 {output_file}")

# 读取CSV文件
df = pd.read_csv('pickup_points.csv')
pickup_points = df[['lon', 'lat']].values.tolist()

# 计算 lon 和 lat 的最大最小值
lon_min = df['lon'].min()
lon_max = df['lon'].max()
lat_min = df['lat'].min()
lat_max = df['lat'].max()

x_min_y_min = util.wgs84_to_mercator(lon_min, lat_min)
x_max_y_max = util.wgs84_to_mercator(lon_max, lat_max)

# 定义 k 的范围
k_values = range(100, 1001, 100)  # 从 10 米到 100 米，步长为 5 米
lameda = 50
fai = 1.3

# 存储不同算法的结果
silhouette_scores = []
execution_times = []
dbscan_silhouette_scores = []
dbscan_execution_times = []
agglomerative_silhouette_scores = []
agglomerative_execution_times = []

# 定义DBSCAN参数
eps = 0.001  # 邻域半径
min_samples = 5  # 最小样本数

# 使用现有算法进行聚类
for k in tqdm(k_values):
    limit = fai * k
    delta_k = fai * k
    cols, rows = get_grids_i_j(lon_min, lat_min, lon_max, lat_max, k)
    grids = [[Grid() for _ in range(cols)] for _ in range(rows)]
    
    for pickup_point in pickup_points:
        i, j = get_i_j_of_pickup_point(x_min_y_min, x_max_y_max, k, pickup_point)
        grids[i][j].pickup_points.append(pickup_point)
    
    G = {}
    count_hot_point = 0
    for row in range(rows):
        for col in range(cols):
            if len(grids[row][col].pickup_points) > lameda:
                grids[row][col].locating_of_grid = get_Grid_locating_point(grids[row][col])
                G[count_hot_point] = grids[row][col].locating_of_grid
                count_hot_point += 1
    
    if count_hot_point > 1:
        # 现有算法
        start_time = time.time()
        clustering_result = process_coordinates(G, delta_k)
        end_time = time.time()
        execution_time = end_time - start_time
        execution_times.append(execution_time)
        
        # 将聚类结果展平为二维数组
        clustered_points = [point for cluster in clustering_result for point in cluster]
        # 计算轮廓系数
        silhouette_avg = silhouette_score(clustered_points, [i for i, cluster in enumerate(clustering_result) for _ in cluster])
        silhouette_scores.append(silhouette_avg)
        
        # DBSCAN
        labels_dbscan, silhouette_avg_dbscan, execution_time_dbscan = dbscan_clustering(list(G.values()), eps, min_samples)
        dbscan_execution_times.append(execution_time_dbscan)
        dbscan_silhouette_scores.append(silhouette_avg_dbscan)
        
        # Agglomerative Clustering
        points = list(G.values())
        n_samples = len(points)
        if count_hot_point >= n_samples:
            count_hot_point = n_samples - 1
            print(f"Adjusted number of clusters to: {count_hot_point}")
        
        if count_hot_point > 1:
            labels_agglomerative, silhouette_avg_agglomerative, execution_time_agglomerative = agglomerative_clustering(points, n_clusters=count_hot_point)
            agglomerative_execution_times.append(execution_time_agglomerative)
            agglomerative_silhouette_scores.append(silhouette_avg_agglomerative)
        else:
            agglomerative_execution_times.append(float('inf'))
            agglomerative_silhouette_scores.append(-1)
    else:
        silhouette_scores.append(-1)  # 如果没有足够的热点网格，轮廓系数设为 -1
        execution_times.append(float('inf'))  # 如果没有足够的热点网格，运行时间设为无穷大
        dbscan_silhouette_scores.append(-1)
        dbscan_execution_times.append(float('inf'))
        agglomerative_silhouette_scores.append(-1)
        agglomerative_execution_times.append(float('inf'))

# 找到最优的 k 值（现有算法）
optimal_k = k_values[np.argmax(silhouette_scores)]
optimal_silhouette_score = silhouette_scores[np.argmax(silhouette_scores)]

print(f"现有算法最优的 k 值: {optimal_k} 米")
print(f"现有算法对应的轮廓系数: {optimal_silhouette_score}")

# 找到最优的 k 值（DBSCAN）
optimal_k_dbscan = k_values[np.argmax(dbscan_silhouette_scores)]
optimal_silhouette_score_dbscan = dbscan_silhouette_scores[np.argmax(dbscan_silhouette_scores)]

print(f"DBSCAN 最优的 k 值: {optimal_k_dbscan} 米")
print(f"DBSCAN 对应的轮廓系数: {optimal_silhouette_score_dbscan}")

# 找到最优的 k 值（Agglomerative Clustering）
optimal_k_agglomerative = k_values[np.argmax(agglomerative_silhouette_scores)]
optimal_silhouette_score_agglomerative = agglomerative_silhouette_scores[np.argmax(agglomerative_silhouette_scores)]

print(f"Agglomerative Clustering 最优的 k 值: {optimal_k_agglomerative} 米")
print(f"Agglomerative Clustering 对应的轮廓系数: {optimal_silhouette_score_agglomerative}")

# 输出每个 k 的轮廓系数和运行时间（现有算法）
for k, silhouette, exec_time in zip(k_values, silhouette_scores, execution_times):
    print(f"k: {k} 米, 现有算法轮廓系数: {silhouette}, 现有算法运行时间: {exec_time:.2f} 秒")

# 输出每个 k 的轮廓系数和运行时间（DBSCAN）
for k, silhouette, exec_time in zip(k_values, dbscan_silhouette_scores, dbscan_execution_times):
    print(f"k: {k} 米, DBSCAN 轮廓系数: {silhouette}, DBSCAN 运行时间: {exec_time:.2f} 秒")

# 输出每个 k 的轮廓系数和运行时间（Agglomerative Clustering）
for k, silhouette, exec_time in zip(k_values, agglomerative_silhouette_scores, agglomerative_execution_times):
    print(f"k: {k} 米, Agglomerative Clustering 轮廓系数: {silhouette}, Agglomerative Clustering 运行时间: {exec_time:.2f} 秒")

# 可视化结果
plt.figure(figsize=(14, 6))

# 绘制轮廓系数与 k 值的关系图
plt.subplot(1, 2, 1)
plt.plot(k_values, silhouette_scores, marker='o', label='现有算法')
plt.plot(k_values, dbscan_silhouette_scores, marker='x', label='DBSCAN')
plt.plot(k_values, agglomerative_silhouette_scores, marker='^', label='Agglomerative Clustering')
plt.title('轮廓系数与 k 值的关系')
plt.xlabel('k (网格大小, 米)')
plt.ylabel('轮廓系数')
plt.legend()
plt.grid(True)

# 绘制运行时间与 k 值的关系图
plt.subplot(1, 2, 2)
plt.plot(k_values, execution_times, marker='o', color='orange', label='现有算法')
plt.plot(k_values, dbscan_execution_times, marker='x', color='blue', label='DBSCAN')
plt.plot(k_values, agglomerative_execution_times, marker='^', color='red', label='Agglomerative Clustering')
plt.title('运行时间与 k 值的关系')
plt.xlabel('k (网格大小, 米)')
plt.ylabel('运行时间 (秒)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 使用最优的 k 值进行最终聚类（现有算法）
k = optimal_k
cols, rows = get_grids_i_j(lon_min, lat_min, lon_max, lat_max, k)
grids = [[Grid() for _ in range(cols)] for _ in range(rows)]

for pickup_point in tqdm(pickup_points):
    i, j = get_i_j_of_pickup_point(x_min_y_min, x_max_y_max, k, pickup_point)
    grids[i][j].pickup_points.append(pickup_point)

G = {}
count_hot_point = 0
for row in range(rows):
    for col in range(cols):
        if len(grids[row][col].pickup_points) > lameda:
            grids[row][col].locating_of_grid = get_Grid_locating_point(grids[row][col])
            G[count_hot_point] = grids[row][col].locating_of_grid
            count_hot_point += 1

if count_hot_point > 1:
    clustering_result = process_coordinates(G, delta_k)
    export_to_geojson(clustering_result)
else:
    print("没有足够的热点网格进行聚类（现有算法）。")

# 使用最优的 k 值进行最终聚类（DBSCAN）
k_dbscan = optimal_k_dbscan
cols_dbscan, rows_dbscan = get_grids_i_j(lon_min, lat_min, lon_max, lat_max, k_dbscan)
grids_dbscan = [[Grid() for _ in range(cols_dbscan)] for _ in range(rows_dbscan)]

for pickup_point in tqdm(pickup_points):
    i, j = get_i_j_of_pickup_point(x_min_y_min, x_max_y_max, k_dbscan, pickup_point)
    grids_dbscan[i][j].pickup_points.append(pickup_point)



G_dbscan = {}
count_hot_point_dbscan = 0
for row in range(rows_dbscan):
    for col in range(cols_dbscan):
        if len(grids_dbscan[row][col].pickup_points) > lameda:
            grids_dbscan[row][col].locating_of_grid = get_Grid_locating_point(grids_dbscan[row][col])
            G_dbscan[count_hot_point_dbscan] = grids_dbscan[row][col].locating_of_grid
            count_hot_point_dbscan += 1

if count_hot_point_dbscan > 1:
    labels_dbscan, silhouette_avg_dbscan, _ = dbscan_clustering(list(G_dbscan.values()), eps, min_samples)
    export_to_geojson([list(G_dbscan.values())], output_file="clusters_dbscan.geojson")
else:
    print("没有足够的热点网格进行聚类（DBSCAN）。")

# 使用最优的 k 值进行最终聚类（Agglomerative Clustering）
k_agglomerative = optimal_k_agglomerative
cols_agglomerative, rows_agglomerative = get_grids_i_j(lon_min, lat_min, lon_max, lat_max, k_agglomerative)
grids_agglomerative = [[Grid() for _ in range(cols_agglomerative)] for _ in range(rows_agglomerative)]

for pickup_point in tqdm(pickup_points):
    i, j = get_i_j_of_pickup_point(x_min_y_min, x_max_y_max, k_agglomerative, pickup_point)
    grids_agglomerative[i][j].pickup_points.append(pickup_point)

G_agglomerative = {}
count_hot_point_agglomerative = 0
for row in range(rows_agglomerative):
    for col in range(cols_agglomerative):
        if len(grids_agglomerative[row][col].pickup_points) > lameda:
            grids_agglomerative[row][col].locating_of_grid = get_Grid_locating_point(grids_agglomerative[row][col])
            G_agglomerative[count_hot_point_agglomerative] = grids_agglomerative[row][col].locating_of_grid
            count_hot_point_agglomerative += 1

if count_hot_point_agglomerative > 1:
    labels_agglomerative, silhouette_avg_agglomerative, _ = agglomerative_clustering(list(G_agglomerative.values()), n_clusters=count_hot_point_agglomerative)
    export_to_geojson([list(G_agglomerative.values())], output_file="clusters_agglomerative.geojson")
else:
    print("没有足够的热点")