from datetime import datetime
import os
import csv

from tqdm import tqdm

from some_func import util

class Grid:
    def __init__(self):
        locating_point = []  # 位于当前网格的定位点
        locating_of_grid = [] # 网格的位置，里面所有点的经纬度平均

def Distance(pointa,pointb):
    distance = 0
    # todo
    return distance

def load_data():# data:key->taxi_id,value->[[time],[coordinates(mercator)],[status]],
    data = {}
    traj_folder = os.path.join(os.getcwd(), "traj")
    for file_name in tqdm(os.listdir(traj_folder)) :
        if file_name.startswith("traj_") and file_name.endswith(".csv"):
            file_path = os.path.join(traj_folder, file_name)
            with open(file_path, mode='r', encoding='utf-8') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    ID = row['ID']
                    time_str = row['Time']
                    time = datetime.strptime(time_str, "%H:%M:%S").time()  # 转换为 time 对象
                    longitude = float(row['Longitude'])
                    latitude = float(row['Latitude'])
                    status = int(row['Status'])
                    data.setdefault(ID, [[], [], []])
                    data[ID][0].append(time)
                    data[ID][1].append([longitude,latitude])
                    # data[ID][1].append(util.mercator_to_wgs84(longitude, latitude))
                    data[ID][2].append(status)
                    # print(ID)
                    # print(time,type(time))
                    # print([longitude, latitude])
                    # print(status,type(status))
                    # print("----------------------")

    return data
########################################################################################

# 参数
# 采用墨卡托坐标系
# lon经度 lat纬度
lon_min = 0
lon_max = 0
lat_min = 0
lat_max = 0

k = 20# 网格的大小，单位为米
lameda = 0# 网格密度阈值
fai = 1.3 # 定义9里面那个参数

######################################################################################

# 读入数据

traj_data = load_data()

all_longitudes = []
all_latitudes = []

for taxi_id, values in traj_data.items():
    coordinates = values[1]
    for coord in coordinates:
        all_longitudes.append(coord[0])  # 经度
        all_latitudes.append(coord[1])  # 纬度

# 确定经纬度的最大最小值
lon_min = min(all_longitudes)
lon_max = max(all_longitudes)
lat_min = min(all_latitudes)
lat_max = max(all_latitudes)

# 打印结果
print(f"Longitude Range: {lon_min} to {lon_max}")
print(f"Latitude Range: {lat_min} to {lat_max}")
