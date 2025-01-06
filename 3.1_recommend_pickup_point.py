import numpy as np

def manhattan_distance(coord1, coord2):
    """计算两个点在 Mercator 坐标系下的曼哈顿距离。"""
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

def recommend_pickup_point(taxi_coord, passenger_coord, existing_points, a, b, c):
    """
    根据输入的坐标和权重推荐一个上车点。

    参数：
        taxi_coord (tuple): 出租车的 Mercator 坐标 (x, y)。
        passenger_coord (tuple): 乘客的 Mercator 坐标 (x, y)。
        existing_points (list): 现有上下车点的坐标列表 (x, y)。
        a (float): 出租车距离的权重。
        b (float): 乘客距离的权重。
        c (float): 与现有上下车点接近程度的权重。

    返回值：
        tuple: 推荐的上车点坐标 (lon, lat)。
    """
    # 在乘客位置附近定义候选点的网格
    search_radius = 1000  # 搜索半径（单位：米）
    grid_step = 10  # 网格步长（单位：米）

    passenger_x, passenger_y = passenger_coord

    candidate_points = [
        (passenger_x + dx, passenger_y + dy)
        for dx in range(-search_radius, search_radius + 1, grid_step)
        for dy in range(-search_radius, search_radius + 1, grid_step)
    ]

    best_point = None
    best_rate = float('inf')

    for candidate in candidate_points:
        taxi_distance = manhattan_distance(taxi_coord, candidate) / 15
        passenger_distance = manhattan_distance(passenger_coord, candidate) / 2
        existing_distance = min(manhattan_distance(candidate, point) for point in existing_points)

        rate = a * taxi_distance + b * passenger_distance + c * existing_distance

        if rate < best_rate:
            best_rate = rate
            best_point = candidate

    return best_point

def convert_to_mercator(lon, lat):
    """将经纬度转换为 Mercator 坐标（示例实现）。"""
    R = 6378137.0  # 地球半径（单位：米）
    x = R * np.radians(lon)
    y = R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
    return x, y

def convert_to_lonlat(x, y):
    """将 Mercator 坐标转换回经纬度。"""
    R = 6378137.0
    lon = np.degrees(x / R)
    lat = np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)
    return lon, lat


# 自动将 (lat, lon) 转换为 (lon, lat)
def swap_latlon_to_lonlat(latlon_list):
    """
    将输入的 (lat, lon) 坐标列表转换为 (lon, lat) 格式。

    Parameters:
        latlon_list (list): 包含 (lat, lon) 元组的列表。

    Returns:
        list: 转换后的 (lon, lat) 元组列表。
    """
    return [(lon, lat) for lat, lon in latlon_list]












# 示例输入坐标（经纬度）
taxi_lonlat = (22.548335516393347, 114.1287778033209)
passenger_lonlat = (22.545655191123327, 114.12468475242909)
existing_lonlat_points = [
    (22.546283856481708, 114.12570407976479),
    (22.545676996993222, 114.1258412369469),
    (22.54780099356031, 114.12445690626753)
]










taxi_coord = convert_to_mercator(*taxi_lonlat)
passenger_coord = convert_to_mercator(*passenger_lonlat)
existing_points = [convert_to_mercator(lon, lat) for lon, lat in existing_lonlat_points]

taxi_lonlat = taxi_lonlat[::-1]  # 单个元组交换
passenger_lonlat = passenger_lonlat[::-1]
existing_lonlat_points = swap_latlon_to_lonlat(existing_lonlat_points)

print("Taxi coordinates (lon, lat):", taxi_lonlat)
print("Passenger coordinates (lon, lat):", passenger_lonlat)
print("Existing points (lon, lat):", existing_lonlat_points)
# 权重
a, b, c = 0.2, 0.5, 0.3

# 推荐一个上车点
recommended_point_mercator = recommend_pickup_point(taxi_coord, passenger_coord, existing_points, a, b, c)

# 将推荐点转换回经纬度（Mercator 投影的逆运算）

recommended_point_lonlat = convert_to_lonlat(*recommended_point_mercator)

print("推荐的上车点（经纬度）:", recommended_point_lonlat)