from math import radians, cos, sin, asin, sqrt, atan, pi, log, tan, exp, atan2, degrees, fabs
import numpy as np
from geopy import distance
from scipy import interpolate
from pyproj import Transformer


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

# 坐标点转为墨卡托坐标系 list
def transform_points_wgs84_to_mercator(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(wgs84_to_mercator(item[0], item[1]))
    return temp_result


def transform_points_mercator_to_wgs84(coordinates):
    temp_result = []
    for i, item in enumerate(coordinates):
        temp_result.append(mercator_to_wgs84(item[0], item[1]))
    return temp_result
