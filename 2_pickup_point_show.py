# import csv
# from datetime import datetime
# import matplotlib.pyplot as plt
# from collections import Counter
#
#
# def read_csv(file_name):
#     """
#     从 CSV 文件读取数据并返回时间戳列表
#
#     Args:
#         file_name: CSV 文件名
#
#     Returns:
#         timestamps: 时间戳列表（格式：'HH:MM:SS'）
#     """
#     timestamps = []
#     with open(file_name, mode='r', encoding='utf-8') as file:
#         reader = csv.reader(file)
#         next(reader)  # 跳过表头
#         for row in reader:
#             timestamps.append(row[2])  # 假设 timestamp 在第3列
#     return timestamps
#
#
# def count_data_by_hour(timestamps):
#     """
#     根据时间戳统计每小时的数据量
#
#     Args:
#         timestamps: 时间戳列表（格式：'HH:MM:SS'）
#
#     Returns:
#         hourly_counts: 每小时数据量的字典，键为小时，值为数据量
#     """
#     hours = [datetime.strptime(ts, '%H:%M:%S').hour for ts in timestamps]
#     hourly_counts = Counter(hours)
#     return hourly_counts
#
#
# def plot_hourly_data(hourly_counts):
#     """
#     绘制柱状图
#
#     Args:
#         hourly_counts: 每小时数据量的字典
#     """
#     hours = range(24)  # 24小时
#     counts = [hourly_counts.get(hour, 0) for hour in hours]  # 确保每小时都有数据
#
#     plt.figure(figsize=(10, 6))
#     plt.bar(hours, counts, color='skyblue', alpha=0.8)
#     plt.xlabel('Hour of Day (0-23)')
#     plt.ylabel('Number of Data Points')
#     plt.title('Number of Data Points Per Hour')
#     plt.xticks(hours)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()
#
#
# # 主程序
# file_name = 'pickup_points.csv'  # 替换为你的 CSV 文件路径
# timestamps = read_csv(file_name)
# hourly_counts = count_data_by_hour(timestamps)
# plot_hourly_data(hourly_counts)
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter


def read_csv(file_name):
    """
    从 CSV 文件读取数据并返回时间戳列表

    Args:
        file_name: CSV 文件名

    Returns:
        timestamps: 时间戳列表（格式：'HH:MM:SS'）
    """
    timestamps = []
    with open(file_name, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            timestamps.append(row[2])  # 假设 timestamp 在第3列
    return timestamps


def count_data_by_two_hour_groups(timestamps):
    """
    根据时间戳统计每两个小时的数据量

    Args:
        timestamps: 时间戳列表（格式：'HH:MM:SS'）

    Returns:
        two_hour_counts: 每两个小时数据量的字典，键为时间段，值为数据量
    """
    # 获取每个时间戳对应的小时，并分组到两个小时的区间
    two_hour_groups = [(datetime.strptime(ts, '%H:%M:%S').hour // 2) for ts in timestamps]
    two_hour_counts = Counter(two_hour_groups)
    return two_hour_counts


def plot_two_hourly_data(two_hour_counts):
    """
    绘制柱状图

    Args:
        two_hour_counts: 每两个小时数据量的字典
    """
    group_labels = [f'{i * 2:02d}-{i * 2 + 1:02d}' for i in range(12)]  # 生成时间段标签
    counts = [two_hour_counts.get(i, 0) for i in range(12)]  # 确保每组都有数据

    plt.figure(figsize=(10, 6))
    plt.bar(group_labels, counts, color='skyblue', alpha=0.8)
    plt.xlabel('Time Group (2-hour intervals)')
    plt.ylabel('Number of Data Points')
    plt.title('Number of Data Points Per 2-Hour Group')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# 主程序
file_name = 'pickup_points.csv'  # 替换为你的 CSV 文件路径
timestamps = read_csv(file_name)
two_hour_counts = count_data_by_two_hour_groups(timestamps)
plot_two_hourly_data(two_hour_counts)
