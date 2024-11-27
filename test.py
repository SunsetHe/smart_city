
from some_func import util
def calculate_distance(g, h):
    """
    计算两个点之间的欧几里得距离。
    """
    g = util.wgs84_to_mercator(g[0],g[1])
    h = util.wgs84_to_mercator(h[0],h[1])
    return ((g[0] - h[0]) ** 2 + (g[1] - h[1]) ** 2) ** 0.5

def process_coordinates(coords_dict, delta_k):

    G = list(coords_dict.values())  # 提取所有坐标点作为初始集合 G
    O = []  # 最终结果集合
    i = 0   # 初始化 i

    while G:  # 当 G 非空时
        print(len(G))
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

# 示例数据
def read_coordinates(file_path):
    coordinates = {}

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            # 移除行末的换行符并将字符串转换为列表
            coords = eval(line.strip())
            coordinates[i] = coords

    return coordinates


# 使用示例
file_path = 'file_grid_locate_1.txt'
coords_dict = read_coordinates(file_path)

delta_k = 26  # 设置 δ.k 的值

# 处理数据
result = process_coordinates(coords_dict, delta_k)

# 打印结果
for value in result:
    print(value)

outputfile = "result.txt"


with open(outputfile, mode='w', encoding='utf-8') as f:
    for point in result:
        f.write(str(point) + "\n")  # 每个文件名写入一行
