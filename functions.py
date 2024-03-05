import numpy as np


def calculate_packet_error_rate(target1, target2):
    """
    计算两点之间的数据包错误率（PER）。

    参数:
    target1 (tuple): 第一个点（客户端）的 (x, y) 坐标。
    target2 (tuple): 第二个点（服务器）的 (x, y) 坐标。

    返回:
    float: 计算出的数据包错误率。
    """
    # 常量
    transmission_power_client = 10e-3  # 客户端的传输功率 10 mW 转换为瓦特
    noise_power_density = -173  # 噪声功率谱密度 dBm/Hz
    bandwidth_uplink = 1e6  # 上行链路带宽 1 Mbps 转换为 Hz
    tau = 1  # 阈值参数，需要根据系统设置来定义或获取

    # 将噪声功率谱密度从 dBm/Hz 转换为 W/Hz
    noise_power_density_w = 10 ** ((noise_power_density - 30) / 10)

    # 提取坐标
    x1, y1 = target1
    x2, y2 = target2

    # 计算客户端和服务器之间的距离
    distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    # 路径损耗模型（简化为自由空间模型）
    path_loss = (4 * np.pi * distance * 1e9 / (3e8)) ** 2  # 这里使用 1 GHz 频率作为示例

    # 信道增益计算
    channel_gain = 1 / path_loss

    # 信噪比 (SNR) 计算
    snr = (transmission_power_client * channel_gain) / (noise_power_density_w * bandwidth_uplink)

    # 根据提供的公式计算数据包错误率 (PER)
    per = 1 - np.exp(-tau * bandwidth_uplink * noise_power_density_w / (transmission_power_client * channel_gain))

    return per


def Single_Value(Client):
    Ki = Client.Ki
    qiB = Client.qiB
    Vi = Ki * (1 - qiB)
    return Vi

def Pair_Value(Client1,Client2):
    K1 = Client1.Ki
    q1B = Client1.qiB

    K2 = Client2.Ki
    q2B = Client2.qiB

    Vij = None

    pass














