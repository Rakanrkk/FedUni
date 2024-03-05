import numpy as np
from torch.utils.data import DataLoader
import torch

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

    return per * 100 # 暂时提高1
#此函数用于计算qiB
def calculate_packet_error_rate_d2d(target1, target2):
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

    return per * 100 # 暂时提高1
#此函数用于计算qijD


def create_client_loaders_noniid(train_dataset, num_clients, batch_size):
    client_data_size = len(train_dataset) // num_clients
    client_data_indices = list(range(len(train_dataset)))
    client_data_list = []

    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = (i + 1) * client_data_size
        client_indices = client_data_indices[start_idx:end_idx]
        client_data = torch.utils.data.Subset(train_dataset, client_indices)
        client_data_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
        client_data_list.append(client_data_loader)

    return client_data_list



def create_client_loaders_noniid_linear(train_dataset, num_clients, batch_size):
    # 根据客户端的索引，索引大的分配更多的数据量,均匀递增
    total_size = len(train_dataset)
    sizes = [int(total_size * (i / (num_clients * (num_clients + 1) / 2))) for i in range(1, num_clients + 1)]
    client_data_indices = list(range(total_size))
    client_data_list = []
    start_idx = 0

    for i in range(num_clients):
        end_idx = start_idx + sizes[i]
        client_indices = client_data_indices[start_idx:end_idx]
        client_data = torch.utils.data.Subset(train_dataset, client_indices)
        client_data_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
        client_data_list.append(client_data_loader)
        start_idx = end_idx

    return client_data_list




def create_client_loaders_noniid_dirichlet(train_dataset, num_clients, batch_size, alpha=0.5):
    total_size = len(train_dataset)
    # 使用Dirichlet分布生成数据分配比例
    proportions = np.random.dirichlet(np.array([alpha] * num_clients))
    # 计算每个客户端的数据量
    sizes = [int(prop * total_size) for prop in proportions]

    # 确保所有数据点都被分配
    sizes[-1] = total_size - sum(sizes[:-1])

    client_data_indices = torch.randperm(total_size).tolist()
    client_data_list = []
    start_idx = 0

    for i in range(num_clients):
        end_idx = start_idx + sizes[i]
        client_indices = client_data_indices[start_idx:end_idx]
        client_data = torch.utils.data.Subset(train_dataset, client_indices)
        client_data_loader = torch.utils.data.DataLoader(client_data, batch_size=batch_size, shuffle=True)
        client_data_list.append(client_data_loader)
        start_idx = end_idx

    return client_data_list


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.view(inputs.size(0), -1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

1






