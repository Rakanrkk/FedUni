import numpy as np
import functions
from torch.optim import SGD
import torch.nn as nn
import copy

class BaseStation:
    def __init__(self, position,model,optimizer):
        self.position = position  # 基站位置
        self.model = model  # 进行联邦的模型部署
        self.optimizer = optimizer

    def ClientSelection(self, N, client_sequence, unique_pairs):
        # 挑选S集合的Top-N个
        S = client_sequence[0:N]
        # 挑选P集合的Top-N个
        P = unique_pairs[0:N]
        # 联合两个集合，但是类不同
        R = S + P
        #print(len(R))
        # 拟删除设备的列表
        items_to_remove = []

        for client in S:
            #print('此刻的单个设备id',client.client_id)
            for pair in P:
                #print('此刻的pairD2D设备id', pair.clientB.client_id)
                #print('此刻的pairD2B设备id', pair.clientD.client_id)
                #print(pair.clientB.client_id,',',pair.clientD.client_id)
                if client.client_id == pair.clientB.client_id or client.client_id == pair.clientD.client_id:
                    if client.Vi <= pair.Vi :
                        items_to_remove.append(client)
                        #print('移除',client.client_id)
                    else :
                        items_to_remove.append(pair)
                        #print('移除', pair.clientB.client_id)

        for item in items_to_remove:
            R.remove(item)

        R = sorted(R, key=lambda x: x.Vi, reverse=True)
        print(len(R))
        return R

    def aggregate_gradients(self, gradients, Name_Ki_list):
        # 梯度聚合
        '''
        :param gradients: num_clients累加的分类器梯度和
        :param gradients_auto: num_clients累加的编码器梯度和
        :param num_clients: 客户端的个数
        '''
        K = sum(Name_Ki_list)

        # 在全局分类器模型上应用聚合后的梯度
        for param in self.model.parameters():
            param.grad = None  # 清空梯度

        for client_idx, client_grad in enumerate(gradients):
            client_weight = Name_Ki_list[client_idx] / K  # 计算该客户端权重的归一化值
            for idx, param in enumerate(self.model.parameters()):
                if param.grad is None:
                    param.grad = client_grad[idx] * client_weight
                else:
                    param.grad += client_grad[idx] * client_weight

        # 更新模型参数(如果是聚合模型就没那么多事情，后续改，目前还是聚合梯度)
        self.optimizer.step()

    def get_model(self):
        return self.model

    def findposition(self):
        return self.position


class Client:
    """客户端类"""

    def __init__(self, client_id, min_distance, radius, BaseStation, dataloader):
        ###场景用
        self.client_id = client_id  # 从0开始的
        self.position = setting_position(min_distance, radius)
        ###训练用
        self.BaseStation = BaseStation
        self.dataloader = dataloader  # 分给客户端的本地数据集。初始化客户端要做数据并发
        ###调度用
        self.Ki = len(self.dataloader)
        self.qiB = functions.calculate_packet_error_rate(self.position, BaseStation.findposition())
        self.Vi = self.Ki * (1- self.qiB)

    def train(self, aggregate, Name_Ki_list, local_epochs, lr, device):

        # 拷贝当前全局模型（但是必须得想好，全局模型是通过广播还是其他方式进行下发）
        model = copy.deepcopy(self.BaseStation.get_model())

        # 优化器SGD
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for batch_idx, (inputs, labels) in enumerate(self.dataloader):
                # print(inputs.shape)
                inputs = inputs.view(inputs.size(0), -1)
                # print(inputs.shape)
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()

        grad = []  # 初始化中间变量grad,负责向server发送所有客户端的梯度

        for param in model.parameters():
            # 遍历模型每层的参数：
            # print(param.grad.shape,'梯度大小')
            grad.append(param.grad)

        return grad, self.Ki



    # 调用该客户端位置
    def findposition(self):
        print('客户端%s的位置是:' % (self.client_id), self.position)
        return self.position



class D2D_pair:
    """客户端对"""

    def __init__(self, client1, client2):
        if client1.qiB < client2.qiB:
            self.clientB = client1
            self.clientD = client2
        else:
            self.clientB = client2
            self.clientD = client1
        self.Dij=counting_distance(self.clientB.position, self.clientD.position)
        self.Q = functions.calculate_packet_error_rate_d2d(self.clientB.position, self.clientD.position)
        self.Vij = (1 - self.clientB.qiB) * (self.clientB.Ki + self.clientD.Ki) * (1 - self.clientD.qiB) * (
                    1 - self.clientD.qiB)
        self.Vi = (1 - self.clientB.qiB) * (self.clientB.Ki + self.clientD.Ki)


def setting_position(min_distance, radius):
    # 创建随机位置，确保所有客户端都在圆内，同时不过于接近基站

    while True:
        r = np.sqrt(np.random.uniform(min_distance ** 2, radius ** 2))  # 随机生成半径，带最小限制
        theta = np.random.uniform(0, 2 * np.pi)  # 随机生成角度
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return (x, y)


def counting_distance(target1, target2):
    target1 = np.array(target1)
    target2 = np.array(target2)
    distance = np.linalg.norm(target1 - target2)
    return distance

def remove_duplicate_pairs(sorted_client_pair_sequence):
    unique_pairs = []
    seen_client_ids = set()

    for pair in sorted_client_pair_sequence:
        clientB_id = pair.clientB.client_id
        clientD_id = pair.clientD.client_id

        if clientB_id not in seen_client_ids and clientD_id not in seen_client_ids:
            unique_pairs.append(pair)
            seen_client_ids.add(clientB_id)
            seen_client_ids.add(clientD_id)
            #print(clientB_id, clientD_id)
    return unique_pairs