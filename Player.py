import numpy as np
import torch
from torch.optim import SGD
import torch.nn as nn
import copy
import random

class BaseStation:
    def __init__(self, position,model,optimizer):
        self.position = position  # 基站位置
        self.model = model  # 进行联邦的模型部署
        self.optimizer = optimizer

    def random_selection(self, client_sequence, N):
        return random.sample(client_sequence, N)


    def aggregate_gradients(self, gradients, Name_Ki_list):

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
        self.optimizer.step()


    def aggregate_models(self, models):
        # 联邦平均，FedAvg
        if not models:
            pass
        else:

            aggregated_model = {}
            for key in models[0].keys():
                aggregated_model[key] = torch.zeros_like(models[0][key])

            # Calculate the total number of models
            num_models = len(models)

            print('本次聚合的模型总数:', num_models)

            # Aggregate models by calculating the average of each parameter
            for model in models:
                for key in aggregated_model.keys():
                    aggregated_model[key] += model[key].float() / num_models

            # Load the aggregated parameters into the global model at the base station
            self.model.load_state_dict(aggregated_model, strict=True)




    def aggregate_weightedmodels(self, models, weights=None):
        # 加权平均
        if models == []:
            pass
        else:
            if weights is None:
                weights = [1.0] * len(models)
            else:
                assert len(weights) == len(models), "Number of weights should match number of models."

            # Initialize aggregated model parameters
            aggregated_model = {}
            for key in models[0].keys():
                aggregated_model[key] = torch.zeros_like(models[0][key].float())


            total_weight = sum(weights)
            print('本次聚合样本总大小:',total_weight)

            # Convert models to float and Aggregate models
            for model, weight in zip(models, weights):
                model_float = {k: v.float() for k, v in model.items()}  # Convert all parameters to float
                for key in aggregated_model.keys():
                    aggregated_model[key] += model_float[key] * (weight / total_weight)

            # Ensure the aggregated model is also in float to match the converted model
            aggregated_model = {k: v.float() for k, v in aggregated_model.items()}

            # Load the aggregated parameters into the global model at the base station
            self.model.load_state_dict(aggregated_model, strict=True)




    def get_model(self):
        return self.model

    def findposition(self):
        return self.position

    def print_model_parameters(self):
        """
        打印模型参数的名称、形状和部分数据。
        """
        for name, param in self.model.named_parameters():
            print(f"参数名称: {name}")
            print(f"形状: {param.size()}")
            print(f"数据的一部分: {param.data}")
            print("-" * 50)  # 分割线



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

    def grad_computing(self, local_epochs, lr, device):
        # 拷贝当前全局模型
        model = copy.deepcopy(self.BaseStation.get_model())
        # 优化器SGD
        optimizer = SGD(model.parameters(), lr=lr)
        #optimizer = optim.Adam(model.parameters(), lr=0.001)

        criterion = nn.CrossEntropyLoss()
        #criterion = nn.MSELoss()
        for _ in range(local_epochs):
            for batch_idx, (inputs, labels) in enumerate(self.dataloader):

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

    def param_computing(self, local_epochs, lr, device):
        # 拷贝当前全局模型
        model = copy.deepcopy(self.BaseStation.get_model())
        # 优化器SGD
        optimizer = SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        #criterion = nn.MSELoss()
        for _ in range(local_epochs):
            for batch_idx, (inputs, labels) in enumerate(self.dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                #print(loss.item())
                #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
                optimizer.step()
        #print('客户端训练完成',self.client_id)
        return model.state_dict(), self.Ki

    # 调用该客户端位置
    def findposition(self):
        print('客户端%s的位置是:' % (self.client_id), self.position)
        return self.position








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


