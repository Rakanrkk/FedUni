######
# 工程文件
import DataSet
import Player
import Networks
import Functions
import noniid
import participant
######
# 扩展库
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt



###
# 场景设置
###


################### 设置参数
# 场景参数
num_devices = 25 #客户端数量
radius = 100  # 半径为500米
min_distance = radius / 3  # 客户端至基站的最小距离
np.random.seed(0)  # 伪随机锁，注释后会成为真正随机


# 训练参数
batch_size = 64
global_epochs = 1000
local_epochs = 1
learning_rate = 0.001
device = torch.device('cuda')

################### 实例化基站类
#model = Networks.MLP().to(device)
#model = Networks.CNN().to(device)
#model = Networks.AlexNet().to(device)
model = Networks.VGG().to(device)
#model = Networks.resnet.to(device)


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
BS = Player.BaseStation((0, 0),model,optimizer)  # 基站类实例化
base_station = BS.findposition()

################### 实例化客户端
client_sequence = []  # 构建客户端列表，方便遍历


# 加载数据集
#train_dataset,test_dataset = DataSet.MNIST()
#train_dataset,test_dataset = DataSet.FashionMNIST()
train_dataset,test_dataset = DataSet.CIFAR()

# 创建客户端数据加载器,non-iid,unbalanced
#client_loaders = noniid.create_client_loaders_iid(train_dataset, num_devices, batch_size)
client_loaders = noniid.create_client_loaders_noniid(train_dataset, num_devices, batch_size, num_classes=10, batchs_per_client=40)
#client_loaders = noniid.create_client_loaders_noniid_unbalanced(train_dataset, num_devices, batch_size, num_classes=10)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

for client_loader in client_loaders:
        num = client_loaders.index(client_loader)
        # 实例化客户端
        client = Player.Client(num ,min_distance,radius,BS,dataloader = client_loader)
        print('检测客户端数据集大小：',client.Ki)
        #print(client.qiB)
        client_sequence.append(client)

###
# 联邦轮次，关于时间t的大循环
###

accuracy_list = []  # 准确率表用于可视化
for epoch in range(global_epochs):

    aggregate = [] #模型参数\梯度列表
    Name_Ki_list = [] #数据集大小列表（权重）
    # 只有被选中的客户端才进行训练，train谁基站将全局模型发谁

    #N = participant.Greedy[epoch]
    #N = participant.KM[epoch]
    #N = participant.rand[epoch]
    N = 15
    #N = participant.UCB[epoch]



    selected = BS.random_selection(client_sequence,N)
    print(len(selected))
    for client in selected:

        param,Ki = client.param_computing(local_epochs,learning_rate,device)
        aggregate.append(param)
        Name_Ki_list.append(Ki)


    # Server进行聚合
    BS.aggregate_weightedmodels(aggregate, Name_Ki_list)
    #BS.aggregate_models(aggregate)


    ################### 测试模型
    accuracy = Functions.evaluate(BS.get_model(), test_loader, device)
    accuracy_list.append(accuracy * 100)
    #print(client_sequence[2].Vi)
    print(f'Global Epoch: {epoch + 1}, Test Accuracy: {accuracy * 100:.2f}%')

plt.plot(range(len(accuracy_list)), accuracy_list)
plt.xlabel("Epoch")  # 给x轴起名字
plt.ylabel("Accuracy")  # 给y轴起名字
print(accuracy_list)
plt.show()

