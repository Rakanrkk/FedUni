######
# 工程文件
import Player
import Networks
import functions
######
# 扩展库
import numpy as np
import random
from torchvision import datasets, transforms
from torch.optim import SGD
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt



###
# 场景设置
###


################### 设置参数
# 场景参数
num_devices = 100
radius = 100  # 半径为500米
D2D_distance = 30  # D2D的最远距离
min_distance = radius / 3  # 客户端至基站的最小距离
np.random.seed(0)  # 伪随机锁，注释后会成为真正随机
N = 30  # 基站客户端容量
C = 30  # D2D最大通信半径
# 训练参数
batch_size = 128
global_epochs = 50
local_epochs = 2
learning_rate = 0.001
device = torch.device('cpu') #'cuda' if torch.cuda.is_available() else  cuda:0

################### 实例化基站类
model = Networks.MLP().to(device)
optimizer = torch.optim.Adam(model.parameters())
BS = Player.BaseStation((0, 0),model,optimizer)  # 基站类实例化
base_station = BS.findposition()

################### 实例化客户端
client_sequence = []  # 构建客户端列表，方便遍历
pair_sequence = []  # 构建D2D配对列表，方便遍历


# 数据集准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

# 创建客户端数据加载器
#client_loaders = functions.create_client_loaders_noniid(train_dataset, num_devices, batch_size)
client_loaders = functions.create_client_loaders_noniid_dirichlet(train_dataset, num_devices, batch_size, alpha=0.8)


for client_loader in client_loaders:
        num = client_loaders.index(client_loader)
        # 实例化客户端
        client = Player.Client(num ,min_distance,radius,BS,dataloader = client_loader)
        #print(client.Ki)
        client_sequence.append(client)


#   对客户端列表排序，但不先取
sorted_client_sequence = sorted(client_sequence, key=lambda x: x.Vi,reverse=True)


################### 构建D2D pair类
#   生成D2D pair（但是有重复）
for i in range(len(client_sequence)):
    for j in range(i + 1, len(client_sequence)):
        # 距离判断：大于D2D通信半径就跳过
        if Player.counting_distance(client_sequence[i].position, client_sequence[j].position) > C:
            continue
        D2D_pair = Player.D2D_pair(client_sequence[i], client_sequence[j])
        pair_sequence.append(D2D_pair)

#   先对Vij进行排序
sorted_client_pair_sequence = sorted(pair_sequence, key=lambda x: x.Vij,reverse=True)
#   调用函数，从高到低，去重sorted_client_pair_sequence
unique_pairs = Player.remove_duplicate_pairs(sorted_client_pair_sequence)




################### 基站进行客户端调度
select = []
R = BS.ClientSelection(N,client_sequence,unique_pairs)
for item in R:
    if isinstance(item, Player.Client):  # 如果是 Client 对象
        #可以加一个调包概率
        if random.random() <= 1 - item.qiB:
            select.append(item)
            print("设备参与联邦:", item.client_id)
        else:
            print("设备调包", item.client_id)

    elif isinstance(item, Player.D2D_pair):  # 如果是 Pair 对象
        #可以加一个D2B调包概率
        p = random.random()
        q = random.random()
        if p <= 1 - item.clientB.qiB:
            select.append(item.clientB)
            print("D2B设备参与联邦:", item.clientB.client_id)
            if random.random() <= 1 - item.Q:
                #加一个D2D调包概率
                select.append(item.clientD)
                print("D2D设备参与联邦:", item.clientD.client_id)
            else:
                print("D2D设备掉包", item.clientD.client_id)
        else:
            print("D2B设备掉包", item.clientB.client_id)



print(len(select),'长度')

################### 联邦轮次
accuracy_list = []
for epoch in range(global_epochs):
    # 初始化一个全0列表，作为中间变量，负责存储每个客户端的梯度，以确保在一个round内能聚合，下一个round则消除

    aggregate = []
    Name_Ki_list = []
    # 每个客户端本地训练，依次更新，不聚合
    for client in select:
        # 调用train的时候就是往列表里加东西（梯度）但是对于梯度，对应维度相加，而且最后加权平均 Client单负责累加，Server负责加权
        grad,Ki = client.train(aggregate,Name_Ki_list,local_epochs,learning_rate,device)
        aggregate.append(grad)  # 将grad列表转换为Tensor类型，并添加到aggregate中
        Name_Ki_list.append(Ki)
    #print(len(aggregate))
    #print(Name_Ki_list)

    # Server进行聚合
    BS.aggregate_gradients(aggregate, Name_Ki_list)
    # 测试模型
    accuracy = functions.evaluate(BS.get_model(), test_loader, device)
    accuracy_list.append(accuracy * 100)
    print(f'Global Epoch: {epoch + 1}, Test Accuracy: {accuracy * 100:.2f}%')

plt.plot(range(len(accuracy_list)), accuracy_list)
plt.xlabel("Epoch")  # 给x轴起名字
plt.ylabel("Accuracy")  # 给y轴起名字
print(accuracy_list)
plt.show()