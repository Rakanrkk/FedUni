######
#工程文件
import Player
import functions
######
#扩展库
import numpy as np


###
#场景设置
###


################### 设置参数
#场景参数
num_devices = 50
radius = 100  # 半径为500米
D2D_distance = 30  # D2D的最远距离
min_distance = radius / 3  # 客户端至基站的最小距离
np.random.seed(0)  # 伪随机锁，注释后会成为真正随机

#训练参数
batch_size = 128
global_epochs = 50
local_epochs = 2
learning_rate = 0.001


################### 实例化基站类

BS = Player.BaseStation((0,0))   #基站类实例化
base_station = BS.findposition()


################### 实例化客户端
client_sequence = []   # 构建客户端列表，方便遍历


for num in range(num_devices):
        # 实例化客户端
        client = Player.Client(num ,min_distance,radius,BS)
        #print(client.Ki)
        #print(client.qiB)
        client_sequence.append(client)




################### 联邦轮次
accuracy_list = []
for epoch in range(global_epochs):
    Client_Selection_List = []
    BS.ClientSelection(client_sequence)
    pass









