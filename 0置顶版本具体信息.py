

#------------------------------------此版本为联邦学习通用框架--------------------------------------
#源于FedD2D，可能存在细节小漏洞，比如加权平均方式，和客户端类没有删除的位置属性，很好改。
#准备了三套数据集和网络，要一一对应
#一种iid,和两种数据异构的划分方式
#传梯度和传模型参数的传参方式
#没有torch-gpu的记得把device = torch.device('cuda')变成device = torch.device('cpu')



#版权为猫猫虫所有，可供个人作为一个初步框架做实验仿真，不要外传。