import numpy as np
import functions


class BaseStation:
    def __init__(self,position):
        self.position = position  # 基站位置


    def ClientSelection(self,client_sequence):
        Client_sequence = client_sequence
        # 距离计算，可以内嵌？

        # 计算所有客户端的Value函数Vi
        for client in Client_sequence:
            Vi = functions.Single_Value(client)
            print('客户端的Value函数',Vi)

        # 对所有客户端的Vi进行排序

    def findposition(self):
        return self.position


class Client:
    """客户端类"""

    def __init__(self, client_id, min_distance,radius,BaseStation):

        ###场景用
        self.client_id = client_id      #从0开始的
        self.position = setting_position(min_distance,radius)
        self.state = 'unavailable'      # 用户状态
        '''
        用户状态
        '''
        ###训练用
        self.BaseStation= BaseStation

        ###调度用
        self.Ki = 50
        self.qiB = functions.calculate_packet_error_rate(self.position, BaseStation.findposition())


    # 调用该客户端位置
    def findposition(self):
        print('客户端%s的位置是:'%(self.client_id),self.position)
        return self.position


    # 进行D2D任务
    def D2D(self):
        if self.state == ' ':
            pass



def setting_position(min_distance,radius):
    # 创建随机位置，确保所有客户端都在圆内，同时不过于接近基站

    while True:
        r = np.sqrt(np.random.uniform(min_distance ** 2, radius ** 2))  # 随机生成半径，带最小限制
        theta = np.random.uniform(0, 2 * np.pi)  # 随机生成角度
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return (x,y)



def counting_distance(target1, target2):
    target1 = np.array(target1)
    target2 = np.array(target2)
    distance = np.linalg.norm(target1 - target2)
    return distance

