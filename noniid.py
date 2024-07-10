import numpy as np
from torch.utils.data import DataLoader, Subset, ConcatDataset
import DataSet
import random
from torch.utils.data import DataLoader, Dataset





def create_client_loaders_noniid(train_dataset, num_clients, batch_size, num_classes=10, batchs_per_client=10):
    #这个函数当设置的每个客户端分配的样本量大于数据集总样本量会报错，可以根据第二个划分函数取一些重复的数据，很好改。
    # 分类数据
    samples_per_client = batchs_per_client * batch_size
    label_to_indices = {i: [] for i in range(num_classes)}
    for index, (_, target) in enumerate(train_dataset):
        label_to_indices[target].append(index)

    client_data_list = []

    # 每个客户端将从不同的标签中随机选择一定数量的数据，保持noniid
    for i in range(num_clients):
        print('划分客户端', i)
        client_data = []

        # 从所有标签中随机选择标签，数量根据noniid的需求调整
        selected_labels = random.sample(list(label_to_indices.keys()), 2)  # 目前选择3个标签

        for label in selected_labels:
            indices = label_to_indices[label]
            sampled_indices = random.sample(indices, k=samples_per_client // len(selected_labels))
            client_data.extend(sampled_indices)

        client_dataset = DataSet.MyDataset([train_dataset[i][0] for i in client_data],
                                   [train_dataset[i][1] for i in client_data])

        client_data_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        client_data_list.append(client_data_loader)

    return client_data_list






def create_client_loaders_noniid_unbalanced(train_dataset, num_clients, batch_size, num_classes=10,
                                            min_size_per_client=64):
    # 分类数据
    label_to_indices = {i: [] for i in range(num_classes)}
    for index, (_, target) in enumerate(train_dataset):
        label_to_indices[target].append(index)

    client_data_list = []

    # 计算不同标签数量和数据集大小的阈值
    threshold_large = int(0.2 * num_clients) #CIFAE为0.2

    for i in range(num_clients):
        print('划分客户端',i)
        if i < threshold_large:  # 前20%得到2个标签

            selected_labels = random.sample(list(label_to_indices.keys()), 2)
            client_data = []

            # 随机确定每个客户端的数据量大小，范围约束在10到20之间
            client_data_size = random.randint(10 * min_size_per_client, 20 * min_size_per_client)
            #print(client_data_size, 'clientdatasize')
            # 从选定的每个类别中收集数据
            for label in selected_labels:
                indices = label_to_indices[label]
                # 如果可选的索引少于客户端应有的数据量一半，则全部索引都用上；否则进行随机选择
                #num_samples = min(len(indices), client_data_size // 3)
                num_samples = client_data_size
                print(num_samples, 'num_samples')
                sampled_indices = random.sample(indices, k=num_samples)
                client_data.extend(sampled_indices)
                print(len(client_data))

            client_dataset = DataSet.MyDataset([train_dataset[i][0] for i in client_data],
                                               [train_dataset[i][1] for i in client_data])

            client_data_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
            print(len(client_data_loader),'dataloader')
            client_data_list.append(client_data_loader)

        else:  # 剩下的50%得到1个标签
            selected_labels = random.sample(list(label_to_indices.keys()), 1)
            client_data = []

            # 随机确定每个客户端的数据量大小，范围约束在10到20之间
            client_data_size = random.randint(1 * min_size_per_client, 3 * min_size_per_client)

            # 从选定的每个类别中收集数据
            for label in selected_labels:
                indices = label_to_indices[label]
                # 如果可选的索引少于客户端应有的数据量一半，则全部索引都用上；否则进行随机选择
                #num_samples = min(len(indices), client_data_size )
                num_samples = client_data_size
                sampled_indices = random.sample(indices, k=num_samples)
                client_data.extend(sampled_indices)

            client_dataset = DataSet.MyDataset([train_dataset[i][0] for i in client_data],
                                               [train_dataset[i][1] for i in client_data])

            # 为客户端创建数据加载器

            client_data_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True,drop_last=True)
            client_data_list.append(client_data_loader)


    return client_data_list




def create_client_loaders_iid(train_dataset, num_clients, batch_size):
    # 获取数据集大小和所有索引
    num_data = len(train_dataset)
    all_indices = list(range(num_data))

    # 随机打乱索引
    np.random.shuffle(all_indices)

    # 计算每个客户端的数据大小
    client_data_size = num_data // num_clients
    client_data_list = []

    # 为每个客户端分配数据
    for i in range(num_clients):
        # 如果是最后一个客户端，取剩余所有数据
        if i == num_clients - 1:
            client_indices = all_indices[i * client_data_size:]
        else:
            client_indices = all_indices[i * client_data_size: (i + 1) * client_data_size]

        # 创建数据子集和数据加载器
        client_data = Subset(train_dataset, client_indices)
        client_data_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
        client_data_list.append(client_data_loader)

    return client_data_list
