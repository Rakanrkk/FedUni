import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import networkx as nx


class GNNModel(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNModel, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels, 16)
        self.conv2 = pyg_nn.GCNConv(16, out_channels)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.sigmoid(self.conv2(x, edge_index))
        return x

def loss_fn(edge_scores, edge_index, weights, lambda_, num_nodes):
    # Compute total weight of selected edges
    selected_edges = edge_scores > 0.5
    edge_weights = weights[selected_edges]
    total_weight = edge_weights.sum()

    # Compute penalty for shared vertices
    node_scores = pyg_utils.degree(edge_index[0][selected_edges], num_nodes=num_nodes) + \
                  pyg_utils.degree(edge_index[1][selected_edges], num_nodes=num_nodes)
    penalty = lambda_ * (node_scores ** 2).sum()

    return -total_weight + penalty

# 创建随机图和权重
G = nx.gnm_random_graph(200, 400)
weights = {e: torch.rand(1).item() for e in G.edges}
nx.set_edge_attributes(G, weights, "weight")

# 转换为GNN输入格式
edge_index = torch.tensor(list(G.edges)).t().contiguous()
edge_attr = torch.tensor([G[u][v]["weight"] for u, v in G.edges], dtype=torch.float)
node_features = torch.ones((G.number_of_nodes(), 1))  # 所有节点的特征设为1

# 将边权重转换为张量
weights_tensor = torch.tensor([weights[e] for e in G.edges], dtype=torch.float)

# 初始化模型
model = GNNModel(in_channels=1, out_channels=1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
lambda_ = 0.1  # 调整该参数以平衡权重和约束

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    edge_scores = model(node_features, edge_index).squeeze()
    loss = loss_fn(edge_scores, edge_index, weights_tensor, lambda_, node_features.size(0))
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# 预测边的选择
model.eval()
with torch.no_grad():
    edge_scores = model(node_features, edge_index).squeeze()
    selected_edges = edge_index[:, edge_scores.topk(20).indices].t().contiguous()
print("Selected edges:", selected_edges)
