import torch.nn as nn
import torch

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()

        self.model = nn.Sequential(
            nn.Linear(28*28, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.Softmax(dim=-1),
            # nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        # Define the forward pass of your model

        out = self.model(x)
        #out = self.linear(x)
        return out








