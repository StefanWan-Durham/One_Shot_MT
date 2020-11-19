import torch
import torch.functional as F
import torch.nn as nn


# init weights
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


# G Net: GAN 788-->4096-->2048
class G_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(G_Net, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.lRelu = nn.LeakyReLU()

        self.apply(weight_init)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_normal(m.weight)
        #         nn.init.constant(m.bias,0)

    def forward(self, input):
        output = self.lRelu(self.fc1(input))
        output = self.relu(self.fc2(output))

        return output


# C Net: for quality classification  2048 --> 1024--> 1
class C_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(C_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lRelu = nn.LeakyReLU(0.2, True)
        self.sigmod = nn.Sigmoid()

        self.apply(weight_init)

    def forward(self, input):
        output = self.lRelu(self.fc1(input))
        output = self.sigmod(self.fc2(output))

        return output


# F Net: for test  2048-->1024-->40 / 2048--1024-->10
class F_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(F_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lRelu = nn.LeakyReLU(0.2, True)

        self.apply(weight_init)

    def forward(self, input):
        output = self.lRelu(self.fc1(input))
        output = self.fc2(output)

        return output  # 之后经过CrossEntropy分类（10类，或者40类）
