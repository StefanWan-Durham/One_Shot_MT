import torch
import torch.nn as nn
import torch.nn.functional as F

class G__Net(nn.Module):
    def __init__(self,pretrain_weight,pretrain_bias,input_dim,hl_num,out_dim):
        super(G__Net, self).__init__()
        self.pretrain_weight = pretrain_weight
        self.pretrain_bias = pretrain_bias

        self.fc1 = nn.Linear(input_dim, hl_num)  # G: fc1 778-->4096
        self.fc2 = nn.Linear(hl_num,out_dim)    # G: fc2  4096-->2048
        self.fc3 = nn.Linear(input_dim,1)       # Q: fc3  2048-->40

    def forward(self,x,y):
        x1 = self.fc1(x)  # G
        x1 = self.fc2(x)  # G
        x1 = x1*self.pretrain_weight + self.pretrain_bias  # G-->T:2048-->40 ? 有其他写法没
        x1 = F.softmax(x1,dim=1)    # Q Labels: 0,1
        x2 = self.fc3(x)
        return x1,x2


