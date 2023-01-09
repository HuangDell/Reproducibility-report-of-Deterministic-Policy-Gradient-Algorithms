import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = torch.FloatTensor(action_bound)
        self.layer_1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer_1.weight, 0., 0.3)
        nn.init.constant_(self.layer_1.bias, 0.1)
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0., 0.3)
        self.output.bias.data.fill_(0.1)

    # 输入是state，输出的是一个确定性的action
    def forward(self, s):
        a = torch.relu(self.layer_1(s))
        a = torch.tanh(self.output(a))
        # 对action范围进行映射
        scaled_a = a * self.action_bound
        return scaled_a


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 30
        # layer
        self.layer_1 = nn.Linear(state_dim, n_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(action_dim, n_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.output = nn.Linear(n_layer, 1)

    # Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
    def forward(self, s, a):
        s = self.layer_1(s)
        a = self.layer_2(a)
        q_val = self.output(torch.relu(s + a))
        return q_val
