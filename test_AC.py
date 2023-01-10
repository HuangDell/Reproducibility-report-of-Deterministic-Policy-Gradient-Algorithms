import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
env=gym.make('MountainCar-v0').unwrapped
'''CartPole的环境状态特征量为推车的位置x、速度x_dot、杆子的角度theta、角速度theta_dot，状态是这四个状态特征所组成的，情况将是无限个，是连续的（即无限个状态），动作是推车向左-0，向右-1，（离散的，有限个，2个）'''
state_number=env.observation_space.shape[0]
action_number=env.action_space.n
LR_A = 0.005    # learning rate for actor
LR_C = 0.01     # learning rate for critic
Gamma = 0.9
Switch=0#训练、测试切换标志
'''AC第一部分 设计actor'''
'''第一步.设计actor和critic的网络部分'''
class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.in_to_y1=nn.Linear(state_number,50)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(50,20)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(20,action_number)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=torch.sigmoid(inputstate)
        act=self.out(inputstate)
        return F.softmax(act,dim=-1)
class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.in_to_y1=nn.Linear(state_number,40)
        self.in_to_y1.weight.data.normal_(0,0.1)
        self.y1_to_y2=nn.Linear(40,20)
        self.y1_to_y2.weight.data.normal_(0,0.1)
        self.out=nn.Linear(20,1)
        self.out.weight.data.normal_(0,0.1)
    def forward(self,inputstate):
        inputstate=self.in_to_y1(inputstate)
        inputstate=F.relu(inputstate)
        inputstate=self.y1_to_y2(inputstate)
        inputstate=torch.sigmoid(inputstate)
        act=self.out(inputstate)
        return act
class Actor():
    def __init__(self):
        self.actor=ActorNet()
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=LR_A)
    '''第二步.编写actor的选择动作函数'''
    def choose(self,inputstate):
        inputstate=torch.FloatTensor(inputstate)
        probs=self.actor(inputstate).detach().numpy()
        action=np.random.choice(np.arange(action_number),p=probs)
        return action
    '''第四步.根据td-error进行学习，编写公式log(p(s,a))*td_e的代码'''
    def learn(self,s,a,td):
        s = torch.FloatTensor(s)
        prob = self.actor(s)
        log_prob = torch.log(prob)
        actor_loss=-log_prob[a]*td
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
'''第二部分 Critic部分'''
class Critic():
    def __init__(self):
        self.critic=CriticNet()
        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=LR_C)
        self.lossfunc=nn.MSELoss()#均方误差（MSE）
    '''第三步.编写td-error的计算代码（V现实减去V估计就是td-error）'''
    def learn(self,s,r,s_):
        '''当前的状态s计算当前的价值，下一个状态s_计算出下一状态的价值v_，然后v_乘以衰减γ再加上r就是v现实'''
        s = torch.FloatTensor(s)
        v=self.critic(s)#输入当前状态，有网络得到估计v
        r=torch.FloatTensor([r])#.unsqueeze(0)#unsqueeze(0)在第一维度增加一个维度
        s_ = torch.FloatTensor(s_)
        reality_v=r+Gamma*self.critic(s_).detach()#现实v
        td_e=self.lossfunc(reality_v,v)
        self.optimizer.zero_grad()
        td_e.backward()
        self.optimizer.step()
        advantage=(reality_v-v).detach()
        return advantage

# 用于测试环境的入口
def test_AC(env_name: str):
    MAX_EPISODES = 300
    MAX_EP_STEPS = 200
    # env = gym.make(env_name)
    # s_dim = env.observation_space.shape[0]
    # a_dim = env.action_space.shape[0]
    start_time = time.time()  # 记录算法运行时间
    a = Actor()
    c = Critic()
    y = []
    for i in range(MAX_EPISODES):
        s = env.reset()[0]
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            action = a.choose(s)
            s_next, r, done, _, info = env.step(action)
            td_error = c.learn(s, r, s_next)  # gradient = grad[r + gamma * V(s_) - V(s)]
            a.learn(s, action, td_error)  # true_gradient = grad[logPi(s,a) * td_error]
            s = s_next
            if done:
                break
            ep_reward += r
            y.append(ep_reward)  # 添加reward，以便后续制图
        print(f'Episode:{i}, Reward: {ep_reward}')
    print('Running time: ', time.time() - start_time)
    return MAX_EPISODES*MAX_EP_STEPS,y,env_name

