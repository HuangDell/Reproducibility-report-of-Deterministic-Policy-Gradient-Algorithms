import gymnasium as gym
import time
import numpy as np
import torch

from plot import show
from DPG import DDPG

def test(env_name:str):
    # hyper parameters
    VAR = 3  # control exploration
    MAX_EPISODES = 300
    MAX_EP_STEPS = 200
    MEMORY_CAPACITY = 10000
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies
    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 尝试使用GPU
    ddpg = DDPG(state_dim=s_dim,
                action_dim=a_dim,
                action_bound=a_bound,
                replacement=REPLACEMENT,
                #device=device,
                memory_capacity=MEMORY_CAPACITY)

    t1 = time.time()
    y = []
    for i in range(MAX_EPISODES):
        s = env.reset()[0]
        ep_reward = 0
        for j in range(MAX_EP_STEPS):

            # Add exploration noise
            a = ddpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -2, 2)  # 在动作选择上添加随机噪声

            s_, r, done, _, _ = env.step(a)

            ddpg.store_transition(s, a, r / 10, s_)

            if ddpg.pointer > MEMORY_CAPACITY:
                VAR *= .9995  # decay the action randomness
                ddpg.learn()

            s = s_
            ep_reward += r
            if j == MAX_EP_STEPS - 1:
                print(f'Episode:{i}, Reward: {ep_reward}, Explore: {VAR}')
                y.append(ep_reward)
                break
    print('Running time: ', time.time() - t1)
    show(range(MAX_EPISODES), y, env_name)
