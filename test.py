import gymnasium as gym
import time
import numpy as np

from plot import show
from DPG import DPG


# 用于测试不同的环境
def test(env_name: str):
    # hyper parameters
    VAR = 3  # control exploration
    MAX_EPISODES = 300
    MAX_EP_STEPS = 200
    MEMORY_CAPACITY = 10000  # 记忆空间大小
    REPLACEMENT = [
        dict(name='soft', tau=0.005),
        dict(name='hard', rep_iter=600)
    ][0]  # you can try different target replacement strategies
    env = gym.make(env_name)
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_bound = env.action_space.high  # 限定动作输出的范围
    dpg = DPG(state_dim=s_dim,
              action_dim=a_dim,
              action_bound=a_bound,
              replacement=REPLACEMENT,
              memory_capacity=MEMORY_CAPACITY)

    start_time = time.time()  # 记录算法运行时间
    y = []
    for i in range(MAX_EPISODES):
        s = env.reset()[0]
        ep_reward = 0
        for j in range(MAX_EP_STEPS):
            a = dpg.choose_action(s)
            a = np.clip(np.random.normal(a, VAR), -2, 2)  # 在动作选择上添加随机噪声
            s_next, r, done, _, info = env.step(a)
            dpg.store_transition(s, a, r / 10, s_next)  # 保存记忆
            if dpg.pointer > MEMORY_CAPACITY:
                VAR *= .9995  # decay the action randomness
                dpg.learn()
            s = s_next
            ep_reward += r
        print(f'Episode:{i}, Reward: {ep_reward}')
        y.append(ep_reward)  # 添加reward，以便后续制图
    print('Running time: ', time.time() - start_time)
    show(range(MAX_EPISODES), y, env_name)
