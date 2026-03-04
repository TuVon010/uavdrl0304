import gym
from gym import spaces
import numpy as np
from envs.env_core1 import EnvCore

class ContinuousActionEnv(object):
    """
    Padding 版环境包装器
    所有智能体对外暴露相同的维度，解决 Runner 报错问题。
    """
    def __init__(self):
        self.env = EnvCore()
        self.num_agent = self.env.agent_num
        
        # 直接使用 env_core 计算好的统一最大维度
        self.obs_dim = self.env.obs_dim       # 63
        self.act_dim = self.env.action_dim    # 12

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        # Critic 输入全局观测 (所有智能体的 obs 拼接)
        share_obs_dim = self.num_agent * self.obs_dim

        for agent_idx in range(self.num_agent):
            # 所有智能体全部统一定义
            
            # Action Space (Box)
            self.action_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(self.act_dim,), dtype=np.float32
            ))

            # Observation Space (Box)
            self.observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(self.obs_dim,), dtype=np.float32
            ))

            # Shared Obs (Critic)
            self.share_observation_space.append(spaces.Box(
                low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32
            ))

    def step(self, actions):
        results = self.env.step(actions)
        obs, rews, dones, infos = results
        # 确保 reward 是 (n_agents, 1) 形状
        rews = np.array(rews).reshape(-1, 1)
        return obs, rews, dones, infos

    def reset(self):
        return self.env.reset()

    def close(self):
        pass

    def seed(self, seed):
        np.random.seed(seed)