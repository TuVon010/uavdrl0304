import time
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain
import torch

from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        
        # 创建轨迹保存目录
        self.plot_dir = str(self.run_dir / 'trajectories')
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # ====================================
            # [统计初始化]
            # ====================================
            ep_real_latency_sum = 0.0 
            ep_time_cost_sum = 0.0    
            ep_real_energy_sum = 0.0  
            
            # 轨迹记录: {agent_id: [[x,y], [x,y]...]}
            trajectories = {i: [] for i in range(self.num_agents)}

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,
                ) = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                
                # ====================================
                # [数据收集]
                # ====================================
                current_info = infos[0] # 取第0个线程的数据
                
                step_latency_sum = 0
                step_cost_sum = 0
                step_energy_sum = 0
                
                for i in range(self.num_agents):
                    agent_info = current_info[i]
                    
                    # 记录轨迹点
                    if 'pos' in agent_info:
                        trajectories[i].append(agent_info['pos'])
                    
                    if i < 10: # User
                        lat = agent_info.get('latency', 0)
                        eng = agent_info.get('energy', 0)
                        t_cost = agent_info.get('time_cost', 0)
                        
                        step_latency_sum += lat
                        step_energy_sum += eng
                        step_cost_sum += t_cost
                    else: # UAV
                        eng = agent_info.get('fly_energy', 0)
                        step_energy_sum += eng
                
                ep_real_latency_sum += step_latency_sum
                ep_time_cost_sum += step_cost_sum
                ep_real_energy_sum += step_energy_sum

                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # ====================================
            # [打印统计]
            # ====================================
            avg_time_cost = ep_time_cost_sum / self.episode_length
            avg_real_latency = ep_real_latency_sum / self.episode_length
            avg_real_energy = ep_real_energy_sum / self.episode_length
            
            print(f"\n[Episode {episode}] Stats:")
            print(f"  > Avg Time Cost Sum (Weighted): {avg_time_cost:.4f}")
            print(f"  > Avg Real Latency Sum (s):     {avg_real_latency:.4f}")
            print(f"  > Avg Real Energy Sum (J):      {avg_real_energy:.4f}")

            # ====================================
            # [画图] 生成带打点的轨迹图
            # ====================================
            self.plot_trajectories(trajectories, episode)

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    " Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def plot_trajectories(self, trajectories, episode):
        """画出 UAV 和 User 的轨迹 (点线结合)"""
        plt.figure(figsize=(8, 8))
        plt.xlim(0, 500)
        plt.ylim(0, 500)
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        
        # 颜色映射
        cmap = plt.get_cmap('tab10')
        
        # --- 1. 画 Users (0-9) ---
        # 为了不让图太乱，User 只画终点或淡色的轨迹
        for i in range(10):
            traj = np.array(trajectories[i])
            if len(traj) > 0:
                # 虚线轨迹
                plt.plot(traj[:, 0], traj[:, 1], linestyle=':', alpha=0.3, color='gray', linewidth=1)
                # 每一个时隙的点 (小灰点)
                plt.scatter(traj[:, 0], traj[:, 1], s=5, marker='.', color='gray', alpha=0.3)
                # 终点 (蓝色圆点)
                plt.scatter(traj[-1, 0], traj[-1, 1], s=30, marker='o', label='Users' if i==0 else None, color='blue', alpha=0.6)

        # --- 2. 画 UAVs (10-12) ---
        for i in range(10, self.num_agents):
            traj = np.array(trajectories[i])
            if len(traj) > 0:
                uav_id = i - 10
                color = cmap(uav_id) # 每个UAV不同颜色
                
                # 实线轨迹
                plt.plot(traj[:, 0], traj[:, 1], linewidth=1.5, label=f'UAV {uav_id}', color=color, alpha=0.8)
                
                # [关键] 每一个时隙的位置打点
                plt.scatter(traj[:, 0], traj[:, 1], s=25, marker='o', color=color, alpha=0.8)
                
                # 起点 (黑色叉叉)
                plt.scatter(traj[0, 0], traj[0, 1], s=80, marker='x', color='black', zorder=10) 
                # 终点 (红色五角星)
                plt.scatter(traj[-1, 0], traj[-1, 1], s=100, marker='*', color='red', zorder=10) 

        plt.title(f'Episode {episode} Trajectories (Points = Time Steps)')
        plt.legend(loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 保存图片
        save_path = os.path.join(self.plot_dir, f'traj_ep_{episode}.png')
        plt.savefig(save_path, dpi=100)
        plt.close()

    def warmup(self):
        obs = self.envs.reset() 
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            
            values.append(_t2n(value))
            action = _t2n(action)
            
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                action_env = action

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        pass

    @torch.no_grad()
    def render(self):
        pass