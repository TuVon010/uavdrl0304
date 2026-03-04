import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # [新增] 必须导入 pandas
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
        
        # [新增] 创建数据文档保存目录
        self.data_dir = str(self.run_dir / 'data_logs')
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # ====================================
            # [统计初始化] 每个轮次开始前清零
            # ====================================
            ep_real_latency_sum = 0.0 # 真实时延和
            ep_user_saving_rate_sum = 0.0 # [NEW] 累计所有用户的节省率
            ep_time_cost_sum = 0.0    # 时间成本和 (加权)
            ep_real_energy_sum = 0.0  # 真实能耗 (User + UAV)
            ep_real_energy_sum = 0.0  # 真实能耗 (User + UAV)
            
            # [新增] 初始化分类能耗统计变量0210zja
            ep_user_energy_sum = 0.0        # 用户总能耗 (Sum of 10 users)
            ep_uav_fly_energy_sum = 0.0     # UAV 飞行总能耗 (Sum of 3 UAVs)
            ep_uav_comp_energy_sum = 0.0    # UAV 计算总能耗 (Sum of 3 UAVs)
            ep_uav_target_dist_sum = 0.0  # [NEW] 累计 UAV 到目标的距离
            ep_uav_min_dist_sum = 0.0     # [NEW] 累计 UAV 之间的最小距离
            # 轨迹记录: {agent_id: [[x,y], [x,y]...]}
            trajectories = {i: [] for i in range(self.num_agents)}

            # ====================================
            # [新增] 最后一轮的数据收集容器
            # ====================================
            is_last_episode = (episode == episodes - 1)
            last_ep_user_data = [] # 存储用户文档数据
            last_ep_uav_data = []  # 存储 UAV 文档数据

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
                # [数据收集] 从 infos 提取数据
                # ====================================
                # 注意：infos 是一个 tuple，长度为 n_rollout_threads (通常是1或更多)
                # 这里假设 n_rollout_threads=1，或者我们只统计第一个线程的数据用于打印/画图
                
                # infos shape: (n_threads, n_agents) -> dict
                current_info = infos[0] # 取第0个线程的数据
                
                step_latency_sum = 0
                step_cost_sum = 0
                step_energy_sum = 0
                # [新增] Step内的临时累加变量0210zja
                step_user_e = 0
                step_uav_fly = 0
                step_uav_comp = 0
                #0214zja
                step_user_saving_rate = 0
                step_uav_dist = 0
                step_uav_min = 0
                
                
                for i in range(self.num_agents):
                    agent_info = current_info[i]
                    
                    # 记录轨迹点
                    if 'pos' in agent_info:
                        trajectories[i].append(agent_info['pos'])
                    
                    if i < 10: # User
                        lat = agent_info.get('latency', 0)
                        eng = agent_info.get('energy', 0)
                        sav = agent_info.get('saving_rate', 0) # [NEW]
                        t_cost = agent_info.get('time_cost', 0)
                        
                        step_latency_sum += lat
                        step_energy_sum += eng
                        step_cost_sum += t_cost
                        # [新增] 累加用户能耗0210zja
                        step_user_e += eng
                        step_user_saving_rate += sav 
                        # [新增] 如果是最后一轮，收集用户详细数据
                        if is_last_episode:
                            assoc_id = agent_info.get('assoc_id', -1)
                            assoc_str = "Local" if assoc_id == -1 else f"UAV_{assoc_id}"
                            
                            last_ep_user_data.append({
                                "Step": step,
                                "User_ID": i,
                                "Association": assoc_str,
                                "Offload_Ratio": agent_info.get('offload_ratio', 0),
                                "Allocated_Freq_Hz": agent_info.get('alloc_freq', 0), # 算力分配
                                "Latency_s": lat,
                                "Saving_Rate": sav, # Save to CSV
                                "Energy_J": eng
                            })

                    else: # UAV
                    #########################0210zja
                        fly_eng = agent_info.get('fly_energy', 0)
                        comp_eng = agent_info.get('uav_comp_energy', 0) # 获取计算能耗
                        
                        step_energy_sum += (fly_eng + comp_eng) # 总能耗增加
                        dist_t = agent_info.get('uav_dist_to_target', 0) # [NEW]
                        min_d = agent_info.get('uav_min_dist', 0)        # [NEW]
                        # [新增] 分别累加 UAV 能耗
                        step_uav_fly += fly_eng
                        step_uav_comp += comp_eng
                        step_uav_dist += dist_t
                        step_uav_min += min_d
                        
                        # [新增] 如果是最后一轮，收集 UAV 详细数据
                        if is_last_episode:
                            uav_id = i - 10
                            pos = agent_info.get('pos', [0,0])
                            connected_users = agent_info.get('connected_users', [])
                            
                            last_ep_uav_data.append({
                                "Step": step,
                                "UAV_ID": uav_id,
                                "Pos_X": pos[0],
                                "Pos_Y": pos[1],
                                "Fly_Energy_J": fly_eng,
                                "Comp_Energy_J": comp_eng,
                                "Total_Energy_J": fly_eng + comp_eng,
                                "Dist_to_Target": dist_t,
                                 "Min_UAV_Dist": min_d,
                                "Connected_Users": str(connected_users) # 转字符串方便保存
                            })
                
                ep_real_latency_sum += step_latency_sum
                ep_time_cost_sum += step_cost_sum
                ep_real_energy_sum += step_energy_sum
                # 累加每个 step 的平均值到 episode 总和
                # 比如：这个 slot 所有用户的平均 saving rate
                ep_user_saving_rate_sum += (step_user_saving_rate / 10.0) 
                # 这个 slot 所有 UAV 的平均距离
                ep_uav_target_dist_sum += (step_uav_dist / 3.0)
                ep_uav_min_dist_sum += (step_uav_min / 3.0)
                # [新增] 累加到 Episode 总和
                ep_user_energy_sum += step_user_e
                ep_uav_fly_energy_sum += step_uav_fly
                ep_uav_comp_energy_sum += step_uav_comp
########################################################################
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
            # [打印统计] 每个轮次结束打印0210zja
            # ====================================
            # 平均值 = 总和 / 时隙数 (得到每个时隙的平均值)
            avg_time_cost = ep_time_cost_sum / self.episode_length
            avg_real_latency = ep_real_latency_sum / self.episode_length
            avg_real_energy = ep_real_energy_sum / self.episode_length
            
            # [新增] 计算分类能耗的平均值
            avg_user_e = ep_user_energy_sum / self.episode_length
            avg_uav_fly = ep_uav_fly_energy_sum / self.episode_length
            avg_uav_comp = ep_uav_comp_energy_sum / self.episode_length
            # 计算每一步(per step)的平均值
            avg_saving = ep_user_saving_rate_sum / self.episode_length
            avg_uav_target = ep_uav_target_dist_sum / self.episode_length
            avg_uav_safe = ep_uav_min_dist_sum / self.episode_length
            
            # 最后一轮生成文档
            if is_last_episode:
                print(f"\n>>> Saving Last Episode Data (Episode {episode}) ...")
                
                # 1. 保存用户数据
                user_df = pd.DataFrame(last_ep_user_data)
                user_csv_path = os.path.join(self.data_dir, 'last_ep_user_metrics.csv')
                user_df.to_csv(user_csv_path, index=False)
                print(f"  [User Doc] Saved to {user_csv_path}")
                
                # 2. 保存 UAV 数据
                uav_df = pd.DataFrame(last_ep_uav_data)
                uav_csv_path = os.path.join(self.data_dir, 'last_ep_uav_metrics.csv')
                uav_df.to_csv(uav_csv_path, index=False)
                print(f"  [UAV Doc] Saved to {uav_csv_path}")

            # 画图与 Log
            self.plot_trajectories(trajectories, episode)

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
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

                # ==========================================================
                # [新增代码] 分组打印 User 和 UAV 的奖励
                # ==========================================================
                user_rewards = []
                uav_rewards = []

                for agent_id in range(self.num_agents):
                    # 1. 计算该智能体的本回合平均奖励
                    avg_rew = np.mean(self.buffer[agent_id].rewards) * self.episode_length
                    train_infos[agent_id].update({"average_episode_rewards": avg_rew})
                    
                    # 2. 根据 ID 分组 (前10个是User，后3个是UAV)
                    if agent_id < 10:
                        user_rewards.append(avg_rew)
                    else:
                        uav_rewards.append(avg_rew)

                # 3. 计算组内统计值
                avg_user_rew = np.mean(user_rewards) if user_rewards else 0
                avg_uav_rew = np.mean(uav_rewards) if uav_rewards else 0
                
                # 4. 打印到控制台 (Console)
                print(f"----------------------------------------------------------------")
                print(f" [User] Mean: {avg_user_rew:.4f} | Min: {np.min(user_rewards):.4f} | Max: {np.max(user_rewards):.4f}")
                print(f" [UAV ] Mean: {avg_uav_rew:.4f}  | Min: {np.min(uav_rewards):.4f} | Max: {np.max(uav_rewards):.4f}")
                print(f"----------------------------------------------------------------")
                print(f"\n[Episode {episode}] Stats:")
                print(f"  > Avg Time Cost Sum (Weighted): {avg_time_cost:.4f}")
                print(f"  > Avg Real Latency Sum (s):     {avg_real_latency:.4f}")
                print(f"  > Avg Real Energy Sum (J):      {avg_real_energy:.4f}")
                print(f"\n[Episode {episode}] Stats (Avg per Step):")
                # [修改] 打印细分的能耗
                print(f"  > User Energy Sum:       {avg_user_e:.4f} J")
                print(f"  > UAV Fly Energy Sum:    {avg_uav_fly:.4f} J")
                print(f"  > UAV Comp Energy Sum:   {avg_uav_comp:.4f} J")
                #0214添加
                print(f"  > User Saving Rate:      {avg_saving:.4f} (Target: > 0.0)")
                print(f"  > UAV Dist to Target:    {avg_uav_target:.2f} m")
                print(f"  > UAV Min Separation:    {avg_uav_safe:.2f} m (Safe: > 50.0)")
                print(f"  > System Energy (U+M):   {avg_user_e + avg_uav_fly+avg_uav_comp:.2f} J")
                
                # === 修复结束 ===
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def plot_trajectorieso(self, trajectories, episode):
        """画出 UAV 和 User 的轨迹"""
        plt.figure(figsize=(6, 6))
        plt.xlim(0, 500)
        plt.ylim(0, 500)
        
        # 颜色映射
        colors = plt.cm.get_cmap('tab10', self.num_agents)
        
        # 画 Users (0-9)
        for i in range(10):
            traj = np.array(trajectories[i])
            if len(traj) > 0:
                # User 移动较慢，画成虚线或点
                plt.plot(traj[:, 0], traj[:, 1], linestyle=':', alpha=0.5, color='gray')
                plt.scatter(traj[-1, 0], traj[-1, 1], s=20, marker='o', label=f'U{i}' if i==0 else None, color='blue', alpha=0.5)

        # 画 UAVs (10-12)
        for i in range(10, self.num_agents):
            traj = np.array(trajectories[i])
            if len(traj) > 0:
                # UAV 画实线，带箭头或明显标记
                plt.plot(traj[:, 0], traj[:, 1], linewidth=2, label=f'UAV{i-10}')
                plt.scatter(traj[0, 0], traj[0, 1], s=50, marker='x', color='black') # 起点
                plt.scatter(traj[-1, 0], traj[-1, 1], s=50, marker='*', color='red') # 终点

        plt.title(f'Episode {episode} Trajectories')
        plt.legend(loc='upper right', fontsize='small')
        plt.grid(True)
        
        # 保存图片
        save_path = os.path.join(self.plot_dir, f'traj_ep_{episode}.png')
        plt.savefig(save_path)
        plt.close() # 关闭画布，防止内存溢出

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
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)  # shape = [env_num, agent_num * obs_dim]

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
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
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
                # TODO 这里改造成自己环境需要的形式即可
                # TODO Here, you can change the action_env to the form you need
                action_env = action
                # raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
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
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[
                            eval_action[:, i]
                        ]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({"eval_average_episode_rewards": eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
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
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs:
            imageio.mimsave(
                str(self.gif_dir) + "/render.gif",
                all_frames,
                duration=self.all_args.ifi,
            )
