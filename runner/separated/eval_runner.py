import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from runner.separated.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class EvalRunner(Runner):
    def __init__(self, config):
        super(EvalRunner, self).__init__(config)
        # 测试结果保存路径
        self.save_dir = str(self.run_dir / 'eval_results')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # ===================================================
        # [核心] 动态获取环境参数
        # ===================================================
        env_core = self.envs.envs[0].env 
        self.n_users = env_core.n_users  # 10
        self.n_uavs = env_core.n_uavs    # 3
        
        print(f"[EvalRunner] Detected: n_users={self.n_users}, n_uavs={self.n_uavs}")

    def run_eval(self):
        print(f"Loading model from {self.model_dir} ...")
        self.restore() 
        
        obs = self.envs.reset()
        rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # === 数据记录容器 ===
        history = {
            'step': [],
            'uav_pos': [],          # [step][uav_id] -> [x, y]
            'uav_energy': [],       # [step][uav_id] -> energy
            'user_pos': [],         # [step][user_id] -> [x, y] (新增)
            'sys_cost': [],         
            'sys_latency': [],      
            'user_details_list': [] # 用于保存所有详细记录导出CSV
        }

        print(f"\n=== 开始测试 (Episode Length: {self.episode_length}) ===")
        
        for step in range(self.episode_length):
            # --- A. 决策 ---
            temp_actions = []
            with torch.no_grad():
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True
                    )
                    action = _t2n(action)
                    temp_actions.append(action)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

            actions_env = [ [] for _ in range(self.n_rollout_threads) ]
            for agent_id in range(self.num_agents):
                actions_env[0].append(temp_actions[agent_id][0])

            # --- B. 环境步进 ---
            next_obs, rewards, dones, infos = self.envs.step(actions_env)
            
            # --- C. 数据提取 ---
            current_info = infos[0] 
            
            # 1. UAV 数据
            step_uav_pos = []
            step_uav_energy = []
            for i in range(self.n_users, self.num_agents):
                u_info = current_info[i]
                total_e = u_info.get('fly_energy', 0) + u_info.get('uav_comp_energy', 0)
                pos = u_info.get('pos', np.array([0,0]))
                step_uav_energy.append(total_e)
                step_uav_pos.append(pos)
            
            # 2. User 数据
            step_latency_sum = 0
            step_cost_sum = 0
            step_user_pos = []
            
            user_actions = [temp_actions[i][0] for i in range(self.n_users)]
            
            for i in range(self.n_users):
                u_info = current_info[i]
                lat = u_info.get('latency', 0)
                cost = u_info.get('time_cost', 0)
                pos = u_info.get('pos', np.array([0,0])) # 获取位置
                
                step_latency_sum += lat
                step_cost_sum += cost
                step_user_pos.append(pos)
                
                # 动作解析
                raw_act = user_actions[i]
                assoc_idx = int(np.round(raw_act[0]))
                assoc_idx = np.clip(assoc_idx, 0, self.n_uavs) 
                ratio = np.clip(raw_act[1], 0.0, 1.0)
                
                status_code = -1
                status_str = "Unknown"
                if assoc_idx == 0 or ratio < 0.01:
                    status_str = "Local"
                    status_code = -1
                    ratio = 0.0
                else:
                    status_str = f"UAV_{assoc_idx-1}"
                    status_code = assoc_idx-1
                
                # 添加到详细记录列表 (用于导出 CSV)
                history['user_details_list'].append({
                    "Step": step,
                    "User_ID": i,
                    "Pos_X": pos[0],
                    "Pos_Y": pos[1],
                    "Association_Str": status_str,
                    "Association_ID": status_code, # -1: Local, 0~2: UAV
                    "Offload_Ratio": ratio,
                    "Latency": lat,
                    "Weighted_Cost": cost
                })

            # --- D. 存入历史 ---
            history['step'].append(step)
            history['uav_pos'].append(step_uav_pos)
            history['uav_energy'].append(step_uav_energy)
            history['user_pos'].append(step_user_pos)
            history['sys_cost'].append(step_cost_sum)
            history['sys_latency'].append(step_latency_sum)
            
            obs = next_obs

        # --- E. 结果处理 ---
        self.process_results(history)

    def process_results(self, history):
        steps = history['step']
        uav_pos_data = np.array(history['uav_pos'])   # (Steps, n_uavs, 2)
        user_pos_data = np.array(history['user_pos']) # (Steps, n_users, 2)
        uav_eng_data = np.array(history['uav_energy'])

        # ==========================
        # 1. 导出 CSV 文档
        # ==========================
        print("\n" + "="*60)
        print(">>> 正在导出数据文档 (CSV) ...")
        
        # (1) User 详细数据文档
        user_df = pd.DataFrame(history['user_details_list'])
        user_csv_path = os.path.join(self.save_dir, 'user_metrics_detailed.csv')
        user_df.to_csv(user_csv_path, index=False)
        print(f"  [1] 用户指标已保存: {user_csv_path}")

        # (2) UAV 轨迹与能耗文档
        uav_records = []
        for t in steps:
            row = {"Step": t}
            for u in range(self.n_uavs):
                row[f"UAV_{u}_X"] = uav_pos_data[t, u, 0]
                row[f"UAV_{u}_Y"] = uav_pos_data[t, u, 1]
                row[f"UAV_{u}_Energy"] = uav_eng_data[t, u]
            uav_records.append(row)
        
        uav_df = pd.DataFrame(uav_records)
        uav_csv_path = os.path.join(self.save_dir, 'uav_traces.csv')
        uav_df.to_csv(uav_csv_path, index=False)
        print(f"  [2] 无人机数据已保存: {uav_csv_path}")

        # ==========================
        # 2. 打印部分概览到控制台
        # ==========================
        print("="*60)
        print(">>> 最后时刻 (Step 100) 用户状态概览:")
        last_step_df = user_df[user_df['Step'] == (self.episode_length - 1)]
        print(last_step_df[['User_ID', 'Association_Str', 'Offload_Ratio', 'Latency']].to_string(index=False))

        # ==========================
        # 3. 绘图 (包含 User 轨迹)
        # ==========================
        plt.style.use('default') 
        
        # Fig 1: 综合轨迹图
        plt.figure(figsize=(10, 10))
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_uavs))
        
        # A. 画用户 (灰色虚线 + 小点)
        for i in range(self.n_users):
            # 获取第 i 个用户的所有位置
            u_traj = user_pos_data[:, i, :] 
            plt.plot(u_traj[:, 0], u_traj[:, 1], color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            # 画终点
            plt.scatter(u_traj[-1, 0], u_traj[-1, 1], s=20, marker='o', color='gray', alpha=0.6)
            if i == 0: # 只给第一个用户加标签，防止图例太乱
                plt.plot([], [], color='gray', linestyle=':', label='Users Trajectory')

        # B. 画 UAV (彩色实线 + 标记)
        for i in range(self.n_uavs):
            traj = uav_pos_data[:, i, :]
            plt.plot(traj[:, 0], traj[:, 1], label=f'UAV {i}', color=colors[i], linewidth=2.5, alpha=0.9)
            # 起点
            plt.scatter(traj[0, 0], traj[0, 1], marker='s', s=80, color=colors[i], edgecolors='k', zorder=5)
            # 终点
            plt.scatter(traj[-1, 0], traj[-1, 1], marker='*', s=150, color=colors[i], edgecolors='k', zorder=5)
            
        plt.xlim(0, 500)
        plt.ylim(0, 500)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Trajectories (Episode Len: {self.episode_length})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.save_dir, 'trajectories_full.png'), dpi=150)
        plt.close()

        # Fig 2: 系统指标 & UAV 能耗 (可选，这里保留)
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Time Slot')
        ax1.set_ylabel('Avg Latency (s)', color='red')
        # 计算平均时延
        avg_lat = np.array(history['sys_latency']) / self.n_users
        ax1.plot(steps, avg_lat, color='red', label='Avg Latency')
        ax1.tick_params(axis='y', labelcolor='red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('UAV Total Energy (J)', color='blue')
        # 计算 UAV 总能耗
        total_uav_e = np.sum(uav_eng_data, axis=1)
        ax2.plot(steps, total_uav_e, color='blue', linestyle='--', label='Total UAV Energy')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        plt.title('System Performance')
        plt.savefig(os.path.join(self.save_dir, 'system_performance.png'))
        plt.close()

        print(f"\n[完成] 结果已保存在: {self.save_dir}")