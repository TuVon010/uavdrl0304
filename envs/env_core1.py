import numpy as np
from envs.Base import Base
from envs.physics_engine import PhysicsEngine
#0215奖励函数修改回线性模型
#0304给“空载”的无人机加上动态的怠工惩罚,修改半径
class EnvCore(object):
    """
    异构多智能体环境核心 (Padding Version - Fixed)
    修正:
    1. 加入无人机计算能耗 (E_comp)
    2. 统一使用 w_energy 权重
    """
    def __init__(self):
        self.base = Base()
        self.engine = PhysicsEngine(self.base)
        
        self.n_uavs = self.base.n_uavs # 3
        self.n_users = self.base.n_users # 10
        self.agent_num = self.n_uavs + self.n_users # 13
        
        # === 1. 确定最大维度 (用于对齐) ===
        # User: Pos(2)+Size(1)+Cyc(1)+Prio(1) + UAV_Pos(6)+Loads(3) = 14
        self.real_user_obs_dim = 14
        
        # UAV: Self(3) + 10 * User_Info(6) = 63
        self.real_uav_obs_dim = 3 + self.n_users * 6 
        
        self.real_user_action_dim = 2
        self.real_uav_action_dim = 2 + self.n_users # 12

        # 统一维度 (Padding Target)
        self.obs_dim = max(self.real_user_obs_dim, self.real_uav_obs_dim)       # 63
        self.action_dim = max(self.real_user_action_dim, self.real_uav_action_dim) # 12

        # 接口暴露
        self.user_obs_dim = self.obs_dim
        self.uav_obs_dim = self.obs_dim
        self.user_action_dim = self.action_dim
        self.uav_action_dim = self.action_dim

    def reset(self):
        # 地图中心
        center_x, center_y = 250.0, 250.0
        
        # ==========================================
        # 1. 初始化 UAV (三角形顶点分布)
        # ==========================================
        self.uavs = []
        # 定义三角形半径 (UAV离中心的距离)
        radius = 120.0 
        # 定义三个角度: 90度(正上方), 210度(左下), 330度(右下)
        angles = [np.pi/2, 7*np.pi/6, 11*np.pi/6]
        
        for i in range(self.n_uavs):
            # 计算顶点坐标
            uav_x = center_x + radius * np.cos(angles[i % 3])
            uav_y = center_y + radius * np.sin(angles[i % 3])
            uav_x = np.clip(uav_x, 0, 500)
            uav_y = np.clip(uav_y, 0, 500)
            self.uavs.append({'id': i, 'pos': np.array([uav_x, uav_y]), 'energy': 1e5})
            
        # ==========================================
        # 2. 初始化 User (中心密集聚集 + GaussMarkov)
        # ==========================================
        self.users = []
        for i in range(self.n_users):
            # 使用高斯分布生成位置 (密集聚集)
            # loc=中心, scale=标准差(聚集程度, 越小越密集)
            user_x = np.random.normal(loc=center_x, scale=20.0) 
            user_y = np.random.normal(loc=center_y, scale=20.0)
            
            # 必须 Clip 防止由于概率极低事件飞出地图
            user_x = np.clip(user_x, 0, 500)
            user_y = np.clip(user_y, 0, 500)
            pos = np.array([user_x, user_y])
            
            self.users.append({
                'id': i,
                'position': pos,
                # 初始速度和方向随机，后续由 physics_engine 的 update_user_positions 接管 (GaussMarkov)
                'velocity': np.random.uniform(0, 0.5), 
                'direction': np.random.uniform(0, 2*np.pi),
                'trajectory': [pos],
                'task_size': np.random.uniform(self.base.task_size_min, self.base.task_size_max),
                'cycles': np.random.uniform(self.base.cycles_min, self.base.cycles_max),
                'priority': np.random.choice([0, 1], p=[0.8, 0.2])
            })
            
        self.steps = 0
        return self._get_obs(prev_loads=[0]*self.n_uavs)

    def step(self, actions):
        self.steps += 1
        
        user_actions_raw = actions[:self.n_users]
        uav_actions_raw = actions[self.n_users:]
        
        # --- 动作解析 ---
        associations = {} 
        offloading_ratios = {}
        uav_loads = [0] * self.n_uavs
        
        # --- 1. 动作解析 ---
        for u_idx, act in enumerate(user_actions_raw):
            assoc_idx = int(np.round(act[0]))
            assoc_idx = np.clip(assoc_idx, 0, self.n_uavs) 
            ratio = np.clip(act[1], 0.0, 1.0)
            
            if assoc_idx == 0 or ratio < 0.01:
                associations[u_idx] = -1 
                offloading_ratios[u_idx] = 0.0
            else:
                target_uav = assoc_idx - 1
                associations[u_idx] = target_uav
                offloading_ratios[u_idx] = ratio
                uav_loads[target_uav] += 1

        # --- 物理演化 ---
        self.engine.update_user_positions(self.users)
        
        uav_fly_energies = []
        uav_out_of_bound = [False] * self.n_uavs # [新增] 记录是否出界

        for m_idx, act in enumerate(uav_actions_raw):
            vel = np.array([act[0], act[1]])
            v_mag = np.linalg.norm(vel)
            if v_mag > self.base.uav_v_max:
                vel = vel / v_mag * self.base.uav_v_max
                v_mag = self.base.uav_v_max
            
            # 更新位置
            new_pos = self.uavs[m_idx]['pos'] + vel * self.base.time_step
            
            # [新增] 检查是否撞墙 (给一个额外惩罚)
            if (new_pos[0] < 0 or new_pos[0] > 500 or 
                new_pos[1] < 0 or new_pos[1] > 500):
                uav_out_of_bound[m_idx] = True

            self.uavs[m_idx]['pos'] = np.clip(new_pos, 0, 550)
            
            e_fly = self.engine.compute_uav_energy(v_mag)
            uav_fly_energies.append(e_fly)

        # --- 计算指标 ---
        user_latencies = np.zeros(self.n_users)
        user_energies = np.zeros(self.n_users)
        uav_comp_energies = np.zeros(self.n_uavs)
        
        # [新增] 用于记录分配给每个用户的频率，用于存入 info
        user_allocated_freq = np.zeros(self.n_users) 
        # [新增] 用于记录每个 UAV 连接的用户 ID 列表
        uav_connected_users_list = [[] for _ in range(self.n_uavs)]
        
        # [核心新增] 计算基准时延 (Baseline)
        t_local_list = np.zeros(self.n_users)
        t_ideal_list = np.zeros(self.n_users)

        for u_id in range(self.n_users):
            user = self.users[u_id]
            # 基准 1: 完全本地计算的时间
            t_loc_base = (user['task_size'] * user['cycles']) / self.base.C_local
            t_local_list[u_id] = t_loc_base
            
            # 基准 2: 理想全卸载时间 (不考虑传输，独占 UAV 算力)
            t_ideal_base = (user['task_size'] * user['cycles']) / self.base.C_uav
            t_ideal_list[u_id] = t_ideal_base

        # 计算真实时延与能耗
        for m in range(self.n_uavs):
            connected_users = [u for u in associations if associations[u] == m]
            
            # [新增] 记录连接列表
            uav_connected_users_list[m] = connected_users 

            if not connected_users: continue
            
            # 资源分配权重
            raw_weights = uav_actions_raw[m][2:] 
            valid_weights = []
            for u_id in connected_users:
                w_idx = u_id if u_id < len(raw_weights) else 0
                valid_weights.append(np.exp(raw_weights[w_idx]))
            
            sum_w = sum(valid_weights) + 1e-9
            alloc_ratios = [w / sum_w for w in valid_weights]
            bw_per_user = self.base.B_total / len(connected_users)
            
            for i, u_id in enumerate(connected_users):
                user = self.users[u_id]
                lamb = offloading_ratios[u_id]
                f_uav = alloc_ratios[i] * self.base.C_uav # 分配到的频率
                
                # [新增] 记录分配的频率
                user_allocated_freq[u_id] = f_uav

                # ... (原有的能耗和时延计算逻辑保持不变) ...
                cycles_total = lamb * user['task_size'] * user['cycles']
                e_comp_uav = self.base.xi_m  * cycles_total
                uav_comp_energies[m] += e_comp_uav
                
                # 信道与时延计算
                g = self.engine.get_channel_gain(user['position'], self.uavs[m]['pos'])
                rate = self.engine.compute_rate(g, bw_per_user, self.base.p_tx_max)
                t_tx = (lamb * user['task_size']) / (rate + 1e-9)
                t_exe = cycles_total / (f_uav + 1e-9)
                t_offload = t_tx + t_exe
                
                # 本地
                t_loc = ((1-lamb) * user['task_size'] * user['cycles']) / self.base.C_local
                e_loc = self.base.k_local * (self.base.C_local**2) * (1-lamb) * user['task_size'] * user['cycles']
                e_tx = self.base.p_tx_max * t_tx
                
                user_latencies[u_id] = max(t_loc, t_offload)
                user_energies[u_id] = e_loc + e_tx # 用户侧能耗 (本地计算+传输)

        # 未关联用户
        for u_id in range(self.n_users):
            if associations[u_id] == -1:
                user = self.users[u_id]
                # 本地计算时的实际时延就是本地基准时延
                user_latencies[u_id] = t_local_list[u_id]
                e_loc = self.base.k_local * (self.base.C_local**2) * user['task_size'] * user['cycles']
                user_energies[u_id] = e_loc

        # ==========================================
        # 4. 奖励计算 (基于时间成本节省率)
        # ==========================================
        rewards = []
        
        # 计算所有用户的节省率 (Saving Rate)
        saving_rates = []
        for u in range(self.n_users):
            numerator = t_local_list[u] - user_latencies[u]          # 实际省下的时间 (可能是负的).0214 17:00,增大调正
            denominator = t_local_list[u] - t_ideal_list[u] + 1e-9   # 理论最大能省下的时间
            rate = numerator / denominator
            # 限制在 [-1, 1] 之间，线性奖励对范围很敏感
            saving_rates.append(np.clip(rate, -10.0, 10.0))
            
        avg_saving_rate = np.mean(saving_rates) 
        
        # --- User Reward ---
        for u in range(self.n_users):
            prio = self.base.omega_H if self.users[u]['priority'] else self.base.omega_L
            
            # [修改 1] 改回线性奖励
            # 逻辑：Rate 从 -0.5 提升到 0.5，奖励线性增加，梯度恒定，动力更足
            r_saving = self.base.w_saving * saving_rates[u] * prio
            
            # 2. 能耗惩罚 (线性归一化)
            eng_norm = user_energies[u] / self.base.norm_energy_user
            r_energy = -self.base.w_energy * eng_norm
            
            # 3. 合作奖励 (共享全局节省率，取代之前的惩罚)
            r_coop = self.base.coop_gamma * avg_saving_rate
            
            # 4. 违规惩罚
            r_penalty = 0
            if user_latencies[u] > self.base.latency_max:
                r_penalty = -self.base.w_penalty

            r = r_saving + r_energy + r_coop + r_penalty
            rewards.append(r)
            
        # --- UAV Reward ---
        all_user_positions = np.array([u['position'] for u in self.users])
        center_of_all_users = np.mean(all_user_positions, axis=0)
        
        # [NEW] 暂存 UAV 调试信息，以便传入 info
        uav_debug_stats = [{} for _ in range(self.n_uavs)]

        for m in range(self.n_uavs):
            # [修正] UAV 总能耗 = 飞行 + 计算
            total_uav_energy = uav_fly_energies[m] + uav_comp_energies[m]
            
            # [修改1] 即使没有连接用户，也要承担全局时延的责任 (Cooperative Reward)
            # 迫使 UAV 即使不服务也要关心系统状态，或者给一个空载惩罚
            # 1. 能耗项 (线性归一化: -E/600)
            uav_eng_norm = total_uav_energy / self.base.norm_energy_uav
            r_uav_energy = -self.base.w_energy * uav_eng_norm # 1. 能耗项 (线性归一化: -E/600),权重乘以归一化后的能耗
            
            # 2. 服务质量项 (基于簇内时延)
            cluster_users = [u for u in associations if associations[u] == m]
            
            if len(cluster_users) > 0:
                # 1. 服务奖励：根据它服务的用户的平均节省率给分
                cluster_rates = [saving_rates[u] for u in cluster_users]
                avg_cluster_rate = np.mean(cluster_rates)
                
                # [修改 2] 改回线性奖励
                # 简单直接：平均节省率 * 服务人数 * 权重
                # 只要能提升一点点节省率，分就蹭蹭涨，这会逼着UAV贴脸服务
                r_service = self.base.w_saving * avg_cluster_rate * len(cluster_users)
                
                cluster_positions = np.array([self.users[uid]['position'] for uid in cluster_users])
                target_pos = np.mean(cluster_positions, axis=0)
            else:
                # [修改此处] 把固定的 -5.0 换成一个能提供移动梯度的惩罚
                # 让它知道：越靠近中心，即便没有用户，被骂得也轻一点
                dist_to_center = np.linalg.norm(self.uavs[m]['pos'] - center_of_all_users)
                r_service = -5.0 - 10.0 * (dist_to_center / self.base.norm_pos) 
                target_pos = center_of_all_users

            # 3. 距离引导 (减弱了引力，让系统更依赖真实的时延节省)
            dist_to_target = np.linalg.norm(self.uavs[m]['pos'] - target_pos)
            dist_norm = dist_to_target / self.base.norm_pos
            r_guide = -self.base.w_guide * dist_norm

            # 4. 防碰撞惩罚 (保持指数级，构建“虚拟墙”)
            r_collision = 0
            min_dist_uav = 9999.0 # [NEW] 记录最小距离
            for other_m in range(self.n_uavs):
                if m == other_m: continue
                dist_uav = np.linalg.norm(self.uavs[m]['pos'] - self.uavs[other_m]['pos'])
                min_dist_uav = min(min_dist_uav, dist_uav) # Update min dist
                
                if dist_uav < self.base.uav_safe_dist:
                    violation_norm = (self.base.uav_safe_dist - dist_uav) / self.base.uav_safe_dist
                    r_collision -= self.base.w_collision * np.exp(np.clip(violation_norm, 0, 5))

            # 5. 撞墙惩罚
            r_bound = -10.0 if uav_out_of_bound[m] else 0.0

            r = r_service + r_uav_energy + r_guide + r_collision + r_bound
            rewards.append(r)
            
            # [NEW] Save debug info
            uav_debug_stats[m] = {
                'uav_dist_to_target': dist_to_target,
                'uav_min_dist': min_dist_uav
            }

        for u in self.users:
            u['task_size'] = np.random.uniform(self.base.task_size_min, self.base.task_size_max)
            
        # ========================================================
        # [修改点] 构建包含详细数据的 Infos，用于Runner绘图和统计
        # ========================================================
        infos = []
        for i in range(self.agent_num):
            info = {}
            if i < self.n_users:
                # User info: 包含真实时延、能耗、位置
                info['latency'] = user_latencies[i]
                info['energy'] = user_energies[i]
                info['pos'] = self.users[i]['position'].copy() 
                # 计算 time cost (delay part cost)
                prio = self.base.omega_H if self.users[i]['priority'] else self.base.omega_L
                info['time_cost'] = prio * user_latencies[i]
                
                # [新增输出] 记录 saving_rate
                info['saving_rate'] = saving_rates[i]
                info['assoc_id'] = associations[i]
                info['offload_ratio'] = offloading_ratios[i]
                info['alloc_freq'] = user_allocated_freq[i]

            else:
                # UAV info: 包含位置、飞行能耗
                uav_idx = i - self.n_users
                info['fly_energy'] = uav_fly_energies[uav_idx]
                info['uav_comp_energy'] = uav_comp_energies[uav_idx]
                info['pos'] = self.uavs[uav_idx]['pos'].copy()
                
                # [新增] 连接的用户列表
                info['connected_users'] = uav_connected_users_list[uav_idx]
                # [UAV Metric] 把刚刚存的调试信息放进去
                info.update(uav_debug_stats[uav_idx]) 
            infos.append(info)

        next_obs = self._get_obs(uav_loads)
        done = [self.steps >= 100] * self.agent_num 
        
        return next_obs, rewards, done, infos

    def _get_obs(self, prev_loads):
        obs = []
        flat_uav_pos = []
        for uav in self.uavs:
            flat_uav_pos.extend(uav['pos'] / self.base.norm_pos)
            
        # 1. User Obs (Padding to 63)
        for u in self.users:
            o = []
            o.extend(u['position'] / self.base.norm_pos)
            o.append(u['task_size'] / self.base.norm_data)
            o.append(u['cycles'] / self.base.norm_cycle)
            o.append(u['priority'])
            o.extend(flat_uav_pos)
            o.extend([l/10.0 for l in prev_loads]) 
            
            # Padding
            padding = [0.0] * (self.obs_dim - len(o))
            o.extend(padding)
            obs.append(np.array(o))
            
        # 2. UAV Obs (Need exactly 63)
        for m in self.uavs:
            o = []
            o.extend(m['pos'] / self.base.norm_pos) # 2
            o.append(m['energy'] / 1e6)             # 1
            
            for u in self.users:
                o.extend(u['position'] / self.base.norm_pos)    # 2
                o.append(u['task_size'] / self.base.norm_data)  # 1
                o.append(u['cycles'] / self.base.norm_cycle)    # 1 [之前少了这行!]
                o.append(u['priority'])                         # 1
                o.append(1.0)                                   # 1
            
            # 安全 Padding (防止算错，强制补齐)
            if len(o) < self.obs_dim:
                o.extend([0.0] * (self.obs_dim - len(o)))
                
            obs.append(np.array(o))
            
        return obs