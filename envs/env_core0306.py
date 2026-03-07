import numpy as np
from envs.Base0306 import Base
from envs.physics_engine0306 import PhysicsEngine
#0215奖励函数修改回线性模型
#0304给“空载”的无人机加上动态的怠工惩罚,修改半径
#单无人机多用户
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
        center_x, center_y = 500.0, 500.0
        
        # ==========================================
        # 1. 初始化 UAV (三角形顶点分布)
        # ==========================================
        self.uavs = []
        # 定义三角形半径 (UAV离中心的距离)
        radius = 250.0 
        # 定义三个角度: 90度(正上方), 210度(左下), 330度(右下)
        angles = [np.pi/2, 7*np.pi/6, 11*np.pi/6]
        
        for i in range(self.n_uavs):
            # 计算顶点坐标
            uav_x = center_x + radius * np.cos(angles[i % 3])
            uav_y = center_y + radius * np.sin(angles[i % 3])
            uav_x = np.clip(uav_x, 0, 1000)
            uav_y = np.clip(uav_y, 0, 1000)
            self.uavs.append({'id': i, 'pos': np.array([uav_x, uav_y]), 'energy': 1e5})
            
        # ==========================================
        # 2. 初始化 User (中心密集聚集 + GaussMarkov)
        # ==========================================
        self.users = []
        for i in range(self.n_users):
            # 使用高斯分布生成位置 (密集聚集)
            # loc=中心, scale=标准差(聚集程度, 越小越密集)
            user_x = np.random.normal(loc=center_x, scale=25.0) 
            user_y = np.random.normal(loc=center_y, scale=25.0)
            
            # 必须 Clip 防止由于概率极低事件飞出地图
            user_x = np.clip(user_x, 0, 1000)
            user_y = np.clip(user_y, 0, 1000)
            pos = np.array([user_x, user_y])
            
            self.users.append({
                'id': i,
                'position': pos,
                # 初始速度和方向随机，后续由 physics_engine 的 update_user_positions 接管 (GaussMarkov)
                'velocity': np.random.uniform(0.0, 0.5), 
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
        #0304修改act映射方式
        # env_core.py 中用户的动作解析部分
        for u_idx, act in enumerate(user_actions_raw):
            # 1. 卸载比例用 tanh 平滑映射到 [0, 1]
            ratio = 0.5 * (np.tanh(act[1]) + 1.0)
            
            # 2. 连接选择：用 tanh 把 act[0] 映射到 [-1, 1] 之间
            # 然后均分成 self.n_uavs + 1 (本地) 份
            choice_val = np.tanh(act[0]) 
            
            # 假设 n_uavs = 3，总共有 4 个选项 (0本地, 1连U0, 2连U1, 3连U2)
            # [-1, -0.5) -> 本地
            # [-0.5, 0) -> UAV_0
            # [0, 0.5) -> UAV_1
            # [0.5, 1] -> UAV_2
            total_options = self.n_uavs + 1
            # 把 [-1, 1] 映射到 [0, total_options - 0.001]
            mapped_val = (choice_val + 1.0) / 2.0 * (total_options - 1e-5)
            assoc_idx = int(mapped_val)
            
            if assoc_idx == 0 or ratio < 0.1:
                associations[u_idx] = -1  # 本地计算
                offloading_ratios[u_idx] = 0.0
            else:
                target_uav = assoc_idx - 1
                associations[u_idx] = target_uav
                offloading_ratios[u_idx] = ratio
                uav_loads[target_uav] += 1

        # --- 物理演化 ---
        self.engine.update_user_positions(self.users)
        uav_fly_energies = []
        uav_out_of_bound = [False] * self.n_uavs 
        
        # 计算全局用户中心 (极简宏观目标)
        all_user_positions = np.array([u['position'] for u in self.users])
        global_center = np.mean(all_user_positions, axis=0)

        for m_idx, act in enumerate(uav_actions_raw):
            
            # 直接计算 UAV 到全局用户中心的距离和角度
            vec_to_target = global_center - self.uavs[m_idx]['pos']
            dist_to_target = np.linalg.norm(vec_to_target)
            ideal_angle = np.arctan2(vec_to_target[1], vec_to_target[0])
            
            v_ratio = 0.5 * (np.tanh(act[0]) + 1.0) 
            
            # 【极简动作防呆】假如远离中心100m以外就往中间引导
            if dist_to_target > 100:
                # 【圈外：赶路模式】限制角度！
                # 必须朝向目标前进，最大允许左右偏移 90 度 (np.pi/2)
                angle = ideal_angle + np.tanh(act[1]) * (np.pi / 2.0)
            else:
                # 【圈内：微调/干活模式】彻底解放！
                # 允许 360 度 (-π 到 π) 全向移动，方便躲避碰撞和覆盖边缘用户
                # 【圈内】依然以目标为基准，但允许 360 度全向微调！
                angle = ideal_angle + np.tanh(act[1]) * np.pi
            
            v_mag = v_ratio * self.base.uav_v_max
            vel = np.array([v_mag * np.cos(angle), v_mag * np.sin(angle)])
            new_pos = self.uavs[m_idx]['pos'] + vel * self.base.time_step
            
            if (new_pos[0] < 0 or new_pos[0] > 1000 or new_pos[1] < 0 or new_pos[1] > 1000):
                uav_out_of_bound[m_idx] = True

            self.uavs[m_idx]['pos'] = np.clip(new_pos, 0, 1000)
            e_fly = self.engine.compute_uav_energy(v_mag)
            uav_fly_energies.append(e_fly)

        # --- 计算指标 ---
        user_latencies = np.zeros(self.n_users)
        user_energies = np.zeros(self.n_users)
        uav_comp_energies = np.zeros(self.n_uavs)
        
        # [新增] 用于记录分配给每个用户的频率，用于存入 info
        user_allocated_freq = np.zeros(self.n_users)
        user_tx_rates = np.zeros(self.n_users) 
        
        # ==========================================================
        # [核心新增] 用于记录每一个细分时间指标的数组
        # ==========================================================
        user_t_tx = np.zeros(self.n_users)   # 传输时间
        user_t_exe = np.zeros(self.n_users)  # UAV处理时间
        user_t_loc = np.zeros(self.n_users)  # 本地处理时间 
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
                # ==================== [核心新增]debug ====================
                user_tx_rates[u_id] = rate # 记录香农速率
                # ==================================================
                t_tx = (lamb * user['task_size']) / (rate + 1e-9)
                t_exe = cycles_total / (f_uav + 1e-9)
                t_offload = t_tx + t_exe
                
                # 本地
                t_loc = ((1-lamb) * user['task_size'] * user['cycles']) / self.base.C_local
                e_loc = self.base.k_local * (self.base.C_local**2) * (1-lamb) * user['task_size'] * user['cycles']
                e_tx = self.base.p_tx_max * t_tx
                
                # ==========================================================
                # [核心新增] 精准记录连接用户的三段时间
                # ==========================================================
                user_t_tx[u_id] = t_tx
                user_t_exe[u_id] = t_exe
                user_t_loc[u_id] = t_loc
                
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

                # ==========================================================
                # [核心新增] 精准记录未连接用户的时间（全本地计算，无传输无无人机执行）
                # ==========================================================
                user_t_tx[u_id] = 0.0
                user_t_exe[u_id] = 0.0
                user_t_loc[u_id] = t_local_list[u_id]

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
            saving_rates.append(np.clip(rate, -1.0, 1.0))
            
        global_avg_saving_rate = np.mean(saving_rates) 
        # 将所有人的节省率平均，代表整体时延体验
        global_saving_reward = self.base.w_saving * global_avg_saving_rate
        # --- 核心指标 2：系统全局能耗 (System Energy Cost) ---
        # 统一将全体 User 和全体 UAV 的能耗做归一化相加
        # total_user_energy_norm = np.sum(user_energies) / (self.base.norm_energy_user * self.n_users)
        # total_uav_energy_norm = np.sum(uav_fly_energies + uav_comp_energies) / (self.base.norm_energy_uav * self.n_uavs)
        total_user_energy_norm = np.sum(user_energies) / (self.base.norm_energy_user * self.n_users)
        total_uav_energy_norm = np.sum(uav_fly_energies + uav_comp_energies) / (self.base.norm_energy_uav * self.n_uavs)
        
        global_energy_reward = -self.base.w_energy * (total_user_energy_norm + total_uav_energy_norm)
        
        # --- 核心指标 3：系统违规总数 (System Constraint) ---
        violation_count = sum([1 for l in user_latencies if l > self.base.latency_max])
        global_penalty = -self.base.w_penalty * violation_count
        
        # 👑 【系统统一奖励】(这就是全村人的共同目标)
        system_reward = global_saving_reward + global_energy_reward + global_penalty
        
        # ==========================================
        # 分配奖励与装载数据
        # ==========================================
        
        # 1. 分发给 User (完全共享系统奖励)
        user_reward_details = []
        for u in range(self.n_users):
            r_user = system_reward
            rewards.append(r_user)
            
            # 借用旧字典的 Key，确保不破坏你的 Runner CSV 记录代码
            user_reward_details.append({
                'r_global_saving': global_saving_reward,
                'r_global_energy': global_energy_reward,
                'r_system': system_reward, 
                'r_global_penalty': global_penalty,
                'step_reward': r_user
            })
            
        ## 2. 分发给 UAV (系统奖励 + 个人驾驶违规)
        uav_debug_stats = [{} for _ in range(self.n_uavs)]
        for m in range(self.n_uavs):
            
            # ========================================================
            # [核心新增：防摸鱼机制] 
            # 如果没拉到客（空载），直接给个人单独扣 5 分！逼它去抢客户！
            # ========================================================
            connected_users = [u for u in associations if associations[u] == m]
            r_idle = -5.0 if len(connected_users) == 0 else 0.0
            
            # A. 物理引导：哪怕是吃大锅饭，也要有鞭子把他们赶到 100 米内干活
            dist_to_center = np.linalg.norm(self.uavs[m]['pos'] - global_center)
            r_guide = 0.0
            if dist_to_center > 100.0:
                r_guide = -self.base.w_guide * (dist_to_center / self.base.norm_pos)
                
            # B. 驾驶安全：防碰撞和出界
            r_collision = 0
            min_dist_uav = 9999.0 
            for other_m in range(self.n_uavs):
                if m == other_m: continue
                dist_uav = np.linalg.norm(self.uavs[m]['pos'] - self.uavs[other_m]['pos'])
                min_dist_uav = min(min_dist_uav, dist_uav)
                if dist_uav < self.base.uav_safe_dist:
                    violation_norm = (self.base.uav_safe_dist - dist_uav) / self.base.uav_safe_dist
                    r_collision -= self.base.w_collision * 10.0 * violation_norm

            r_bound = -10.0 if uav_out_of_bound[m] else 0.0

            # 👑 UAV 最终奖励 = 系统大锅饭 + 自己的扣分项 + 【怠工惩罚】
            r_uav = system_reward + r_guide + r_collision + r_bound + r_idle
            
            
            rewards.append(r_uav)
            
            # 借用旧字典的 Key 存入信息
            uav_debug_stats[m] = {
                'uav_dist_to_target': dist_to_center,
                'uav_min_dist': min_dist_uav,
                # 'r_service': system_reward,  
                'r_global_energy': global_energy_reward,
                'r_guide': r_guide, 
                'r_idle': r_idle,# 怠工惩罚
                'r_collision': r_collision,
                'r_bound': r_bound,
                'r_system': system_reward,
                'step_reward': r_uav 
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
                # ==================== [核心新增] ====================
                info['tx_rate'] = user_tx_rates[i] 
                
                # ==========================================================
                # [核心新增] 将三个时间段放进 infos 供 Runner 提取
                # ==========================================================
                info['t_tx'] = user_t_tx[i]
                info['t_exe'] = user_t_exe[i]
                info['t_loc'] = user_t_loc[i]
                # [核心新增] 装入 User 奖励数据
                info.update(user_reward_details[i])

            else:
                # UAV info: 包含位置、飞行能耗
                uav_idx = i - self.n_users
                info['fly_energy'] = uav_fly_energies[uav_idx]
                info['uav_comp_energy'] = uav_comp_energies[uav_idx]
                info['pos'] = self.uavs[uav_idx]['pos'].copy()
                
                # [新增] 连接的用户列表
                info['connected_users'] = uav_connected_users_list[uav_idx]
                # [UAV Metric] 把刚刚存的调试信息(包含奖励细项)放进去
                info.update(uav_debug_stats[uav_idx]) 
            infos.append(info)

        next_obs = self._get_obs(uav_loads)
        done = [self.steps >= 60] * self.agent_num 
        
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
            o.append(m['energy'] / 1e5)             # 1
            
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