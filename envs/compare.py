# ==========================================
        # 4. 奖励计算 (极致精简：全局共享大锅饭)
        # 严格对应论文公式 Eq.(24)：Cost = Delay + Energy
        # ==========================================
        rewards = []
        
        # --- 核心指标 1：系统全局节省率 (System Delay Cost 的反面) ---
        saving_rates = []
        for u in range(self.n_users):
            rate = (t_local_list[u] - user_latencies[u]) / (t_local_list[u] - t_ideal_list[u] + 1e-9)
            saving_rates.append(np.clip(rate, -1.0, 1.0))
        
        # 将所有人的节省率平均，代表整体时延体验
        global_saving_reward = self.base.w_saving * np.mean(saving_rates)
        
        # --- 核心指标 2：系统全局能耗 (System Energy Cost) ---
        # 统一将全体 User 和全体 UAV 的能耗做归一化相加
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
            r_scaled = system_reward / 10.0 # 缩放保护梯度
            rewards.append(r_scaled)
            
            # 借用旧字典的 Key，确保不破坏你的 Runner CSV 记录代码
            user_reward_details.append({
                'r_saving': global_saving_reward,
                'r_energy': global_energy_reward,
                'r_coop': system_reward, 
                'r_penalty': global_penalty,
                'step_reward': r_scaled
            })

        # 2. 分发给 UAV (系统奖励 + 个人驾驶违规)
        uav_debug_stats = [{} for _ in range(self.n_uavs)]
        for m in range(self.n_uavs):
            
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

            # UAV 最终奖励 = 系统大锅饭 + 自己的扣分项
            r_uav = system_reward + r_guide + r_collision + r_bound
            
            r_scaled = r_uav / 10.0 
            rewards.append(r_scaled)
            
            # 借用旧字典的 Key 存入信息
            uav_debug_stats[m] = {
                'uav_dist_to_target': dist_to_center,
                'uav_min_dist': min_dist_uav,
                'r_service': system_reward,  
                'r_uav_energy': global_energy_reward,
                'r_guide': r_guide,
                'r_collision': r_collision,
                'r_bound': r_bound,
                'r_coop': system_reward,
                'step_reward': r_scaled 
            }