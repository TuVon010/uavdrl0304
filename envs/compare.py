# ==========================================
        # 4. 奖励计算 (回归线性 Linear)
        # ==========================================
        rewards = []
        
        # 计算所有用户的节省率 (Saving Rate)
        saving_rates = []
        for u in range(self.n_users):
            numerator = t_local_list[u] - user_latencies[u]
            denominator = t_local_list[u] - t_ideal_list[u] + 1e-9
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
            
            eng_norm = user_energies[u] / self.base.norm_energy_user
            r_energy = -self.base.w_energy * eng_norm
            
            r_coop = self.base.coop_gamma * avg_saving_rate
            
            r_penalty = 0
            if user_latencies[u] > self.base.latency_max:
                r_penalty = -self.base.w_penalty

            r = r_saving + r_energy + r_coop + r_penalty
            rewards.append(r)
            
        # --- UAV Reward ---
        all_user_positions = np.array([u['position'] for u in self.users])
        center_of_all_users = np.mean(all_user_positions, axis=0)
        
        uav_debug_stats = [{} for _ in range(self.n_uavs)]

        for m in range(self.n_uavs):
            total_uav_energy = uav_fly_energies[m] + uav_comp_energies[m]
            
            cluster_users = [u for u in associations if associations[u] == m]
            
            if len(cluster_users) > 0:
                cluster_rates = [saving_rates[u] for u in cluster_users]
                avg_cluster_rate = np.mean(cluster_rates)
                
                # [修改 2] 改回线性奖励
                # 简单直接：平均节省率 * 服务人数 * 权重
                # 只要能提升一点点节省率，分就蹭蹭涨，这会逼着UAV贴脸服务
                r_service = self.base.w_saving * avg_cluster_rate * len(cluster_users)
                
                cluster_positions = np.array([self.users[uid]['position'] for uid in cluster_users])
                target_pos = np.mean(cluster_positions, axis=0)
            else:
                # 怠工惩罚
                r_service = -5.0 
                target_pos = center_of_all_users

            uav_eng_norm = total_uav_energy / self.base.norm_energy_uav
            r_uav_energy = -self.base.w_energy * uav_eng_norm

            dist_to_target = np.linalg.norm(self.uavs[m]['pos'] - target_pos)
            dist_norm = dist_to_target / self.base.norm_pos
            r_guide = -self.base.w_guide * dist_norm

            r_collision = 0
            min_dist_uav = 9999.0 
            for other_m in range(self.n_uavs):
                if m == other_m: continue
                dist_uav = np.linalg.norm(self.uavs[m]['pos'] - self.uavs[other_m]['pos'])
                min_dist_uav = min(min_dist_uav, dist_uav)
                
                if dist_uav < self.base.uav_safe_dist:
                    violation_norm = (self.base.uav_safe_dist - dist_uav) / self.base.uav_safe_dist
                    r_collision -= self.base.w_collision * np.exp(np.clip(violation_norm, 0, 5))

            r_bound = -10.0 if uav_out_of_bound[m] else 0.0

            r = r_service + r_uav_energy + r_guide + r_collision + r_bound
            rewards.append(r)
            
            uav_debug_stats[m] = {
                'uav_dist_to_target': dist_to_target,
                'uav_min_dist': min_dist_uav
            }