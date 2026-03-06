# 4. 撞墙出界惩罚
            r_bound = -10.0 if uav_out_of_bound[m] else 0.0

            # 最终合并奖励
            r = r_service + r_uav_energy + r_guide + r_collision + r_bound
            
            # ========================================================
            # [提醒] 如果你之前在这里加了 r_scaled = r / 100.0，
            # 请确保把 r_scaled 放入 rewards.append，下面 step_reward 也记录 r_scaled
            # ========================================================
            rewards.append(r)
            
            # 记录调试信息和 [核心新增] 的奖励细项
            uav_debug_stats[m] = {
                'uav_dist_to_target': dist_to_target,
                'uav_min_dist': min_dist_uav,
                'r_service': r_service,
                'r_uav_energy': r_uav_energy,
                'r_guide': r_guide,
                'r_collision': r_collision,
                'r_bound': r_bound,
                'step_reward': r 
            }