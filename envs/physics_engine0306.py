import numpy as np
import math

class PhysicsEngine:
    def __init__(self, base):
        self.base = base

    # ==========================================
    # 1. 严格的高斯-马尔可夫移动模型 (Gauss-Markov)
    # ==========================================
    def MarkovRandom(self, pre_var, memory_level, mean_var, Gauss_variance):
        """论文公式实现"""
        Gauss_process = np.random.normal(0, Gauss_variance)
        # 当前值 = memory * 上一时刻 + (1-memory) * 平均值 + 扰动项
        cur_var = memory_level * pre_var + (1 - memory_level) * mean_var + np.sqrt(
            1 - memory_level ** 2) * Gauss_process
        return cur_var

    def update_user_positions(self, users):
        """
        更新所有用户的位置、速度和方向
        users: list of dict, 必须包含 'velocity', 'direction', 'position'
        """
        for vehicle in users:
            # 1. 更新速度标量
            vehicle['velocity'] = self.MarkovRandom(
                vehicle['velocity'], 
                self.base.user_memory_level_velocity,
                self.base.user_mean_velocity, 
                self.base.user_Gauss_variance_velocity
            )
            
            # 2. 更新方向 (弧度)
            vehicle['direction'] = self.MarkovRandom(
                vehicle['direction'], 
                self.base.user_memory_level_direction,
                self.base.user_mean_direction, 
                self.base.user_Gauss_variance_direction
            )
            
            # 3. 位置更新 (基于当前速度和方向)
            # x_new = x_old + v * cos(theta) * dt
            dx = vehicle['velocity'] * math.cos(vehicle['direction']) * self.base.mobility_slot
            dy = vehicle['velocity'] * math.sin(vehicle['direction']) * self.base.mobility_slot
            
            cur_x = vehicle['position'][0] + dx
            cur_y = vehicle['position'][1] + dy
            
            # 4. 边界限制 (防止跑出地图)
            cur_x = np.clip(cur_x, self.base.field_X[0], self.base.field_X[1])
            cur_y = np.clip(cur_y, self.base.field_Y[0], self.base.field_Y[1])
            
            vehicle['position'] = np.array([cur_x, cur_y])
            
            if 'trajectory' in vehicle:
                vehicle['trajectory'].append(vehicle['position'])

    # ==========================================
    # 2. 通信与能耗模型
    # ==========================================
    def get_channel_gain(self, p1, p2):
        """概率视距 (Probabilistic LoS) 信道增益"""
        dist = np.linalg.norm(p1 - p2)
        d_3d = np.sqrt(dist**2 + self.base.h**2)
        
        # 计算仰角 (degree)
        theta = np.degrees(np.arcsin(self.base.h / d_3d))
        
        # LoS 概率
        p_los = 1.0 / (1.0 + self.base.a * np.exp(-self.base.b * (theta - self.base.a)))
        
        # 路径损耗
        pl_los = d_3d ** -self.base.alpha_los
        pl_nlos = d_3d ** -self.base.alpha_nlos
        
        # 平均路径损耗
        pl_avg = p_los * pl_los + (1 - p_los) * pl_nlos
        
        return self.base.beta0 * pl_avg

    def compute_rate(self, g, bw, p_tx):
        """Shannon 公式"""
        if bw <= 1e-9: return 0.0
        snr = (p_tx * g) / (self.base.sigma2 )#0209
        return bw * np.log2(1 + snr)

    def compute_uav_energy(self, v_mag):
        """UAV 飞行功率模型"""
        v_mag = np.clip(v_mag, 0, 30)
        
        # 叶片型面功率
        term1 = self.base.P0 * (1 + 3 * (v_mag**2) / (self.base.U_tip**2))
        
        # 诱导功率
        ind_sq = np.sqrt(1 + (v_mag**4)/(4 * self.base.v0**4)) - (v_mag**2)/(2 * self.base.v0**2)
        term2 = self.base.Pi * np.sqrt(np.maximum(0, ind_sq)) 
        
        # 寄生功率
        term3 = 0.5 * self.base.d0 * self.base.rho * self.base.s * self.base.A_rotor * (v_mag**3)
        
        power = term1 + term2 + term3
        return power * self.base.time_step # 返回能量 (Joules)