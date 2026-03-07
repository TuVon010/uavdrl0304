import numpy as np
#rl-gym
class Base:
    def __init__(self):
        # =========================
        # 1. 场景设置 (3 UAV, 10 Users)
        # =========================
        self.n_uavs = 3
        self.n_users = 10
        self.field_X = [0, 1000]#修改范围到1000
        self.field_Y = [0, 1000]
        self.h = 50              # UAV高度 (m)
        self.time_step = 1.0      # delta_t (s)
        # [核心新增] 无人机的“免罚覆盖半径” (比如 150 米)
        self.coverage_radius = 80.0
        
        # =========================
        # 2. 归一化参数 (RL收敛核心)
        # =========================
        self.norm_pos = 1000.0             # 位置归一化 (地图边长)
        self.norm_data = 5e6              
        self.norm_cycle = 1000.0          
        self.norm_freq = 10e9             
        
        # [新增] 能耗归一化基准 (Estimate Max Energy per Step)
        # 飞行功率约 500W * 1s = 500J
        self.norm_energy_uav = 600.0      
        # 用户发射功率 0.2W + 本地计算 ~ 1-5J
        self.norm_energy_user = 5.0       
        # (删除了 norm_latency，因为现在用节省率，自带归一化)

        # =========================
        # 3. 通信与信道参数 (Urban LoS)
        # =========================
        self.f_c = 2e9            
        self.c = 3e8              
        self.alpha_los = 2.8      
        self.alpha_nlos = 3.5     
        self.a = 9.61             # LoS 概率参数
        self.b = 0.16
        #self.beta0 = 10**(-50/10) # 参考路径损耗 (-50dB)
        # [修改 1] 提升参考信号强度 (-50dB -> -40dB)
        # 这相当于给无人机装了更好的天线，信号增强10倍
        self.beta0 = 10**(-60/10)
        self.sigma2 = 1e-13       # 噪声功率 (W)
        self.B_total = 20e6       # 每个UAV的总带宽 (50MHz)

        # =========================
        # 4. 智能体物理属性
        # =========================
        # --- UAV ---
        self.uav_v_max = 15.0     # 最大水平速度 (m/s)
        self.C_uav = 10e9         # UAV 总算力 (10 GHz)
        # self.xi_m = 8.2e-27         # UAV 算力能耗系数
        self.xi_m = 8.2e-10         # UAV 算力能耗系数
        
        # 飞行能耗参数 (Rotary-Wing)
        self.P0 = 79.86
        self.Pi = 88.63
        self.U_tip = 120
        self.v0 = 4.03
        self.d0 = 0.6
        self.rho = 1.225
        self.s = 0.05
        self.A_rotor = 0.5

        # --- User (MU) ---
        self.p_tx_max = 0.2       # MU 最大发射功率 (W)
        self.C_local = 1e9        # MU 本地算力 (1 GHz)
        self.k_local = 1e-28      # MU 能耗系数
        
        # =========================
        # 5. Gauss-Markov 移动模型参数 (严格)
        # =========================
        self.mobility_slot = 1.0          # 刷新间隔
        self.user_mean_velocity = 0.5     # 渐进平均速度
        self.user_mean_direction = 0.1    # 渐进平均方向 (rad)
        self.user_memory_level_velocity = 0.6  
        self.user_memory_level_direction = 0.8 
        self.user_Gauss_variance_velocity = 0.5
        self.user_Gauss_variance_direction = 0.5

        # =========================
        # 6. 任务生成
        # =========================
        self.task_size_min = 1e6  
        self.task_size_max = 5e6
        self.cycles_min = 999
        self.cycles_max = 1000
        self.latency_max = 1.0    # 最大容忍时延 (s)

        # =========================
        # 7. 奖励权重 (纯共享全局奖励)
        # =========================
        self.w_saving = 100.0       # 全局节省率权重 (放大一点，作为主要动力)
        self.w_energy = 20.0        # 全局能耗惩罚权重
        self.w_penalty = 1.0       # 全局超时惩罚
        
        self.w_guide = 15.0        # 无人机物理牵引力 
        self.w_collision = 5.0
        
        self.omega_H = 1.2        # 高优先级加权
        self.omega_L = 1.0        
        self.coop_gamma = 0.9     # 合作因子
        
        # 互斥参数
        self.uav_safe_dist = 10.0