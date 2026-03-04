import sys
import os
import torch
import numpy as np
from pathlib import Path

# --- 新增这两行代码 ---
# 获取当前文件的上一级目录（也就是项目根目录）
sys.path.append(str(Path(__file__).resolve().parent.parent))
# --------------------
from config import get_config
from envs.env_wrappers import DummyVecEnv
from runner.separated.eval_runner import EvalRunner

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            from envs.env_continuous import ContinuousActionEnv
            env = ContinuousActionEnv()
            # 设置随机种子，保证每次测试环境行为一致
            env.seed(all_args.seed + rank * 5000)
            return env
        return init_env
    # 强制单线程
    return DummyVecEnv([get_env_fn(0)])

def main(args):
    parser = get_config()
    # 添加必要的参数默认值，防止报错
    parser.add_argument("--scenario_name", type=str, default="MyEnv")
    parser.add_argument("--num_agents", type=int, default=13)
    
    all_args = parser.parse_known_args(args)[0]

    # === 强制配置覆盖 ===
    all_args.use_eval = True
    all_args.n_rollout_threads = 1  # 测试必须单线程
    all_args.cuda = True if torch.cuda.is_available() else False
    
    # === 自动寻找最新的模型路径 ===
    # 如果你想手动指定，请取消下面这行的注释并修改路径:
    all_args.model_dir = "results/MyEnv/MyEnv/mappo/check/run10/models"
    
    if all_args.model_dir is None:
        base_dir = Path("results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
        if not base_dir.exists():
            print(f"Error: 找不到实验目录 {base_dir}")
            return
            
        # 找最大的 runX 文件夹
        runs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("run")], 
                      key=lambda x: int(x.name.replace("run", "")))
        if runs:
            latest_run = runs[-1]
            all_args.model_dir = str(latest_run / "models")
            print(f"自动锁定最新模型路径: {all_args.model_dir}")
        else:
            print("Error: 该实验下没有 run 文件夹")
            return

    # 初始化环境
    envs = make_eval_env(all_args)
    
    # 构造 Config
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": envs, 
        "num_agents": all_args.num_agents,
        "device": torch.device("cuda" if all_args.cuda else "cpu"),
        "run_dir": Path(all_args.model_dir).parent 
    }

    # 运行测试
    runner = EvalRunner(config)
    runner.run_eval()

    # 清理
    envs.close()

if __name__ == "__main__":
    # 示例运行命令: python eval_test.py --experiment_name test_exp
    main(sys.argv[1:])