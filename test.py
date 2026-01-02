import torch
import numpy as np
import time
import os
from game_env import MinesweeperEnv
from model import CNNDQN

# ===========================
# 🛠️ 测试配置
# ===========================
GRID_SIZE = 9       # 你的地图尺寸 (测 9x9 时记得改这里)
N_MINES = 10         # 你的雷数

TEST_EPISODES = 1000
MODEL_PATH = f"saved_models/dqn_{GRID_SIZE}x{GRID_SIZE}.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_test():
    print(f"🚀 开始测试 | 模型: {MODEL_PATH}")
    
    env = MinesweeperEnv(GRID_SIZE, N_MINES)
    model = CNNDQN(GRID_SIZE, GRID_SIZE*GRID_SIZE).to(device)
    
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✅ 权重加载成功")
        except Exception as e:
             print(f"❌ 模型不匹配: {e}")
             print("💡 提示: 网络结构已改为 5x5 卷积，请删除旧权重重新训练！")
             return
    else:
        print("❌ 找不到模型文件！请先运行 trainV2.py")
        return

    model.eval()
    wins = 0
    
    for i in range(TEST_EPISODES):
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                q_values = model(state_tensor)
                # Action Masking
                visible_mask = torch.tensor(env.visible.flatten(), device=device, dtype=torch.bool)
                q_values[0, visible_mask] = -float('inf')
                action = torch.argmax(q_values).item()
            
            state, reward, done = env.step(action)
            
            if done and np.sum(env.visible) == (GRID_SIZE**2 - N_MINES):
                wins += 1
        
        if (i+1) % 100 == 0:
            print(f"进度: {i+1}/{TEST_EPISODES} | 胜率: {wins/(i+1)*100:.2f}%")

    print(f"🏆 最终胜率: {wins/TEST_EPISODES*100:.2f}%")

if __name__ == "__main__":
    run_test()