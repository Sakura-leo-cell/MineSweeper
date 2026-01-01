import torch
import numpy as np
import time
import pygame
from game_env import MinesweeperEnv
from model import CNNDQN
from gui_renderer import MinesweeperGUI

# --- 配置 ---
GRID_SIZE = 6
N_MINES = 6
MODEL_PATH = "minesweeper_cnn_pro.pth" # 确保这里是你训练好的模型文件名
STEP_DELAY = 2 # AI每步思考的时间（秒），调大一点方便看清楚

def run_visual_game():
    # 1. 初始化
    env = MinesweeperEnv(GRID_SIZE, N_MINES)
    gui = MinesweeperGUI(env, cell_size=60) # cell_size控制窗口大小
    
    model = CNNDQN(GRID_SIZE, GRID_SIZE * GRID_SIZE)
    
    # 2. 加载模型
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print(f"成功加载模型: {MODEL_PATH}")
    except FileNotFoundError:
        print("错误: 找不到模型文件，请先运行 train.py")
        gui.close()
        return

    # 3. 游戏循环
    state = env.reset()
    done = False
    print("--- 游戏开始 ---")

    # 初始渲染
    if not gui.render(): return
    time.sleep(1)

    steps = 0
    while not done:
        # 处理 PyGame 事件防止卡死
        
        # --- AI 决策 ---
        state_tensor = torch.FloatTensor(state).unsqueeze(0) # (1, 2, H, W)
        with torch.no_grad():
            q_values = model(state_tensor)
            # Mask
            visible_mask = env.visible.flatten() == 1
            q_values[0, visible_mask] = -float('inf')
            action = torch.argmax(q_values).item()
        
        x, y = divmod(action, GRID_SIZE)
        print(f"Step {steps}: AI 点击 ({x}, {y})")

        # --- 环境交互 ---
        state, reward, done = env.step(action)
        steps += 1

        # --- 更新界面 ---
        # 传入 (x, y) 是为了高亮显示 AI 当前点的格子
        running = gui.render(last_action_xy=(x, y))
        if not running: return # 用户点了关闭窗口

        time.sleep(STEP_DELAY)

    # 4. 游戏结束处理
    print(">>> 游戏结束 <<<")
    if reward > 0:
        print("AI 胜利！")
        pygame.display.set_caption("Game Over - AI WINS!")
    else:
        print("AI 踩雷！")
        pygame.display.set_caption("Game Over - BOOM!")

    # 揭示所有格子给用户看
    gui.render(reveal_all=True)
    
    # 等待用户关闭
    gui.wait_for_quit()

if __name__ == "__main__":
    run_visual_game()