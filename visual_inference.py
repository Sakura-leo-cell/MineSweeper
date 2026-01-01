import torch
import numpy as np
import time
import pygame
from game_env import MinesweeperEnv
from model import CNNDQN
from gui_renderer import MinesweeperGUI

GRID_SIZE = 6
N_MINES = 6
MODEL_PATH = "minesweeper_cnn_flags.pth" # 注意加载新模型
STEP_DELAY = 1.0 

def run_visual_game():
    env = MinesweeperEnv(GRID_SIZE, N_MINES)
    gui = MinesweeperGUI(env, cell_size=60)
    
    n_actions = GRID_SIZE * GRID_SIZE
    model = CNNDQN(GRID_SIZE, n_actions * 2) # 输出维度 * 2
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print(f"模型加载成功: {MODEL_PATH}")
    except FileNotFoundError:
        print("请先运行 train.py 生成新的模型文件")
        gui.close()
        return

    state = env.reset()
    done = False
    
    # 初始渲染
    gui.render()
    time.sleep(1)

    steps = 0
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # --- 获取有效动作 Mask ---
        flatten_visible = state[0].flatten()
        flatten_flags = state[2].flatten()
        
        valid_click = [i for i, v in enumerate(flatten_visible) if v == -1 and flatten_flags[i] == 0]
        valid_flag = [i + n_actions for i, v in enumerate(flatten_visible) if v == -1]
        valid_actions = valid_click + valid_flag
        
        if not valid_actions: valid_actions = [0]

        with torch.no_grad():
            q_values = model(state_tensor)
            full_mask = torch.ones_like(q_values) * -float('inf')
            full_mask[0, valid_actions] = q_values[0, valid_actions]
            action = torch.argmax(full_mask).item()
        
        # 解析动作
        if action < n_actions:
            x, y = divmod(action, GRID_SIZE)
            act_type = 0 # Click
            print(f"Step {steps}: AI 点击 ({x}, {y})")
        else:
            x, y = divmod(action - n_actions, GRID_SIZE)
            act_type = 1 # Flag
            print(f"Step {steps}: AI 插旗 ({x}, {y}) [PREDICTION: MINE]")

        state, reward, done = env.step(action)
        steps += 1

        # 传入动作信息用于高亮 (类型0为绿框，1为橙框)
        running = gui.render(last_action_info=(x, y, act_type))
        if not running: return

        time.sleep(STEP_DELAY)

    if reward > 0:
        print(">>> AI 胜利！ <<<")
    else:
        print(">>> AI 踩雷 / 失败 <<<")

    gui.render(reveal_all=True)
    gui.wait_for_quit()

if __name__ == "__main__":
    run_visual_game()