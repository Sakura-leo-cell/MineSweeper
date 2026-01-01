import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from game_env import MinesweeperEnv
from model import CNNDQN

# --- 配置修改 ---
GRID_SIZE = 6
N_MINES = 6
EPISODES = 5000       # 增加训练轮数
BATCH_SIZE = 128      # 增加 Batch Size 稳定梯度
LR = 0.0005           # 降低学习率，防止震荡
GAMMA = 0.99          # 看得更长远
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.999 # 衰减得非常慢，保证前2000轮都有大量探索

def train():
    env = MinesweeperEnv(GRID_SIZE, N_MINES)
    # 记得这里的 input shape 在 model 内部已经写死为 2 了，不需要传参
    model = CNNDQN(GRID_SIZE, GRID_SIZE * GRID_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=10000) # 增大记忆库
    
    epsilon = EPSILON_START
    win_count = 0
    
    print(f"--- 开始训练 Pro 版扫雷 AI ({GRID_SIZE}x{GRID_SIZE}) ---")

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0) # (1, 2, H, W)
            
            if random.random() < epsilon:
                action = random.randint(0, GRID_SIZE*GRID_SIZE - 1)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    # 关键Mask：强制不点已经点过的地方
                    visible_mask = env.visible.flatten() == 1
                    q_values[0, visible_mask] = -float('inf')
                    action = torch.argmax(q_values).item()

            next_state, reward, done = env.step(action)
            
            # 重要：如果赢了，打印一下
            if reward == 50.0:
                win_count += 1
            
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            if len(replay_buffer) > BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                b_state, b_action, b_reward, b_next_state, b_done = zip(*batch)
                
                b_state = torch.FloatTensor(np.array(b_state))
                b_next_state = torch.FloatTensor(np.array(b_next_state))
                b_action = torch.LongTensor(b_action).unsqueeze(1)
                b_reward = torch.FloatTensor(b_reward).unsqueeze(1)
                b_done = torch.FloatTensor(b_done).unsqueeze(1)
                
                q_current = model(b_state).gather(1, b_action)
                
                with torch.no_grad():
                    # Double DQN 思想 (简化版)：直接取 max
                    q_next = model(b_next_state).max(1)[0].unsqueeze(1)
                    q_target = b_reward + (GAMMA * q_next * (1 - b_done))
                
                loss = criterion(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}, Wins(last 100): {win_count}")
            win_count = 0

    torch.save(model.state_dict(), "minesweeper_cnn_pro.pth")
    print("训练完成。")

if __name__ == "__main__":
    train()