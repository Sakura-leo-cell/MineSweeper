import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from game_env import MinesweeperEnv
from model import CNNDQN

# --- 配置参数 ---
GRID_SIZE = 6
N_MINES = 6
EPISODES = 5000       
BATCH_SIZE = 128      
LR = 0.0005           
GAMMA = 0.99          
EPSILON_START = 1.0
EPSILON_MIN = 0.05
# 修改：稍微加快衰减 (0.999 -> 0.995)，因为现在的探索是"有效探索"，不需要浪费太多时间
EPSILON_DECAY = 0.995 

def train():
    env = MinesweeperEnv(GRID_SIZE, N_MINES)
    # 输入通道固定为2 (Visible层 + Value层)
    model = CNNDQN(GRID_SIZE, GRID_SIZE * GRID_SIZE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=10000) 
    
    epsilon = EPSILON_START
    win_count = 0
    
    print(f"--- 开始训练 Pro 版扫雷 AI ({GRID_SIZE}x{GRID_SIZE}) ---")
    print(">>> 优化策略：智能探索 (只随机点击未知区域)")

    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0) # (1, 2, H, W)
            
            # --- 关键修改：获取所有合法的动作（即未翻开的格子）---
            # state[0] 是 visible 层: -1 代表未知, 1 代表已知
            flatten_visible = state[0].flatten()
            valid_actions = [i for i, val in enumerate(flatten_visible) if val == -1]
            
            # 极少数情况（如赢了但done还没传出来）可能没有合法动作，兜底防报错
            if not valid_actions: 
                valid_actions = [0]

            # --- 动作选择策略 ---
            if random.random() < epsilon:
                # [核心修改] 随机模式下，只在"合法动作"里选
                # 这样模型永远不会浪费步数去点已经点开的格子
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor)
                    # 预测模式下，Mask 掉已知格子 (设为负无穷)
                    # 这里的逻辑和你原来的一样，是非常正确的
                    visible_mask = env.visible.flatten() == 1
                    q_values[0, visible_mask] = -float('inf')
                    action = torch.argmax(q_values).item()

            # 执行动作
            next_state, reward, done = env.step(action)
            
            if reward == 50.0:
                win_count += 1
            
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            # --- 训练步骤 ---
            if len(replay_buffer) > BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                b_state, b_action, b_reward, b_next_state, b_done = zip(*batch)
                
                b_state = torch.FloatTensor(np.array(b_state))
                b_next_state = torch.FloatTensor(np.array(b_next_state))
                b_action = torch.LongTensor(b_action).unsqueeze(1)
                b_reward = torch.FloatTensor(b_reward).unsqueeze(1)
                b_done = torch.FloatTensor(b_done).unsqueeze(1)
                
                # 计算当前 Q 值
                q_current = model(b_state).gather(1, b_action)
                
                # 计算目标 Q 值 (Double DQN 简化版)
                with torch.no_grad():
                    q_next = model(b_next_state).max(1)[0].unsqueeze(1)
                    q_target = b_reward + (GAMMA * q_next * (1 - b_done))
                
                loss = criterion(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    

        # Epsilon 衰减
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.1f}, Epsilon: {epsilon:.2f}, Wins(last 100): {win_count}")
            win_count = 0

    torch.save(model.state_dict(), "minesweeper_cnn_pro.pth")
    print("训练完成。")

if __name__ == "__main__":
    train()