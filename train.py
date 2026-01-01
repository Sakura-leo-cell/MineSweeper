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
EPSILON_DECAY = 0.995 

def train():
    # 1. 设置设备 (优先使用 CUDA)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 训练开始 (Flags Enabled) ---")
    print(f"正在使用计算设备: {device}")

    env = MinesweeperEnv(GRID_SIZE, N_MINES)
    
    # 输出维度翻倍： [0..N^2-1] 是点击, [N^2..2N^2-1] 是插旗
    n_actions = GRID_SIZE * GRID_SIZE
    total_actions = n_actions * 2
    
    # 2. 将模型移动到 GPU
    model = CNNDQN(GRID_SIZE, total_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    replay_buffer = deque(maxlen=10000) 
    
    epsilon = EPSILON_START
    win_count = 0
    
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # 3. 推理时的状态移动到 GPU
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # --- 动作 Mask 逻辑 (保持在 CPU 上计算即可，因为逻辑简单且涉及列表操作) ---
            flatten_visible = state[0].flatten() # -1 unknown, 1 known
            flatten_flags = state[2].flatten()   # 1 flagged, 0 not
            
            valid_click_indices = [i for i, v in enumerate(flatten_visible) 
                                   if v == -1 and flatten_flags[i] == 0]
            
            valid_flag_indices = [i + n_actions for i, v in enumerate(flatten_visible) 
                                  if v == -1] 
            
            valid_actions = valid_click_indices + valid_flag_indices
            
            if not valid_actions: valid_actions = [0] # 兜底

            # --- 动作选择 ---
            if random.random() < epsilon:
                action = random.choice(valid_actions)
            else:
                with torch.no_grad():
                    q_values = model(state_tensor) # [1, 72] (此时 q_values 在 GPU 上)
                    
                    # 构造 Mask (在 GPU 上创建 Tensor)
                    full_mask = torch.ones_like(q_values) * -float('inf')
                    
                    # 只有有效动作的位置填入真实 Q 值
                    full_mask[0, valid_actions] = q_values[0, valid_actions]
                    
                    action = torch.argmax(full_mask).item()

            # 执行动作 (Environment 依然在 CPU 上运行)
            next_state, reward, done = env.step(action)
            
            if reward == 50.0: win_count += 1
            
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            
            # --- 训练步骤 ---
            if len(replay_buffer) > BATCH_SIZE:
                batch = random.sample(replay_buffer, BATCH_SIZE)
                b_state, b_action, b_reward, b_next_state, b_done = zip(*batch)
                
                # 4. 将训练 Batch 移动到 GPU
                b_state = torch.FloatTensor(np.array(b_state)).to(device)
                b_next_state = torch.FloatTensor(np.array(b_next_state)).to(device)
                b_action = torch.LongTensor(b_action).unsqueeze(1).to(device)
                b_reward = torch.FloatTensor(b_reward).unsqueeze(1).to(device)
                b_done = torch.FloatTensor(b_done).unsqueeze(1).to(device)
                
                # 计算 Current Q (在 GPU 上)
                q_current = model(b_state).gather(1, b_action)
                
                with torch.no_grad():
                    # 计算 Target Q (在 GPU 上)
                    q_next = model(b_next_state).max(1)[0].unsqueeze(1)
                    q_target = b_reward + (GAMMA * q_next * (1 - b_done))
                
                loss = criterion(q_current, q_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()    

        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
            
        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.1f}, Eps: {epsilon:.2f}, Wins: {win_count}")
            win_count = 0

    # 保存模型 (保存前建议转回 CPU，方便推理时加载，或者推理代码也要适配 GPU)
    torch.save(model.cpu().state_dict(), "minesweeper_cnn_flags.pth")
    print("训练完成 (含Flag机制)。")

if __name__ == "__main__":
    train()