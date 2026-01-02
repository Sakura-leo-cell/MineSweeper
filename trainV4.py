import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os
from collections import deque
from game_env import MinesweeperEnv
from model import CNNDQN

# ===========================
# 🛠️ 核心配置区域 (独立训练版)
# ===========================
# 1. 训练规格
CURRENT_GRID_SIZE = 9     
CURRENT_N_MINES = 10       

# 2. 训练参数 (已针对独立训练优化)
BATCH_SIZE = 512          
GAMMA = 0.99
EPS_START = 0.9           # 🔥 独立训练必须从高探索开始 (90%乱走)
EPS_END = 0.1             # 最低保持 10% 的随机性
EPS_DECAY = 100000        # 🔥 加速衰减：10万步内完成新手村教学
TARGET_UPDATE = 50      
MEMORY_SIZE = 100000    
LR = 0.0001               # 🔥 关键：学习率设为 1e-4，让它学得快一点！
NUM_EPISODES = 500000000   

# 自动路径
MODEL_DIR = "saved_models"
NEW_MODEL_PATH = os.path.join(MODEL_DIR, f"dqn_{CURRENT_GRID_SIZE}x{CURRENT_GRID_SIZE}.pth")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, f"checkpoint_{CURRENT_GRID_SIZE}x{CURRENT_GRID_SIZE}.tar")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

def save_checkpoint(model, optimizer, steps_done, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps_done': steps_done
    }
    torch.save(checkpoint, filename)
    torch.save(model.state_dict(), NEW_MODEL_PATH)

def load_checkpoint(model, optimizer, filename):
    """只负责加载当前进度的存档"""
    if os.path.exists(filename):
        print(f"🔄 发现存档: {filename}")
        try:
            checkpoint = torch.load(filename, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            steps = checkpoint['steps_done']
            print(f"✅ 成功恢复! 从第 {steps} 步继续。")
            return steps, True
        except:
            print("⚠️ 存档损坏，重新开始")
    return 0, False

def train():
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    
    env = MinesweeperEnv(CURRENT_GRID_SIZE, CURRENT_N_MINES)
    n_actions = CURRENT_GRID_SIZE * CURRENT_GRID_SIZE
    
    policy_net = CNNDQN(CURRENT_GRID_SIZE, n_actions).to(device)
    target_net = CNNDQN(CURRENT_GRID_SIZE, n_actions).to(device)
    
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayMemory(MEMORY_SIZE)
    
    # 🔥 核心修改：只加载断点，不再寻找老师
    steps_done, is_resume = load_checkpoint(policy_net, optimizer, CHECKPOINT_PATH)
    
    if not is_resume:
        print(f"✨ 初始化新训练 (LR={LR}) | 无迁移 | 一切从零开始")
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    print(f"🚀 开始训练 | 目标: {CURRENT_GRID_SIZE}x{CURRENT_GRID_SIZE} | Batch: {BATCH_SIZE}")
    
    recent_wins = deque(maxlen=100)

    try:
        for i_episode in range(NUM_EPISODES):
            state = env.reset()
            done = False
            total_reward = 0
            is_win = False

            while not done:
                eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                                np.exp(-1. * steps_done / EPS_DECAY)
                steps_done += 1
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # --- Action Masking ---
                visible_flat = env.visible.flatten()
                valid_moves = [i for i in range(n_actions) if visible_flat[i] == 0]
                if not valid_moves: break 

                action = 0
                if random.random() < eps_threshold:
                    action = random.choice(valid_moves)
                else:
                    with torch.no_grad():
                        q_values = policy_net(state_tensor)
                        visible_mask = torch.tensor(visible_flat, device=device, dtype=torch.bool)
                        q_values[0, visible_mask] = -float('inf')
                        action = q_values.max(1)[1].item()
                
                next_state, reward, done = env.step(action)
                total_reward += reward
                if reward == 20.0: is_win = True

                memory.push(state, action, reward, next_state, done)
                state = next_state
                
                if len(memory) >= BATCH_SIZE:
                    transitions = memory.sample(BATCH_SIZE)
                    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
                    
                    b_state = torch.FloatTensor(np.array(batch_state)).to(device)
                    b_action = torch.LongTensor(batch_action).unsqueeze(1).to(device)
                    b_reward = torch.FloatTensor(batch_reward).unsqueeze(1).to(device)
                    b_next_state = torch.FloatTensor(np.array(batch_next_state)).to(device)
                    b_done = torch.FloatTensor(batch_done).unsqueeze(1).to(device)
                    
                    # --- Double DQN + Mask Fix ---
                    with torch.no_grad():
                        next_state_q = policy_net(b_next_state)
                        # Mask next state invalid moves
                        invalid_mask = b_next_state[:, 0, :, :].reshape(BATCH_SIZE, -1) > -0.5
                        next_state_q[invalid_mask] = -float('inf')
                        best_action_indices = next_state_q.max(1)[1].unsqueeze(1)
                        max_next_q = target_net(b_next_state).gather(1, best_action_indices)
                    
                    current_q = policy_net(b_state).gather(1, b_action)
                    expected_q = b_reward + (GAMMA * max_next_q * (1 - b_done))
                    
                    loss = nn.SmoothL1Loss()(current_q, expected_q)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    optimizer.step()

            recent_wins.append(1 if is_win else 0)
            
            if i_episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            if (i_episode + 1) % 200 == 0:
                win_rate = sum(recent_wins) / len(recent_wins) * 100
                print(f"Ep {i_episode+1} | R: {total_reward:.2f} | Win: {win_rate:.1f}% | Eps: {eps_threshold:.2f}")
                save_checkpoint(policy_net, optimizer, steps_done, CHECKPOINT_PATH)

    except KeyboardInterrupt:
        print("\n🛑 保存进度...")
        save_checkpoint(policy_net, optimizer, steps_done, CHECKPOINT_PATH)

if __name__ == "__main__":
    train()