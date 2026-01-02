import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import os
import glob
from collections import deque
from game_env import MinesweeperEnv
from model import CNNDQN

# ===========================
# 🛠️ 核心配置区域
# ===========================
# 1. 你现在要训练的规格
CURRENT_GRID_SIZE = 9     # 目标尺寸 (想练 9x9 就改这里)
CURRENT_N_MINES = 10       # 目标雷数 (9x9 建议 10 雷)

# 2. 老师配置 (迁移学习)
TEACHER_FOLDER = "saved_models"
TEACHER_GRID_SIZE = 6     # 想继承谁的经验？(通常填上一级尺寸)

# 3. 训练参数 (精调版)
BATCH_SIZE = 512          # 🔥 加大 Batch，让训练更稳
GAMMA = 0.99
EPS_START = 0.5           # 如果是继承老模型，不用从 1.0 开始瞎猜
EPS_END = 0.1
EPS_DECAY = 100000       
TARGET_UPDATE = 50      
MEMORY_SIZE = 100000    
LR = 0.0001              # 🔥 极低学习率，适合精细打磨
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

def find_best_teacher(folder, size):
    """自动寻找文件夹里最新、最强的老师模型"""
    if not os.path.exists(folder): return None
    # 找 .pth 或 .tar
    search_pattern = os.path.join(folder, f"*{size}x{size}*.pth")
    files = glob.glob(search_pattern)
    if not files:
        search_pattern = os.path.join(folder, f"*{size}x{size}*.tar")
        files = glob.glob(search_pattern)
    
    if not files:
        print(f"⚠️ 没找到 {size}x{size} 的老师模型，将从零开始。")
        return None

    # 按修改时间找最新的
    latest_file = max(files, key=os.path.getmtime)
    print(f"🕵️ 锁定最佳老师: {latest_file}")
    return latest_file

def load_transfer_model(model, teacher_path):
    """只继承卷积层(眼睛)，重置全连接层(大脑)"""
    print(f"🔄 正在提取 {teacher_path} 的视觉经验...")
    try:
        checkpoint = torch.load(teacher_path, map_location=device)
        teacher_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        student_dict = model.state_dict()
        
        # 只加载形状匹配的层 (卷积层)
        pretrained_dict = {k: v for k, v in teacher_dict.items() if k in student_dict and v.size() == student_dict[k].size()}
        
        student_dict.update(pretrained_dict)
        model.load_state_dict(student_dict)
        print(f"✅ 成功继承 {len(pretrained_dict)} 层参数 (Conv层已保留，FC层已重置)")
    except Exception as e:
        print(f"❌ 继承失败: {e}")

def load_checkpoint(model, optimizer, filename):
    """加载断点续训"""
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
    
    # 1. 优先断点续训 (自己接自己)
    steps_done, is_resume = load_checkpoint(policy_net, optimizer, CHECKPOINT_PATH)
    
    # 2. 如果没断点，尝试找老师 (迁移学习)
    if not is_resume:
        teacher_path = find_best_teacher(TEACHER_FOLDER, TEACHER_GRID_SIZE)
        if teacher_path:
            load_transfer_model(policy_net, teacher_path)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    print(f"🚀 开始训练 | 目标: {CURRENT_GRID_SIZE}x{CURRENT_GRID_SIZE} | Batch: {BATCH_SIZE} | LR: {LR}")
    
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