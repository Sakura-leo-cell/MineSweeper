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
# 🛠️ 核心配置区域 (文件夹继承版)
# ===========================
# 1. 当前你要训练的目标尺寸
CURRENT_GRID_SIZE = 6    # 目标：训练 9x9
CURRENT_N_MINES = 4      # 目标雷数

# 2. 老师模型在哪里？(迁移学习配置)
TEACHER_FOLDER = "saved_models"   # 老师们住的文件夹
TEACHER_GRID_SIZE = 6             # 你想继承哪种尺寸的经验？(例如：从 6x6 继承)

# 3. 训练参数
BATCH_SIZE = 256        
GAMMA = 0.99
# 迁移学习时，如果是从小图转大图，可以设为 0.5；如果是同尺寸继续练，设为 0.3
EPS_START = 0.5         
EPS_END = 0.05
EPS_DECAY = 50000       # 稍微慢一点，适应新环境
TARGET_UPDATE = 50      
MEMORY_SIZE = 100000    
LR = 0.00001            # 微调阶段，学习率要低
NUM_EPISODES = 500000   # 大图需要更多时间

# 自动生成保存路径
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
    """
    🕵️ 智能星探：在文件夹里找最好的老师
    策略：寻找文件名包含 dqn_{size}x{size} 的文件，并返回修改时间最新的那个
    """
    if not os.path.exists(folder):
        return None
    
    # 搜索模式：例如 saved_models/dqn_6x6*.pth
    # 这样可以兼容 dqn_6x6.pth, dqn_6x6_v2.pth 等命名
    search_pattern = os.path.join(folder, f"dqn_{size}x{size}*.pth")
    files = glob.glob(search_pattern)
    
    if not files:
        # 再试着找找 checkpoint
        search_pattern = os.path.join(folder, f"checkpoint_{size}x{size}*.tar")
        files = glob.glob(search_pattern)
    
    if not files:
        print(f"⚠️ 在 {folder} 里没找到 {size}x{size} 的老师模型。")
        return None

    # 按修改时间排序，取最新的
    latest_file = max(files, key=os.path.getmtime)
    print(f"🕵️ 自动锁定最佳老师: {latest_file}")
    return latest_file

def load_transfer_model(model, teacher_path):
    """继承逻辑：保留卷积层 (眼睛)，重置全连接层 (大脑)"""
    print(f"🔄 正在提取 {teacher_path} 的视觉经验...")
    
    try:
        checkpoint = torch.load(teacher_path, map_location=device)
        
        # 兼容性处理：如果是 checkpoint 字典，取 model_state_dict，如果是纯权重直接用
        if 'model_state_dict' in checkpoint:
            teacher_dict = checkpoint['model_state_dict']
        else:
            teacher_dict = checkpoint
            
        student_dict = model.state_dict()
        
        # 核心算法：只加载形状完全匹配的层 (即卷积层)
        pretrained_dict = {k: v for k, v in teacher_dict.items() if k in student_dict and v.size() == student_dict[k].size()}
        
        model_layers = len(student_dict)
        loaded_layers = len(pretrained_dict)
        
        student_dict.update(pretrained_dict)
        model.load_state_dict(student_dict)
        
        print(f"📊 基因融合完成: 继承了 {loaded_layers}/{model_layers} 层参数 (主要是卷积层)")
        print("🧠 全连接层已重置，准备适应新地图...")
        
    except Exception as e:
        print(f"❌ 继承失败: {e}")
        return

def load_checkpoint(model, optimizer, filename):
    """断点续训加载"""
    if os.path.exists(filename):
        print(f"🔄 发现当前任务进度: {filename}")
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
    
    # 1. 优先尝试断点续训 (继续练当前的 9x9)
    steps_done, is_resume = load_checkpoint(policy_net, optimizer, CHECKPOINT_PATH)
    
    # 2. 如果没有断点，则去文件夹里找个最好的老师 (从 6x6 继承)
    if not is_resume:
        teacher_path = find_best_teacher(TEACHER_FOLDER, TEACHER_GRID_SIZE)
        if teacher_path:
            load_transfer_model(policy_net, teacher_path)
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    print(f"🚀 开始训练 V2 | 目标: {CURRENT_GRID_SIZE}x{CURRENT_GRID_SIZE} | 雷数: {CURRENT_N_MINES}")
    
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
                    
                    with torch.no_grad():
                        next_state_q = policy_net(b_next_state)
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