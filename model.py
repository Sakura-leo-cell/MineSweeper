import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNDQN(nn.Module):
    def __init__(self, grid_size, output_dim):
        super(CNNDQN, self).__init__()
        self.grid_size = grid_size
        self.output_dim = output_dim
        
        # --- 卷积层 (视觉系统) ---
        # Layer 1: 使用 5x5 大卷积核，直接看清 "邻居的邻居" (关键优化)
        # padding=2 保证输出尺寸不变
        self.conv1 = nn.Conv2d(2, 64, kernel_size=5, padding=2)
        
        # Layer 2 & 3: 继续提取深层逻辑特征
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # 计算全连接层的输入维度
        self.flatten_dim = 128 * grid_size * grid_size
        
        # --- Dueling DQN 结构 (决策系统) ---
        
        # 1. 价值流 (V): 评估局面好坏
        self.value_stream = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        
        # 2. 优势流 (A): 评估动作好坏
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.flatten_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.reshape(x.size(0), -1) # Flatten
        
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Q(s,a) = V(s) + (A(s,a) - mean(A))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals