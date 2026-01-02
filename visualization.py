import pygame
import torch
import numpy as np
import time
import os
import sys
from game_env import MinesweeperEnv
from model import CNNDQN

# ===========================
# 🛠️ 配置区域
# ===========================
GRID_SIZE = 6          # 你的地图尺寸 (如果要看 9x9 请改这里)
N_MINES = 4            # 雷数
MODEL_PATH = f"saved_models/dqn_{GRID_SIZE}x{GRID_SIZE}.pth"

# 界面设置
CELL_SIZE = 60         # 每个格子的大小 (像素)
HEADER_HEIGHT = 60     # 顶部状态栏高度
BG_COLOR = (192, 192, 192)
GRID_COLOR = (128, 128, 128)
HIDDEN_COLOR = (220, 220, 220)
REVEALED_COLOR = (180, 180, 180)
TEXT_COLOR = (0, 0, 0)
MINE_COLOR = (255, 0, 0)
HIGHLIGHT_COLOR = (0, 255, 0) # AI 当前操作的高亮颜色

# 数字颜色 (仿照 Windows 经典扫雷)
NUM_COLORS = {
    1: (0, 0, 255),    # 蓝
    2: (0, 128, 0),    # 绿
    3: (255, 0, 0),    # 红
    4: (0, 0, 128),    # 深蓝
    5: (128, 0, 0),    # 深红
    6: (0, 128, 128),  # 青
    7: (0, 0, 0),      # 黑
    8: (128, 128, 128) # 灰
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MinesweeperVis:
    def __init__(self):
        pygame.init()
        self.width = GRID_SIZE * CELL_SIZE
        self.height = GRID_SIZE * CELL_SIZE + HEADER_HEIGHT
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(f"AI Minesweeper Bot - {GRID_SIZE}x{GRID_SIZE}")
        
        self.font = pygame.font.SysFont('arial', int(CELL_SIZE * 0.6), bold=True)
        self.ui_font = pygame.font.SysFont('arial', 20)
        
        self.clock = pygame.time.Clock()
        self.env = MinesweeperEnv(GRID_SIZE, N_MINES)
        self.model = self.load_model()
        
        self.delay = 500  # 毫秒，每步延迟
        self.paused = False
        self.running = True
        
        self.reset_game()

    def load_model(self):
        print(f"Loading model from {MODEL_PATH}...")
        model = CNNDQN(GRID_SIZE, GRID_SIZE*GRID_SIZE).to(device)
        if os.path.exists(MODEL_PATH):
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print("✅ Model loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                print("Using random weights (AI will be stupid).")
        else:
            print("⚠️ Model file not found. Running with random weights.")
        model.eval()
        return model

    def reset_game(self):
        self.state = self.env.reset()
        self.done = False
        self.last_action = None
        self.steps = 0
        self.result_text = ""
        self.result_timer = 0

    def get_ai_action(self):
        state_tensor = torch.FloatTensor(self.state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            
            # Action Masking (只看没翻开的)
            visible_mask = torch.tensor(self.env.visible.flatten(), device=device, dtype=torch.bool)
            q_values[0, visible_mask] = -float('inf')
            
            action = torch.argmax(q_values).item()
        return action

    def draw_cell(self, r, c):
        x = c * CELL_SIZE
        y = r * CELL_SIZE + HEADER_HEIGHT
        rect = pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)
        
        val = self.env.grid[r, c]
        is_visible = self.env.visible[r, c]
        
        # 1. 绘制背景
        if is_visible:
            if val == -1: # 雷
                pygame.draw.rect(self.screen, MINE_COLOR, rect)
            else:
                pygame.draw.rect(self.screen, REVEALED_COLOR, rect)
        else:
            pygame.draw.rect(self.screen, HIDDEN_COLOR, rect)
            # 绘制立体的边缘效果
            pygame.draw.line(self.screen, (255,255,255), (x,y), (x+CELL_SIZE, y), 2)
            pygame.draw.line(self.screen, (255,255,255), (x,y), (x, y+CELL_SIZE), 2)
            pygame.draw.line(self.screen, (100,100,100), (x+CELL_SIZE-2,y), (x+CELL_SIZE-2, y+CELL_SIZE), 2)
            pygame.draw.line(self.screen, (100,100,100), (x,y+CELL_SIZE-2), (x+CELL_SIZE, y+CELL_SIZE-2), 2)

        pygame.draw.rect(self.screen, GRID_COLOR, rect, 1)

        # 2. 绘制内容
        if is_visible:
            if val == -1:
                # 画个简单的雷 (圆圈)
                pygame.draw.circle(self.screen, (0,0,0), (x+CELL_SIZE//2, y+CELL_SIZE//2), CELL_SIZE//4)
            elif val > 0:
                text = self.font.render(str(val), True, NUM_COLORS.get(val, (0,0,0)))
                text_rect = text.get_rect(center=rect.center)
                self.screen.blit(text, text_rect)

        # 3. 高亮 AI 刚刚点击的格子
        if self.last_action is not None:
            lr, lc = divmod(self.last_action, GRID_SIZE)
            if lr == r and lc == c:
                pygame.draw.rect(self.screen, HIGHLIGHT_COLOR, rect, 4)

    def draw_ui(self):
        pygame.draw.rect(self.screen, (50, 50, 50), (0, 0, self.width, HEADER_HEIGHT))
        
        status = f"Steps: {self.steps} | Mines: {N_MINES}"
        if self.paused: status += " | PAUSED"
        
        text = self.ui_font.render(status, True, (255, 255, 255))
        self.screen.blit(text, (10, 10))
        
        help_text = self.ui_font.render("Space:Pause | Up/Down:Speed | R:Reset", True, (200, 200, 200))
        self.screen.blit(help_text, (10, 35))

        if self.done:
            res_color = (0, 255, 0) if "WIN" in self.result_text else (255, 0, 0)
            res_surf = self.font.render(self.result_text, True, res_color)
            res_rect = res_surf.get_rect(center=(self.width//2, self.height//2))
            # 画个背景框让字清楚点
            bg_rect = res_rect.inflate(20, 20)
            pygame.draw.rect(self.screen, (50,50,50, 200), bg_rect)
            self.screen.blit(res_surf, res_rect)

    def run(self):
        last_move_time = 0
        
        while self.running:
            current_time = pygame.time.get_ticks()
            
            # --- 1. 事件处理 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_UP:
                        self.delay = max(50, self.delay - 50) # 加速
                    elif event.key == pygame.K_DOWN:
                        self.delay += 50 # 减速
                    elif event.key == pygame.K_r:
                        self.reset_game()

            # --- 2. AI 逻辑 ---
            if not self.paused and not self.done:
                if current_time - last_move_time > self.delay:
                    action = self.get_ai_action()
                    self.last_action = action
                    
                    self.state, reward, self.done = self.env.step(action)
                    self.steps += 1
                    last_move_time = current_time
                    
                    # 检查结果
                    if self.done:
                        # 胜利条件 (所有非雷都翻开了)
                        if np.sum(self.env.visible) == (GRID_SIZE**2 - N_MINES):
                            self.result_text = "YOU WIN!"
                            print("🎉 AI 赢了！")
                        else:
                            self.result_text = "BOOM! LOST"
                            print("💥 AI 踩雷了...")
                        self.result_timer = current_time

            # --- 3. 自动重开逻辑 ---
            if self.done and current_time - self.result_timer > 2000: # 结束后停顿2秒
                self.reset_game()

            # --- 4. 绘制 ---
            self.screen.fill(BG_COLOR)
            
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    self.draw_cell(r, c)
            
            self.draw_ui()
            pygame.display.flip()
            self.clock.tick(240) # 保持 240 FPS 的刷新率

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    vis = MinesweeperVis()
    vis.run()