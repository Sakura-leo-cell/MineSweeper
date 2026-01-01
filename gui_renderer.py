import pygame
import numpy as np

# 颜色定义
COLOR_BG = (192, 192, 192)      # 背景灰
COLOR_UNKNOWN = (160, 160, 160) # 未知区域深灰
COLOR_REVEALED = (220, 220, 220)# 已知区域浅灰
COLOR_BORDER = (100, 100, 100)  # 边框
COLOR_MINE = (255, 0, 0)        # 雷是红色
COLOR_TEXT = (0, 0, 0)          # 默认字色
COLOR_HIGHLIGHT = (0, 255, 0)   # AI当前点击的高亮色

# 数字颜色 (仿 Windows 扫雷配色)
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

class MinesweeperGUI:
    def __init__(self, env, cell_size=50):
        self.env = env
        self.grid_size = env.grid_size
        self.cell_size = cell_size
        self.width = self.grid_size * cell_size
        self.height = self.grid_size * cell_size
        
        # 初始化 Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI Minesweeper Visualization")
        self.font = pygame.font.SysFont('Arial', int(cell_size * 0.6), bold=True)
        self.clock = pygame.time.Clock()

    def render(self, last_action_xy=None, reveal_all=False):
        # 处理退出事件，防止窗口卡死
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill(COLOR_BG)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                   self.cell_size, self.cell_size)
                
                # 绘制基础方块
                if self.env.visible[x, y] == 0 and not reveal_all:
                    # 未知区域
                    pygame.draw.rect(self.screen, COLOR_UNKNOWN, rect)
                    pygame.draw.rect(self.screen, COLOR_BORDER, rect, 2)
                else:
                    # 已知区域 (或者游戏结束全显示)
                    val = self.env.grid[x, y]
                    
                    if val == -1: # 雷
                        pygame.draw.rect(self.screen, COLOR_MINE, rect)
                        pygame.draw.circle(self.screen, (0,0,0), rect.center, self.cell_size//4)
                    else: # 安全数字
                        pygame.draw.rect(self.screen, COLOR_REVEALED, rect)
                        if val > 0:
                            text = self.font.render(str(val), True, NUM_COLORS.get(val, COLOR_TEXT))
                            text_rect = text.get_rect(center=rect.center)
                            self.screen.blit(text, text_rect)
                    
                    pygame.draw.rect(self.screen, COLOR_BORDER, rect, 1)

                # 高亮 AI 刚刚点击的位置
                if last_action_xy and x == last_action_xy[0] and y == last_action_xy[1]:
                    pygame.draw.rect(self.screen, COLOR_HIGHLIGHT, rect, 3)

        pygame.display.flip()
        return True

    def close(self):
        pygame.quit()

    def wait_for_quit(self):
        print("按窗口右上角 X 或按任意键退出可视化...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
            self.clock.tick(15)
        self.close()