import pygame
import numpy as np

COLOR_BG = (192, 192, 192)
COLOR_UNKNOWN = (160, 160, 160)
COLOR_REVEALED = (220, 220, 220)
COLOR_BORDER = (100, 100, 100)
COLOR_MINE = (255, 0, 0)
COLOR_FLAG = (255, 255, 0)      # 新增：旗子颜色 (黄色)
COLOR_TEXT = (0, 0, 0)
COLOR_HIGHLIGHT = (0, 255, 0)   # 点击高亮 (绿色)
COLOR_HIGHLIGHT_FLAG = (255, 165, 0) # 插旗高亮 (橙色)

NUM_COLORS = {
    1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 4: (0, 0, 128),
    5: (128, 0, 0), 6: (0, 128, 128), 7: (0, 0, 0), 8: (128, 128, 128)
}

class MinesweeperGUI:
    def __init__(self, env, cell_size=50):
        self.env = env
        self.grid_size = env.grid_size
        self.cell_size = cell_size
        self.width = self.grid_size * cell_size
        self.height = self.grid_size * cell_size
        
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("AI Minesweeper (Flags Version)")
        self.font = pygame.font.SysFont('Arial', int(cell_size * 0.6), bold=True)
        self.clock = pygame.time.Clock()

    def render(self, last_action_info=None, reveal_all=False):
        # last_action_info: (x, y, type) -> type 0 for click, 1 for flag
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill(COLOR_BG)

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, 
                                   self.cell_size, self.cell_size)
                
                # 1. 绘制已翻开区域
                if self.env.visible[x, y] == 1 or reveal_all:
                    if self.env.grid[x, y] == -1: # 雷
                        pygame.draw.rect(self.screen, COLOR_MINE, rect)
                        pygame.draw.circle(self.screen, (0,0,0), rect.center, self.cell_size//4)
                    else: # 数字
                        pygame.draw.rect(self.screen, COLOR_REVEALED, rect)
                        val = self.env.grid[x, y]
                        if val > 0:
                            text = self.font.render(str(val), True, NUM_COLORS.get(val, COLOR_TEXT))
                            text_rect = text.get_rect(center=rect.center)
                            self.screen.blit(text, text_rect)
                    pygame.draw.rect(self.screen, COLOR_BORDER, rect, 1)
                
                # 2. 绘制未知区域
                else:
                    pygame.draw.rect(self.screen, COLOR_UNKNOWN, rect)
                    pygame.draw.rect(self.screen, COLOR_BORDER, rect, 2)
                    
                    # 3. 绘制旗子 (只在未知区域显示)
                    if self.env.flags[x, y] == 1:
                        # 画个黄色小方块代表旗子
                        flag_rect = rect.inflate(-self.cell_size//3, -self.cell_size//3)
                        pygame.draw.rect(self.screen, COLOR_FLAG, flag_rect)

                # 4. 高亮 AI 操作
                if last_action_info:
                    lx, ly, ltype = last_action_info
                    if x == lx and y == ly:
                        color = COLOR_HIGHLIGHT if ltype == 0 else COLOR_HIGHLIGHT_FLAG
                        pygame.draw.rect(self.screen, color, rect, 4)

        pygame.display.flip()
        return True

    def close(self):
        pygame.quit()

    def wait_for_quit(self):
        print("Waiting for quit...")
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                    waiting = False
            self.clock.tick(15)
        self.close()