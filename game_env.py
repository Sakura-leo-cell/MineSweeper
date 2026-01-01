import numpy as np
import random

class MinesweeperEnv:
    def __init__(self, grid_size=6, n_mines=6):
        self.grid_size = grid_size
        self.n_mines = n_mines
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.visible = np.zeros((grid_size, grid_size), dtype=int) 
        self.reset()

    def reset(self):
        self.grid.fill(0)
        self.visible.fill(0)
        # 简单起见，随机布雷（为了训练稳定性，暂时不搞"首点无雷"逻辑，依赖AI去学）
        mines_placed = 0
        while mines_placed < self.n_mines:
            x, y = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            if self.grid[x, y] != -1:
                self.grid[x, y] = -1
                mines_placed += 1
        
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] == -1: continue
                self.grid[x, y] = self._count_mines(x, y)
        
        return self._get_state()

    def _count_mines(self, x, y):
        count = 0
        for i in range(max(0, x-1), min(self.grid_size, x+2)):
            for j in range(max(0, y-1), min(self.grid_size, y+2)):
                if self.grid[i, j] == -1: count += 1
        return count

    def step(self, action):
        x, y = divmod(action, self.grid_size)
        
        # 1. 重复点击惩罚 (稍微降低一点惩罚，别让它太害怕)
        if self.visible[x, y] == 1:
            return self._get_state(), -1.0, False # 之前是 -5

        # 2. 踩雷
        if self.grid[x, y] == -1:
            self.visible[x, y] = 1
            return self._get_state(), -5.0, True # 之前是 -10，稍微宽容点

        # 3. 点击安全格
        # 计算这次操作前后的可见数量，以此计算"开疆拓土"的奖励
        visible_before = np.sum(self.visible)
        self.visible[x, y] = 1
        if self.grid[x, y] == 0:
            self._flood_fill(x, y)
        visible_after = np.sum(self.visible)
        
        # 奖励机制的核心修改：
        # 基础奖励 0.3 + 每多翻开一个格子额外奖励 0.5
        # 这会极大地鼓励 AI 去找 0 (因为点 0 会连锁翻开很多)
        newly_revealed = visible_after - visible_before
        reward = 0.3 + (newly_revealed * 0.5)

        # 4. 胜利
        if np.sum(self.visible) == (self.grid_size**2 - self.n_mines):
            return self._get_state(), 50.0, True # 巨额奖励

        return self._get_state(), reward, False

    def _flood_fill(self, x, y):
        queue = [(x, y)]
        visited = set([(x,y)])
        while queue:
            cx, cy = queue.pop(0)
            for i in range(max(0, cx-1), min(self.grid_size, cx+2)):
                for j in range(max(0, cy-1), min(self.grid_size, cy+2)):
                    if (i, j) not in visited:
                        visited.add((i, j))
                        self.visible[i, j] = 1 # 标记可见
                        if self.grid[i, j] == 0:
                            queue.append((i, j))

    def _get_state(self):
        # --- 关键修改：双通道输入 ---
        # Channel 0: 可视状态 (-1: 未知, 0: 已知) -> 方便卷积区分边界
        # Channel 1: 数值状态 (显示归一化的数字，未知区域填 0)
        
        channel_visible = np.zeros((self.grid_size, self.grid_size))
        channel_value = np.zeros((self.grid_size, self.grid_size))
        
        # 未知的地方：Visible层设为 -1
        channel_visible[self.visible == 0] = -1
        # 已知的地方：Visible层设为 0
        channel_visible[self.visible == 1] = 1 # 或者 0.5, 只要区分开就行
        
        # 数值层：只填已知区域的数字
        mask = (self.visible == 1)
        channel_value[mask] = self.grid[mask] / 8.0
        
        # 堆叠成 (2, H, W)
        state = np.array([channel_visible, channel_value])
        return state