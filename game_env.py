import numpy as np
import random

class MinesweeperEnv:
    def __init__(self, grid_size=6, n_mines=6):
        self.grid_size = grid_size
        self.n_mines = n_mines
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.visible = np.zeros((grid_size, grid_size), dtype=int)
        self.flags = np.zeros((grid_size, grid_size), dtype=int) # 新增：插旗层
        self.reset()

    def reset(self):
        self.grid.fill(0)
        self.visible.fill(0)
        self.flags.fill(0) # 重置旗子
        
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
        # 动作空间现在是 2 * grid_size * grid_size
        # 0 ~ N^2-1 : 点击
        # N^2 ~ 2N^2-1 : 插旗
        total_cells = self.grid_size * self.grid_size
        
        if action < total_cells:
            # === 动作类型：点击 (Click) ===
            x, y = divmod(action, self.grid_size)
            
            # 1. 如果这里插了旗，为了安全起见，禁止点击（或者给惩罚）
            # 这里我们设定：AI如果想点旗子，直接无效并给微小惩罚，强制它先取消旗子
            if self.flags[x, y] == 1:
                return self._get_state(), -0.5, False

            # 2. 重复点击已探索区域
            if self.visible[x, y] == 1:
                return self._get_state(), -1.0, False 

            # 3. 踩雷
            if self.grid[x, y] == -1:
                self.visible[x, y] = 1
                return self._get_state(), -20.0, True 

            # 4. 点击安全格
            visible_before = np.sum(self.visible)
            self.visible[x, y] = 1
            if self.grid[x, y] == 0:
                self._flood_fill(x, y)
            visible_after = np.sum(self.visible)
            
            newly_revealed = visible_after - visible_before
            reward = 0.1 + (newly_revealed * 0.5)

            # 检查胜利 (胜利条件：所有非雷区都翻开了)
            if np.sum(self.visible) == (total_cells - self.n_mines):
                return self._get_state(), 50.0, True

            return self._get_state(), reward, False

        else:
            # === 动作类型：插旗 (Flag) ===
            flag_idx = action - total_cells
            x, y = divmod(flag_idx, self.grid_size)

            # 1. 只能在未知区域插旗
            if self.visible[x, y] == 1:
                return self._get_state(), -1.0, False # 浪费步数

            # 2. 切换插旗状态 (如果已有旗则取消，没有则插上)
            if self.flags[x, y] == 1:
                self.flags[x, y] = 0
                return self._get_state(), -0.1, False # 取消旗子轻微惩罚（避免反复横跳）
            else:
                self.flags[x, y] = 1
                # --- 关键：立即验证插旗是否正确 ---
                if self.grid[x, y] == -1:
                    # 正确插旗！给予奖励！
                    return self._get_state(), 1.0, False
                else:
                    # 错误插旗！这是个安全格，你却插了旗
                    # 为了不误导自己，强制把旗拔掉，并给予惩罚
                    self.flags[x, y] = 0 
                    return self._get_state(), -1.0, False

    def _flood_fill(self, x, y):
        queue = [(x, y)]
        visited = set([(x,y)])
        while queue:
            cx, cy = queue.pop(0)
            for i in range(max(0, cx-1), min(self.grid_size, cx+2)):
                for j in range(max(0, cy-1), min(self.grid_size, cy+2)):
                    if (i, j) not in visited:
                        visited.add((i, j))
                        # 自动翻开时，如果有旗子，自动把旗子去掉（因为已经证实是安全的了）
                        if self.flags[i, j] == 1:
                            self.flags[i, j] = 0
                        
                        self.visible[i, j] = 1 
                        if self.grid[i, j] == 0:
                            queue.append((i, j))

    def _get_state(self):
        # --- 修改：三通道输入 ---
        # Channel 0: 可视状态 (-1: 未知, 1: 已知)
        # Channel 1: 数值状态 (归一化数字)
        # Channel 2: 插旗状态 (1: 有旗, 0: 无旗) -> 这一层非常重要，让AI记住它判断哪里是雷
        
        channel_visible = np.zeros((self.grid_size, self.grid_size))
        channel_value = np.zeros((self.grid_size, self.grid_size))
        channel_flag = self.flags.copy()
        
        channel_visible[self.visible == 0] = -1
        channel_visible[self.visible == 1] = 1
        
        mask = (self.visible == 1)
        channel_value[mask] = self.grid[mask] / 8.0
        
        state = np.array([channel_visible, channel_value, channel_flag])
        return state