# harness/astar.py
import numpy as np
import heapq

class GridMap:
    """從 world 檔案硬編碼障礙物，建立 2D occupancy grid"""

    def __init__(self, resolution=0.25):
        self.res = resolution          # 每格 0.25m
        self.x_min, self.x_max = -10, 10
        self.y_min, self.y_max = -10, 10

        # grid 大小
        self.nx = int((self.x_max - self.x_min) / self.res)
        self.ny = int((self.y_max - self.y_min) / self.res)
        self.grid = np.zeros((self.nx, self.ny), dtype=bool)

        self._add_obstacles()

    def _world_to_grid(self, x, y):
        gx = int((x - self.x_min) / self.res)
        gy = int((y - self.y_min) / self.res)
        return gx, gy

    def _mark_obstacle(self, x, y, radius=1.0):
        """在 (x,y) 周圍 radius 範圍標記為障礙"""
        gx, gy = self._world_to_grid(x, y)
        r = int(radius / self.res) + 1
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                nx, ny = gx+dx, gy+dy
                if 0 <= nx < self.nx and 0 <= ny < self.ny:
                    self.grid[nx][ny] = True

    def _add_obstacles(self):
        # 四面牆（邊界加 buffer）
        for i in range(self.nx):
            for j in range(1): self.grid[i][j] = True # 2-> 只標記牆本身，buffer 1格
            for j in range(self.ny-1, self.ny): self.grid[i][j] = True # 2-> 1
        for j in range(self.ny):
            for i in range(1): self.grid[i][j] = True
            for i in range(self.nx-1, self.nx): self.grid[i][j] = True

        # 四個角落箱子（1x1x1m，飛行高度 >1m 可以飛越，但保守起見標記）
        boxes = [(9.4,-9.4), (9.4,9.3), (-9.3,9.3), (-9.2,-9.4)]
        for x, y in boxes:
            self._mark_obstacle(x, y, radius=0.75) # 1.5 -> 1.0

        # Dumpster
        self._mark_obstacle(1.7, -7.0, radius=0.75) # 2.0 -> 1.0

        # Construction Cones（小，縮到 0.3）
        cones = [
            (3.44, 0.67), (2.94, 0.92), (2.41, 1.07), (1.91, 1.23),
            (1.41, 1.36), (0.92, 1.47), (0.38, 1.55), (-0.14, 1.60),
            (-0.67, 1.77), (-1.19, 1.96), (-1.72, 2.19), (-2.25, 2.45)
        ]
        for x, y in cones:
            self._mark_obstacle(x, y, radius=0.2) # 0.5 -> 0.3

    def is_free(self, x, y):
        gx, gy = self._world_to_grid(x, y)
        if gx < 0 or gx >= self.nx or gy < 0 or gy >= self.ny:
            return False
        return not self.grid[gx][gy]


def astar(grid_map: GridMap, start_xy, goal_xy):
    """
    在 2D grid 上跑 A*，回傳路徑長度（公尺）
    start_xy, goal_xy: (x, y) 世界座標
    """
    sx, sy = grid_map._world_to_grid(*start_xy)
    gx, gy = grid_map._world_to_grid(*goal_xy)

    def heuristic(a, b):
        return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2) * grid_map.res

    open_set = [(0, (sx, sy))]
    came_from = {}
    g_score = {(sx, sy): 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == (gx, gy):
            # 還原路徑長度
            path_len = g_score[current] * grid_map.res
            return path_len

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            neighbor = (current[0]+dx, current[1]+dy)
            if not (0 <= neighbor[0] < grid_map.nx and
                    0 <= neighbor[1] < grid_map.ny):
                continue
            if grid_map.grid[neighbor[0]][neighbor[1]]:
                continue

            move_cost = np.sqrt(dx**2 + dy**2)
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, (gx, gy))
                heapq.heappush(open_set, (f, neighbor))

    # 找不到路徑，回傳直線距離
    return np.linalg.norm(np.array(goal_xy) - np.array(start_xy))
