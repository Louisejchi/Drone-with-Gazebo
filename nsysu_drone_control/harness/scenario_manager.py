# harness/scenario_manager.py
import numpy as np
from harness.astar import GridMap, astar

class ScenarioManager:
    """
    自動產生測試場景，分級為 Easy / Medium / Hard
    並計算 SPL 所需的最短路徑長度
    """

    def __init__(self):
        self.grid_map = GridMap()
        # 無人機起飛後的固定起始位置（世界原點附近）
        self.start_xy = (0.0, 0.0)

    def generate_scenario(self, difficulty: str = None):
        """
        產生一個測試場景
        回傳: {
            'target': np.array([x, y, z]),
            'difficulty': 'Easy' / 'Medium' / 'Hard',
            'straight_dist': float,
            'optimal_path_len': float,
            'detour_ratio': float   # optimal / straight，越大代表越難繞路
        }
        """
        for _ in range(1000):  # 最多嘗試 1000 次
            target = self._sample_target(difficulty)
            if target is None:
                continue

            tx, ty, tz = target

            # 確認目標點本身不在障礙物內
            if not self.grid_map.is_free(tx, ty):
                continue

            straight = np.linalg.norm(
                np.array([tx, ty]) - np.array(self.start_xy)
            )

            # 距離太近不算
            if straight < 1.5:
                continue

            optimal = astar(self.grid_map, self.start_xy, (tx, ty))
            detour_ratio = optimal / straight

            actual_difficulty = self._classify(straight, detour_ratio)

            # 如果指定了難度，不符合就重新取樣
            if difficulty and actual_difficulty != difficulty:
                continue

            return {
                'target': np.array([tx, ty, tz], dtype=np.float32),
                'difficulty': actual_difficulty,
                'straight_dist': straight,
                'optimal_path_len': optimal,
                'detour_ratio': detour_ratio,
            }

        raise RuntimeError(f'無法產生 difficulty={difficulty} 的場景，請放寬條件')

    def _sample_target(self, difficulty):
        """根據難度決定取樣範圍"""
        if difficulty == 'Easy':
            # 近距離，空曠區域
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-8, -2)  # 南方空曠
            z = np.random.uniform(0.5, 4.0)
        elif difficulty == 'Hard':
            # 遠距離，靠近障礙物或牆壁
            x = np.random.uniform(-8, 8)
            y = np.random.uniform(-8, 8)
            z = np.random.uniform(0.5, 2.5)  # 低空，牆壁影響更大
        else:
            # Medium 或 None（完全隨機）
            x = np.random.uniform(-5, 5)
            y = np.random.uniform(-5, 5)
            z = np.random.uniform(0.5, 4.0)

        return (x, y, z)

    def _classify(self, straight_dist: float, detour_ratio: float):
        """
        根據直線距離和繞路比值分級
        Easy:   近距離 + 幾乎不用繞路
        Medium: 中距離 或 稍微繞路
        Hard:   遠距離 或 需要大幅繞路
        """
        if straight_dist < 4.0 and detour_ratio < 1.1:
            return 'Easy'
        elif straight_dist > 7.0 or detour_ratio > 1.3:
            return 'Hard'
        else:
            return 'Medium'

    def generate_test_suite(self, n_easy=10, n_medium=10, n_hard=10):
        """產生完整測試集"""
        scenarios = []
        for diff, n in [('Easy', n_easy), ('Medium', n_medium), ('Hard', n_hard)]:
            count = 0
            while count < n:
                try:
                    s = self.generate_scenario(difficulty=diff)
                    scenarios.append(s)
                    count += 1
                except RuntimeError:
                    break
        return scenarios
