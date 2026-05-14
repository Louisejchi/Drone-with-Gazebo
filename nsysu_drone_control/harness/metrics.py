# harness/metrics.py
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class EpisodeResult:
    """單一 episode 的測試結果"""
    difficulty: str
    target: np.ndarray
    straight_dist: float
    optimal_path_len: float
    detour_ratio: float

    success: bool = False
    actual_path_len: float = 0.0
    steps_taken: int = 0
    final_dist: float = 0.0
    trajectory: List[np.ndarray] = field(default_factory=list)

    @property
    def spl(self):
        """
        SPL = success * (optimal / max(actual, optimal))
        成功才有 SPL，失敗為 0
        """
        if not self.success:
            return 0.0
        return self.optimal_path_len / max(self.actual_path_len, self.optimal_path_len)


class MetricsTracker:
    """收集所有 episode 結果並計算指標"""

    def __init__(self):
        self.results: List[EpisodeResult] = []

    def add(self, result: EpisodeResult):
        self.results.append(result)

    def summary(self, difficulty: Optional[str] = None):
        """
        回傳指標摘要
        difficulty: 'Easy' / 'Medium' / 'Hard' / None（全部）
        """
        results = self.results
        if difficulty:
            results = [r for r in results if r.difficulty == difficulty]

        if not results:
            return None

        n = len(results)
        n_success = sum(r.success for r in results)

        sr = n_success / n
        spl = np.mean([r.spl for r in results])
        avg_final_dist = np.mean([r.final_dist for r in results])
        avg_steps = np.mean([r.steps_taken for r in results])

        # 成功的 episode 平均繞路比（actual / optimal）
        success_results = [r for r in results if r.success]
        if success_results:
            avg_path_ratio = np.mean([
                r.actual_path_len / r.optimal_path_len
                for r in success_results
            ])
        else:
            avg_path_ratio = float('nan')

        return {
            'n': n,
            'n_success': n_success,
            'SR': sr,
            'SPL': spl,
            'avg_final_dist': avg_final_dist,
            'avg_steps': avg_steps,
            'avg_path_ratio': avg_path_ratio,
        }

    def print_report(self):
        print("=" * 60)
        print("Harness Test Report")
        print("=" * 60)

        # 整體
        overall = self.summary()
        print(f"\n{'Overall':10s} | n={overall['n']:3d} | "
              f"SR={overall['SR']:.2%} | "
              f"SPL={overall['SPL']:.3f} | "
              f"AvgFinalDist={overall['avg_final_dist']:.2f}m")

        # 各難度
        print()
        for diff in ['Easy', 'Medium', 'Hard']:
            s = self.summary(difficulty=diff)
            if s is None:
                continue
            print(f"{diff:10s} | n={s['n']:3d} | "
                  f"SR={s['SR']:.2%} | "
                  f"SPL={s['SPL']:.3f} | "
                  f"AvgFinalDist={s['avg_final_dist']:.2f}m | "
                  f"PathRatio={s['avg_path_ratio']:.2f}")

        print("=" * 60)
