# harness/report_generator.py
import json
import numpy as np
from harness.metrics import MetricsTracker

def generate_html_report(tracker: MetricsTracker, output_path: str):
    """把測試結果輸出成 HTML 報告"""

    overall = tracker.summary()
    easy = tracker.summary('Easy')
    medium = tracker.summary('Medium')
    hard = tracker.summary('Hard')

    # 收集所有 episode 資料
    episodes = []
    for r in tracker.results:
        episodes.append({
            'difficulty': r.difficulty,
            'target_x': float(r.target[0]),
            'target_y': float(r.target[1]),
            'target_z': float(r.target[2]),
            'success': r.success,
            'spl': float(r.spl),
            'final_dist': float(r.final_dist),
            'actual_path': float(r.actual_path_len),
            'optimal_path': float(r.optimal_path_len),
            'steps': r.steps_taken,
        })

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Harness Test Report</title>
<style>
  body {{ font-family: monospace; padding: 20px; background: #1e1e1e; color: #d4d4d4; }}
  h1 {{ color: #569cd6; }}
  h2 {{ color: #9cdcfe; border-bottom: 1px solid #444; padding-bottom: 5px; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
  th {{ background: #2d2d2d; color: #569cd6; padding: 8px; text-align: left; }}
  td {{ padding: 8px; border-bottom: 1px solid #333; }}
  .success {{ color: #4ec9b0; }}
  .fail {{ color: #f44747; }}
  .easy {{ color: #b5cea8; }}
  .medium {{ color: #dcdcaa; }}
  .hard {{ color: #f44747; }}
  .metric {{ font-size: 2em; color: #569cd6; }}
  .metric-box {{ display: inline-block; background: #2d2d2d; padding: 15px 25px; margin: 10px; border-radius: 5px; text-align: center; }}
  .metric-label {{ font-size: 0.8em; color: #888; }}
</style>
</head>
<body>
<h1>🚁 Drone Navigation Harness Report</h1>

<h2>Overall Metrics</h2>
<div>
  <div class="metric-box">
    <div class="metric">{overall['SR']:.0%}</div>
    <div class="metric-label">Success Rate</div>
  </div>
  <div class="metric-box">
    <div class="metric">{overall['SPL']:.3f}</div>
    <div class="metric-label">SPL</div>
  </div>
  <div class="metric-box">
    <div class="metric">{overall['avg_final_dist']:.2f}m</div>
    <div class="metric-label">Avg Final Dist</div>
  </div>
  <div class="metric-box">
    <div class="metric">{overall['n']}</div>
    <div class="metric-label">Total Episodes</div>
  </div>
</div>

<h2>By Difficulty</h2>
<table>
  <tr>
    <th>Difficulty</th><th>n</th><th>SR</th><th>SPL</th>
    <th>Avg Final Dist</th><th>Path Ratio</th>
  </tr>
  {"".join(f'''
  <tr>
    <td class="{d.lower()}">{d}</td>
    <td>{s["n"]}</td>
    <td>{s["SR"]:.0%}</td>
    <td>{s["SPL"]:.3f}</td>
    <td>{s["avg_final_dist"]:.2f}m</td>
    <td>{s["avg_path_ratio"]:.2f}</td>
  </tr>''' for d, s in [('Easy', easy), ('Medium', medium), ('Hard', hard)] if s)}
</table>

<h2>Episode Details</h2>
<table>
  <tr>
    <th>#</th><th>Difficulty</th><th>Target</th>
    <th>Result</th><th>Final Dist</th><th>SPL</th>
    <th>Actual Path</th><th>Optimal Path</th><th>Steps</th>
  </tr>
  {"".join(f'''
  <tr>
    <td>{i+1}</td>
    <td class="{ep["difficulty"].lower()}">{ep["difficulty"]}</td>
    <td>({ep["target_x"]:.2f}, {ep["target_y"]:.2f}, {ep["target_z"]:.2f})</td>
    <td class="{"success" if ep["success"] else "fail"}">{"✅" if ep["success"] else "❌"}</td>
    <td>{ep["final_dist"]:.2f}m</td>
    <td>{ep["spl"]:.3f}</td>
    <td>{ep["actual_path"]:.2f}m</td>
    <td>{ep["optimal_path"]:.2f}m</td>
    <td>{ep["steps"]}</td>
  </tr>''' for i, ep in enumerate(episodes))}
</table>

</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"HTML 報告已存到 {output_path}")
