# harness/report_from_json.py
import json
import sys
import argparse

def generate_html_from_json(json_path: str, output_path: str):
    with open(json_path) as f:
        data = json.load(f)

    overall = data['overall']
    episodes = data['episodes']
    by_diff = data['by_difficulty']
    timestamp = data.get('timestamp', 'N/A')
    model = data.get('model', 'N/A')

    diff_rows = ""
    for d in ['Easy', 'Medium', 'Hard']:
        s = by_diff.get(d)
        if not s:
            continue
        path_ratio = s.get('avg_path_ratio')
        path_ratio_str = f"{path_ratio:.2f}" if path_ratio and path_ratio == path_ratio else "N/A"
        diff_rows += f"""
  <tr>
    <td class="{d.lower()}">{d}</td>
    <td>{s['n']}</td>
    <td>{s['SR']:.0%}</td>
    <td>{s['SPL']:.3f}</td>
    <td>{s['avg_final_dist']:.2f}m</td>
    <td>{path_ratio_str}</td>
  </tr>"""

    episode_rows = ""
    for i, ep in enumerate(episodes):
        tx, ty, tz = ep['target']
        status_class = "success" if ep['success'] else "fail"
        status_icon = "✅" if ep['success'] else "❌"
        episode_rows += f"""
  <tr>
    <td>{i+1}</td>
    <td class="{ep['difficulty'].lower()}">{ep['difficulty']}</td>
    <td>({tx:.2f}, {ty:.2f}, {tz:.2f})</td>
    <td class="{status_class}">{status_icon}</td>
    <td>{ep['final_dist']:.2f}m</td>
    <td>{ep['spl']:.3f}</td>
    <td>{ep['actual_path_len']:.2f}m</td>
    <td>{ep['optimal_path_len']:.2f}m</td>
    <td>{ep['steps_taken']}</td>
  </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Harness Report</title>
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
  .meta {{ color: #888; font-size: 0.9em; margin-bottom: 20px; }}
</style>
</head>
<body>
<h1>🚁 Drone Navigation Harness Report</h1>
<div class="meta">Model: {model} | Generated: {timestamp}</div>

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
  {diff_rows}
</table>

<h2>Episode Details</h2>
<table>
  <tr>
    <th>#</th><th>Difficulty</th><th>Target</th>
    <th>Result</th><th>Final Dist</th><th>SPL</th>
    <th>Actual Path</th><th>Optimal Path</th><th>Steps</th>
  </tr>
  {episode_rows}
</table>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f"HTML 報告已存到 {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',  default='harness_result.json')
    parser.add_argument('--output', default='public/index.html')
    args = parser.parse_args()
    generate_html_from_json(args.input, args.output)
