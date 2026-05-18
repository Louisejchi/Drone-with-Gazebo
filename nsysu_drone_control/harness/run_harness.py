# harness/run_harness.py
import argparse
import numpy as np
import time
import rclpy
import threading
from rclpy.executors import MultiThreadedExecutor

from harness.scenario_manager import ScenarioManager
from harness.metrics import MetricsTracker, EpisodeResult

# 從主程式 import
import sys
sys.path.append('/ros2_ws/src/nsysu_drone_control')
from rl_fly_to_target import DroneROSInterface, DroneGymEnv
from stable_baselines3 import PPO


def run_episode(env, model, scenario):
    """
    跑一個測試 episode，回傳 EpisodeResult
    """
    target = scenario['target']
    env.target = target

    # 手動 reset，不重新抽 target
    env.ros.reset_drone()
    env.step_count = 0
    pose, _ = env.ros.get_state()
    env.prev_dist = np.linalg.norm(pose - target)

    obs = env._get_obs()
    total_path_len = 0.0
    prev_pose = pose.copy()
    trajectory = [pose.copy()]

    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)

        pose, _ = env.ros.get_state()
        trajectory.append(pose.copy())

        # 累積實際飛行距離
        total_path_len += np.linalg.norm(pose - prev_pose)
        prev_pose = pose.copy()

        if terminated or truncated:
            break

    final_dist = np.linalg.norm(pose - target)
    success = final_dist < 1.0

    return EpisodeResult(
        difficulty=scenario['difficulty'],
        target=target,
        straight_dist=scenario['straight_dist'],
        optimal_path_len=scenario['optimal_path_len'],
        detour_ratio=scenario['detour_ratio'],
        success=success,
        actual_path_len=total_path_len,
        steps_taken=step + 1,
        final_dist=final_dist,
        trajectory=trajectory,
    )


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model', default='ppo_drone_random_target',
    parser.add_argument('--model', default='checkpoints/ppo_drone_10000_steps',
                        help='模型檔案路徑')
    parser.add_argument('--n-easy',   type=int, default=10)
    parser.add_argument('--n-medium', type=int, default=10)
    parser.add_argument('--n-hard',   type=int, default=10)
    parser.add_argument('--output',   default='harness_result.txt',
                        help='報告輸出路徑')
    args = parser.parse_args()

    # 啟動 ROS2
    rclpy.init()
    ros = DroneROSInterface()
    executor = MultiThreadedExecutor()
    executor.add_node(ros)
    spin_thread = threading.Thread(target=executor.spin)
    spin_thread.start()

    # 1. 先停止 ROS callback 來源
    executor.shutdown()

    # 2. 等 spin 完全結束
    spin_thread.join()

    # 3. 再 shutdown rclpy
    rclpy.shutdown()

    # 等第一筆 pose
    print("等待 pose...")
    timeout = time.time() + 5.0
    while time.time() < timeout:
        pose, _ = ros.get_state()
        if np.any(pose != 0.0):
            break
        time.sleep(0.1)

    env = DroneGymEnv(ros)
    model = PPO.load(args.model)

    # 產生測試集
    sm = ScenarioManager()
    suite = sm.generate_test_suite(
        n_easy=args.n_easy,
        n_medium=args.n_medium,
        n_hard=args.n_hard
    )
    print(f"測試集共 {len(suite)} 個場景")

    # 跑測試
    tracker = MetricsTracker()
    for i, scenario in enumerate(suite):
        print(f"\n[{i+1}/{len(suite)}] {scenario['difficulty']:6s} | "
              f"target: ({scenario['target'][0]:.2f}, "
              f"{scenario['target'][1]:.2f}, "
              f"{scenario['target'][2]:.2f}) | "
              f"最短路徑: {scenario['optimal_path_len']:.2f}m")

        result = run_episode(env, model, scenario)
        tracker.add(result)

        status = '✅ 成功' if result.success else '❌ 失敗'
        print(f"  {status} | 實際路徑: {result.actual_path_len:.2f}m | "
              f"最終距離: {result.final_dist:.2f}m | "
              f"SPL: {result.spl:.3f}")

    # 印報告
    tracker.print_report()

    # 存報告
    with open(args.output, 'w') as f:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tracker.print_report()
        f.write(buf.getvalue())
    print(f"\n報告已存到 {args.output}")
        
    # html 報告
    from harness.report_generator import generate_html_report
    generate_html_report(tracker, 'harness_result.html')

    result_data = {
        'model': args.model,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'overall': tracker.summary(),
        'by_difficulty': {
            d: tracker.summary(d)
            for d in ['Easy', 'Medium', 'Hard']
            if tracker.summary(d)
        },
        'episodes': [
            {
                'difficulty': r.difficulty,
                'target': r.target.tolist(),
                'success': r.success,
                'spl': float(r.spl),
                'final_dist': float(r.final_dist),
                'actual_path_len': float(r.actual_path_len),
                'optimal_path_len': float(r.optimal_path_len),
                'detour_ratio': float(r.detour_ratio),
                'steps_taken': r.steps_taken,
            }
            for r in tracker.results
        ]
    }

    json_path = args.output.replace('.txt', '.json')
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"JSON 已存到 {json_path}")
        
    executor.shutdown()
    spin_thread.join(timeout=2.0)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
