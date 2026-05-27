import os
import argparse
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
from std_srvs.srv import Empty as EmptySrv

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

import threading
from rclpy.executors import MultiThreadedExecutor
import time
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

# system archeticture
'''
ROS2/Gazebo
    ↓
DroneROSInterface
    ↓
DroneGymEnv
    ↓
PPO 訓練
'''

# ROS 跟 Python RL 的橋樑
# 負責: 1. 收 ROS 資料 2. 發 ROS 指令 3. 幫 RL 拿到目前狀態
class DroneROSInterface(Node):
    def __init__(self):
        super().__init__('rl_drone_interface',
                         parameter_overrides=[
                             rclpy.parameter.Parameter(
                                 'use_sim_time',
                                 rclpy.parameter.Parameter.Type.BOOL,
                                 True
                             )
                         ])

        # 目前無人機位置
        self.current_pose = np.zeros(3)
        # 儲存目前速度
        self.current_vel = np.zeros(3)
        # 紀錄上一筆時間
        self.last_time = self.get_clock().now()

        # 多執行緒保護: ROS callback 與 RL thread 共同存取 pose/vel
        self._pose_lock = threading.Lock()

        # 建立 ROS publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.reset_pub = self.create_publisher(Empty, '/simple_drone/reset', 10)
        self.land_pub = self.create_publisher(Empty, '/simple_drone/land', 10)

        # 訂閱無人機真實位置 (ground truth pose)
        self.pose_sub = self.create_subscription(Pose, '/simple_drone/gt_pose', self._pose_cb, 10)

        # 建立 ROS service client：用來呼叫 Gazebo 的 /reset_world service
        self.reset_world_client = self.create_client(EmptySrv, '/reset_world')

    def _pose_cb(self, msg: Pose):
        """ROS topic callback: 更新當前位置與速度."""
        new_pose = np.array([msg.position.x, msg.position.y, msg.position.z])
        now = self.get_clock().now()

        # 處理 /reset_world 導致的模擬時間歸零/倒退問題
        if now.nanoseconds < self.last_time.nanoseconds: # 模擬時間倒退了，重置狀態
            with self._pose_lock:
                self.last_time = now
                self.current_pose = new_pose
                self.current_vel = np.zeros(3)
            return
        # 計算時間差，更新速度與位置
        dt = (now - self.last_time).nanoseconds / 1e9 # 轉換成秒
        if dt > 0.001: # 避免除以零或過小的 dt
            with self._pose_lock:
                self.current_vel = (new_pose - self.current_pose) / dt
                self.current_pose = new_pose
                self.last_time = now

    def get_state(self):
        """Thread-safe 讀取目前 pose 與速度."""
        with self._pose_lock: # 確保讀取時不會被 ROS callback 同時修改
            return self.current_pose.copy(), self.current_vel.copy()   # 回傳副本，避免外部修改內部狀態

    def send_velocity(self, vx, vy, vz):
        """將速度指令封裝成 Twist 並發布到無人機。"""
        msg = Twist()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = float(vz)
        self.cmd_vel_pub.publish(msg)

    def reset_drone(self):
        """重置無人機狀態：降落、重置場景，再起飛。"""
        # 步驟零：清空速度指令
        self.send_velocity(0.0, 0.0, 0.0)
        time.sleep(0.3)

        # 步驟一：先降落，讓插件回到乾淨狀態
        self.land_pub.publish(Empty())
        #self._wait_for_low_height(max_z=0.2, timeout=5.0).
        time.sleep(3.0)  # 等降落完成

        # 步驟二：如果 reset_world service ready，就呼叫它
        if self.reset_world_client.service_is_ready():
            future = self.reset_world_client.call_async(EmptySrv.Request())
            while not future.done():
                time.sleep(0.1)

        # 等待重置後 pose 穩定，避免直接起飛前狀態還沒就緒
        #self._wait_for_pose_stable(timeout=6.0)
        time.sleep(2.0)

        # 步驟三：起飛
        # 在發 takeoff 前，確認有 subscriber 在聽這個 topic（plugin/driver 是否就緒）
        try:
            wait_start = time.time()
            while time.time() - wait_start < 3.0:
                try:
                    subs = self.takeoff_pub.get_subscription_count()
                except Exception:
                    subs = 0
                if subs > 0:
                    break
                time.sleep(0.1)
            if subs == 0:
                self.get_logger().warn('No subscribers for /simple_drone/takeoff; takeoff may not be handled by simulator')
            else:
                self.get_logger().info(f'/simple_drone/takeoff subscribers: {subs}')
        except Exception:
            pass

        # 確認 gt_pose 有 publisher（sim 正在發 pose）
        try:
            pub_count = 0
            wait_start = time.time()
            while time.time() - wait_start < 3.0:
                try:
                    pub_count = self.pose_sub.get_publisher_count()
                except Exception:
                    pub_count = 0
                if pub_count > 0:
                    break
                time.sleep(0.1)
            if pub_count == 0:
                self.get_logger().warn('No publishers for /simple_drone/gt_pose; simulator may not be publishing pose')
            else:
                self.get_logger().info(f'/simple_drone/gt_pose publishers: {pub_count}')
        except Exception:
            pass

        self.takeoff_pub.publish(Empty())
        # 步驟四：等待高度穩定
        self._wait_for_stable_height(min_z=0.5, timeout=12.0)

    def _wait_for_low_height(self, max_z: float = 0.2, timeout: float = 5.0):
        """等待降落到低高度，避免直接重置時機過早。"""
        timeout_ns = int(timeout * 1e9) # 轉換成奈秒
        deadline = self.get_clock().now().nanoseconds + timeout_ns # 計算截止時間
        while self.get_clock().now().nanoseconds < deadline:
            time.sleep(0.1)
            pose, _ = self.get_state()
            if pose[2] <= max_z:
                return

        self.get_logger().warn(f'Landing timeout，目前高度: {self.get_state()[0][2]:.2f}m')

    def _wait_for_pose_stable(self, timeout: float = 6.0):
        """等待 reset 後 pose 穩定，避免在無人機尚未初始化時起飛。"""
        timeout_ns = int(timeout * 1e9)
        deadline = self.get_clock().now().nanoseconds + timeout_ns
        stable_count = 0
        prev_pose = None
        while self.get_clock().now().nanoseconds < deadline:
            time.sleep(0.1)
            pose, _ = self.get_state()
            if prev_pose is not None and np.allclose(pose, prev_pose, atol=1e-3):
                stable_count += 1
                if stable_count >= 5:
                    return
            else:
                stable_count = 0
            prev_pose = pose

        self.get_logger().warn('Reset world 之後 pose 未能穩定，本次起飛可能失敗。')

    def _wait_for_stable_height(self, min_z: float = 0.5, timeout: float = 12.0):
        """等待高度達到最低門檻，並持續穩定一段時間。"""
        timeout_ns = int(timeout * 1e9)
        deadline = self.get_clock().now().nanoseconds + timeout_ns
        stable_count = 0
        last_z = None
        last_publish = self.get_clock().now().nanoseconds
        republish_interval_ns = int(1.0 * 1e9)

        while self.get_clock().now().nanoseconds < deadline:
            time.sleep(0.1)
            now = self.get_clock().now().nanoseconds
            pose, _ = self.get_state()
            if pose[2] > min_z:
                stable_count += 1
                if stable_count >= 3:
                    return
            else:
                stable_count = 0
                if last_z is not None and pose[2] - last_z < 0.01 and now - last_publish > republish_interval_ns:
                    self.get_logger().info('Takeoff 未上升，重新發送 takeoff 指令...')
                    self.takeoff_pub.publish(Empty())
                    last_publish = now
            last_z = pose[2]

        self.get_logger().warn(f'Takeoff timeout，目前高度: {self.get_state()[0][2]:.2f}m')


class DroneGymEnv(gym.Env):
    def __init__(self, ros_interface: DroneROSInterface, verbose_step: bool = False):
        super().__init__()
        self.ros = ros_interface
        self.verbose_step = verbose_step
        self.action_space = spaces.Box(low=-0.6, high=0.6, shape=(3,), dtype=np.float32) # 3 維速度指令：vx, vy, vz
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(6,), dtype=np.float32) # 6 維觀測：相對位置 (x,y,z) 與速度 (vx,vy,vz)

        self.max_steps = 500 #250
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32) # 目標位置
        self.prev_dist = 0.0

    def _get_obs(self):
        """回傳當前 observation：相對位置與速度。"""
        # 把觀測值正規化到約 [-1, 1] 範圍，讓 PPO 更好學習
        pose, vel = self.ros.get_state()
        rel_pos = (self.target - pose) / 10.0 # 將距離縮放到約 [-1, 1] 範圍
        vel_norm = vel / 5.0 # 假設最大速度約 5 m/s，將速度縮放到約 [-1, 1] 範圍
        return np.concatenate([rel_pos, vel_norm]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """環境重置：重置無人機並生成新目標。"""
        super().reset(seed=seed) # 呼叫父類別的 reset 以處理隨機種子
        # 先叫 ROS 端重置並起飛
        self.ros.reset_drone()
        self.step_count = 0

        # 等待 pose 穩定：reset/drone 起飛後，gt_pose 可能還沒更新
        # 使用 rclpy.spin_once 以確保 ROS callback 被執行
        wait_deadline = time.time() + 6.0
        pose, vel = self.ros.get_state()
        while time.time() < wait_deadline:
            time.sleep(0.1)
            pose, vel = self.ros.get_state()
            # 要求：pose 有數值、z 高度超過 0.4m（已起飛）且速度不是異常大
            if np.any(pose != 0.0) and pose[2] > 0.4 and np.linalg.norm(vel) < 10.0:
                break
        else:
            print("⚠️ reset() 超時：在 6 秒內未取得穩定 pose，將接著生成目標（可能需要檢查 sim）")

        # 取得當前 pose 再隨機生成一個距離較遠的目標
        pose, _ = self.ros.get_state()
        while True:
            self.target = np.random.uniform(low=[-8.0, -8.0, 0.5], high=[8.0, 8.0, 4.0]).astype(np.float32) # 生成新目標
            self.prev_dist = np.linalg.norm(pose - self.target) # 計算初始距離
            if self.prev_dist > 1.5: # 確保目標不會太近，讓訓練更有挑戰性
                break

        return self._get_obs(), {}

    def step(self, action):
        """執行一步動作並計算獎勵、終止條件。"""
        vx, vy, vz = action
        self.ros.send_velocity(vx, vy, vz)
        time.sleep(0.1)
        self.step_count += 1

        pose, _ = self.ros.get_state()
        obs = self._get_obs()
        curr_dist = np.linalg.norm(pose - self.target) # 計算當前距離
        
        # 獎勵設計：距離越近獎勵越高，快速接近目標也有額外獎勵，停滯不前有輕微懲罰
        reward = 3.0 * math.exp(-curr_dist)
        if curr_dist < 2.0:
            reward += 5.0 * math.exp(-curr_dist * 2)

        if abs(self.prev_dist - curr_dist) < 0.01 and self.step_count > 15:
            reward -= 0.5

        reward += (self.prev_dist - curr_dist) * 1.5
        self.prev_dist = curr_dist

        terminated = False
        success = False
        # 當距離小於 1.0m 且已經過 15 步，視為成功到達目標
        # 剛 reset 後，無人機可能還沒穩定，或目標剛好生成在附近。因此給予一個緩衝期，讓無人機有時間調整位置。
        if curr_dist < 1.0 and self.step_count > 15:
            reward += 10.0
            terminated = True
            success = True

        # 如果高度過低或過高，視為失敗
        if self.step_count > 15:
            if pose[2] < 0.1 or pose[2] > 7.0:
                reward -= 10.0
                terminated = True
            elif pose[2] < 0.5 or pose[2] > 4.0:
                reward -= 0.1

        truncated = self.step_count >= self.max_steps
        if self.verbose_step:
            print(
                f"Step {self.step_count:3d} | action: [{vx:.2f}, {vy:.2f}, {vz:.2f}] | "
                f"pose: [{pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}] | "
                f"target: [{self.target[0]:.2f}, {self.target[1]:.2f}, {self.target[2]:.2f}] | "
                f"dist: {curr_dist:.2f} | reward: {reward:.2f} | "
                f"done: {terminated or truncated}"
            )
        return obs, reward, terminated, truncated, {'is_success': success}

# 自訂 callback：定期輸出訓練進度與評估模型表現
class TrainLogCallback(BaseCallback):
    def __init__(self, log_interval=2048):
        super().__init__()
        self.log_interval = log_interval

    def _on_step(self):
        if self.n_calls % self.log_interval == 0: # 每 log_interval 步輸出一次訓練進度
            if len(self.model.ep_info_buffer) > 0: # 確保有 episode 資訊可用
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                print(f"Steps: {self.num_timesteps:6d} | mean_reward: {mean_reward:8.2f} | mean_ep_len: {mean_len:6.1f}")
        return True

# 自訂 callback：定期評估模型表現，並在表現提升時保存最佳模型
class BestModelCallback(BaseCallback):
    def __init__(self, save_path="./checkpoints2/best/best_model.zip", eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.best_score = -np.inf

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # 直接從訓練過程的 episode buffer 拿資料，不需要額外跑 episode
            if len(self.model.ep_info_buffer) > 0:
                mean_score = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                # 計算成功率：reward > 某個門檻視為成功
                success_count = sum(1 for ep in self.model.ep_info_buffer if ep['r'] > 50)
                success_rate = success_count / len(self.model.ep_info_buffer)

                print(f"[Eval] steps={self.num_timesteps} | "
                      f"score={mean_score:.2f} | "
                      f"ep_len={mean_len:.1f} | "
                      f"success_rate={success_rate:.2%}")

                if mean_score > self.best_score:
                    self.best_score = mean_score
                    print("🔥 New best model saved!")
                    os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
                    self.model.save(self.save_path)
        return True


def find_latest_checkpoint(folder='./checkpoints2/last'):
    """從 checkpoint 資料夾找出最後建立的模型檔。"""
    if not os.path.isdir(folder):
        return None

    checkpoint_files = [f for f in os.listdir(folder) if f.endswith('.zip')]
    if not checkpoint_files:
        return None

    def checkpoint_step(name):
        parts = name.replace('.zip', '').split('_')
        for i, part in enumerate(parts):
            if part == 'steps' and i > 0:
                try:
                    return int(parts[i-1])
                except ValueError:
                    continue
        return 0

    checkpoint_files.sort(key=lambda f: checkpoint_step(f), reverse=True)
    return os.path.join(folder, checkpoint_files[0])

def plot_trajectory(trajectory, target):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    traj = np.array(trajectory)
    fig = plt.figure(figsize=(12, 5))
    fig.patch.set_facecolor('#1e1e1e')

    # 左圖：3D 軌跡
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_facecolor('#1e1e1e')

    colors = plt.cm.cool(np.linspace(0, 1, len(traj)))
    for i in range(len(traj)-1):
        ax1.plot(traj[i:i+2, 0], traj[i:i+2, 1], traj[i:i+2, 2],
                color=colors[i], linewidth=1.5)

    ax1.scatter(*traj[0],  color='#4ec9b0', s=100, label='Start', zorder=5)
    ax1.scatter(*traj[-1], color='#f44747', s=100, label='End',   zorder=5)
    ax1.scatter(*target,   color='#dcdcaa', s=200, marker='*', label='Target', zorder=5)

    # 成功半徑球
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    r = 1.0
    x = target[0] + r * np.outer(np.cos(u), np.sin(v))
    y = target[1] + r * np.outer(np.sin(u), np.sin(v))
    z = target[2] + r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x, y, z, alpha=0.1, color='#dcdcaa')

    ax1.set_xlabel('X', color='white')
    ax1.set_ylabel('Y', color='white')
    ax1.set_zlabel('Z', color='white')
    ax1.set_title('3D Flight Trajectory', color='white')
    ax1.legend(facecolor='#2d2d2d', labelcolor='white')
    ax1.tick_params(colors='white')

    # 右圖：距離隨時間變化
    ax2 = fig.add_subplot(122)
    ax2.set_facecolor('#1e1e1e')

    dists = [np.linalg.norm(p - target) for p in traj]
    ax2.plot(dists, color='#569cd6', linewidth=2)
    ax2.axhline(y=1.0, color='#4ec9b0', linestyle='--', alpha=0.7, label='Success (1.0m)')
    ax2.fill_between(range(len(dists)), 0, 1.0, alpha=0.1, color='#4ec9b0')

    ax2.set_xlabel('Steps', color='white')
    ax2.set_ylabel('Distance to Target (m)', color='white')
    ax2.set_title('Distance over Time', color='white')
    ax2.legend(facecolor='#2d2d2d', labelcolor='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2)
    for spine in ax2.spines.values():
        spine.set_color('#444')

    plt.tight_layout()
    plt.savefig('flight_trajectory.png', dpi=150,
                bbox_inches='tight', facecolor='#1e1e1e')
    print("軌跡圖存到 flight_trajectory.png")

def train(env, resume=False, checkpoint_path=None):
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints2/last/',
        name_prefix='ppo_drone'
    )

    best_callback = BestModelCallback(
        save_path="./checkpoints2/best/best_model.zip",
        eval_freq=10000
    )

    if resume:
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint('./checkpoints2/last')
            if checkpoint_path is not None:
                print(f"Resume training from latest checkpoint: {checkpoint_path}")
        else:
            print(f"Resume training from specified checkpoint: {checkpoint_path}")

    if resume and checkpoint_path is not None and os.path.isfile(checkpoint_path):
        model = PPO.load(checkpoint_path, env=env)
        # 降低 learning rate
        new_lr = 1e-4
        model.learning_rate = new_lr
        model.lr_schedule = lambda _: new_lr

        for param_group in model.policy.optimizer.param_groups:
            param_group['lr'] = new_lr
            print(f"Learning rate 調整為 1e-4")

        print("optimizer lr =", model.policy.optimizer.param_groups[0]['lr'])
        print("schedule lr =", model.lr_schedule(1.0))
    else:
        if resume:
            print("⚠️ 找不到可用的 checkpoint，將從頭開始訓練。")
        model = PPO(
            'MlpPolicy', env, verbose=1,
            learning_rate=3e-4, #1E-4
            n_steps=2048,
            batch_size=64, # 128
            tensorboard_log='./ppo_drone_logs/'
        )

    model.learn(
        total_timesteps=500_000,
        reset_num_timesteps=False,
        callback=CallbackList([checkpoint_callback, TrainLogCallback(), best_callback])
    )
    model.save('./checkpoints2/final/final_last_model')



def test(env, n_episodes=10):
    model = PPO.load('checkpoints2/best/best_model')
    results = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        success = False
        trajectory = []

        for step in range(env.max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            pose, _ = env.ros.get_state()
            trajectory.append(pose.copy())

            if info.get('is_success', False):
                success = True

            if terminated or truncated:
                break

        results.append({'success': success, 'reward': total_reward, 'steps': step})
        print(f"Episode {ep+1:2d} | 總獎勵: {total_reward:.2f} | 步數: {step} | success: {success}")

    sr = sum(r['success'] for r in results) / n_episodes
    avg_reward = np.mean([r['reward'] for r in results])
    print(f"\n=== 總結 ===")
    print(f"Success Rate: {sr:.0%}")
    print(f"Avg Reward:   {avg_reward:.2f}")

    plot_trajectory(trajectory, env.target)


def sanity_check(ros: DroneROSInterface):
    print("等待第一筆 pose 進來...")
    timeout = time.time() + 5.0
    while time.time() < timeout:
        pose, _ = ros.get_state()
        if np.any(pose != 0.0):
            break
        time.sleep(0.1)
    else:
        print("❌ 超時：5 秒內沒有收到任何 pose，請確認 Gazebo 有在跑且 topic 名稱正確")
        return

    print("測試 reset_drone()...")
    ros.reset_drone()

    pose, vel = ros.get_state()
    print(f"Reset 後 Pose: {pose}")
    print(f"Reset 後 Vel:  {vel}")

    if pose[2] < 0.5:
        print(f"❌ 高度只有 {pose[2]:.2f}m，takeoff 可能還沒完成")
    else:
        print(f"✅ 高度 {pose[2]:.2f}m，reset 正常")

    env = DroneGymEnv(ros)
    obs, _ = env.reset()
    print(f"\nTarget: {env.target}")
    print(f"Init dist: {env.prev_dist:.2f}m")
    print(f"Obs: {obs}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        pose, _ = ros.get_state()
        print(f"Step {i+1} | pose z: {pose[2]:.2f}m | reward: {reward:.2f} | done: {terminated or truncated}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test', 'check'], default='train')
    parser.add_argument('--resume', action='store_true', help='Continue training from the latest checkpoint in checkpoints2/last')
    parser.add_argument('--checkpoint', type=str, default=None, help='Explicit checkpoint file to resume training from')
    parser.add_argument('--verbose-steps', action='store_true', help='Print information for every environment step')
    args = parser.parse_args()
    rclpy.init()
    ros = DroneROSInterface()

    executor = MultiThreadedExecutor()
    executor.add_node(ros)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    env = DroneGymEnv(ros, verbose_step=args.verbose_steps)

    try:
        if args.mode == 'check':
            sanity_check(ros)
        elif args.mode == 'train':
            train(env, resume=args.resume, checkpoint_path=args.checkpoint)
        else:
            test(env, n_episodes=10)
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
