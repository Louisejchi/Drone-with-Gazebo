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
        self._wait_for_low_height(max_z=0.2, timeout=5.0)

        # 步驟二：如果 reset_world service ready，就呼叫它
        if self.reset_world_client.service_is_ready():
            future = self.reset_world_client.call_async(EmptySrv.Request())
            while not future.done():
                time.sleep(0.1)

        time.sleep(0.5)

        # 步驟三：起飛
        self.takeoff_pub.publish(Empty())
        # 步驟四：等待高度穩定
        self._wait_for_stable_height(min_z=0.5, timeout=10.0)

    def _wait_for_low_height(self, max_z: float = 0.2, timeout: float = 5.0):
        """等待降落到低高度，避免直接重置時機過早。"""
        timeout_ns = int(timeout * 1e9)
        deadline = self.get_clock().now().nanoseconds + timeout_ns

        while self.get_clock().now().nanoseconds < deadline:
            time.sleep(0.1)
            pose, _ = self.get_state()
            if pose[2] <= max_z:
                return

        self.get_logger().warn(f'Landing timeout，目前高度: {self.get_state()[0][2]:.2f}m')

    def _wait_for_stable_height(self, min_z: float = 0.5, timeout: float = 10.0):
        """等待高度達到最低門檻，並持續穩定一段時間。"""
        timeout_ns = int(timeout * 1e9)
        deadline = self.get_clock().now().nanoseconds + timeout_ns
        stable_count = 0

        while self.get_clock().now().nanoseconds < deadline:
            time.sleep(0.1)
            pose, _ = self.get_state()
            if pose[2] > min_z:
                stable_count += 1
                if stable_count >= 3:
                    return
            else:
                stable_count = 0

        self.get_logger().warn(f'Takeoff timeout，目前高度: {self.get_state()[0][2]:.2f}m')


class DroneGymEnv(gym.Env):
    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface
        self.action_space = spaces.Box(low=-0.6, high=0.6, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(6,), dtype=np.float32)

        self.max_steps = 250
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_dist = 0.0

    def _get_obs(self):
        """回傳當前 observation：相對位置與速度。"""
        pose, vel = self.ros.get_state()
        rel_pos = (self.target - pose) / 10.0
        vel_norm = vel / 5.0
        return np.concatenate([rel_pos, vel_norm]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """環境重置：重置無人機並生成新目標。"""
        super().reset(seed=seed)
        self.ros.reset_drone()
        self.step_count = 0

        pose, _ = self.ros.get_state()
        while True:
            self.target = np.random.uniform(low=[-5.0, -5.0, 0.5], high=[5.0, 5.0, 4.0]).astype(np.float32)
            self.prev_dist = np.linalg.norm(pose - self.target)
            if self.prev_dist > 1.5:
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
        curr_dist = np.linalg.norm(pose - self.target)

        reward = 3.0 * math.exp(-curr_dist)
        if curr_dist < 1.5:
            reward += 2.0 * math.exp(-curr_dist * 3)

        if abs(self.prev_dist - curr_dist) < 0.01 and self.step_count > 15:
            reward -= 0.2

        reward += (self.prev_dist - curr_dist) * 1.5
        self.prev_dist = curr_dist

        terminated = False
        if curr_dist < 1.0 and self.step_count > 15:
            reward += 10.0
            terminated = True

        if self.step_count > 15:
            if pose[2] < 0.1 or pose[2] > 7.0:
                reward -= 10.0
                terminated = True
            elif pose[2] < 0.5 or pose[2] > 4.0:
                reward -= 0.1

        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}


class TrainLogCallback(BaseCallback):
    def __init__(self, log_interval=2048):
        super().__init__()
        self.log_interval = log_interval

    def _on_step(self):
        if self.n_calls % self.log_interval == 0:
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep['r'] for ep in self.model.ep_info_buffer])
                mean_len = np.mean([ep['l'] for ep in self.model.ep_info_buffer])
                print(f"Steps: {self.num_timesteps:6d} | mean_reward: {mean_reward:8.2f} | mean_ep_len: {mean_len:6.1f}")
        return True


class BestModelCallback(BaseCallback):
    def __init__(self, env, save_path="./checkpoints2/best/best_model.zip", verbose=1):
        super().__init__(verbose)
        self.env = env
        self.save_path = save_path
        self.best_score = -np.inf

    def _on_step(self):
        if self.n_calls % 10000 == 0:
            scores = []
            for _ in range(3):
                obs, _ = self.env.reset()
                done = False
                total_reward = 0
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, _ = self.env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                scores.append(total_reward)

            mean_score = np.mean(scores)
            print(f"[Eval] score = {mean_score:.3f}")

            if mean_score > self.best_score:
                self.best_score = mean_score
                print("🔥 New best model saved!")
                self.model.save(self.save_path)
        return True


def train(env):
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./checkpoints2/last/',
        name_prefix='ppo_drone'
    )

    best_callback = BestModelCallback(
        env=env,
        save_path="./checkpoints2/best/best_model.zip"
    )

    model = PPO(
        'MlpPolicy', env, verbose=1,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        tensorboard_log='./ppo_drone_logs/'
    )

    model.learn(
        total_timesteps=500_000,
        reset_num_timesteps=False,
        callback=CallbackList([checkpoint_callback, TrainLogCallback(), best_callback])
    )
    model.save('./checkpoints2/final/final_last_model')


def test(env):
    model = PPO.load('checkpoints2/best/best_model')
    obs, _ = env.reset()
    total_reward = 0

    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        dist = np.linalg.norm(obs[:3] * 10.0)
        if step % 20 == 0 or terminated:
            print(f"Step {step:3d} | 距離目標: {dist:.2f}m | 獎勵: {reward:.2f}")

        if terminated or truncated:
            print(f"🏁 測試結束 | 總獎勵: {total_reward:.2f} | 最終步數: {step}")
            break


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
    args = parser.parse_args()
    rclpy.init()
    ros = DroneROSInterface()

    executor = MultiThreadedExecutor()
    executor.add_node(ros)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if args.mode == 'check':
            sanity_check(ros)
        elif args.mode == 'train':
            env = DroneGymEnv(ros)
            train(env)
        else:
            env = DroneGymEnv(ros)
            test(env)
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        rclpy.shutdown()


if __name__ == '__main__':
    main()
