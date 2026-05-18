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
        self.current_pose = np.zeros(3)
        self.current_vel = np.zeros(3)
        self.last_time = self.get_clock().now()
        self._pose_lock = threading.Lock() # 保護共享資料

        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.takeoff_pub = self.create_publisher(Empty, '/simple_drone/takeoff', 10)
        self.reset_pub = self.create_publisher(Empty, '/simple_drone/reset', 10)
        self.pose_sub = self.create_subscription(Pose, '/simple_drone/gt_pose', self._pose_cb, 10)
        self.reset_world_client = self.create_client(EmptySrv, '/reset_world')
        self.land_pub = self.create_publisher(Empty, '/simple_drone/land', 10
                )
    def _pose_cb(self, msg: Pose):
        new_pose = np.array([msg.position.x, msg.position.y, msg.position.z])
        now = self.get_clock().now()
        
        # 處理 /reset_world 導致的模擬時間歸零/倒退問題
        if now.nanoseconds < self.last_time.nanoseconds:
            self.last_time = now
            self.current_pose = new_pose
            self.current_vel = np.zeros(3)
            return

        dt = (now - self.last_time).nanoseconds / 1e9
        if dt > 0.001:
            self.current_vel = (new_pose - self.current_pose) / dt
            self.current_pose = new_pose
            self.last_time = now

    def get_state(self):
        """thread-safe 讀取狀態"""
        with self._pose_lock:
            return self.current_pose.copy(), self.current_vel.copy()

    def send_velocity(self, vx, vy, vz):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(vx), float(vy), float(vz)
        self.cmd_vel_pub.publish(msg)

    def reset_drone(self):
        

        # 步驟零：確保上一回合的速度指令清空，避免干擾起飛狀態
        self.send_velocity(0.0, 0.0, 0.0)

        # 確保上一個 episode 的指令都清空了
        time.sleep(1.0)
        # 步驟一：先 land，讓 plugin 回到乾淨狀態
        self.land_pub.publish(Empty())
        time.sleep(3.0)  # 等降落完成

        # 步驟二：reset world
        # 依照論文建議進行世界與無人機重置 [cite: 101, 102]
        if self.reset_world_client.service_is_ready():
            future = self.reset_world_client.call_async(EmptySrv.Request())
            
            # 等 future 完成
            while not future.done():
                time.sleep(0.1)
        
        #self.reset_pub.publish(Empty())
        time.sleep(2.0)

        # 步驟三：takeoff
        self.takeoff_pub.publish(Empty())
        self._wait_for_stable_height(min_z=0.5, timeout=10.0)
        '''
        # 等到 future 完成，最多等 3 秒
        
        deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=3)
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.get_clock().now() > deadline:
                self.get_logger().warn('reset_world timeout')
                break
        
        self.reset_pub.publish(Empty())
        rclpy.spin_once(self, timeout_sec=0.5)
        self.takeoff_pub.publish(Empty())
        
        # 修正三：等無人機高度穩定再離開，而不是固定等 1.5 秒
        self._wait_for_stable_height(target_z=1.0, tolerance=0.3, timeout=5.0)
        '''
    def _wait_for_stable_height(self, min_z: float = 0.5, timeout: float = 10.0):
        """等到無人機高度進入目標範圍才返回"""
        # deadline = time.time() + timeout
        """等高度穩定超過 min_z 即可，不限定要到達特定高度"""
        #prev_z = 0.0
        #stable_count = 0

        # while time.time() < deadline:
        #    #rclpy.spin_once(self, timeout_sec=0.1)
        #    time.sleep(0.1)
            
        #    pose, _ = self.get_state()
        #    z = pose[2]

            # 高度夠高，且變化量很小（穩定了）
        #    if z > min_z:
                #stable_count += 1
                #if stable_count >= 5:  # 連續 0.5 秒穩定
        #        time.sleep(1.0)
        #        return
            #else:
            #    stable_count = 0

            # prev_z = z

            # pose, _ = self.get_state()
            #if abs(pose[2] - target_z) < tolerance:
            #    time.sleep(0.3)
            #    return
        # 將 timeout 轉換為奈秒，並加上當前的 ROS 時間
        timeout_ns = int(timeout * 1e9)
        start_time = self.get_clock().now().nanoseconds
        deadline = start_time + timeout_ns

        while self.get_clock().now().nanoseconds < deadline:
            time.sleep(0.1)  # 讓出線程給背景的 ROS executor 接收資料

            pose, _ = self.get_state()
            z = pose[2]

            if z > min_z:
                time.sleep(1.0)
                return


        self.get_logger().warn(f'Takeoff timeout，目前高度: {self.get_state()[0][2]:.2f}m')

class DroneGymEnv(gym.Env):
    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface
        # 參考論文：動作空間僅控制 vy, vz [cite: 118]
        self.action_space = spaces.Box(low=-0.6, high=0.6, shape=(3,), dtype=np.float32)
        # 參考論文：狀態空間包含相對位置與速度 [cite: 110, 113]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(6,), dtype=np.float32)
        
        self.max_steps = 250
        self.target = np.array([0.0, 0.0, 0.0])
        self.prev_dist = 0.0

    def _get_obs(self):
        # 歸一化處理，讓模型更容易學習特徵 
        #rel_pos = (self.target - self.ros.current_pose) / 10.0
        #vel = self.ros.current_vel / 5.0
        #return np.concatenate([rel_pos, vel]).astype(np.float32)
        pose, vel = self.ros.get_state()
        rel_pos = (self.target - pose) / 10.0
        vel_norm = vel / 5.0
        return np.concatenate([rel_pos, vel_norm]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.ros.reset_drone()
        self.step_count = 0
        
        
        # 額外等待 pose 更新穩定
        #for _ in range(20):
        #    rclpy.spin_once(self.ros, timeout_sec=0.1)

        #self.step_count = 0
        # 隨機生成前方的目標點 [cite: 105]
        pose, _ = self.ros.get_state()
        while True:

            self.target = np.random.uniform(low=[-5.0, -5.0, 0.5], high=[5.0, 5.0, 4.0]).astype(np.float32)
            self.prev_dist = np.linalg.norm(pose - self.target)
            if self.prev_dist > 1.5:  # 確保起始就不在成功範圍內
                break

        
        return self._get_obs(), {}

    def step(self, action):
        vx, vy, vz = action
        # 參考論文：固定前進速度 vx=0.4 [cite: 119]
        self.ros.send_velocity(vx, vy, vz)
        #rclpy.spin_once(self.ros, timeout_sec=0.1)
        time.sleep(0.1)
        self.step_count += 1
        
        pose, _ = self.ros.get_state() # thread-safe 讀取 

        obs = self._get_obs()
        curr_dist = np.linalg.norm(pose - self.target)

        # 論文核心：指數距離獎勵 
        reward = 3.0 * math.exp(-curr_dist) # 30->3 解決 value function 不收斂
        # 近距離額外獎勵
        if curr_dist < 1.5:
            reward += 2.0 * math.exp(-curr_dist * 3) # 10->1
        
        ## 距離變化太小，代表無人機停住了
        if abs(self.prev_dist - curr_dist) < 0.01 and self.step_count > 15:
            reward -= 0.2  # 停著不動就扣分
        
        # 進度回饋：鼓勵縮短距離
        reward += (self.prev_dist - curr_dist) * 1.5 # 15.0 -> 1.5
        self.prev_dist = curr_dist

        terminated = False
        #if curr_dist < 0.5 and self.step_count > 15: # 成功判定 
        if curr_dist < 1.0 and self.step_count > 15: # 成功判定 
            reward += 10.0 # 100->10
            terminated = True
        
        if self.step_count > 15: # 保護期過後的碰撞判定 
            if pose[2] < 0.1 or pose[2] > 7.0:
                reward -= 10.0  # 100 -> 10
                terminated = True
            # 如果只是高度偏離，扣分但不結束 (引導它飛回來)
            elif pose[2] < 0.5 or pose[2] > 4.0:
                reward -= 0.1 # 1 -> 1

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
                print(f"Steps: {self.num_timesteps:6d} | "
                      f"mean_reward: {mean_reward:8.2f} | "
                      f"mean_ep_len: {mean_len:6.1f}")
        return True

class BestModelCallback(BaseCallback):
    def __init__(self, env, save_path="./checkpoints2/best_model.zip", verbose=1):
        super().__init__(verbose)
        self.env = env
        self.save_path = save_path
        self.best_score = -np.inf

    def _on_step(self):
        # 每 10k steps 評估一次
        if self.n_calls % 10000 == 0:
            scores = []

            for _ in range(3):  # 5 episodes average
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
    
    # last model
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,        # 每 10000 步存一次
        save_path='./checkpoints2/last/',
        name_prefix='ppo_drone'
    )

    # best model
    
    best_callback = BestModelCallback(
        env=env,
        save_path="./checkpoints2/best/best_model.zip"
    )

    # 使用 PPO 演算法並調整學習參數 [cite: 11, 74]
    model = PPO(
        'MlpPolicy', env, verbose=1,
        learning_rate=1e-4,
        n_steps=2048, # 論文推薦較長的觀察步數 [cite: 154]
        batch_size=128,
        tensorboard_log='./ppo_drone_logs/'
    )

    # checkpoint
    #model = PPO.load(
    #    './checkpoints/last/ppo_drone_20000_steps',
    #    env=env,
    #    tensorboard_log='./ppo_drone_logs/'
    #)
    model.learn(
            total_timesteps=500_000, 
            reset_num_timesteps=False, # 避免 timestep 重置
            callback=CallbackList([checkpoint_callback, TrainLogCallback(), best_callback]) 
            )
    # 建議增加訓練時間
    model.save('./checkpoints2/final/final_last_model')

def test(env):
    #model = PPO.load('ppo_drone_random_target')
    model = PPO.load('checkpoints2/best/best_model')

    obs, _ = env.reset()
    total_reward = 0

    for step in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        
        pose, _ = env.ros.get_state()  # 加這行
        dist = np.linalg.norm(obs[:3] * 10.0)

        if step % 20 == 0 or terminated:
            #dist = np.linalg.norm(obs[:3] * 10.0) # 還原歸一化前的距離
            print(f"Step {step:3d} | 距離目標: {dist:.2f}m | 獎勵: {reward:.2f}")

        if terminated or truncated:
            print(f"🏁 測試結束 | 總獎勵: {total_reward:.2f} | 最終步數: {step}")
            break

# 測試
def sanity_check(ros: DroneROSInterface):
    print("等待第一筆 pose 進來...")

    # 等到 pose 不再是零
    timeout = time.time() + 5.0
    while time.time() < timeout:
        pose, _ = ros.get_state()
        if np.any(pose != 0.0):
            break
        time.sleep(0.1)
    else:
        print("❌ 超時：5 秒內沒有收到任何 pose，請確認 Gazebo 有在跑且 topic 名稱正確")
        return
    
    """
    before_pose, before_vel = ros.get_state()
    print(f"Pose before: {before_pose}")

    time.sleep(2.0)  # 等 2 秒看 pose 有沒有變化

    after_pose, after_vel = ros.get_state()
    print(f"Pose after:  {after_pose}")
    print(f"Vel:         {after_vel}")

    # 判讀
    if np.allclose(before_pose, after_pose, atol=1e-4):
        print("⚠️  Pose 沒有變化，無人機可能是靜止的（正常），或 callback 仍有問題")
    else:
        print("✅ Pose 有在更新")

    if np.any(np.abs(after_vel) > 1e3):
        print("❌ Vel 異常，sim time 可能還沒對齊")
    else:
        print("✅ Vel 數值合理")
    """
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

    # 用 MultiThreadedExecutor 在背景持續 spin
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
        executor.shutdown()   # 先停 executor
        spin_thread.join(timeout=2.0)  # 等執行緒結束
        rclpy.shutdown()

if __name__ == '__main__':
    main()
