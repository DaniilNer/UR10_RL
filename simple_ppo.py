import time
import argparse
import numpy as np
from tqdm import trange
from gym.wrappers import FlattenObservation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from envirement import UR10_env, REWARDS


N_TRAIN_STEP = 5e5
traing_args = { "policy": "MlpPolicy",
                "learning_rate": 0.0003,
                "n_steps": 2048, 
                "batch_size": 64, 
                "n_epochs": 10, 
                "gamma": 0.99, 
                "gae_lambda": 0.95, 
                "clip_range": 0.2,
                "policy_kwargs": dict(net_arch=[256, 256]), 
                "device": 'cpu'
              }

def run_episode(policy, env, viz=False, video=False):
    state = env.reset()
    if video:
        env.start_video("run.mp4")
    c = 0
    mr = 0
    while True:
        action = policy.predict(state, deterministic=True)[0]
        state, reward, done, info = env.step(action)
        mr += reward
        c += 1
        if viz:
            time.sleep(0.1)
        if done:
            success = int(info['reach'])
            print("SUCCESS", success)
            break
    if video:
         env.stop_video()
    return mr/c, success

def evaluate(policy, env, nepisodes=100, viz=False):
    success_rate = 0
    mean_reward = 0
    for _ in trange(nepisodes):
        reward, success = run_episode(policy, env, viz)
    mean_reward /= nepisodes
    success_rate /= nepisodes
    print("mean reward", mean_reward, "mean success rate", success_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_mode', type=str, default='end_effector', choices=['end_effector', 'all_joints'])
    parser.add_argument('--mode', type=str, default='viz', choices=['train', 'eval', 'viz', 'video'])
    parser.add_argument('--model_path', type=str, default="models\deter_reward_end_effector")
    parser.add_argument('--random_reward', type=bool, default=True)
    parser.add_argument('--reward', type=str, default='sparce_reward', choices=REWARDS.keys())
    args = parser.parse_args()

    if args.mode == 'train':
            train_env = DummyVecEnv([lambda: UR10_env(goal_env=False, random_reward=args.random_reward)] *1)
            eval_callback = EvalCallback(train_env, best_model_save_path="./models/",
                             log_path="./logs/", eval_freq=5000,
                             deterministic=True, render=False)
            model = PPO(env=train_env,
                            verbose=1,# tensorboard_log='log',
                            **traing_args
                        )
            model.learn(N_TRAIN_STEP, callback=eval_callback)
    elif args.mode == 'eval':
            eval_env = UR10_env(mode = 'visulization', goal_env=False, random_reward=args.random_reward)
            model = PPO.load(args.model_path, env=eval_env)
            evaluate(model, eval_env)
    elif args.mode == 'viz':
          eval_env = UR10_env(mode = 'visulization', goal_env=False, random_reward=args.random_reward)
          model = PPO.load(args.model_path, env=eval_env)
          run_episode(model, eval_env, viz=True)
    elif args.mode == 'video':
          eval_env = UR10_env(mode = 'visulization', goal_env=False, random_reward=args.random_reward)
          model = PPO.load(args.model_path, env=eval_env)
          run_episode(model, eval_env, viz=True, video=True)
    

