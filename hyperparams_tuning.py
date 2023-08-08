from typing import Any
from typing import Dict

import gym
import optuna
import numpy as np
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from gym.wrappers import FlattenObservation
import torch
import torch.nn as nn

from hyperparams_tuning import *
from envirement import UR10_env


DEFAULT_HYPERPARAMS_HER = {
    "policy": "MultiInputPolicy",
    "replay_buffer_class": HerReplayBuffer,
    "replay_buffer_kwargs": dict(
        n_sampled_goal=4,
        goal_selection_strategy="future",
    max_episode_length=300,
    ),
}
DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}


N_TRIALS = 100
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e5)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

def sample_sac_params(trial):
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    batch_size = trial.suggest_categorical(
        "batch_size", [16, 32, 64, 128, 256])
    buffer_size = trial.suggest_categorical(
        "buffer_size", [int(1e3), int(1e4), int(1e5)])
    learning_starts = trial.suggest_categorical(
        "learning_starts", [0, 1000, 10000, 20000])
    train_freq = trial.suggest_categorical(
        "train_freq", [8, 16, 32, 64, 128, 256, 512])
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02])
    gradient_steps = train_freq
    ent_coef = "auto"
    log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    net_arch = trial.suggest_categorical(
        "net_arch", ["small", "medium", "big"])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
    }[net_arch]

    target_entropy = "auto"

    return {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }

class TrialEvalCallback(EvalCallback):
    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS_HER.copy()
    kwargs.update(sample_sac_params(trial))
    #model = SAC(env=DummyVecEnv([lambda: FlattenObservation(UR10_env())] *4), **kwargs)
    #model = SAC(env=DummyVecEnv([lambda: UR10_env()] *4), **kwargs)
    model = SAC(env=UR10_env(), device='cpu', **kwargs)
    eval_env =  UR10_env()
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    nan_encountered = False
    try:
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        print(e)
        nan_encountered = True
    finally:
        model.env.close()
        eval_env.close()

    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    try:
        study.optimize(objective, n_trials=N_TRIALS, timeout=600)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))