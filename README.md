## Reach task with reinforcement learning
This repository include envirement for solving reaching task.
Available action spaces:
1) Control end effector(default)
2) Control all joints
Available formulation of task:
1) Reward is fixed during training and test
2) Reward is random generated in some box on table every episode(default)


Simple_ppo.py is main python executable file with few options:
1) Train PPO
2) Eval PPO 100 episodes, output is success rate(% of reached positions) and mean reward
3) visualizate work of agent 1 episode

Hyperparams_tuning.py is optuna heperparams research for SAC with HER, but ppo works fine on this task and search params is long task, so i dont use it.BBB

Available rewards:
1) Lp distance between EEF and target
2) Lp distance between EEF and target + panalty action norm
3) Different between current distance and distance from last step
4) Sparce reward in common form:
    c1 - alpha1 * distance if distace > epsilon
    c2 - alpha2 * distance if disance < epsilin
5) Sparce reward with change in distance:
    delta distance if distace > epsilon
    delta distance + C1 if distace epsilon

## How to run
Run simple_PPO.py with the following arguments:
--mode in ['train', 'eval', 'viz', 'video'] for mode in which agent will run
--control_mode in ['end_effector, all_joints'] for type of action space
--random_reward - bool that indicate is reward is random every epidode
--reward_type - type of reward, sparce with distance by default
--model_path for weights of NN in eval, video or viz mode.
    There is few prerained options in models folder:
        end_effector.zip - with EEF action space
        all_joints.zip - with all joints action space


