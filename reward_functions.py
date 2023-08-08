import numpy as np
from numpy import linalg as LA


def Lp_reward(cur_pos, target_pos, p=2, **kwargs):
    distance = LA.norm(cur_pos - target_pos, ord=p)
    return -distance

def inverse_Lp_reward(cur_pos, target_pos, p=2, **kwargs):
    distance = LA.norm(cur_pos - target_pos, ord=p)
    return 1/distance

def Lp_reward_action_penalty(cur_pos, target_pos, actions=None, p=2, penalty_coef= 0.1, **kwargs):
    distance = LA.norm(cur_pos - target_pos, ord=p)
    reward = -distance - penalty_coef*LA.norm(actions, ord=p)
    return reward

def change_in_distance_rewards(cur_pos, target_pos, last_diffs=None, p=2, **kwargs):
    distance = LA.norm(cur_pos - target_pos, ord=p)
    last_distance = LA.norm(last_diffs, ord=p)
    reward = distance - last_distance
    return reward

def sparce_reward(cur_pos, target_pos, alpha1=1, alpha2=0.5, c1=0, c2=1, epsilon=0.05, p=1, **kwargs):
    # c1 - alpha1 * distance if distace > epsilon
    # c2 - alpha2 * distance if disance < epsilin
    distance = LA.norm(target_pos - cur_pos)#, ord=p)
    if distance > epsilon:
        reward = c1 - alpha1 * distance
        return reward
    else:
        reward = c2 - alpha2 * distance

        print('========================================')
        print(reward)
        return reward
    

def change_in_distance_rewards_sparce(cur_pos, target_pos, last_diffs=None, epsilon=0.15, p=2, **kwargs):
    distance = LA.norm(cur_pos - target_pos, ord=p)
    last_distance = LA.norm(last_diffs, ord=p)
    dist_change = distance - last_distance
    if distance > epsilon:
        reward = dist_change
        return reward
    else:
        reward = dist_change + 10
        return reward
