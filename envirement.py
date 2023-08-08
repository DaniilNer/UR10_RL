import gym
import time
import numpy as np
import pybullet
import pybullet_data
from typing import Optional
from numpy import linalg as LA
from functools import partial
from reward_functions import *


REWARDS = {"L1": partial(Lp_reward, p=1),
           "L2": partial(Lp_reward, p=2),
           "L4": partial(Lp_reward, p=4),
           "L1_with_action_penalty": partial(Lp_reward_action_penalty, p=1, penalty_coef=0.1),
           "L2_with_action_penalty": partial(Lp_reward_action_penalty, p=2, penalty_coef=0.1),
           "sparce_reward": partial(sparce_reward, alpha1=1, alpha2=0.5, c1=0, c2=1, epsilon=0.1, p=2),
           "inverse_Lp_reward": partial(inverse_Lp_reward, p=2)
           } 

class UR10_env(gym.Env):
    #observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(15,), dtype='float32')

    #action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=float)

    def __init__(self, 
                 mode: str='train', 
                 reward_type: str = "sparce_reward",
                 reward_kwargs: Optional[dict] = None,
                 control_mode: str = "end_effector", #"end_effector", #"end_effector",
                 max_episode_length: Optional[int] = 300,
                 goal_env=True,
                 random_reward=True,
                 epsilon=0.02):
        

        assert mode in ("train", "test", "visulization"), "invalid mode"
        assert reward_type in REWARDS, "invalid reward type"
        assert control_mode in ("end_effector", "all_joints"), "invalid control mode"
        self.control_mode = control_mode
        self.random_reward = random_reward

        self.shape_obs = 9 if self.control_mode == 'end_effector' else 15
        self.shape_act = 3 if self.control_mode == 'end_effector' else 6
        self.epsilon = epsilon
        self.goal_env = goal_env
        if not goal_env:
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.shape_obs,), dtype='float32') 
        else:
            self.observation_space = gym.spaces.Dict(dict(
                                                desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                                                achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype='float32'),
                                                observation=gym.spaces.Box(-np.inf, np.inf, shape=(self.shape_obs,), dtype='float32'),
                                                ))

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.shape_act,), dtype='float32')

        self.max_episode_steps = max_episode_length
        self.current_step = 0
        self.dv = 0.05

        self.reward = partial(REWARDS[reward_type], **(reward_kwargs if reward_kwargs is not None else {}))
        if mode == "visulization":
            self.vis = True
        else:
            self.vis = False
        self.client = pybullet.connect(pybullet.GUI if self.vis else pybullet.DIRECT)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        #self.init_ur10()

    #def compute_terminated(self, achieved_goal, desired_goal, info):
    #    pass
    #def compute_truncated(self, achieved_goal, desired_goal, info):
    #    pass

    def init_ur10(self):
        pybullet.setGravity(0, 0, -10)
        self.planeID = pybullet.loadURDF("plane.urdf")
        self.tableID = pybullet.loadURDF('table/table.urdf', globalScaling=1, basePosition=[0.5, 0, 0.])
        _, m = pybullet.getAABB(self.tableID)
        ur10_position = [0, 0, m[-1]+0.25] 
        self.ur10_orientation = pybullet.getQuaternionFromEuler([0, 0, 0]) 
        self.ur10 = pybullet.loadURDF("UR10_r.urdf", ur10_position, self.ur10_orientation)
        self.__parse_joint_info__()
        self.__bese_position__()
        self.position_restrict = [(0.5, 1.0), (-0.35, 0.35), (m[-1]+0.05, 1)]
        self.current_step = 0
    
    def start_video(self, filename):
        pybullet.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, filename)
        time.sleep(3)

    def stop_video(self):
        pybullet.stopStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4)

    def move(self, coords):
        if self.control_mode == 'all_joints':
            joint_poses = coords
        else:
            joint_poses = pybullet.calculateInverseKinematics(self.ur10, 
                                                            self.end_point, 
                                                            coords,
                                                            pybullet.getQuaternionFromEuler([np.pi, 0, 0]),
                                                            #self.ur10_orientation,
                                                            #lowerLimits = self.arm_lower_limits, 
                                                            #upperLimits = self.arm_upper_limits, 
                                                            #jointRanges = self.arm_joint_ranges, 
                                                            maxNumIterations=100,
                                                            residualThreshold=0.01)
        
        for i, joint_id in enumerate(self.controllable_joints):
            pybullet.setJointMotorControl2(self.ur10, 
                                           joint_id, 
                                           pybullet.POSITION_CONTROL, 
                                           targetPosition=joint_poses[i],
                                           #force=self.joints[joint_id]["jointMaxForce"], 
                                           #maxVelocity=self.joints[joint_id]['jointMaxVelocity']
                                           )
        pybullet.stepSimulation()

    def __bese_position__(self):
        #rest_pose = []
        #for ranges in zip(self.arm_lower_limits, self.arm_upper_limits):
        #    rest_pose.append(np.random.uniform(low=ranges[0], high=ranges[1]))
        rest_pose = [0,-1.7, 1.5, 0, 0, 0]
        for i, joint_id in enumerate(self.controllable_joints):
            pybullet.resetJointState(self.ur10, joint_id, rest_pose[i])
        pybullet.stepSimulation()

    def __parse_joint_info__(self): 
        self.joints = []
        self.controllable_joints = []
        for i in range(pybullet.getNumJoints(self.ur10)):
            jointInfo = pybullet.getJointInfo(self.ur10, i)
            info = {
                'jointID': jointInfo[0],
                'jointName': jointInfo[1].decode('utf-8'),
                'jointType': jointInfo[2], # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
                'jointLowerLimit': jointInfo[8],
                'jointUpperLimit': jointInfo[9],
                'jointMaxForce': jointInfo[10],
                'jointMaxVelocity': jointInfo[11],
                "controllable": jointInfo[2] != pybullet.JOINT_FIXED
            }
            if info["controllable"]:
                self.controllable_joints.append(info["jointID"])
                #pybullet.setJointMotorControl2(self.ur10, info["jointID"], pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints.append(info)
            if info['jointName'] == 'flange-tool0':
                self.end_point = info['jointID']
        self.arm_lower_limits = [info['jointLowerLimit'] for info in self.joints if info["controllable"]]
        self.arm_upper_limits = [info['jointUpperLimit'] for info in self.joints if info["controllable"]]
        self.arm_joint_ranges = [info['jointUpperLimit'] - info['jointLowerLimit'] for info in self.joints if info["controllable"]]

    def reset(self):
        pybullet.resetSimulation()
        self.init_ur10()

        if self.random_reward:
            x_position = np.random.uniform(0.6, 1)
            y_position = np.random.uniform(-0.3, 0.3)
        else:
            x_position =  0.7 
            y_position = 0 
        self.current_target = np.array([x_position, y_position, 0.7])
        if self.control_mode == 'all_joints':
            self.joint_values = [0,-1.7, 1.5, 0, 0, 0]
        else:
            self.joint_values = np.array([1.0, 0.0, 1.0])
        return self.get_observation()

    def rescale_for_restrict(self, coords):
        new_coords = np.zeros(3)
        for i in range(len(self.position_restrict)):
             new_coords[i] = (coords[i] + 1) / 2 * (self.position_restrict[i][1] - self.position_restrict[i][0]) + self.position_restrict[i][0]
        return new_coords
    
    def get_observation(self):
        state = np.zeros(self.shape_obs)
        eef_pos, _, _, _, _, _, eef_v, eef_av = pybullet.getLinkState(self.ur10, linkIndex=self.end_point, computeLinkVelocity=True)
        state[:3] = eef_pos
        #state[9:12] = eef_v
        #state[12:15] = eef_av
        #state[9:12] = eef_pos - self.current_target
        #state[12:15] = self.current_target
        state[3:6] = self.current_target - eef_pos
        state[6:9] = self.current_target
        self.current_diff = self.current_target - eef_pos
        if self.control_mode == 'all_joints':
            positions = []
            velocities = []
            for joint_id in self.controllable_joints:
                pos, vel, _, _ = pybullet.getJointState(self.ur10, joint_id)
                positions.append(pos)
                velocities.append(vel)
        if self.goal_env:
            return {'observation': np.float32(state), 'desired_goal': self.current_target, 'achieved_goal': eef_pos} #state
        else:
            return np.float32(state)
    
    def get_reward(self):
        eef_pos, _, _, _, _, _, eef_v, eef_av = pybullet.getLinkState(self.ur10, linkIndex=self.end_point, computeLinkVelocity=True)
        reward = self.compute_reward(eef_pos, self.current_target, None)
        return reward
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        reward = self.reward(achieved_goal, desired_goal, actions=self.last_action, last_diffs=self.current_diff, epsilon=self.epsilon)
        return reward

    
    def done_gen(self):
        if LA.norm(self.current_diff) < self.epsilon:
            done, reached = True, True
        elif self.current_step == self.max_episode_steps:
            done, reached = True, False
        else:
            done, reached = False, False
        return done, reached
    
    def step(self, action):
        self.last_action = np.clip(action * self.dv, -1, 1)
        if self.control_mode == 'all_joints':
            self.joint_values += self.last_action
            target_pos = self.joint_values
        else: 
            #self.joint_values = np.array(pybullet.getLinkState(self.ur10, self.end_point)[0])
            self.joint_values += self.last_action
            target_pos = self.rescale_for_restrict(self.joint_values)
        self.move(target_pos)
        self.current_diff = self.current_target - np.array(pybullet.getLinkState(self.ur10, self.end_point)[0])
        reward = self.get_reward()
        done, reach_flag = self.done_gen()
        self.current_step += 1
        return self.get_observation(), reward, done, {"reach": reach_flag}
    
    #def __del__(self):
    #    pybullet.disconnect()

