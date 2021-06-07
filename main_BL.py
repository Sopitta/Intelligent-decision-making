import glob
import os
import sys
import random
import time
import numpy as np
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q

try:
    #sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('D:/self-driving cars/simulator/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('Z:/Documents/Carla/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('C:/School/Carla sim/CARLA_0.9.11/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
#from env import CarEnv, World, HUD
from Env.env_discrete import World, HUD
#from agent.myagent import Agent
from high_level_sc import HighLevelSC
from stable_baselines import DQN #get action from DQN and evn.step(action)
from stable_baselines.common.env_checker import check_env


train = True
training_number = 4
train_time_step = 12000000
method = 'dqn'
log_fold = "./sopitta_logs_{}/".format(method)
model_name = "./sopitta_logs_{}/{}_{}".format(method, method, training_number)
log_name = "log_{}_{}".format(method, training_number)
env = World()
if train:
        # TRAIN
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_fold)
        model.learn(total_timesteps=train_time_step, tb_log_name=log_name)
        model.save(model_name)
        print('Done training')

else:
        # EVALUATE
        model = DQN.load(model_name)
        obs = env.reset()
        for i in range(eval_range):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
        print('Done evaluation')

#check_env(env)

        
    





