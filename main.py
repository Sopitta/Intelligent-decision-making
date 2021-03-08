import glob
import os
import sys
import random
import time
import numpy as np
import pygame

try:
    #sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('D:/self-driving cars/simulator/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
     #sys.path.append(glob.glob('Z:/Documents/Carla/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from env import CarEnv
from agent.myagent import Agent




def game_loop():
    
    env = None
    pygame.init()
    
    try:
        env = CarEnv()
        agent = Agent(env.player)
        while True:
            ahead_dist = env
            control = agent.run_step()
            #print(control)
            env.player.apply_control(control)
    finally:
        if  env is not None:
            print('im here')
            env.destroy()
        pygame.quit()


        
try:
        game_loop()
except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

        
    





