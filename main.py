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
        spawn_point = env.map.get_spawn_points()[0]
        agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))
        
        while True:
            
            #action = agent.safeaction(dist)
            #set destination again when return from local planner to global planner
            if len(agent.local_plan._waypoints_queue)==0 and agent.local_plan._global_plan == False :#and car is in safe state
                agent.set_destination((spawn_point.location.x,
                                   spawn_point.location.y,
                                   spawn_point.location.z))
            
            control = agent.run_step()
            #print(control)
            env.player.apply_control(control)
    finally:
        if  env is not None:
            env.destroy()
        pygame.quit()


        
try:
        game_loop()
except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

        
    





