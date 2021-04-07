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
from env import CarEnv
from agent.myagent import Agent
from high_level_sc import HighLevelSC




def game_loop():
    
    env = None
    pygame.init()
    
    try:
        env = CarEnv()
        player = env.player
        car1 = env.car1
        agent = Agent(player)
        spawn_point = env.map.get_spawn_points()[0]
        dest = carla.Transform(carla.Location(x=150, y=-193.3, z= 0.27530714869499207))
        #agent.set_destination((spawn_point.location.x,spawn_point.location.y,spawn_point.location.z))
        agent.set_destination((dest.location.x,dest.location.y,dest.location.z))
        highlevel_sc = HighLevelSC()
        prev_action = None
        action = None 
        run = True
        
        while run:
            
            #action = agent.safeaction(dist)
            #set destination again when return from local planner to global planner
                #if len(agent.local_plan._waypoints_queue)==0 and agent.local_plan._global_plan == False :#and car is in safe state
                #    agent.set_destination((spawn_point.location.x,
                #                   spawn_point.location.y,
                #                   spawn_point.location.z))
                player_obs = highlevel_sc.get_obs(player)
                car1_obs = highlevel_sc.get_obs(car1)
                dist = highlevel_sc.euclidean_dist(player_obs,car1_obs)
                #print(dist)
                #control = agent.run_step()
                action = highlevel_sc.safe_action(dist)
                #print(action)
                #control = agent.run_step()
                control = agent.run_step2(action,prev_action)
                player.apply_control(control)
                prev_action = action
            #except KeyboardInterrupt:
            #    print('interrupt')
            #    run = False
    finally:
        #if  env is not None:
        print('destroy')
        env.destroy()
        pygame.quit()


        
try:
        game_loop()
except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')

        
    





