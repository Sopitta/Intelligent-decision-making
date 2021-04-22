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
from env import CarEnv, World, HUD
from agent.myagent import Agent
from high_level_sc import HighLevelSC




def game_loop():
    
    env = None
    pygame.init()
    pygame.font.init()
    
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(4.0)
        display = pygame.display.set_mode((1280, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(1280, 720)
        #env = CarEnv()
        world = client.load_world('Town06')
        env = World(world, hud)
        #env = World(client.get_world(), hud)
        player = env.player
        car1 = env.car1
        agent = Agent(player)
        spawn_point = env.map.get_spawn_points()[0]
        #dest = carla.Transform(carla.Location(x=150, y=-193.3, z= 0.27530714869499207))
        dest = carla.Transform(carla.Location(x=8.4, y=205.1, z= 0.0))
        #agent.set_destination((spawn_point.location.x,spawn_point.location.y,spawn_point.location.z))
        agent.set_destination((dest.location.x,dest.location.y,dest.location.z))
        highlevel_sc = HighLevelSC(world)
        prev_action = None
        action = None 
        run = True
        clock = pygame.time.Clock()
        while run:
                
                env.tick(clock)
                env.render(display)
                pygame.display.flip()
            
            #action = agent.safeaction(dist)
            #set destination again when return from local planner to global planner
                #if len(agent.local_plan._waypoints_queue)==0 and agent.local_plan._global_plan == False :#and car is in safe state
                #    agent.set_destination((spawn_point.location.x,
                #                   spawn_point.location.y,
                #                   spawn_point.location.z))
                player_obs = highlevel_sc.get_obs(player)
                car1_obs = highlevel_sc.get_obs(car1)
                dist = highlevel_sc.euclidean_dist(player_obs,car1_obs)
                ref = highlevel_sc.get_xy_ref(player)
                player_xy = highlevel_sc.get_xy_player(player)
                car1_xy = highlevel_sc.get_xy_car(car1)
                theta = highlevel_sc.get_theta_car(ref,car1_xy,player_xy)
                print(theta)
                control = agent.run_step()
                action = highlevel_sc.safe_action(dist)
                #control = agent.run_step2(action,prev_action)
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
except KeyboardInterrupt or K_ESCAPE:
        print('\nCancelled by user. Bye!')

        
    





