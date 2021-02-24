#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import time

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    #sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('D:/self-driving cars/simulator/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import random
import time
import numpy as np
import cv2

#from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
#from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
#from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error



#from agents.navigation.global_route_planner import GlobalRoutePlanner
#from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

IM_WIDTH = 640
IM_HEIGHT = 480

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================



def game_loop(args):
    """ Main loop for agent"""

    pygame.init()
    pygame.font.init()
    world = None
    tot_target_reached = 0
    num_min_waypoints = 21
    
    actor_list = []

    try:
        
        ##Modifiable Variables
        targetLane = -3

        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        blueprint = world.get_blueprint_library().filter('vehicle.*model3*')[0]
        
        map = world.get_map()
        
        #loc = carla.Location(args.x, args.y, args.z)
        #print("Initial location: ", loc)
        #current_w = map.get_waypoint(loc)
        
        
    
        '''
        dao = GlobalRoutePlannerDAO(map, 2.0)
        grp = GlobalRoutePlanner(dao)
        grp.setup()

        a = carla.Location(x=96.0,
                           y=4.45,
                           z=0)
        b = carla.Location(x=215.0,
                           y=6.23,
                           z=0)
        w1 = grp.trace_route(a, b)
        '''
        waypoint_list = map.generate_waypoints(40)
        #print(waypoint_list[0])
        waypoint = waypoint_list[0]
        location = waypoint.transform.location + carla.Vector3D(0, 0, 1.5)
        rotation = waypoint.transform.rotation
        vehicle = world.spawn_actor(blueprint, carla.Transform(location, rotation))
        actor_list.append(vehicle)
        print("SPAWNED!")
        #print(waypoint.lane_type)
        
        #Vehicle properties setup
        physics_control = vehicle.get_physics_control()
        max_steer = physics_control.wheels[0].max_steer_angle
        rear_axle_center = (physics_control.wheels[2].position + physics_control.wheels[3].position)/200
        offset = rear_axle_center - vehicle.get_location()
        wheelbase = np.linalg.norm([offset.x, offset.y, offset.z])
        vehicle.set_simulate_physics(True)
       
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=-10,z=10), carla.Rotation(-45,0,0))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        camera.listen(lambda data: process_img(data))
        #time.sleep(10)
        
        vehicle.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
        time.sleep(10)
        #vehicle.get_location
        a1=vehicle.get_location()
        current_w = map.get_waypoint(a1)
        w = current_w.get_left_lane()
        # Get the next waypoint(s) at particular distance using
        #next_w = left_w.next(20)
        #next_w.insert(0,left_w)
        #print(len(next_w))
        
        #w = left_w 
        i = 0
        while True:
            i =  i + 1
        #for w in next_w:
            #print("here")
            world.debug.draw_point(w.transform.location, life_time=5)
            #Control vehicle's throttle and steering
            throttle = 0.4
            vehicle_transform = vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            steer = control_pure_pursuit(vehicle_transform, w.transform, max_steer, wheelbase)
            control = carla.VehicleControl(throttle, steer)
            vehicle.apply_control(control)
            #time.sleep(0.15)
            w = w.next(1.5)[0]
            if i == 1000:
                break
            
                
        
        #print(left_w.lane_type)
        #vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=-0.2))
        #time.sleep(2)
        #vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.2))
        #time.sleep(2)
        #vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
        #time.sleep(5)
        
    
    
    finally:
        #if world is not None:
            #world.destroy()
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')

        pygame.quit()



#Returns only the waypoints that are not along the straights
def get_curvy_waypoints(waypoints):
    curvy_waypoints = []
    for i in range(len(waypoints) - 1):
        x1 = waypoints[i].transform.location.x
        y1 = waypoints[i].transform.location.y
        x2 = waypoints[i+1].transform.location.x
        y2 = waypoints[i+1].transform.location.y
        if (abs(x1 - x2) > 1) and (abs(y1 - y2) > 1):
            print("x1: " + str(x1) + "  x2: " + str(x2))
            print(abs(x1 - x2))
            print("y1: " + str(y1) + "  y2: " + str(y2))
            print(abs(y1 - y2))
            curvy_waypoints.append(waypoints[i])
      
    #To make the path reconnect to the starting location
    curvy_waypoints.append(curvy_waypoints[0])

    return curvy_waypoints


def control_pure_pursuit(vehicle_tr, waypoint_tr, max_steer, wheelbase):
    # TODO: convert vehicle transform to rear axle transform
    wp_loc_rel = relative_location(vehicle_tr, waypoint_tr.location) + carla.Vector3D(wheelbase, 0, 0)
    wp_ar = [wp_loc_rel.x, wp_loc_rel.y]
    d2 = wp_ar[0]**2 + wp_ar[1]**2
    steer_rad = math.atan(2 * wheelbase * wp_loc_rel.y / d2)
    steer_deg = math.degrees(steer_rad)
    steer_deg = np.clip(steer_deg, -max_steer, max_steer)
    return steer_deg / max_steer

def relative_location(frame, location):
  origin = frame.location
  forward = frame.get_forward_vector()
  right = frame.get_right_vector()
  up = frame.get_up_vector()
  disp = location - origin
  x = np.dot([disp.x, disp.y, disp.z], [forward.x, forward.y, forward.z])
  y = np.dot([disp.x, disp.y, disp.z], [right.x, right.y, right.z])
  z = np.dot([disp.x, disp.y, disp.z], [up.x, up.y, up.z])
  return carla.Vector3D(x, y, z)

# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
