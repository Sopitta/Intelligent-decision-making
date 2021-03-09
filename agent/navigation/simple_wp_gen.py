import glob
import os
import sys
try:
    #sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('D:/self-driving cars/simulator/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
     #sys.path.append(glob.glob('Z:/Documents/Carla/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from enum import Enum
from collections import deque
import random

sys.path.insert(1, 'D:/Master thesis/agent/navigation/controller')
sys.path.insert(1, 'D:/Master thesis')
from PIDcontroller import VehiclePIDController
#from navigation.controller import VehiclePIDController

import carla


class WaypointGen(object):
    MIN_DISTANCE_PERCENTAGE = 0.9
    def __init__(self, vehicle,opt_dict):
        self._vehicle = vehicle
        self._map = self.vehicle.get_world().get_map()
        self._target_speed = None
        self._sampling_radius = None
        self._min_distance = None
        
    def __del__(self):
        if self._vehicle:
            self._vehicle.destroy()
            print("Destroying ego-vehicle!")
            
    def _init_controller(self,opt_dict):
        """
        Controller initialization.

        :param opt_dict: dictionary of arguments.
        :return:
        """
        # default params
        self._target_speed = opt_dict['target_speed']
        self._dt = 1.0 / self._target_speed
        #self._dt = 1.0 / 20.0
        #self._target_speed = 20.0  # Km/h
        self._sampling_radius = self._target_speed * 1 / 3.6  # 1 seconds horizon
        self._min_distance = self._sampling_radius * self.MIN_DISTANCE_PERCENTAGE
        self._max_brake = 0.3
        self._max_throt = 0.75
        self._max_steer = 0.8
        args_lateral_dict = {
            'K_P': 1.95,
            'K_D': 0.2,
            'K_I': 0.07,
            'dt': self._dt}
        args_longitudinal_dict = {
            'K_P': 1.0,
            'K_D': 0,
            'K_I': 0.05,
            'dt': self._dt}

        # parameters overload
        #self._target_speed = opt_dict['target_speed']
       
        self._current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self._vehicle_controller = VehiclePIDController(self._vehicle,
                                                        args_lateral=args_lateral_dict,
                                                        args_longitudinal=args_longitudinal_dict,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer,)

        self._global_plan = False

        # compute initial waypoints
        self._waypoints_queue.append((self._current_waypoint.next(self._sampling_radius)[0])

        # fill waypoint trajectory queue
        action = None
        self.gen_wp(action,k=5) 
    def gen_wp(self,action,k=5):
        
        #stay
        if action == 0: #stay
            for _ in range(k):
                last_waypoint = self._waypoints_queue[-1]
                next_waypoints = last_waypoint.next(self._sampling_radius)[0]
                self._waypoints_queue.append(next_waypoint)
        elif action == 1: #generate waypoints in a left or right 
            for _ in range(k):
                last_waypoint = self._waypoints_queue[-1]
                
                if _ == 1:
                    eq_left_last_waypoint = last_waypoint.get_right_lane()
                    
            
            
        
        
        
        

