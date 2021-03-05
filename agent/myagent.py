#modified from https://github.com/Sentdex/Carla-RL
import glob
import os
import sys
try:
    #sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('D:/self-driving cars/simulator/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from navigation.waypoint_gen import WaypointGen

class Agent:
    def __init__(self, vehicle, target_speed = 20):
         self.vehicle = vehicle
         self.waypoint_gen = WaypointGen(self._vehicle, opt_dict={'target_speed' : target_speed)
         self.targ                                                         
    def safeaction(self):
        #if condition
            #action = 0
        #else:
            #action = 1
        action = 0
        return action
    def genwaypoints(self):
        #geneate way points based on the chosen action (velocity and acceleration)
        #will need a controller here
        next_wp = self.waypoint_gen._compute_next_waypoints()
        return next_wp 
    def getcontrol(self):
        control = self.waypoint_gen._compute_next_waypoints()
        return control
    
        
        
    

