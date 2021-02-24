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

class Agent:
    def __init__(self, vehicle):
         self.vehicle = vehicle
    def safeaction(self):
        #if condition
            #action = 0
        #else:
            #action = 1
        action = 0
        return action
    def genwaypoints(self,action):
        #geneate way points based on the chosen action (velocity and acceleration)
        #will need a controller here
        pass
        
    

