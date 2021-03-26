import glob
import os
import sys
import numpy as np
import math
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
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D:/Master thesis/agent/navigation')

class HighLevelSC(object):
    def __init__(self):
        self.a = None
        
    def get_obs(self, car):
        '''
        car:carla.Actor, type carla.Vehicle
        return: Observation 
        
        '''
        loc_xyz = car.get_location()
        t = car.get_transform()
        a_xyz = car.get_acceleration() #m/s2^2
        v_xyz = car.get_velocity() #m/s
        heading_deg = t.rotation.yaw #deg
        heading_rad = heading_deg * (np.pi/180) #rad
        v_ms = math.sqrt(loc_xyz.x**2 + loc_xyz.z**2 + loc_xyz.z**2)
        return [loc_xyz.x,loc.xyz.y,heading_rad,v_ms]
    
    def euclidean_dist(self,obs1,obs2):
        '''
        obs1/2:Observations(list) 
        return: 
        dist :Distance between two vehicles calulated from their observations
        '''
        a = np.array((obs1[0],obs1[1])) #(2,)
        b = np.array((obs2[0],obs2[1])) #(2,)
        dist = np.linalg.norm(a-b)
        return dist
    
    def safe_action(dist):
        
        if dist < = 5:
            action = 1 #change lane
        else:
            action = 0 #stay in the same lane
            
        
        
        
        
        
        
        
        
        