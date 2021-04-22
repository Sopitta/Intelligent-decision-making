import glob
import os
import sys
import numpy as np
import math
'''
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
'''
import carla
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, 'D:/Master thesis/agent/navigation')
sys.path.insert(1, 'C:/School/Master thesis/agent/navigation')
class HighLevelSC(object):
    def __init__(self,world):
        #self.a = None 
        self.world_h = world
        self.map_h = self.world_h.get_map()
    def get_obs(self, car):
        '''
        car:carla.Actor, type carla.Vehicle
        return: Observation 
        
        '''
        #print('getting obs')
        loc_xyz = car.get_location()
        t = car.get_transform()
        a_xyz = car.get_acceleration() #m/s2^2
        v_xyz = car.get_velocity() #m/s
        heading_rad = t.rotation.yaw * (np.pi/180) #rad
        v_ms = math.sqrt(v_xyz.x**2 + v_xyz.z**2 + v_xyz.z**2)
        x = loc_xyz.x
        y = loc_xyz.y
        
        return [x,y,heading_rad,v_ms]
    
    def euclidean_dist(self,obs1,obs2):
        '''
        obs1,obs2:Observations(list) 
        return: 
        dist :Distance between two vehicles calulated from their observations
        '''
        
        a = np.array((obs1[0],obs1[1])) #(2,)
        b = np.array((obs2[0],obs2[1])) #(2,)
        dist = np.linalg.norm(a-b)
        return dist
    
    def safe_action(self,dist):
        
        if dist <= 12:
            action = 1 #change lane
        else:
            action = 0 #stay in the same lane
            
        return action
    def get_xy_ref(self, player):
        '''
        player: a player, type carla.Vehicle
        return: 
        ref_wp :a carla waypoint that is used as a reference point in angle calculation
        '''
        #get player's current location.
        cur_t = player.get_transform()
        #the reference is in front of the player
        ref_xy = np.array((cur_t.location.x-10,cur_t.location.y))
        
        return ref_xy
    
    def get_xy_player(self, player):
        '''
        player: a player, type carla.Vehicle
        return: 
        ref_wp :a carla waypoint that is used as a reference point in angle calculation
        '''
        #get player's current location.
        cur_t = player.get_transform()
        #the reference is in front of the player
        player_xy = np.array((cur_t.location.x,cur_t.location.y))
        return player_xy
    
    def get_xy_car(self, car):
        '''
        player: a player, type carla.Vehicle
        return: 
        ref_wp :a carla waypoint that is used as a reference point in angle calculation
        '''
        #get player's current location.
        cur_t = car.get_transform()
        #the reference is in front of the player
        car_xy = np.array((cur_t.location.x,cur_t.location.y))
        return car_xy
    
    
    def get_theta_car(self,ref_xy,car_xy,player_xy):
        
        v0 = ref_xy - car_xy
        v1 = ref_xy - player_xy
        angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)) #radian
        return np.degrees(angle) #degree
        
    
        
        
        
        
        
        
        
        
        