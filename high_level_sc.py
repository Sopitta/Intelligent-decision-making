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
        self.center_y = [-17.7,-16.5]
        self.left_y = [-14.3,-13.0]
        self.right_y = [-20.8,-19.5]
        self.num_car = 0
        self.t_dist = 8
        self.safe_dist = 50
    
    def count_car(self,car1,car2,car3):
        car_list = []
        car_list.append(car1)
        car_list.append(car2)
        car_list.append(car3)
        self.num_car = 0
        for car in car_list:
            if car is not None:
                self.num_car = self.num_car + 1
        
                
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
    def get_action(self,player,car1,car2,car3):
        '''
        action
        stay = 1, change left = 2, change right =3
        '''
        #when there is no car in the road
        if self.num_car == 0:
            action = 1 #stay
        #when there is just one car in the road
        if self.num_car == 1:
            action = self.get_action_1(player,car1,car2,car3)
        if self.num_car == 2:
            action = self.get_action_2(player,car1,car2,car3)
        return action

    def get_action_1(self,player,car1,car2,car3): #get an action when there is one car in the environment.

         # player is in the middle lane.
         #if self.get_obs(player)[1] in self.center_y:
         if self.center_y[0] <= self.get_obs(player)[1] <= self.center_y[1]:
                if car1 is None:
                    action = 1 #stay
                else:
                    obs_player = self.get_obs(player)
                    obs_1 = self.get_obs(car1)
                    dist1 = self.euclidean_dist(obs_player,obs_1)
                    
                    #distance is large enough
                    if dist1 >  self.t_dist:
                        action = 1 #stay
                        
                    #in close distance and player is behind and player is slower   
                    if dist1 <=  self.t_dist and obs_player[0] > obs_1[0] and obs_player[3] < obs_1[3]:
                        action = 1 #stay
                    
                    #in close distance and player is behind and player is faster    
                    if dist1 <=  self.t_dist and obs_player[0] > obs_1[0] and obs_player[3] > obs_1[3]:
                        action = 2 #go lane left
                    
                    #in close distance and player is in the front and player is faster    
                    if dist1 <=  self.t_dist and obs_player[0] < obs_1[0] and obs_player[3] > obs_1[3]:
                        action = 1 #stay
                    
                    #in close distance and player is in the front and player is slow  
                    if dist1 <=  self.t_dist and obs_player[0] < obs_1[0] and obs_player[3] < obs_1[3]:
                        action = 2 #go lane left

         #player is in left lane            
         elif self.left_y[0] <= self.get_obs(player)[1] <= self.left_y[1]:
                if car2 is None: 
                    action = 1 #stay
                else:
                    obs_player = self.get_obs(player)
                    obs_2 = self.get_obs(car2)
                    dist2 = self.euclidean_dist(obs_player,obs_2)
                    
                    #distance is large enough
                    if dist2 >  self.t_dist:
                        action = 1 #stay
                        
                    #in close distance and player is behind and player is slower   
                    if dist2 <=  self.t_dist and obs_player[0] > obs_2[0] and obs_player[3] < obs_2[3]:
                        action = 1 #stay
                    
                    #in close distance and player is behind and player is faster    
                    if dist2 <=  self.t_dist and obs_player[0] > obs_2[0] and obs_player[3] > obs_2[3]:
                        action = 3 #go lane right
                    
                    #in close distance and player is in the front and player is faster    
                    if dist2 <=  self.t_dist and obs_player[0] < obs_2[0] and obs_player[3] > obs_2[3]:
                        action = 1 #stay
                    
                    #in close distance and player is in the front and player is faster    
                    if dist2 <=  self.t_dist and obs_player[0] < obs_2[0] and obs_player[3] < obs_2[3]:
                        action = 3 #go lane right  

         #player is in right lane            
         elif self.right_y[0] <= self.get_obs(player)[1] <= self.right_y[1]:
                if car3 is None: 
                    action = 1 #stay
                else:
                    obs_player = self.get_obs(player)
                    obs_3 = self.get_obs(car3)
                    dist3 = self.euclidean_dist(obs_player,obs_3)
                    
                    #distance is large enough
                    if dist3 >  self.t_dist:
                        action = 1 #stay
                        
                    #in close distance and player is behind and player is slower   
                    if dist3 <=  self.t_dist and obs_player[0] > obs_3[0] and obs_player[3] < obs_3[3]:
                        action = 1 #stay
                    
                    #in close distance and player is behind and player is faster    
                    if dist3 <=  self.t_dist and obs_player[0] > obs_3[0] and obs_player[3] > obs_3[3]:
                        action = 2 #go lane left
                    
                    #in close distance and player is in the front and player is faster    
                    if dist3 <=  self.t_dist and obs_player[0] < obs_3[0] and obs_player[3] > obs_3[3]:
                        action = 1 #stay
                    
                    #in close distance and player is in the front and player is faster    
                    if dist3 <=  self.t_dist and obs_player[0] < obs_3[0] and obs_player[3] < obs_3[3]:
                        action = 2 #go lane left
         else:
            action = 1

         return action

    def get_action_2(self,player,car1,car2,car3): #get an action when there is two cars in the environment.
         # player is in the middle lane.
         if self.center_y[0] <= self.get_obs(player)[1] <= self.center_y[1]:
                if car1 is None:
                    action = 1
                else:
                    obs_player = self.get_obs(player)
                    obs_1 = self.get_obs(car1)
                    dist1 = self.euclidean_dist(obs_player,obs_1)
                    
                    #distance is large enough
                    if dist1 >  self.t_dist:
                        action = 1 #stay
                        
                    #in close distance and player is behind and player is slower   
                    if dist1 <=  self.t_dist and obs_player[0] > obs_1[0] and obs_player[3] < obs_1[3]:
                        action = 1 #stay
                    
                    #in close distance and player is behind and player is faster    
                    if dist1 <=  self.t_dist and obs_player[0] > obs_1[0] and obs_player[3] > obs_1[3]:
                        if car2 is None:
                            action = 2 #go to lane left
                        else:
                            action = 3 #go to lane right
                    
                    #in close distance and player is in the front and player is faster    
                    if dist1 <=  self.t_dist and obs_player[0] < obs_1[0] and obs_player[3] > obs_1[3]:
                        action = 1 #stay
                    
                    #in close distance and player is in the front and player is slower   
                    if dist1 <=  self.t_dist and obs_player[0] < obs_1[0] and obs_player[3] < obs_1[3]:
                         if car2 is None:
                            action = 2 #go to lane left
                         else:
                            action = 3 #go to lane right

         elif self.left_y[0] <= self.get_obs(player)[1] <= self.left_y[1]: #player on the left lane with two cars
                if car2 is None:
                    action = 1
                else:
                    obs_player = self.get_obs(player)
                    obs_2 = self.get_obs(car2)
                    dist2 = self.euclidean_dist(obs_player,obs_2)
                    
                    #distance is large enough
                    if dist2 >  self.t_dist:
                        action = 1 #stay
                        
                    #in close distance and player is behind and the player is slower   
                    if dist2 <=  self.t_dist and obs_player[0] > obs_2[0] and obs_player[3] < obs_2[3]:
                        action = 1 #stay
                    
                    #in close distance and player is behind and the player is faster    
                    if dist2 <=  self.t_dist and obs_player[0] > obs_2[0] and obs_player[3] > obs_2[3]:
                        if car1 is None:
                            action = 3 #go to middel lane
                        else:
                            action = 3 #go to middle lane
                    
                    #in close distance and player is in the front and player is faster    
                    if dist2 <=  self.t_dist and obs_player[0] < obs_2[0] and obs_player[3] > obs_2[3]:
                        action = 1 #stay
                    
                    #in close distance and player is in the front and player is slower  
                    if dist2 <=  self.t_dist and obs_player[0] < obs_2[0] and obs_player[3] < obs_2[3]:
                        action = 3 #go to middle lane

         elif self.right_y[0] <= self.get_obs(player)[1] <= self.right_y[1]: #player on the right lane with two cars
                if car3 is None:
                    action = 1
                else:
                    obs_player = self.get_obs(player)
                    obs_3 = self.get_obs(car3)
                    dist3 = self.euclidean_dist(obs_player,obs_3)
                    
                    #distance is large enough
                    if dist3 >  self.t_dist:
                        action = 1 #stay
                        
                    #in close distance and player is behind and the player is slower   
                    if dist3 <=  self.t_dist and obs_player[0] > obs_3[0] and obs_player[3] < obs_3[3]:
                        action = 1 #stay
                    
                    #in close distance and player is behind and the player is faster    
                    if dist3 <=  self.t_dist and obs_player[0] > obs_3[0] and obs_player[3] > obs_3[3]:
                        if car1 is None:
                            action = 2 #go to middel lane
                        else:
                            action = 2 #go to middle lane
                    
                    #in close distance and player is in the front and player is faster    
                    if dist3 <=  self.t_dist and obs_player[0] < obs_3[0] and obs_player[3] > obs_3[3]:
                        action = 1 #stay
                    
                    #in close distance and player is in the front and player is slower  
                    if dist3 <=  self.t_dist and obs_player[0] < obs_3[0] and obs_player[3] < obs_3[3]:
                        action = 2 #go to middle lane
         else:
            action = 1
         return action
    
    def get_action_2_1(self,player,car1,car2): #get an action when there is two cars in the environment.
         # player is in the middle lane.
         if self.center_y[0] <= self.get_obs(player)[1] <= self.center_y[1]:
                if car1 is None:
                    action = 1
                else:
                    obs_player = self.get_obs(player)
                    obs_1 = self.get_obs(car1)
                    dist1 = self.euclidean_dist(obs_player,obs_1)
                    
                    #distance is large enough
                    if dist1 >  self.t_dist:
                        action = 1 #stay
                        
                    #in close distance and player is behind and player is slower   
                    if dist1 <=  self.t_dist and obs_player[0] > obs_1[0] and obs_player[3] < obs_1[3]:
                        action = 1 #stay
                    
                    #in close distance and player is behind and player is faster    
                    if dist1 <=  self.t_dist and obs_player[0] > obs_1[0] and obs_player[3] > obs_1[3]:
                        if car2 is None:
                            action = 2 #go to lane left
                        else:
                            obs_player = self.get_obs(player)
                            obs_2 = self.get_obs(car2)
                            dist2 = self.euclidean_dist(obs_player,obs_2)

                            # distance long enough and player is behind and player is slower.
                            if dist2 > self.safe_dist and obs_player[0] > obs_2[0] and obs_player[3] < obs_2[3]:
                                 action = 2 #go to lane left
                            # distance long enough and player is behind and player is faster.
                            elif dist2 > self.safe_dist and obs_player[0] > obs_2[0] and obs_player[3] > obs_2[3]:
                                 action = 2
                            # distance long enough and player is in the front and player is faster.
                            elif dist2 > self.safe_dist and obs_player[0] < obs_2[0] and obs_player[3] > obs_2[3]:
                                 action = 2
                            else:
                                action = 1 #stay

                    #in close distance and player is in the front and player is faster    
                    if dist1 <=  self.t_dist and obs_player[0] < obs_1[0] and obs_player[3] > obs_1[3]:
                        action = 1 #stay
                    
                    #in close distance and player is in the front and player is slower   
                    if dist1 <=  self.t_dist and obs_player[0] < obs_1[0] and obs_player[3] < obs_1[3]:
                         if car2 is None:
                            action = 2 #go to lane left
                         else:
                            obs_player = self.get_obs(player)
                            obs_2 = self.get_obs(car2)
                            dist2 = self.euclidean_dist(obs_player,obs_2)

                            # distance long enough and player is behind and player is slower.
                            if dist2 > self.safe_dist and obs_player[0] > obs_2[0] and obs_player[3] < obs_2[3]:
                                 action = 2 #go to lane left
                            # distance long enough and player is behind and player is faster.
                            elif dist2 > self.safe_dist and obs_player[0] > obs_2[0] and obs_player[3] > obs_2[3]:
                                 action = 2
                            # distance long enough and player is in the front and player is faster.
                            elif dist2 > self.safe_dist and obs_player[0] < obs_2[0] and obs_player[3] > obs_2[3]:
                                 action = 2
                            else:
                                action = 1 #stay



         elif self.left_y[0] <= self.get_obs(player)[1] <= self.left_y[1]: #player on the left lane with two cars
                if car2 is None:
                    action = 1
                else:
                    obs_player = self.get_obs(player)
                    obs_2 = self.get_obs(car2)
                    dist2 = self.euclidean_dist(obs_player,obs_2)
                    
                    #distance is large enough
                    if dist2 >  self.t_dist:
                        action = 1 #stay
                        
                    #in close distance and player is behind and the player is slower   
                    if dist2 <=  self.t_dist and obs_player[0] > obs_2[0] and obs_player[3] < obs_2[3]:
                        action = 1 #stay
                    
                    #in close distance and player is behind and the player is faster    
                    if dist2 <=  self.t_dist and obs_player[0] > obs_2[0] and obs_player[3] > obs_2[3]:
                        if car1 is None:
                            action = 3 #go to middel lane
                        else:
                            obs_player = self.get_obs(player)
                            obs_1 = self.get_obs(car1)
                            dist1 = self.euclidean_dist(obs_player,obs_1)

                            # distance long enough and player is behind and player is slower.
                            if dist1 > self.safe_dist and obs_player[0] > obs_1[0] and obs_player[3] < obs_1[3]:
                                 action = 3 #go to lane right
                            # distance long enough and player is behind and player is faster.
                            elif dist1 > self.safe_dist and obs_player[0] > obs_1[0] and obs_player[3] > obs_1[3]:
                                 action = 3
                            # distance long enough and player is in the front and player is faster.
                            elif dist1 > self.safe_dist and obs_player[0] < obs_1[0] and obs_player[3] > obs_1[3]:
                                 action = 3
                            else:
                                action = 1 #stay
                    
                    #in close distance and player is in the front and player is faster    
                    if dist2 <=  self.t_dist and obs_player[0] < obs_2[0] and obs_player[3] > obs_2[3]:
                        action = 1 #stay
                    
                    #in close distance and player is in the front and player is slower  
                    if dist2 <=  self.t_dist and obs_player[0] < obs_2[0] and obs_player[3] < obs_2[3]:
                        action = 3 #go to middle lane

         else:
            action = 1 #stay

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
        
    
        
        
        
        
        
        
        
        
        