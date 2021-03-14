#modified from https://github.com/Sentdex/Carla-RL
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

import carla
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'D:/Master thesis/agent/navigation')
#sys.path.insert(0, '..')
#from navigation.waypoint_gen import WaypointGen
from waypoint_gen import WaypointGen
from global_route_planner import GlobalRoutePlanner
from global_route_planner_dao import GlobalRoutePlannerDAO

class Agent:
    def __init__(self, vehicle, target_speed = 20):
         self.vehicle = vehicle
         self.waypoint_gen = WaypointGen(self.vehicle, opt_dict={'target_speed' : target_speed}) 
         self._hop_resolution = 2.0
         self.action = None
    def safeaction(self,distance):
        if distance < 5:
            self.action = 1 #change
        else:
            self.action = 0 #stay
        
        return self.action
    '''
    def genwaypoints(self):
        #geneate way points based on the chosen action (velocity and acceleration)
        #will need a controller here
        next_wp = self.waypoint_gen._compute_next_waypoints()
        return next_wp 
    '''
    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route
    
    def run_step(self,action):
        control = self.waypoint_gen.run_step(action)
        return control
    
    
        
        
    

