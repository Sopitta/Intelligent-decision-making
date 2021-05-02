#modified from https://github.com/Sentdex/Carla-RL
import glob
import os
import sys
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
#sys.path.insert(1, 'C:/School/Master thesis/agent/navigation')
#sys.path.append('C:/School/Master thesis/agent/navigation')
#print(sys.path)
#sys.path.insert(0, '..')
#from navigation.waypoint_gen import WaypointGen
from agent.navigation.local_planner import LocalPlanner
from agent.navigation.global_route_planner import GlobalRoutePlanner
from agent.navigation.global_route_planner_dao import GlobalRoutePlannerDAO


class Agent:
    def __init__(self, vehicle, target_speed = 20):
         self.vehicle = vehicle
         self.local_plan = LocalPlanner(self.vehicle, opt_dict={'target_speed' : target_speed}) 
         self.world = self.vehicle.get_world()
         self.map = self.world.get_map()
         
         self._hop_resolution = 2.0
         self.action = None
         self._grp = None
          
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
    def trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self.vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route
    
    def run_step(self):
        control = self.local_plan.run_step()
        return control
    
    def run_step2(self,action,prevaction):
        control = self.local_plan.run_step2(action,prevaction)
        return control

    def run_step3(self,action,prevaction):
        control = self.local_plan.run_step3(action,prevaction)
        return control
    
    
    
    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self.map.get_waypoint(self.vehicle.get_location())
        end_waypoint = self.map.get_waypoint(carla.Location(location[0], location[1], location[2]))

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        print(len(route_trace))
        self.local_plan.set_global_plan(route_trace)
    
    
        
    

