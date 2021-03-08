import glob
import os
import sys
import random
import time
import numpy as np
import pygame
import weakref

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


def process_ods(event):
    other = event.other_actor #carla.Actor
    if "vehicle" in other.type_id:
        dist = event.distance
        print("distance from the front car is" + str(dist))
        return dist
'''    
#https://github.com/copotron/sdv-course/blob/master/lesson0/camera.py   
def process_img(disp, image):
    
    org_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
    array = np.reshape(org_array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:,:,::-1]
    array = array.swapaxes(0,1)
    surface = pygame.surfarray.make_surface(array)
    disp.blit(surface, (200,0))
    pygame.display.flip()

display = pygame.display.set_mode(
        (1200, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF
    )

'''
#https://www.programcreek.com/python/?code=erdos-project%2Fpylot%2Fpylot-master%2Fscripts%2Ftest_canny_lane_detection.py
def spawn_driving_vehicle(client, world):
    """ This function spawns the driving vehicle and puts it into
    an autopilot mode.

    Args:
        client: The carla.Client instance representing the simulation to
          connect to.
        world: The world inside the current simulation.

    Returns:
        A carla.Actor instance representing the vehicle that was just spawned.
    """
    # Get the blueprint of the vehicle and set it to AutoPilot.
    vehicle_bp = random.choice(
        world.get_blueprint_library().filter('vehicle.*'))
    while not vehicle_bp.has_attribute('number_of_wheels') or not int(
            vehicle_bp.get_attribute('number_of_wheels')) == 4:
        vehicle_bp = random.choice(
            world.get_blueprint_library().filter('vehicle.*'))
    vehicle_bp.set_attribute('role_name', 'autopilot')

    # Get the spawn point of the vehicle.
    start_pose = random.choice(world.get_map().get_spawn_points())

    # Spawn the vehicle.
    batch = [
        carla.command.SpawnActor(vehicle_bp, start_pose).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True))
    ]
    vehicle_id = client.apply_batch_sync(batch)[0].actor_id

    # Find the vehicle and return the carla.Actor instance.
    time.sleep(0.5)  # This is so that the vehicle gets registered in the actors.
    return world.get_actors().find(vehicle_id)


class CarEnv(object):    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        #self.world = self.client.load_world('Town01')
        #self.map = self.world.get_map()
        self.player = None
        self.rgb_cam =  None
        
        self.reset()
        
        
    def reset(self):
        
        #self.actor_list = []
        
        #self.blueprint_library = self.world.get_blueprint_library()
        model_3 = self.world.get_blueprint_library().filter("model3")[0]
        
        '''
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(model_3, spawn_point)
            
        while self.player is None:
            spawn_points = self.map.get_spawn_points()
            spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
            self.player = self.world.try_spawn_actor(model_3, spawn_point)
        '''
        
        
        #spawn a player at a random spawn points
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.player = self.world.spawn_actor(model_3, self.transform)
        #self.actor_list.append(self.player)
        print(self.player)
        
        self.rgb_cam = RGBCamera(self.player)
        #self.ods_sensor = ObjectDetectionSensor(self.player)
        
        #self.actor_list.append(self.rgb_cam.sensor)
        #self.actor_list.append(self.ods_sensor.sensor)
        
        
        '''
        #spawn a rgb camera attached to the player.
        self.blueprint_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.rgb_cam = self.world.spawn_actor(self.blueprint_cam, self.cam_transform, attach_to=self.player,attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.rgb_cam)
        self.rgb_cam.listen(lambda data: process_img(display,data))
         
         
        #spawn an object detection sensor attached to the player.
        self.blueprint_ods = self.blueprint_library.find('sensor.other.obstacle')
        self.blueprint_ods.set_attribute('only_dynamics', 'TRUE')
        self.blueprint_ods.set_attribute('debug_linetrace', 'TRUE')
        self.ods_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0)) # Put this sensor on the windshield of the car.
        self.ods_sensor = self.world.spawn_actor(self.blueprint_ods, self.ods_transform, attach_to=self.player)
        self.actor_list.append(self.ods_sensor)
        self.adist = self.ods_sensor.listen(lambda event: process_ods(event))
        '''
        
        
        #spawn other vehicles.
        #self.player.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
        #self.player.set_autopilot(True)
        #time.sleep(60)
        
        #print('destroying actors')
        #for actor in self.actor_list:
        #    actor.destroy()
        #print('done.')

        #pygame.quit()
    def destroy(self):
        
         """Destroys all actors"""
         actors = [self.rgb_cam.sensor, self.player]
         print('destroying actors')
         for actor in actors:
             
             if actor is not None:
                 actor.destroy()
        #pygame.quit()
 
    
# ==============================================================================
# -- ObjectDetectionSensor -----------------------------------------------------------
# ==============================================================================   
            
class ObjectDetectionSensor(object):
     def __init__(self, parent_actor):
         self.sensor = None
         self.parent = parent_actor
         self.ahead_dist = 100
         world = self.parent.get_world()
         bp = world.get_blueprint_library().find('sensor.other.obstacle')
         bp.set_attribute('only_dynamics', 'TRUE')
         bp.set_attribute('debug_linetrace', 'TRUE')
         self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self.parent)
         weak_self = weakref.ref(self)
         self.sensor.listen(lambda event: ObjectDetectionSensor.on_detection(weak_self, event))
         
     @staticmethod
     def on_detection(weak_self,event):
         self = weak_self()
         if not self:
             
             return
         other = event.other_actor #carla.Actor
         if "vehicle" in other.type_id:
             dist = event.distance
             print("distance from the front car is" + str(dist))
             self.ahead_dist = dist
             
# ==============================================================================
# -- RGBCamera -----------------------------------------------------------
# ==============================================================================   
             
class RGBCamera(object):
     def __init__(self, parent_actor):
         self.sensor = None
         self.parent = parent_actor
         world = self.parent.get_world()
         bp = world.get_blueprint_library().find('sensor.camera.rgb')
         transform = carla.Transform(carla.Location(x=2.5, z=0.7))
         self.sensor = world.spawn_actor(bp, transform, attach_to=self.parent,attachment_type=carla.AttachmentType.Rigid)
         weak_self = weakref.ref(self)
         display = pygame.display.set_mode((1200, 600),pygame.HWSURFACE | pygame.DOUBLEBUF)
         self.sensor.listen(lambda data: RGBCamera.process_img(weak_self,display,data))
         
     @staticmethod
     def process_img(weak_self,disp,image):
         self = weak_self()
         #if not self:
         #    return
        
         org_array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
         array = np.reshape(org_array, (image.height, image.width, 4))
         array = array[:, :, :3]
         array = array[:,:,::-1]
         array = array.swapaxes(0,1)
         surface = pygame.surfarray.make_surface(array)
         disp.blit(surface, (200,0))
         pygame.display.flip()


        
         
    
        
        
    
        
        
         
         
         
         