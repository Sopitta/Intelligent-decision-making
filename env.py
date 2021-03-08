import glob
import os
import sys
import random
import time
import numpy as np
import pygame

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


class CarEnv:    
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        #self.world = self.client.load_world('Town01')
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]
        self.player = None
        self.adist = None
        
    def reset(self):
        
        self.actor_list = []
         
        #spawn a player at a random spawn points
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.player = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.player)
         
        #spawn a rgb camera attached to the player.
        self.blueprint_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.cam_transform = carla.Transform(carla.Location(x=2.5, z=0.7))

        #spawn the sensor and attach to vehicle.
        self.rgb_cam = self.world.spawn_actor(self.blueprint_cam, self.cam_transform, attach_to=self.player,attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.rgb_cam)
        #sensor.listen(lambda data: process_img(data))
        self.rgb_cam.listen(lambda data: process_img(display,data))
         
        #spawn an object detection sensor attached to the player.
        self.blueprint_ods = self.blueprint_library.find('sensor.other.obstacle')
        self.blueprint_ods.set_attribute('only_dynamics', 'TRUE')
        self.blueprint_ods.set_attribute('debug_linetrace', 'TRUE')
        self.ods_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0)) # Put this sensor on the windshield of the car.
        self.ods_sensor = self.world.spawn_actor(self.blueprint_ods, self.ods_transform, attach_to=self.player)
        self.actor_list.append(self.ods_sensor)
        self.adist = self.ods_sensor.listen(lambda event: process_ods(event))
        
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
        for actor in self.actor_list:
            actor.destroy()
 
#create sensors object    
        
        
    
        
        
         
         
         
         