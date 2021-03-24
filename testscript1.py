import glob
import os
import sys
try:
    #sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
    sys.path.append(glob.glob('D:/self-driving cars/simulator/CARLA_0.9.10.1/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('Z:/Documents/Carla/CARLA_0.9.10/WindowsNoEditor/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
    #sys.path.append(glob.glob('../self-driving/simulator/PythonAPI/carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2
import pygame

IM_WIDTH = 640
IM_HEIGHT = 480

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0


def process_ods(event):
    other = event.other_actor #carla.Actor
    if "vehicle" in other.type_id:
        dist = event.distance
        print("distance from the front car is" + str(dist))
        return dist
    
#https://github.com/copotron/sdv-course/blob/master/lesson0/camera.py   
def process_img2(disp, image):
    #image.save_to_disk('output/%05d.png' % image.frame, 
    #   carla.ColorConverter.Raw)
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

def game_loop():
    """ Main loop for agent"""

    #pygame.init()
    #pygame.font.init()
    world = None
    
    
   
    actor_list = []
    
    try:
        ##Modifiable Variables
        #targetLane = -3
        client = carla.Client('127.0.0.1', 2000)
        client.set_timeout(30.0)
        world = client.get_world()
        #world = client.load_world('Town01')
        #client.reload_world()
        #world = client.get_world()
        #map = client.get_available_maps()
        print(world)
        mymap = world.get_map()
        print(map)
        model_3 = world.get_blueprint_library().filter('vehicle.*model3*')[0]
        
        #transform = random.choice(world.get_map().get_spawn_points())
        waypoint_list = mymap.generate_waypoints(40)
        #print(waypoint_list[0])
        waypoint = waypoint_list[0]
        location = waypoint.transform.location + carla.Vector3D(0, 0, 1.5)
        rotation = waypoint.transform.rotation
        #vehicle = world.spawn_actor(model_3,transform)
        vehicle = world.spawn_actor(model_3, carla.Transform(location, rotation))
        #vehicle.set_autopilot(True)
        actor_list.append(vehicle)
        
        #add rgb cam
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors
        # get the blueprint for this sensor
        blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        # change the dimensions of the image
        #blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
        #blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
        #blueprint.set_attribute('fov', '110')
        # Adjust sensor relative to vehicle
        spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        #spawn the sensor and attach to vehicle.
        sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle,attachment_type=carla.AttachmentType.Rigid)
        actor_list.append(sensor)
        #sensor.listen(lambda data: process_img(data))
        sensor.listen(lambda data: process_img2(display,data))
        
        
        #add object detecion sensor
        blueprint_ods = world.get_blueprint_library().find('sensor.other.obstacle')
        blueprint_ods.set_attribute('only_dynamics', 'TRUE')
        blueprint_ods.set_attribute('debug_linetrace', 'TRUE')
        ods_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(yaw=0)) # Put this sensor on the windshield of the car.
        ods_sensor = world.spawn_actor(blueprint_ods, ods_transform, attach_to=vehicle,attachment_type=carla.AttachmentType.Rigid)
        actor_list.append(ods_sensor)
        ods_sensor.listen(lambda event: process_ods(event))
        
        vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1.0))
        vehicle.set_autopilot(True)
        #vehicle.set_light_state(self._lights)
        
        time.sleep(20)
        
    finally:
        #if world is not None:
            #world.destroy()
        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')

        pygame.quit()
        
def main():
     try:
        game_loop()

     except KeyboardInterrupt:  
         print('\nCancelled by user. Bye!')
    
if __name__ == '__main__':
    main()

        

