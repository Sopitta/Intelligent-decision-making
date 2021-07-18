"""Example of automatic vehicle control from client side."""

import glob
import os
import sys
import random
import time
import numpy as np
import pygame
from pygame.locals import KMOD_CTRL
from pygame.locals import K_ESCAPE
from pygame.locals import K_q
import weakref

import argparse
import collections
import datetime
import logging
import math
import os
import re
import matplotlib.pyplot as plt

import carla
from carla import ColorConverter as cc
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from RL.myRL import RL
import gym
from gym import spaces
from stable_baselines.common.env_checker import check_env
from agent.myagent import Agent
from high_level_sc import HighLevelSC
from RL.safety_rules import safety_rules


class World(gym.Env):    
    def __init__(self):

        super(World, self).__init__()
        
        client = carla.Client("localhost", 2000)
        client.set_timeout(4.0)

        #initialize pygame
        pygame.init()
        pygame.font.init()

        hud = HUD(1280, 720)
        self.world = client.load_world('Town06')
        self.map = self.world.get_map()


        #self.action_space = spaces.Discrete(3) #action 0,1,2 
        #self.action_space = spaces.Box(np.array([-1, 0]), np.array([+1, +1]), dtype=np.float32 )  # steer, throttle
        self.action_space = spaces.Box(np.array([-1]), np.array([1]), dtype=np.float32 )  # throttle

        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            #shape=(9,336,150), 
            shape = (453600,),
            dtype=np.float32)
        
        self.player = None
        self.car1 = None
        self.car2 = None
        self.car3 = None
        self.car4 = None
        self.walker1 = None
        self.camera_manager = None
        self.collision_sensor = None
        self.hud = hud
        self.actor_list = []
        self._gamma = 2.2
        self.cumulative_reward = 0.0
        self.total_step = 0
        self.episode = 0
        self.STEER_AMT = 1.0
        self.collision_hist = []
        self.RL = RL()
        self.agent = None
        self.high_level = None
        self.previous_safe_action = None
        self.safety_rules = None

        #pygame
        self.display = pygame.display.set_mode((1280, 720),pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

    def step(self,action):
        """apply steer and throttle from Reinforcement learning"""
       

        #pygame related stuff
        self.tick(self.clock)
        self.render(self.display)
        pygame.display.flip()

        print(action)
        print(action.shape)

        #emergency brake stuff
        Emergency_brake = False

        #walker
        control_w = carla.WalkerControl()
        control_w.speed = 0.5
        control_w.direction.y = 1
        control_w.direction.x = 0
        control_w.direction.z = 0
        self.walker1.apply_control(control_w)
        
        #player
        control_p = carla.VehicleControl()

        #find euclidian Distance
        obs_player = self.get_obs(self.player)
        #print(obs_player)
        obs_walker = self.get_obs(self.walker1)
        e_dist = self.euclidean_dist(obs_player,obs_walker)

        #safety rules
        #print(abs(self.player.get_location().x - self.walker1.get_location().x))
        #if abs(self.player.get_location().x - self.walker1.get_location().x) < 10:

        print(self.is_dest(self.player)) #test this function first, it all is good, use it.
        #if e_dist < 7 or self.is_dest(self.player):
        if e_dist < 7:
            Emergency_brake = True
            control_p = carla.VehicleControl()
            control_p.steer = 0.0
            control_p.throttle = 0.0
            control_p.brake = 1.0
            control_p.hand_brake = False
            control_p.manual_gear_shift = False    
        else:
            if abs(self.player.get_location().x - self.walker1.get_location().x) <= 30: #activate > RL
                #use control_p from RL
                control_s= self.agent.run_step()
                control_p = carla.VehicleControl()
                throt_RL = action[0]
                print(throt_RL)
                print(type(throt_RL))
                if throt_RL >= 0:
                    control_p.throttle = float(throt_RL)
                else:
                    control_p.brake = abs(float(throt_RL)) 
                control_p.steer = control_s.steer #throttle from RL, steer from safety controller.
            else:
                #use safety rule when not in range.
                control_p = self.agent.run_step()
            #use RL
            #use safety control
        
        #control_p = self.agent.run_step()
        self.player.apply_control(control_p)

        #getting reward and done
        player_state = self.get_state(self.player)
        done,reward = self.RL.cal_reward(self.collision_hist,player_state)
        
        #getting observation_space
        obs = self.process_bev()
        info = dict()
       
        #if self.player.get_location().z > 0: #if the car is at the floor  
            
        # if action from RL leads to out of range or collision, activate the control signal from safe action module
        if self.total_step == 200:
            done = True
        
        

        #print(done)
        if done == True:
            self.episode == self.episode + 1
            self.destroy()

       
        self.total_step = self.total_step+1


        
        #print(action)
        return obs, reward, done, info
        
        
    def reset(self):
        
        self.actor_list = []
        self.collision_hist = []
        self.cumulative_reward = 0.0
        self.total_step = 0.0
        self.episode = 0.0
        self.previous_safe_action = None
        self.spawn_player()
        #self.spawn_others()
        self.spawn_sensors()
        self.spawn_walker()
        self.agent = Agent(self.player)
        #self.high_level = HighLevelSC(self.world)
        #self.safety_rules = safety_rules()
        self.dest = carla.Transform(carla.Location(x=400, y=-17.2, z= 0.0))
        self.agent.set_destination((self.dest.location.x,self.dest.location.y,self.dest.location.z))
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

        # return obsrevation
        obs = self.process_bev()
        #print(obs.shape)
        return obs
       

    def spawn_player(self):
        """spawn player to the env"""
        model_3 = self.world.get_blueprint_library().filter("model3")[0]
        self.transform = carla.Transform(carla.Location(x=550, y=-17.2, z=10),carla.Rotation(yaw=-180)) #map6
        if self.player is not None:
            self.player.destroy()
        self.player = self.world.spawn_actor(model_3, self.transform)
        self.actor_list.append(self.player)
        print(f'Player model is {self.player}.')
       

    def spawn_others(self):
        """spawn other cars to the env"""

        #car1
        vehicle_bp = self.world.get_blueprint_library().filter("mercedes-benz")[0]
        self.transform1 = carla.Transform(carla.Location(x=490, y=-17.2, z= 10.0),carla.Rotation(yaw=-180))
        if self.car1 is not None:
            self.car1.destroy()
        self.car1 = self.world.spawn_actor(vehicle_bp, self.transform1)
        #self.car1.set_autopilot(True)
        #self.car1.apply_control(carla.VehicleControl(throttle=0.35, steer=0))
        self.actor_list.append(self.car1)
        print(f'Car1 model is {self.car1}.')
        
    def spawn_sensors(self):
        """spawn sensors to the env"""
        #camera manager
        print('spawning sensors')
        if self.camera_manager is not None:
            self.camera_manager.sensor.destroy()
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        
        #bird eye view
        self.birdview_producer = BirdViewProducer(
        self.player,  # carla.Client
        target_size=PixelDimensions(width=150, height=336),
        pixels_per_meter=4,
        crop_type=BirdViewCropType.FRONT_AND_REAR_AREA)

        #collision sensor
        if self.collision_sensor is not None:
            self.collision_sensor.sensor.destroy()
        self.collision_sensor = CollisionSensor(self.player, self.hud)

    def spawn_walker(self):
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker'))
        self.transform_walk = carla.Transform(carla.Location(x=490, y=-21.2, z= 10.0),carla.Rotation(yaw=-180))
        if self.walker1 is not None:
            self.walker1.destroy()
        self.walker1 = self.world.spawn_actor(walker_bp, self.transform_walk)
        print(f'Walker is {self.walker1}.')


    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        #print('rendering')
        self.camera_manager.render(display)
        self.hud.render(display)
    '''    
    def destroy(self):
        
         """Destroys all actors"""
         #self.actors = [self.rgb_cam.sensor, self.player, self.car1]
         print('destroying actors')
         for actor in self.actor_list:
             if actor is not None:
                 actor.destroy()
        #pygame.quit()
    '''
    def destroy(self):
        """Destroys all actors"""
        sensors = [self.camera_manager, self.collision_sensor]
        cars = [self.car1, self.car2, self.car3, self.car4,self.player,self.walker1]
        for s in sensors:
            if s is not None:
                s.sensor.destroy()
        for c in cars:
            if c is not None:
                c.destroy()
        

    

    def process_bev(self):
        """process bird eye views images"""
        birdview = self.birdview_producer.produce(
        agent_vehicle=self.player)  # carla.Actor (spawned vehicle)

        # produces np.ndarray of shape (height, width, 3)
        #rgb = self.birdview_producer.as_rgb(birdview)
        obs = birdview.flatten()
        #print(obs.shape)

        return obs

    def get_state(self, car):
        '''
        car:carla.Actor, type carla.Vehicle
        return: State of the car 
        
        '''
        loc_xyz = car.get_location()
        t = car.get_transform()
        a_xyz = car.get_acceleration() #m/s2^2
        v_xyz = car.get_velocity() #m/s
        heading_rad = t.rotation.yaw * (np.pi/180) #rad
        v_ms = math.sqrt(v_xyz.x**2 + v_xyz.z**2 + v_xyz.z**2)
        x = loc_xyz.x
        y = loc_xyz.y
        
        return [x,y,heading_rad,v_ms]

    def get_obs(self, car):
        '''
        car:carla.Actor, type carla.Vehicle
        return: State of the car 
        
        '''
        loc_xyz = car.get_location()
        #t = car.get_transform()
        
        x = loc_xyz.x
        y = loc_xyz.y
        
        return [x,y]

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
    def is_dest(self,player):
        loc_xyz = player.get_location()
        player_x = loc_xyz.x
        player_y = loc_xyz.y
        dest_x = self.dest.location.x
        dest_y = self.dest.location.y
        a = np.array((player_x,player_y)) #(2,)
        b = np.array((dest_x,dest_y)) #(2,)
        dist = np.linalg.norm(a-b)
        

        return dist < 2



    #get car state
    #get obs for RL (for example,images)
    #



'''
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

'''             
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


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)  


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self.collision_hist = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.collision_hist.append(event)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
            
            
# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        #print('font '+str(mono))
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        #colhist = world.collision_sensor.get_collision_history()
        #collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        #max_col = max(1.0, max(collision))
        #collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            #'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            0,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)
         
    
def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name        
        
    
        
        
         
         
         
         