import glob
import os
import sys
import random
import time
import numpy as np

class RL(object):
	
	def __init__(self):
		self.done = False
		
	
	def R_eff(self,speed):
		speed_kmh = speed*3.6
		#reward_speed = -abs(speed_kmh-20)+13
		'''
		if speed_kmh >= 0 and speed_kmh <=1:
			#reward_speed = 0
			reward_speed = -2
		elif speed_kmh > 1 and speed_kmh <=3:
			#reward_speed = 1
			reward_speed = 2
		elif speed_kmh > 3 and speed_kmh <=15:
			#reward_speed = 3
			reward_speed = 4
			#reward_speed = 0.2*speed_kmh + 2
		elif speed_kmh > 15 and speed_kmh <=20:
			reward_speed = 5
			#reward_speed = 4
		elif speed_kmh > 20 :
			#reward_speed = -6 * speed_kmh + 138
			#reward_speed = 0
			reward_speed = -2
		'''
		if speed_kmh >= 0 and speed_kmh <1:
			reward_speed = -2
		elif speed_kmh >= 1 and speed_kmh <5:
			reward_speed = ((0.25*speed_kmh)-0.25)
		elif speed_kmh >= 5 and speed_kmh <=20:
			reward_speed = (((4/15)*speed_kmh)-(1/3))
		elif speed_kmh > 20 :
			reward_speed = -2
		

		return reward_speed

	def R_comfort(self,a_ms,a_sim):
		acc_kmh = a_ms*3.6
		if acc_kmh <= 7.2 and a_sim >=0:
			reward_comfort = 2	
		else:
			reward_comfort = 0
		return reward_comfort
  
	def R_collide(self,collision_hist):
		if len(collision_hist) != 0:
			self.done = True
			reward_collision = -100
		else:
			reward_collision = 0
		return reward_collision

	def reset(self):
		self.done = False
		
			
	
		
		


			
	




