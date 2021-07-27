import glob
import os
import sys
import random
import time
import numpy as np

class RL(object):
	
	def __init__(self):
		self.done = False
		self.cumulative_reward = 0.0
		self.reward = 0.0
	def calculate_jerk(self):
		pass
	def cal_reward(self,collision_hist,player_state):
		#print('calculating reward')
		if len(collision_hist) != 0:
			self.done = True
			self.reward = -100
		elif player_state[3]*3.6 < 10 : #speed lower than 10 km/h
			self.reward = -1
		else:
			self.reward = 1

		#self.cumulative_reward = self.cumulative_reward+self.reward
		return self.done,self.reward
	def R_safe(self,emergency_brake):
		if emergency_brake:
			reward_safe = -30
		else:
			reward_safe = 0
		return reward_safe
	def R_eff(self,speed):
		speed_kmh = speed*3.6
		#reward_speed = -abs(speed_kmh-20)+13
		if speed_kmh >= 0 and speed_kmh <=1:
			reward_speed = 0
		elif speed_kmh > 1 and speed_kmh <=3:
			reward_speed = 1
		elif speed_kmh > 3 and speed_kmh <=15:
			reward_speed = 3
		elif speed_kmh > 15 and speed_kmh <=20:
			reward_speed = 5
		elif speed_kmh > 20 :
			#reward_speed = -6 * speed_kmh + 138
			reward_speed = 0
		'''
		if speed_kmh >= 0 and speed_kmh <=3:
			reward_speed = reward_speed - 10
		
		if player_break > 0.7:
			reward_break = -25
		else:
			reward_break = 0
		reward_eff = reward_speed + reward_break
		'''
		return reward_speed
	def R_comfort(self,acc):
		acc_kmh = acc*3.6
		if acc_kmh <= 7.2:
			reward_comfort = 2
		else:
			#reward_comfort = (-1.5 * acc_kmh) + 7.2
			reward_comfort = 0
		return reward_comfort

	def R_break(self,player_break):
		if player_break >= 0.7:
			reward_break = -2
		else:
			reward_break = 0
		return reward_break

	def R_throt(self,player_throt):
		if player_throt >= 0.7:
			reward_throt = -2
		else:
			reward_throt = 0
		return reward_throt

	def R_collide(self,collision_hist):
		if len(collision_hist) != 0:
			self.done = True
			reward_collision = -100
		else:
			reward_collision = 0
		return reward_collision
	def reset(self):
		self.done = False
		
			
	
		
		


			
	#if collision -1000, else 1




