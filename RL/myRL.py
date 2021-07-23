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
	def R_eff(self,speed,player_break):
		speed_kmh = speed*3.6
		reward_speed = -abs(speed_kmh-20)+13
		if player_break > 0.7:
			reward_break = -15
		else:
			reward_break = 0
		reward_eff = reward_speed + reward_break
		return reward_eff
	def R_comfort(self,acc):
		acc_kmh = acc*3.6
		if acc_kmh <= 7.2:
			reward_comfort = 0
		else:
			reward_comfort = -acc_kmh+7.2
		return reward_comfort
	def R_collide(self,collision_hist):
		if len(collision_hist) != 0:
			self.done = True
			reward_collision = -100
		else:
			reward_collision = 0
		return reward_collision
			
	
		
		


			
	#if collision -1000, else 1




