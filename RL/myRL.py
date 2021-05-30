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
			self.reward = - 100
		elif player_state[3]*3.6 < 10 : #speed lower than 10 km/h
			self.reward = -1
		else:
			self.reward = 1

		#self.cumulative_reward = self.cumulative_reward+self.reward
		return self.done,self.reward


			
	#if collision -1000, else 1




