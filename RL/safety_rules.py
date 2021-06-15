import glob
import os
import sys
import random
import time
import numpy as np

class safety_rules(object):
	
	def __init__(self):
		self.left_edge = -13.0
		self.right_edge = -21.0
		
	def is_out_of_range(self,next_waypoint):
		"""
		check if the RL next waypoint is out of range or not 

		"""
		out_of_range = False
		if next_waypoint.transform.location.y < 21.0 or  next_waypoint.transform.location.y > 13.0:
			out_of_range = True
		return out_of_range 

	def is_action_safe(self,player,action):
		"""
		check if the RL action is safe.

		"""
		
		
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




