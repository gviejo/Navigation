#!/usr/bin/python
# encoding: utf-8
"""
setup.py    
	


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""



import numpy as np
import sys

from Models import *


class Agent(object):

	def __init__(self, model, parameters):
		self.model = model
		self.parameters = parameters
		self.model.setAllParameters(self.parameters)
		self.agent_direction = 0 # Relative to a allocentric coordinate from the agent
		self.landmark_position = np.random.uniform(-1,1,2)	# Relative to a allocentric coordinate from the environnment
		self.direction = np.random.uniform(0,2*np.pi) # Angle between agent direction and landmark direction
		self.distance = 0  # Distance between the agent and the landmark relative to a allocentric coordinate from the environnment
		self.position = np.random.uniform(-1,1,2) # Relative to a allocentric coordinate from the environnment
		self.action = 0.0
		self.d = 0.1
		



	def update(self):		
		self.agent_direction = self.agent_direction+self.action		
		if self.agent_direction>=2*np.pi: self.agent_direction -= 2*np.pi 		
		self.position[0] += self.d * np.cos(self.agent_direction)
		self.position[1] += self.d * np.sin(self.agent_direction)
		self.distance = np.sqrt(np.sum(np.power(self.position-self.landmark_position, 2)))
		#tmp1 = np.cos(self.agent_direction) * (self.landmark_position[0]-self.position[0])
		#tmp2 = np.sin(self.agent_direction) * (self.landmark_position[1]-self.position[1])
		#self.direction = np.arccos((tmp1 + tmp2)/self.distance)
		l_coord = self.landmark_position-self.position
		a_coord = [np.cos(self.agent_direction), np.sin(self.agent_direction)]
		self.direction = (np.arctan2(l_coord[1], l_coord[0])-np.arctan2(a_coord[1], a_coord[0]))+2*np.pi

	def step(self):
		self.update()
		#self.model.setPosition(self.direction, self.distance, self.position)
		#self.action = self.model.getAction()
		
