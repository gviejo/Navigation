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
		self.landmark_direction = 0 # Relative to a allocentric coordinate
		self.agent_direction = 0 # Relative to a allocentric coordinate
		self.landmark_position = np.random.uniform(-1,1,2)		
		self.direction = np.random.uniform(0,2*np.pi) # Angle between agent direction and landmark direction
		self.distance = 0
		self.position = np.random.uniform(-1,1,2)
		self.action = 0.0
		self.d = 0.1


	def update(self):
		self.agent_direction = self.agent_direction+self.action
		if self.agent_direction>=np.pi: self.agent_direction -= np.pi 		
		self.position[0] = self.d * np.cos(self.agent_direction)
		self.position[1] = self.d * np.sin(self.agent_direction)
		self.distance = np.sqrt(np.sum(np.power(self.position-self.landmark_position, 2)))
		self.landmark_direction = np.cosh((self.position[0]-self.landmark_position[0])/self.distance)
		self.direction = self.landmark_direction - self.agent_direction

	def step(self):
		self.update()
		self.model.setPosition(self.direction, self.distance, self.position)
		self.action = self.model.getAction()
		
