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

	def __init__(self, model, parameters, stats = False):
		self.model = model
		self.parameters = parameters
		self.model.setAllParameters(self.parameters)
		self.agent_direction = 0 # Relative to a allocentric coordinate from the agent
		self.landmark_position = np.random.uniform(-1,1,2)	# Relative to a allocentric coordinate from the environnment
		self.direction = np.random.uniform(0,2*np.pi) # Angle between agent direction and landmark direction
		self.position = np.random.uniform(-1,1,2) # Relative to a allocentric coordinate from the environnment
		self.distance = np.sqrt(np.sum(np.power(self.position-self.landmark_position, 2))) # Distance between the agent and the landmark relative to a allocentric coordinate from the environnment
		self.action = 0.0
		self.d = 0.1
		self.stats = stats
		if self.stats:
			self.colors = dict({'t':(1.0, 0.0, 0.0),'e':(0.0, 0.0, 1.0),'p':(0.0, 1.0, 0.0)})
			self.positions = list()
			self.directions = list()
			self.distances = list()
			self.experts = list()
			self.actions = list()
			self.gates = dict({k:[] for k in self.model.k_ex})

	def computeDirection(self):
		"""Counter clockwise angle is computed in an allocentric coordinate 
		centered on the agent. Angle betwen the direction of the agent and 
		the direction of the landmark"""
		landmark_coordinate = self.landmark_position-self.position
		landmark_angle = np.arctan2(landmark_coordinate[1], landmark_coordinate[0])
		agent_angle = np.arctan2(np.sin(self.agent_direction), np.cos(self.agent_direction))
		if agent_angle <= landmark_angle:
			self.direction = np.abs(landmark_angle-agent_angle)
		else:
			self.direction = 2*np.pi - np.abs(landmark_angle-agent_angle)

	def computePosition(self):
		""" New position of the agent in the allocentric coordinate of the 
		world. position must be between [-1,1] so one simple condition to
		keep boundaries """
		self.position[0] += self.d * np.cos(self.agent_direction)
		self.position[1] += self.d * np.sin(self.agent_direction)
		self.position[self.position>1.0] = 1.0
		self.position[self.position<-1.0] = -1.0

	def update(self):		
		self.agent_direction = self.agent_direction+self.action		
		if self.agent_direction>=2*np.pi: self.agent_direction -= 2*np.pi 		
		self.computePosition()
		self.distance = np.sqrt(np.sum(np.power(self.position-self.landmark_position, 2)))
		self.computeDirection()		

	def step(self):		
		self.model.setPosition(self.direction, self.distance, self.position)
		self.action = self.model.getAction()
		self.update()
		
		
		if self.stats:
			self.getStats()

	def getStats(self):
		self.positions.append(list(self.position))
		self.directions.append(self.direction)
		self.distances.append(self.distance)
		self.experts.append(self.colors[self.model.winner])
		self.actions.append(self.action)		
		for k in self.model.k_ex: self.gates[k].append(dict(zip(self.model.g.values(),self.model.g.keys()))[k])