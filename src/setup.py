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

	def __init__(self, model, world, parameters, stats = False):
		self.model = model
		self.world = world
		self.parameters = parameters
		self.model.setAllParameters(self.parameters)
		self.n_steps = [] # NUmber of step before reaching reward
		self.d = 0.8
		self.vitesse_max = 0.2
		self.stats = stats
		if self.stats:
			self.positions = []

	def start(self):
		self.agent_direction = self.world.start_direction # Relative to a allocentric coordinate from the agent
		self.landmark_position = self.world.landmark_position # Relative to a allocentric coordinate from the environnment		
		self.position = np.array(self.world.start_position) # Relative to a allocentric coordinate from the environnment
		self.distance = np.sqrt(np.sum(np.power(self.position-self.landmark_position, 2))) # Distance between the agent and the landmark relative to a allocentric coordinate from the environnment		
		self.computeDirection() # Angle between agent direction and landmark direction		
		self.action_angle = 0.0
		self.action_speed = 0.0
		self.wall = self.world.computeWallInformation(self.position, self.agent_direction)
		self.reward = False
		self.world.reward_found = False
		self.model.setPosition(self.direction, self.distance, self.position, self.wall)
		self.n_steps.append(float(0))

		if self.stats:
			self.colors = dict({'t':(1.0, 0.0, 0.0),'e':(0.0, 0.0, 1.0),'p':(0.0, 1.0, 0.0)})
			self.positions.append([])			
			self.directions = [self.agent_direction]
			self.distances = [list([self.distance, self.world.distance])]
			self.experts = list(['t'])
			self.actions = list([[self.action_angle, self.action_speed]])
			self.gates = dict({k:[] for k in self.model.k_ex})
			self.walls = [list(self.wall)]
			self.rewards = []
			self.speeds = []
			self.winners = []

	def computeDirection(self):
		"""Counter clockwise angle is computed in an allocentric coordinate 
		centered on the agent. Angle betwen the direction of the agent and 
		the direction of the landmark in the interval [-pi, pi]"""
		landmark_coordinate = self.landmark_position-self.position
		landmark_angle = np.arctan2(landmark_coordinate[1], landmark_coordinate[0])		
		self.direction = landmark_angle-self.agent_direction

	def computePosition(self):
		""" New position of the agent in the allocentric coordinate of the 
		world. position must be between [-1,1] so one simple condition to
		keep boundaries """
		self.position[0] += np.cos(self.agent_direction) * self.action_speed
		self.position[1] += np.sin(self.agent_direction) * self.action_speed
		self.position[self.position>1.0] = 1.0
		self.position[self.position<-1.0] = -1.0

	def update(self):
		""" Fonction to move the agent in a new position
		Action should be in range [-pi, pi]"""
		self.agent_direction = self.agent_direction+self.action_angle		
		if self.agent_direction<=-np.pi: self.agent_direction = 2*np.pi - np.abs(self.agent_direction)
		if self.agent_direction>np.pi: self.agent_direction = self.agent_direction - 2*np.pi
		self.computePosition()
		self.wall = self.world.computeWallInformation(self.position, self.agent_direction)
		self.distance = np.sqrt(np.sum(np.power(self.position-self.landmark_position, 2)))
		self.computeDirection()

	def learn(self):
		""" Check from class world if agent get reward
		then update Experts. Reward can be boolean or value"""
		reward = self.world.getReward(self.position)
		self.reward = reward>0.0
		self.model.learn(reward)
		

	def step(self):
		self.action_angle, d = self.model.getAction()		
		self.action_speed = self.vitesse_max/(1.+np.exp(-self.d*d))
		self.update()
		self.learn()
		self.model.setPosition(self.direction, self.distance, self.position, self.wall)
		self.n_steps[-1] += 1			
		if self.stats:
			self.getStats()

	def getStats(self):
		self.positions[-1].append(list(self.position))
		self.directions.append(self.agent_direction)
		self.distances.append(list([self.distance, self.world.distance]))
		self.experts.append(self.colors[self.model.winner])
		self.actions.append(list([self.action_angle, self.action_speed]))
		#for k in self.model.k_ex: self.gates[k].append(dict(zip(self.model.g.values(),self.model.g.keys()))[k])
		self.winners.append((self.model.winner=='t')*1.0)
		#self.walls.append(list(self.wall))
		self.rewards.append(self.reward)
		self.speeds.append(self.action_speed)

class World(object):

	def __init__(self):		
		self.landmark_position = np.array([0.1, 0.0])
		self.reward_position = np.array([0., 0.5])
		self.reward_size = 0.15 # Radius of the reward position
		tmp = np.arange(0, 2*np.pi, 0.1)
		self.reward_circle = np.vstack((np.cos(tmp), np.sin(tmp))).T * self.reward_size + self.reward_position
		self.start_position = np.array([0.0, -0.5])
		self.start_direction = np.pi/2.
		self.distance = np.sqrt(np.sum(np.power(self.start_position-self.reward_position, 2)))  # Distance between the agent and the reward
		self.reward_found = False

	def getReward(self, position):
		self.distance = np.sqrt(np.sum(np.power(position-self.reward_position, 2)))
		if self.distance <= self.reward_size:
			if self.distance <= self.reward_size/2.:
				self.reward_found = True
			return 1.0
		else:
			return 0.0

	def computeWallInformation(self, position, angle):		
		ind = np.argmax(np.abs(position))
		distance = 1.0 - np.abs(position[ind])
		if ind == 0 and position[0] >= 0.0: 
			return np.array([-angle, distance])
		elif ind == 0 and position[0] < 0.0:
			if position[1] > 0.0:
				return np.array([np.pi-angle, distance])
			if position[1] < 0.0:
				return np.array([-np.pi-angle, distance])
		elif ind == 1 and position[1] >= 0.0:
			return np.array([(np.pi/2.)-angle, distance])
		elif ind == 1 and position[1] < 0.0:
			return np.array([-(np.pi/2.)-angle, distance])



		
