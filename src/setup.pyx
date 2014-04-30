#!/usr/bin/cython
# encoding: utf-8
"""
setup.pyx
	
Cython version : 
	class Agent 
	class World

To compile : python csetup.py build_ext -i

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
"""

import numpy as np
cimport numpy as np
cimport cython
#cimport cModels as Models
from cModels import Dolle

cdef class Agent(object):	
	cdef public dict parameters, colors
	cdef public float reward, agent_direction, distance, action_angle, action_speed, direction
	cdef public str stats
	cdef public list positions, n_steps, experts, gating, distances, actions, pgates
	cdef public object model, world
	cdef public np.ndarray landmark_position, position, wall, 
	cdef public int reward_found

	def __init__(self, tuple experts, dict parameters, str stats = "train"):
		self.model = Dolle(experts, parameters)
		self.world = World()
		self.parameters = parameters
		self.model.setAllParameters(self.parameters)		
		self.reward = 0.0
		self.agent_direction = 0.0
		self.stats = stats
		self.colors = dict({'t':(1.0, 0.0, 0.0),'e':(0.0, 0.0, 1.0),'p':(0.0, 1.0, 0.0)})		
		self.positions = [] # Positions in the world
		self.n_steps = [] # Number of step before reaching reward	
		self.experts = [] # Experts used at each time step
		self.gating = [] # Gate value at each time step

	cpdef start(self):
		""" Should be called at the beginning of a trial"""
		self.agent_direction = self.world.start_direction # Relative to a allocentric coordinate from the agent
		self.landmark_position = self.world.landmark_position # Relative to a allocentric coordinate from the environnment		
		self.position = np.array(self.world.start_position) # Relative to a allocentric coordinate from the environnment
		self.distance = np.sqrt(np.sum(np.power(self.position-self.landmark_position, 2))) # Distance between the agent and the landmark relative to a allocentric coordinate from the environnment		
		self.computeDirection() # Angle between agent direction and landmark direction		
		self.action_angle = 0.0
		self.action_speed = 0.0
		self.wall = self.world.computeWallInformation(self.position, self.agent_direction)
		self.world.reward_found = 0
		self.model.setPosition(self.direction, self.distance, self.position, self.wall, self.agent_direction)		
		# Minimum stats is n_steps, positions and experts for training session
		self.n_steps.append(0)
		self.positions.append([])
		self.experts.append([0]*self.model.n_ex)
		if self.stats == "test":								
			self.distances = [list([self.distance, self.world.distance])]			
			self.actions = list([[self.action_angle, self.action_speed]])			
			self.pgates = []

	cpdef computeDirection(self):
		"""Counter clockwise angle is computed in an allocentric coordinate 
		centered on the agent. Angle betwen the direction of the agent and 
		the direction of the landmark in the interval [-pi, pi]"""
		cdef float landmark_angle
		landmark_coordinate = self.landmark_position-self.position
		landmark_angle = np.arctan2(landmark_coordinate[1], landmark_coordinate[0])		
		self.direction = landmark_angle-self.agent_direction

	cpdef computePosition(self):
		""" New position of the agent in the allocentric coordinate of the 
		world. position must be between [-1,1] so one simple condition to
		keep boundaries """
		cdef np.ndarray new_position
		new_position = self.position.copy()
		new_position[0] += np.cos(self.agent_direction) * self.action_speed
		new_position[1] += np.sin(self.agent_direction) * self.action_speed		
		self.position = self.world.checkPosition(self.position, new_position)

	cpdef update(self):
		""" Fonction to move the agent in a new position
		Action should be in range [-pi, pi]"""
		self.agent_direction = self.agent_direction+self.action_angle		
		if self.agent_direction<=-np.pi: self.agent_direction = 2*np.pi - np.abs(self.agent_direction)
		if self.agent_direction>np.pi: self.agent_direction = self.agent_direction - 2*np.pi
		self.computePosition()
		self.wall = self.world.computeWallInformation(self.position, self.agent_direction)
		self.distance = np.sqrt(np.sum(np.power(self.position-self.landmark_position, 2)))
		self.computeDirection()

	cpdef learn(self):
		""" Check from class world if agent get reward
		then update Experts. Reward can be boolean or value"""
		self.reward = self.world.getReward(self.position)				
		self.model.learn(self.reward)
		

	cpdef step(self):
		""" Only function to call when running the agent.
		The order must be respected for taxon learning """
		self.action_angle, self.action_speed = self.model.getAction()				
		self.update()
		self.model.setPosition(self.direction, self.distance, self.position, self.wall, self.agent_direction)
		self.learn()		
		
		if self.stats:
			self.getStats()

	@cython.cdivision(True)
	cpdef getStats(self):
		self.positions[-1].append(list(self.position))
		self.experts[-1][self.model.k_ex.index(self.model.winner)] += 1
		self.n_steps[-1] += 1
		self.gating.append(self.model.g.copy())
		if self.stats == "test":
			self.distances.append(list([self.distance, self.world.distance]))
			self.actions.append(list([self.action_angle, self.action_speed]))
			self.pgates.append(np.exp(self.model.g)/(np.exp(self.model.g).sum()))		
				

cdef class World(object):
	cdef public np.ndarray landmark_position, reward_position, tmp, reward_circle, start_position
	cdef public float reward_size, start_direction, distance
	cdef public dict walls	
	cdef public int reward_found

	def __cinit__(self):		
		self.landmark_position = np.array([0.1, 0.0])
		self.reward_position = np.array([0., 0.5])
		self.reward_size = 0.08 # Radius of the reward position
		tmp = np.arange(0, 2*np.pi, 0.1)
		self.reward_circle = np.vstack((np.cos(tmp), np.sin(tmp))).T * self.reward_size + self.reward_position
		self.start_position = np.array([0.0, -0.5])
		self.start_direction = np.pi/2.
		self.distance = np.sqrt(np.sum(np.power(self.start_position-self.reward_position, 2)))  # Distance between the agent and the reward
		self.reward_found = 0
		self.walls = dict()

	cpdef getReward(self, np.ndarray position):
		self.distance = np.sqrt(np.sum(np.power(position-self.reward_position, 2)))
		if self.distance <= self.reward_size:			
			self.reward_found = 1
			return 1.0
		else:
			return 0.0

	cpdef computeWallInformation(self, np.ndarray position, float angle):
		cdef int ind
		cdef float distance
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

	cpdef checkPosition(self, np.ndarray old_position, np.ndarray new_position):
		new_position[new_position>1.0] = 1.0
		new_position[new_position<-1.0] = -1.0
		return new_position


cdef class Maze(object):
	cdef public np.ndarray landmark_position, reward_position, tmp, reward_circle, start_position
	cdef public float reward_size, start_direction, distance
	cdef public dict walls
	cdef public int reward_found

	def __cinit__(self):
		self.landmark_position = np.array([0.5, 0.5])
		self.reward_position = np.array([-0.5, 0.5])
		self.reward_size = 0.15
		tmp = np.arange(0, 2*np.pi, 0.1)
		self.reward_circle = np.vstack((np.cos(tmp), np.sin(tmp))).T * self.reward_size + self.reward_position
		self.start_position = np.array([-0.5, -0.5])
		self.start_direction = 0.0
		self.distance = np.sqrt(np.sum(np.power(self.start_position-self.reward_position, 2)))
		self.reward_found = 0
		# Wall
		self.walls = dict({1:np.array([[-1.0,0.0],[-0.4,0.0]]),
						   2:np.array([[0.4,0.0],[1.0,0.0]])})

	cpdef getReward(self, np.ndarray position):
		self.distance = np.sqrt(np.sum(np.power(position-self.reward_position, 2)))
		if self.distance <= self.reward_size:			
			self.reward_found = 1
			return 1.0
		else:
			return 0.0
	cpdef checkPosition(self, np.ndarray old_position, np.ndarray new_position):
		#super(Maze, self).checkPosition(old_position, new_position)
		cdef int i
		new_position[new_position>1.0] = 1.0
		new_position[new_position<-1.0] = -1.0
		for i in self.walls.iterkeys():
			I = self.intersect(self.walls[i][0], self.walls[i][1], old_position, new_position)
			if I is not False:
				new_position = old_position + 0.1 * (old_position - I)
				new_position[new_position>1.0] = 1.0
				new_position[new_position<-1.0] = -1.0
				return new_position
		return new_position

	cpdef intersect(self, np.ndarray A1, np.ndarray A2, np.ndarray B1, np.ndarray B2):
		""" intersection between line A and B
		A1 and A2 are two points in line A
		Return None if parallel lines
		Return (Cx, Cy) intersection coordinates
		"""
		cdef float x, y
		cdef np.ndarray I		
		if (A1[0]-A2[0])*(B1[1]-B2[1]) - (A1[1]-A2[1])*(B1[0]-B2[0]) == 0:
			return False
		else:
			x = ((A1[0]*A2[1]-A1[1]*A2[0])*(B1[0]-B2[0])-(A1[0]-A2[0])*(B1[0]*B2[1]-B1[1]*B2[0]))/((A1[0]-A2[0])*(B1[1]-B2[1])-(A1[1]-A2[1])*(B1[0]-B2[0]))
			y = ((A1[0]*A2[1]-A1[1]*A2[0])*(B1[1]-B2[1])-(A1[1]-A2[1])*(B1[0]*B2[1]-B1[1]*B2[0]))/((A1[0]-A2[0])*(B1[1]-B2[1])-(A1[1]-A2[1])*(B1[0]-B2[0]))
			I = np.array([x, y])			
			c1 = np.sqrt(np.sum(np.power(I-A1, 2))) < np.sqrt(np.sum(np.power(A2-A1, 2)))
			c2 = np.sqrt(np.sum(np.power(I-B1, 2))) < np.sqrt(np.sum(np.power(B2-B1, 2)))
			if c1 and c2:
				return I
			else:
				return False