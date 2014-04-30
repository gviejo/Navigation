#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Expert.pyx

Cython version

To compile : python csetup.py build_ext -i

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
"""

import numpy as np
cimport numpy as np
from itertools import izip
cimport cython
from cpython cimport bool

cdef class Expert(object):
	cdef public dict parameters

	def __cinit__(self):
		self.parameters = dict({'speed':0.1})

	cpdef setParameter(self, str name, float value):		
		if name in self.parameters.keys() : 			
			self.parameters[name] = value

	cpdef setAllParameters(self, dict parameters):		
		for i in parameters.keys(): self.setParameter(i, parameters[i])

	cpdef learn(self, float angle,  float reward):
		pass

	cpdef setCellInput(self, float direction, float distance, np.ndarray position, np.ndarray wall, float agent_direction = 0.0):
		pass

	cpdef computeNextAction(self):
		return (np.random.uniform(-np.pi, np.pi), np.random.uniform(0, 1)*self.parameters['speed'])


cdef class Taxon(object):
	cdef public np.ndarray lc_direction, lc, vc_direction, vc, ac_direction, ac, trace, W
	cdef public float action, norm, delta
	cdef public dict parameters

	def __cinit__(self, dict parameters={}):
		self.parameters = { 'nlc': 100,		 		    # Number of landmarks cells
							'sigma_lc': 0.475,			# Normalized landmark width
							'sigma_vc': 0.001, 			# Visual cell width
							'sigma':0.392,				# Number of action cells
							'nac': 36,					# Standard deviation of the generalization profile
							'eta': 0.001,				# Learning rate
							'lambda': 0.76,				# Eligibility trace decay factor
							'gamma' : 0.8,				# Discount factor
							'speed' : 0.1 }				
		self.setAllParameters(parameters)							
		# Landmarks cells
		self.lc_direction = np.arange(-np.pi, np.pi, (2*np.pi)/float(self.parameters['nlc']))
		self.lc = np.zeros((self.parameters['nlc']), dtype=np.double)
		# Visual cells
		self.vc_direction = np.arange(-np.pi, np.pi, (2*np.pi)/float(self.parameters['nac']))
		self.vc = np.zeros((self.parameters['nac']), dtype=np.double)
		# Action cells
		self.ac_direction = np.arange(-np.pi, np.pi, (2*np.pi)/float(self.parameters['nac']))
		self.ac = np.zeros((self.parameters['nac']), dtype=np.double)
		# Connection
		self.W = np.random.normal(0.0, 0.1, size=(self.parameters['nac'], self.parameters['nlc']))
		# Proposed direction		
		self.action = 0.0 # The proposed direction
		self.norm = 0.0 # The distance if action is choosen
		# Learning initialization		
		self.delta = 0.0
		self.trace = np.zeros((self.parameters['nac'], self.parameters['nlc']), dtype=np.double)

	cpdef setParameter(self, str name, float value):		
		if name in self.parameters.keys() : 			
			self.parameters[name] = value

	cpdef setAllParameters(self, dict parameters):		
		for i in parameters.keys(): self.setParameter(i, parameters[i])

	@cython.cdivision(True)
	cpdef setCellInput(self, float direction, float distance, np.ndarray position, np.ndarray wall, float agent_direction = 0.0):
		""" Direction should be in [-pi, pi] interval 
		Null angle is the curent direction of the agent"""
		cdef np.ndarray delta		
		delta = np.arccos(np.cos(direction)*np.cos(self.lc_direction)+np.sin(direction)*np.sin(self.lc_direction))		
		self.lc = np.exp(-(np.power(delta,2))/(2*(self.parameters['sigma_lc']/float(distance))**2))
		delta = np.arccos(np.cos(wall[0])*np.cos(self.vc_direction)+np.sin(wall[0])*np.sin(self.vc_direction))
		self.vc = np.exp(-(np.power(delta, 2))/(2*(self.parameters['sigma_vc']/float(wall[1]-0.0001))**2))
		self.computeActionActivity()		

	@cython.cdivision(True)
	cpdef computeActionActivity(self):
		cdef list xy
		self.ac = np.dot(self.W, self.lc) - self.vc
		self.ac = np.tanh(self.ac)
		xy = [(self.ac*np.sin(self.ac_direction)).sum(), (self.ac*np.cos(self.ac_direction)).sum()]
		self.action = np.arctan2(xy[0], xy[1])
		self.norm = np.sqrt(np.sum(np.power(xy, 2)))
		self.norm = self.parameters['speed']/(1.+np.exp(-self.norm))			

	cpdef updateTrace(self, float action):
		cdef np.ndarray delta, ac
		delta = np.arccos(np.cos(action)*np.cos(self.ac_direction)+np.sin(action)*np.sin(self.ac_direction))		
		ac = np.exp(-(np.power(delta,2))/(2*self.parameters['sigma']**2))		
		self.trace = self.parameters['lambda']*self.trace+np.outer(ac, self.lc)

	cpdef learn(self, float action, float reward):
		""" Action performed selected from a mixture of experts"""
		#super(Taxon, self).learn(action, reward)		
		self.updateTrace(action)
		self.delta = reward + self.parameters['gamma']*self.ac.max()-action		
		self.W = self.W+self.parameters['eta']*(np.tile(self.delta, (self.parameters['nlc'],1))).T*self.trace

	cpdef computeNextAction(self):
		""" Called by general model for choosing action
		if mixture of experts, return action angle and distance to "walk" """
		#super(Taxon, self).computeNextAction()
		return (self.action, self.norm)


cdef class Planning(object):
	cdef public np.ndarray position, pc, pc_position
	cdef public dict pc_nodes, nodes, nodes_position, edges, values, parameters
	cdef public list path
	cdef public float direction, action
	cdef public int nb_nodes, current_node, goal_node	
	cdef public bool goal_found

	def __cinit__(self, parameters={}):
		self.parameters = { 'theta_pc': 0.2,			# Activity threshold for place cells node linking
							'theta_node': 0.3,			# Activity threshold for node creation
							'alpha': 0.7, 				# Decay factor of the goal value
							'npc': 1681,				# Number of simulated Place cells
							'sigma_pc': 0.2, 
							'speed' : 0.1 }				# Place field size
		self.setAllParameters(parameters)
		self.direction = 0.0 # Direction of the agent in a allocentric frame [-pi, pi]
		self.position = np.zeros(2)
		# Place cells
		self.pc = np.zeros((self.parameters['npc']))
		self.pc_position = np.random.uniform(-1,1, (self.parameters['npc'],2))
		# Graph planning		
		self.nb_nodes = 0
		self.pc_nodes = dict() #indices : weight for place cells - nodes links
		self.nodes = dict() # Nodes activity
		self.nodes_position = dict() # Mean position of place cells linked to nodes | Bad but no choice
		self.current_node = 0 # Current node
		self.goal_node = 0
		self.goal_found = False
		# Planning
		self.edges = dict({0:[]}) # Links between nodes
		self.values = dict() # Weight associated to each nodes, should change every time step
		self.path = []		
		# Return 
		self.action = 0.0

	cpdef setParameter(self, str name, float value):
		if name in self.parameters.keys() : 			
			self.parameters[name] = value

	cpdef setAllParameters(self, dict parameters):		
		for i in parameters.keys(): self.setParameter(i, parameters[i])

	@cython.cdivision(True)
	cpdef setCellInput(self, float direction, float distance, np.ndarray position, np.ndarray wall, float agent_direction = 0.0):
		""" Only position is used """
		cdef np.ndarray distance_to_pc
		if np.max(position)>1.0 or np.min(position)<-1.0: raise Warning("Place cells position should be normalized between [-1,1]")
		self.direction = agent_direction
		self.position = position
		distance_to_pc = np.sqrt((np.power(self.pc_position-position, 2)).sum(1))
		self.pc = np.exp(-distance_to_pc/(2*self.parameters['sigma_pc']**2))
		self.computeGraphNodeActivity()	

	cpdef computeGraphNodeActivity(self):
		""" Dot product of place cells activity and pc->nodes links"""
		cdef int i
		for i in self.nodes.iterkeys(): self.nodes[i] = np.dot(self.pc[self.pc_nodes[i].keys()],self.pc_nodes[i].values())
		if len(self.nodes.keys()) == 0 or np.max(self.nodes.values()) < self.parameters['theta_node']:
			self.createNewNode()
		elif not self.goal_found:
			self.connectNode()
		else:
			self.current_node = np.argmax(self.nodes.values())+1

	cpdef createNewNode(self):
		""" Store a list of place cells indice
		Each indice indicates the position of the place field in the environment """
		cdef np.ndarray ind
		self.nb_nodes+=1
		ind = np.where(self.pc>self.parameters['theta_pc'])[0]		# The indices to the place cells
		self.pc_nodes[self.nb_nodes]  = dict(izip(ind, self.pc[ind]))   	# key : PC ind | values : PC activity
		self.nodes[self.nb_nodes] = np.dot(self.pc[self.pc_nodes[self.nb_nodes].keys()],self.pc_nodes[self.nb_nodes].values())
		self.edges[self.nb_nodes] = [self.current_node]
		self.edges[self.current_node].append(self.nb_nodes)
		self.values[self.nb_nodes] = 0.0
		self.nodes_position[self.nb_nodes] = np.mean(self.pc_position[ind], 0)		
		self.current_node = self.nb_nodes		

	cpdef createGoalNode(self):
		""" Called only if reward is explicitly found.
		Allow to set a goal node over a specific place 
		Also propagate values """
		cdef int ind, i, x		
		self.nb_nodes+=1
		ind = np.argmax(self.pc)
		self.pc_nodes[self.nb_nodes] = dict({ind:self.pc[ind]})
		self.nodes_position[self.nb_nodes] = self.pc_position[ind]
		self.nodes[self.nb_nodes] = np.power(self.pc[ind], 2)
		self.edges[self.nb_nodes] = [self.current_node]
		self.edges[self.current_node].append(self.nb_nodes)
		for i in self.values.iterkeys(): self.values[i] = 0.0
		self.values[self.nb_nodes] = 1.0
		self.current_node = self.nb_nodes
		self.goal_node = self.nb_nodes
		for x in self.edges[self.current_node]: self.propagate(x, [self.current_node], self.parameters['alpha'])

	cpdef connectNode(self):
		""" Check if connection exist already """
		cdef int new_node
		new_node = np.argmax(self.nodes.values())+1
		if self.current_node not in self.edges[new_node] and new_node != self.current_node:
			self.edges[new_node].append(self.current_node)
			self.edges[self.current_node].append(new_node)		
		self.current_node = new_node		

	cpdef learn(self, float action, float reward):
		""" Only if reward not found """
		#super(Planning, self).learn(action, reward)	
		if self.goal_found: 
			for x in self.edges[self.goal_node]: 
				self.propagate(x, [self.goal_node], self.parameters['alpha'])			
		elif reward > 0.0 and not self.goal_found:
			self.goal_found = True			
			self.createGoalNode()			

	cpdef propagate(self, int new_node, list visited, float value):
		""" Propagate discounted value starting from goal node """
		cdef list next_node
		cdef int x
		if self.values[new_node]<value: self.values[new_node] = value
		visited.append(new_node)				
		next_node = list(set(self.edges[new_node])-set(visited))
		if new_node-1:
			for x in next_node:
				self.propagate(x, visited, self.parameters['alpha']*value)
			#map(lambda x:self.propagate(x, visited, self.parameters['alpha'] * value), next_node)

	cpdef computeNextAction(self):
		""" Return tuple (direction, speed) """
		#super(Planning, self).computeNextAction()
		if self.goal_found:					
			if self.current_node == self.goal_node: 
				return (0.0, 0.0)
			else:
				self.path = []
				self.exploreGraph(self.edges[self.current_node], [0, self.current_node])			
				self.computeActionAngle()
				return (self.action, self.parameters['speed'])
		else : 
			return (np.random.uniform(0,2*np.pi), np.random.uniform(0, 1)*self.parameters['speed'])

	cpdef exploreGraph(self, list next_nodes, list visited):
		cdef int i, node		
		next_nodes = list(set(next_nodes)-set(visited))
		node = next_nodes[np.argmax([self.values[i] for i in next_nodes])]		
		visited.append(node)		
		self.path.append(node)
		if node == self.goal_node:
			return
		else:
			self.exploreGraph(self.edges[node], visited)

	cpdef computeActionAngle(self):
		cdef np.ndarray aim_position
		cdef float angle
		aim_position = self.nodes_position[self.path[0]]
		aim_position = aim_position - self.position
		angle = np.arctan2(aim_position[1], aim_position[0])
		self.action = angle - self.direction


