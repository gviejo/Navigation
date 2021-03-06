#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Expert.py

    

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
"""

import numpy as np
from itertools import izip


class Expert(object):

	def __init__(self, parameters = {}):
		self.parameters = dict({'speed':0.1})
		self.setAllParameters(parameters)		

	def setParameter(self, name, value):
		if name in self.parameters.keys() : 			
			self.parameters[name] = value

	def setAllParameters(self, parameters):		
		for i in parameters.keys(): self.setParameter(i, parameters[i])

	def reset(self):
		pass

	def learn(self, angle, reward):
		pass

	def setCellInput(self, direction, distance, position, wall, agent_direction = 0):
		pass

	def computeNextAction(self):
		#return (np.random.uniform(-np.pi, np.pi), np.random.uniform(0, 1)*self.parameters['speed'])
		return (np.random.uniform(-np.pi, np.pi), self.parameters['speed'])		

class Taxon(Expert):

	def __init__(self, parameters={}):
		self.parameters = { 'nlc': 100,		 		    # Number of landmarks cells
							'sigma_lc': 0.475,			# Normalized landmark width
							#'sigma_vc': 0.001, 		# Visual cell width
							'sigma':0.392,				# Standard deviation of the generalization profile
							'nac': 36,					# Number of action cells
							'eta': 0.001,				# Learning rate
							'lambda': 0.76,				# Eligibility trace decay factor
							'gamma' : 0.8,				# Discount factor
							'speed' : 0.1 }

		self.setAllParameters(parameters)							
		# Landmarks cells
		self.lc_direction = np.arange(-np.pi, np.pi, (2*np.pi)/float(self.parameters['nlc']))
		self.lc = np.zeros((self.parameters['nlc']))
		# Visual cells
		self.vc_direction = np.arange(-np.pi, np.pi, (2*np.pi)/float(self.parameters['nac']))
		self.vc = np.zeros((self.parameters['nac']))
		# Action cells
		self.ac_direction = np.arange(-np.pi, np.pi, (2*np.pi)/float(self.parameters['nac']))
		self.ac = np.zeros((self.parameters['nac']))
		# Connection
		self.W = np.random.normal(0.0, 0.1, size=(self.parameters['nac'], self.parameters['nlc']))
		# Proposed direction		
		self.action = 0.0 # The proposed direction
		self.norm = 0.0 # The distance if action is choosen
		# Learning initialization		
		self.delta = 0.0
		self.trace = np.zeros((self.parameters['nac'], self.parameters['nlc']))		

	def reset(self):
		super(Taxon, self).reset()
		self.W = np.random.normal(0.0, 0.1, size=(self.parameters['nac'], self.parameters['nlc']))
		self.trace = np.zeros((self.parameters['nac'], self.parameters['nlc']))		

	def setCellInput(self, direction, distance, position, wall, agent_direction = 0):
		""" Direction should be in [-pi, pi] interval 
		Null angle is the curent direction of the agent"""
		delta = np.arccos(np.cos(direction)*np.cos(self.lc_direction)+np.sin(direction)*np.sin(self.lc_direction))		
		self.lc = np.exp(-(np.power(delta,2))/(2*(self.parameters['sigma_lc']/float(distance))**2))
		#delta = np.arccos(np.cos(wall[0])*np.cos(self.vc_direction)+np.sin(wall[0])*np.sin(self.vc_direction))
		#self.vc = np.exp(-(np.power(delta, 2))/(2*(self.parameters['sigma_vc']/float(wall[1]-0.0001))**2))
		self.computeActionActivity()		

	def computeActionActivity(self):		
		self.ac = np.dot(self.W, self.lc) #- self.vc
		self.ac = np.tanh(self.ac)
		xy = [(self.ac*np.sin(self.ac_direction)).sum(), (self.ac*np.cos(self.ac_direction)).sum()]
		self.action = np.arctan2(xy[0], xy[1])
		self.norm = np.sqrt(np.sum(np.power(xy, 2)))
		#self.norm = self.parameters['speed']/(1.+np.exp(-self.norm))			
		self.norm = self.parameters['speed']

	def updateTrace(self, action):
		delta = np.arccos(np.cos(action)*np.cos(self.ac_direction)+np.sin(action)*np.sin(self.ac_direction))		
		ac = np.exp(-(np.power(delta,2))/(2*self.parameters['sigma']**2))		
		self.trace = self.parameters['lambda']*self.trace+np.outer(ac, self.lc)

	def learn(self, action, reward):
		""" Action performed selected from a mixture of experts"""
		super(Taxon, self).learn(action, reward)
		self.updateTrace(action)
		self.delta = reward + self.parameters['gamma']*self.ac.max()-action
		self.W = self.W+self.parameters['eta']*(np.tile(self.delta, (self.parameters['nlc'],1))).T*self.trace


	def computeNextAction(self):
		""" Called by general model for choosing action
		if mixture of experts, return action angle and distance to "walk" """
		super(Taxon, self).computeNextAction()
		return self.action, self.norm
	
class Planning(Expert):

	def __init__(self, parameters={}):
		self.parameters = { 'theta_pc': 0.2,			# Activity threshold for place cells node linking
							'theta_node': 0.3,			# Activity threshold for node creation
							'alpha': 0.7, 				# Decay factor of the goal value
							'npc': 1681,				# Number of simulated Place cells
							'sigma_pc': 0.2, 
							'speed' : 0.1 }				# Place field size

		self.setAllParameters(parameters)
		self.direction = None # Direction of the agent in a allocentric frame [-pi, pi]
		self.position = None
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

	def reset(self):
		super(Planning, self).reset()
		self.pc_position = np.random.uniform(-1,1, (self.parameters['npc'],2))
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

	def setCellInput(self, direction, distance, position, wall, agent_direction = 0):
		""" Only position is used """
		if np.max(position)>1.0 or np.min(position)<-1.0: raise Warning("Place cells position are normalized between [-1,1]")
		self.direction = agent_direction
		self.position = position
		distance = np.sqrt((np.power(self.pc_position-position, 2)).sum(1))
		self.pc = np.exp(-distance/(2*self.parameters['sigma_pc']**2))
		self.computeGraphNodeActivity()	

	def computeGraphNodeActivity(self):
		""" Dot product of place cells activity and pc-> nodes links """
		for i in self.nodes.iterkeys(): self.nodes[i] = np.dot(self.pc[self.pc_nodes[i].keys()],self.pc_nodes[i].values())
		if len(self.nodes.keys()) == 0 or np.max(self.nodes.values()) < self.parameters['theta_node']:
			self.createNewNode()
		elif not self.goal_found:
			self.connectNode()
		else:
			self.current_node = np.argmax(self.nodes.values())+1

	def createNewNode(self):
		""" Store a list of place cells indice
		 Each indice indicates the position of the place field in the environment """
		ind = np.where(self.pc>self.parameters['theta_pc'])[0]		# The indices to the place cells		
		if len(ind): # New node are created only if place cell activity is larger than threshold
			self.nb_nodes+=1
			self.pc_nodes[self.nb_nodes]  = dict(izip(ind, self.pc[ind]))   	# key : PC ind | values : PC activity
			self.nodes[self.nb_nodes] = np.dot(self.pc[self.pc_nodes[self.nb_nodes].keys()],self.pc_nodes[self.nb_nodes].values())
			self.edges[self.nb_nodes] = [self.current_node]
			self.edges[self.current_node].append(self.nb_nodes)
			self.values[self.nb_nodes] = 0.0
			self.nodes_position[self.nb_nodes] = np.mean(self.pc_position[ind], 0)		
			self.current_node = self.nb_nodes

	def createGoalNode(self):
		""" Called only if reward is explicitly found.
		ALlow to set a goal node over a specific place
		Also propagate values"""
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
		map(lambda x: self.propagate(x, [self.current_node], self.parameters['alpha']), self.edges[self.current_node])

	def connectNode(self):
		new_node = np.argmax(self.nodes.values())+1
		if self.current_node not in self.edges[new_node] and new_node != self.current_node:
			self.edges[new_node].append(self.current_node)
			self.edges[self.current_node].append(new_node)		
		self.current_node = new_node		

	def learn(self, action, reward):
		""" Only if reward not found """
		super(Planning, self).learn(action, reward)		
		if self.goal_found:
			for x in self.edges[self.goal_node]:
				self.propagate(x, [self.goal_node], self.parameters['alpha'])
		elif reward > 0.0 and not self.goal_found:
			self.goal_found = True			
			self.createGoalNode()			

	def propagate(self, new_node, visited, value):
		""" Propagate discounted value starting from goal node"""
		if self.values[new_node]<value: self.values[new_node] = value
		visited.append(new_node)				
		next_node = list(set(self.edges[new_node])-set(visited))
		if new_node-1:
		 	map(lambda x: self.propagate(x, visited, self.parameters['alpha']*value), next_node)		

	def computeNextAction(self):
		""" Return tuple (direction, speed) """
		super(Planning, self).computeNextAction()		
		if self.goal_found:
			if self.current_node == self.goal_node: 
			 	return (0.0, 0.0)
			else:
				self.path = []
				self.exploreGraph(self.edges[self.current_node], [0, self.current_node])			
				self.computeActionAngle()
				return (self.action, self.parameters['speed'])
		else : 
		 	#return (np.random.uniform(0,2*np.pi), np.random.uniform(0, 1)*self.parameters['speed'])
		 	return (np.random.uniform(0,2*np.pi), self.parameters['speed'])

	def exploreGraph(self, next_nodes, visited):		
		next_nodes = list(set(next_nodes)-set(visited))		
		node = next_nodes[np.argmax([self.values[i] for i in next_nodes])]		
		visited.append(node)		
		self.path.append(node)
		if node == self.goal_node:
			return
		else:
			self.exploreGraph(self.edges[node], visited)

	def computeActionAngle(self):		
		aim_position = self.nodes_position[self.path[0]]
		aim_position = aim_position - self.position
		angle = np.arctan2(aim_position[1], aim_position[0])
		self.action = angle - self.direction