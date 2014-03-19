#!/usr/bin/python
# encoding: utf-8
"""
Expert.py

    

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import numpy as np
from itertools import izip
import sys

class Expert(object):

	def __init__(self):
		self.parameters = dict()

	def setParameter(self, name, value):
		if name in self.parameters.keys() : self.parameters[name] = value

	def setAllParameters(self, parameters):
		for i in parameters.keys(): self.setParameter(i, parameters[i])

	def learn(self):
		pass

	def computeNextAction(self):
		return np.random.uniform(0,2*np.pi)

class Taxon(Expert):

	def __init__(self):
		self.parameters = { 'nlc': 100,		 		    # Number of landmarks cells
							'sigma_lc': 0.475,			# Normalized landmark width
							'sigma':0.392,				# Number of action cells
							'nac': 36,					# Standard deviation of the generalization profile
							'eta': 0.001,				# Learning rate
							'lambda': 0.76,				# Eligibility trace decay factor
							'gamma' : 0.8 }				# Discount factor
		# Landmarks cells
		self.lc_direction = (np.arange(self.parameters['nlc'])*2.0*np.pi)/float(self.parameters['nlc'])
		self.lc = np.zeros((self.parameters['nlc']))
		# Action cells
		self.ac_direction = (np.arange(self.parameters['nac'])*2.0*np.pi)/float(self.parameters['nac'])
		self.ac = np.zeros((self.parameters['nac']))
		# Connection
		self.W = np.random.rand(self.parameters['nac'], self.parameters['nlc'])	
		# Proposed direction
		self.actual_direction = 0.0
		self.direction = 0.0 # The proposed direction
		# Learning initialization		
		self.delta = 0.0
		self.trace = np.zeros((self.parameters['nac'], self.parameters['nlc']))

	def computeLandmarkActivity(self, direction, distance):
		self.actual_direction = float(direction)
		self.lc = np.exp(-(np.power((float(direction)-self.lc_direction),2))/(2*(self.parameters['sigma_lc']/float(distance))**2))
		self.computeActionActivity()

	def computeActionActivity(self):
		self.ac = np.dot(self.W, self.lc)		
		self.direction = np.arctan((self.ac*np.sin(self.ac_direction)).sum()/(self.ac*np.cos(self.ac_direction)).sum())
		self.updateTrace()

	def updateTrace(self):
		ac = np.exp(-(np.power((self.direction-self.ac_direction),2))/(2*self.parameters['sigma']**2))
		self.trace = self.parameters['lambda']*self.trace+np.outer(ac, self.lc)

	def learn(self, reward):
		super(Taxon, self).learn()
		self.delta = reward + self.parameters['gamma']*self.ac.max()-self.ac
		self.W = self.W+self.parameters['eta']*(np.tile(self.delta, (self.parameters['nlc'],1))).T*self.trace

	def computeNextAction(self):
		super(Taxon, self).computeNextAction()
		return self.direction
	
class Planning(Expert):

	def __init__(self):
		self.parameters = { 'theta_pc': 0.2,			# Activity threshold for place cells node linking
							'theta_node': 0.3,			# Activity threshold for node creation
							'alpha': 0.7, 				# Decay factor of the goal value
							'npc': 1681,				# Number of simulated Place cells
							'sigma_pc': 0.2 }				# Place field size
		# Place cells
		self.pc = np.zeros((self.parameters['npc']))
		self.pc_position = np.random.uniform(-1,1, (self.parameters['npc'],2))
		# Graph planning		
		self.nb_nodes = 0
		self.pc_nodes = dict() #indices : weight for place cells - nodes links
		self.nodes = dict() # Nodes activity
		self.current_node = 0 # Current node
		self.goal_node = 0
		self.goal_found = False
		# Planning
		self.edges = dict({0:[]}) # Links between nodes
		self.values = dict() # Weight associated to each nodes, should change every time step
		
	def computePlaceCellActivity(self, position):
		if np.max(position)>1.0 or np.min(position)<-1.0: raise Warning("Place cells position should be normalized between [-1,1]")
		distance = np.sqrt((np.power(self.pc_position-position, 2)).sum(1))
		self.pc = np.exp(-distance/(2*self.parameters['sigma_pc']**2))
		self.computeGraphNodeActivity()	

	def computeGraphNodeActivity(self):			
		for i in self.nodes.iterkeys(): self.nodes[i] = np.dot(self.pc[self.pc_nodes[i].keys()],self.pc_nodes[i].values())		
		if len(self.nodes.keys()) == 0 or np.max(self.nodes.values()) < self.parameters['theta_node']:
			self.createNewNode()
		else:
			self.connectNode()

	def createNewNode(self):
		# Store a list of place cells indice
		# Each indice indicates the position of the place field in the environment
		self.nb_nodes+=1
		ind = np.where(self.pc>self.parameters['theta_pc'])[0]		# The indices to the place cells
		self.pc_nodes[self.nb_nodes]  = dict(izip(ind, self.pc[ind]))   	# key : PC ind | values : PC activity
		self.nodes[self.nb_nodes] = np.dot(self.pc[self.pc_nodes[self.nb_nodes].keys()],self.pc_nodes[self.nb_nodes].values())
		self.edges[self.nb_nodes] = [self.current_node]
		self.edges[self.current_node].append(self.nb_nodes)
		self.values[self.nb_nodes] = 0.0
		self.current_node = self.nb_nodes

	def connectNode(self):
		new_node = np.argmax(self.nodes.values())+1
		if self.current_node not in self.edges[new_node] and new_node != self.current_node:
			self.edges[new_node].append(self.current_node)
			self.edges[self.current_node].append(new_node)		
		self.current_node = new_node

	def isGoalNode(self):
		# should be used only when reward is found
		self.goal_found = True
		for i in self.values.iterkeys(): self.values[i] = 0.0
		self.values[self.current_node] = 1.0		
		map(lambda x: self.propagate(x, [self.current_node], self.parameters['alpha']), self.edges[self.current_node])		

	def propagate(self, new_node, visited, value):
		if self.values[new_node]<value: self.values[new_node] = value
		visited.append(new_node)				
		next_node = list(set(self.edges[new_node])-set(visited))
		if new_node-1:
		 	map(lambda x: self.propagate(x, visited, self.parameters['alpha']*value), next_node)		

	def computeNextAction(self):
		super(Planning, self).computeNextAction()			
		if self.goal_found:
			self.current_node = np.argmax(self.nodes.values())
			self.goal_node = np.argmax(self.values.values())+1
			self.path = []
			self.exploreGraph(self.edges[self.current_node], [self.current_node])			
		else : 
		 	return np.random.uniform(0, 2*np.pi)

	def exploreGraph(self, next_nodes, visited):		
		next_nodes = list(set(next_nodes)-set(visited))		
		node = next_nodes[np.argmax([self.values[i] for i in next_nodes])]		
		visited.append(node)		
		self.path.append(node)		
		if node == self.goal_node:
			return
		else:
			self.exploreGraph(self.edges[node], visited)


class Exploration(Expert):

	def __init__(self):
		self.parameters = {}