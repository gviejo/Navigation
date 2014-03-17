#!/usr/bin/python
# encoding: utf-8
"""
Expert.py

    

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import numpy as np
from itertools import izip

class Expert(object):

	def __init__(self):
		self.parameters = dict()

	def setParameter(self, name, value):
		if name in self.parameters.keys() : self.parameters[name] = value

	def setAllParameters(self, parameters):
		for i in parameters.keys(): self.setParameter(i, parameters[i])

	def learn(self):
		pass

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
		self.direction = 0.0
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
	
class Planning(Expert):

	def __init__(self):
		self.parameters = { 'theta_pc': 0.2,			# Activity threshold for place cells node linking
							'theta_node': 0.3,			# Activity threshold for node creation
							'alpha': 0.7, 				# Decay factor of the goal value
							'npc': 1681,				# Number of simulated Place cells
							'sigma_pc': 0.3 }				# Place field size
		# Place cells
		self.pc = np.zeros((self.parameters['npc']))
		self.pc_position = np.random.uniform(-1,1, (self.parameters['npc'],2))
		# Graph planning		
		self.nb_nodes = 0
		self.pc_nodes = dict() #indices : weight for place cells - nodes links
		self.nodes = dict() # Nodes activity
		

	def computePlaceCellActivity(self, position):
		if np.max(position)>1.0 or np.min(position)<-1.0: raise Warning("Place cells position should be normalized between [-1,1]")
		distance = np.sqrt((np.power(self.pc_position-position, 2)).sum(1))
		self.pc = np.exp(-distance/(2*self.parameters['sigma_pc']**2))

	def computeGraphNodeActivity(self):
		for i in self.nodes.iterkeys(): self.nodes[i] = np.dot(self.pc[self.pc_nodes[i].keys()],self.pc_nodes[i].values())
		if len(self.nodes.keys()) == 0 or np.max(self.nodes.values()) < self.parameters['theta_node']: self.createNewNode()

	def createNewNode(self):
		# Store a list of place cells indice
		# Each indice indicates the position of the place field in the environment
		self.nb_nodes+=1
		ind = np.where(self.pc>self.parameters['theta_pc'])[0]		
		self.pc_nodes[self.nb_nodes]  = dict(izip(ind, self.pc[ind]))
		self.nodes[self.nb_nodes] = 0.0


class Exploration(Expert):

	def __init__(self):
		pass
