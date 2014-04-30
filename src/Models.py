#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Models.py    

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

from itertools import izip
from Expert import *
import numpy as np

class Model(object):

	def __init__(self, parameters = dict()):
		self.parameters = parameters

	def setParameter(self, name, value):
		if name in self.parameters.keys() : self.parameters[name] = value

	def setAllParameters(self, parameters):
		for i in parameters.keys(): self.setParameter(i, parameters[i])		

	def retrieveAction(self):
		pass

class Dolle(Model):

	def __init__(self, experts, parameters):
		self.parameters = {	'epsilon': 0.01,
							'gamma': 0.8,
							'lambda': 0.76,
							'nlc':100 }		
		self.setAllParameters(parameters)
		self.experts = {'t':Taxon(), 
						'p':Planning(),
						'e':Expert()}
		self.experts = dict(filter(lambda i:i[0] in experts, self.experts.iteritems())) # Which expert to keep								
		self.n_ex = len(self.experts.keys()) # Number of experts
		self.k_ex = self.experts.keys() # Keys of experts | faster to declare here
		self.n_lc = self.parameters['nlc'] # NUmber of landmarks cells		
		self.n_nodes = 0 # Number of nodes | not constant
		self.actions = dict.fromkeys(self.k_ex) # Proposed action from each expert
		self.action_angle = 0.0
		self.action_distance = 0.0
		self.g = np.zeros(self.n_ex) # Gate Value
		self.g_max = 0.0 # Maximum value
		self.winner = None # Winner expert 
		self.w_nodes = dict() 
		self.w_lc = dict()
		self.trace_lc = dict()
		self.trace_nodes = dict()		
		self.psi = lambda x: np.exp(-x**2.)-np.exp(-np.pi/2.)
		for k in self.k_ex:
			self.experts[k].setAllParameters(self.parameters)
			self.w_nodes[k] = dict()  # Empty dict for weight between nodes and gate
			self.w_lc[k] = np.random.uniform(0,0.01, size = (1,self.n_lc)) # array of w for lc and gate
			self.trace_lc[k] = np.zeros((1,self.n_lc))
			self.trace_nodes[k] = dict()		

	def setPosition(self, direction, distance, position, wall, agent_direction = 0):	
		for k in self.k_ex:
			self.experts[k].setCellInput(direction, distance, position, wall, agent_direction)
		if 'p' in self.k_ex and self.n_nodes != len(self.experts['p'].nodes.keys()):
			self.n_nodes = len(self.experts['p'].nodes.keys())
			new_nodes = set(self.experts['p'].nodes.keys())-set(self.w_nodes.keys())
			for e in self.k_ex: 
				self.w_nodes[e].update(izip(new_nodes,np.random.uniform(0,0.01,size=len(new_nodes))))			
				self.trace_nodes[e].update(izip(new_nodes,np.zeros(len(new_nodes))))

	def computeGateValue(self):		
		tmp2 = 0.0
		for i in xrange(self.n_ex):			
			tmp1 = np.array([self.experts['p'].nodes[j]*self.w_nodes[self.k_ex[i]][j] for j in self.w_nodes[self.k_ex[i]].keys()])			
			if 't' in self.k_ex: tmp2 = np.dot(self.w_lc[self.k_ex[i]], self.experts['t'].lc)
			self.g[i] = tmp2+np.sum(tmp1)

	def retrieveAction(self):
		super(Dolle, self).retrieveAction()
		for e in self.k_ex:
			self.actions[e] = self.experts[e].computeNextAction()
		self.computeGateValue()
		
	def getAction(self):
		self.retrieveAction()
		# CHOOSE EXPERTS						
		self.winner = self.k_ex[np.argmax(self.g)]
		self.g_max = np.max(self.g)
		self.action_angle, self.action_distance = self.actions[self.winner]		
		self.updateTrace()
		return (self.action_angle, self.action_distance)

	def updateTrace(self):
		for e in self.k_ex:
			if 't' in self.k_ex:
				self.trace_lc[e]=self.parameters['lambda']*self.trace_lc[e]+self.psi(self.action_angle-self.actions[e][0])*self.experts['t'].lc
			if 'p' in self.k_ex:
				for i in self.trace_nodes[e].iterkeys():
					self.trace_nodes[e][i]=self.parameters['lambda']*self.trace_nodes[e][i]+self.experts['p'].nodes[i]*self.psi(self.action_angle-self.actions[e][0])

	def learn(self, reward):
		for e in self.k_ex:
			self.experts[e].learn(self.action_angle, reward)
		delta = float(reward)+self.parameters['gamma']*np.max(self.g)-self.g_max
		tmp = delta*self.parameters['epsilon']
		for e in self.k_ex:
			self.w_lc[e] = self.w_lc[e] + tmp*self.trace_lc[e]
			for i in self.w_nodes[e].iterkeys():
				self.w_nodes[e][i] = self.w_nodes[e][i] + tmp*self.trace_nodes[e][i]

		
		






		


