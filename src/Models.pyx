#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Models.pyx

Cython version

To compile : python csetup.py build_ext -i

Copyright (c) 2014 Guillaume VIEJO. All rights reserved.
"""

import numpy as np
cimport numpy as np
from itertools import izip
cimport cython
from cExpert import *

cdef class Model(object):
	cdef public dict parameters

	def __cinit__(self):
		self.parameters = dict()

	cpdef setParameter(self, str name, float value):	
		if name in self.parameters.keys() : self.parameters[name] = value

	cpdef setAllParameters(self, dict parameters):
		cdef str i
		for i in parameters.keys(): self.setParameter(i, parameters[i])		

	# def retrieveAction(self):
	# 	pass

cdef class Dolle(Model):
	cdef public dict experts, actions, w_nodes, w_lc, trace_lc, trace_nodes
	cdef public int n_ex, n_lc, n_nodes
	cdef public list k_ex
	cdef public float action_angle, action_distance, g_max
	cdef public str winner	
	cdef public np.ndarray g
	
	def __cinit__(self, tuple experts, dict parameters):
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
		self.g = np.zeros(self.n_ex) # Gate value
		self.g_max = 0.0 # Maximum value
		self.winner = None # Winner expert 
		self.w_nodes = dict() 
		self.w_lc = dict()
		self.trace_lc = dict()
		self.trace_nodes = dict()		

		for k in self.k_ex:
			self.experts[k].setAllParameters(self.parameters)
			self.w_nodes[k] = dict()  # Empty dict for weight between nodes and gate
			self.w_lc[k] = np.random.uniform(0,0.01, size = (1,self.n_lc)) # array of w for lc and gate
			self.trace_lc[k] = np.zeros((1,self.n_lc))
			self.trace_nodes[k] = dict()		

	cpdef psi(self, float x): return np.exp(-x**2.)-np.exp(-np.pi/2.)

	cpdef setPosition(self, float direction, float distance, np.ndarray position, np.ndarray wall, float agent_direction = 0.0):
		cdef str k, e
		cdef set new_nodes
		for k in self.k_ex:			
			self.experts[k].setCellInput(direction, distance, position, wall, agent_direction)
		if 'p' in self.k_ex and self.n_nodes != len(self.experts['p'].nodes.keys()):
			self.n_nodes = len(self.experts['p'].nodes.keys())
			new_nodes = set(self.experts['p'].nodes.keys())-set(self.w_nodes.keys())
			for e in self.k_ex: 
				self.w_nodes[e].update(izip(new_nodes,np.random.uniform(0,0.01,size=len(new_nodes))))			
				self.trace_nodes[e].update(izip(new_nodes,np.zeros(len(new_nodes))))

	cpdef computeGateValue(self):
		cdef np.ndarray tmp1
		cdef float tmp2 = 0.0
		cdef int i
		for i in xrange(self.n_ex):			
			tmp1 = np.array([self.experts['p'].nodes[j]*self.w_nodes[self.k_ex[i]][j] for j in self.w_nodes[self.k_ex[i]].keys()])			
			if 't' in self.k_ex: tmp2 = np.dot(self.w_lc[self.k_ex[i]], self.experts['t'].lc)
			self.g[i] = tmp2+np.sum(tmp1)

	cpdef retrieveAction(self):
		#super(Dolle, self).retrieveAction()
		cdef str e
		for e in self.k_ex:
			self.actions[e] = self.experts[e].computeNextAction()
		self.computeGateValue()
		
	cpdef getAction(self):
		self.retrieveAction()
		# CHOOSE EXPERTS						
		self.winner = self.k_ex[np.argmax(self.g)]
		self.g_max = np.max(self.g)
		self.action_angle, self.action_distance = self.actions[self.winner]		
		self.updateTrace()
		return (self.action_angle, self.action_distance)

	cpdef updateTrace(self):
		cdef str e
		cdef int i
		for e in self.k_ex:
			if 't' in self.k_ex:
				self.trace_lc[e]=self.parameters['lambda']*self.trace_lc[e]+self.psi(self.action_angle-self.actions[e][0])*self.experts['t'].lc
			if 'p' in self.k_ex:
				for i in self.trace_nodes[e].iterkeys():
					self.trace_nodes[e][i]=self.parameters['lambda']*self.trace_nodes[e][i]+self.experts['p'].nodes[i]*self.psi(self.action_angle-self.actions[e][0])

	cpdef learn(self, float reward):
		cdef str e
		cdef int i
		cdef float delta, tmp
		for e in self.k_ex:			
			self.experts[e].learn(self.action_angle, reward)
		delta = reward+self.parameters['gamma']*np.max(self.g)-self.g_max
		tmp = delta*self.parameters['epsilon']
		for e in self.k_ex:
			self.w_lc[e] = self.w_lc[e] + tmp*self.trace_lc[e]
			for i in self.w_nodes[e].iterkeys():
				self.w_nodes[e][i] = self.w_nodes[e][i] + tmp*self.trace_nodes[e][i]

		
		






		


