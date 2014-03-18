#!/usr/bin/python
# encoding: utf-8
"""
Models.py    

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

from itertools import izip
from Expert import Taxon, Planning, Exploration
import numpy as np
import sys

class Model(object):

	def __init__(self):
		self.parameters = dict()

	def setParameter(self, name, value):
		if name in self.parameters.keys() : self.parameters[name] = value

	def setAllParameters(self, parameters):
		for i in parameters.keys(): self.setParameter(i, parameters[i])
		map(lambda x: x.setAllParameters(parameters), self.experts.values())

	def getAction(self):
		pass

class Dolle(Model):

	def __init__(self):
		self.parameters = {'epsilon': 0.01}		
		self.experts = { 't':Taxon(), 
						'p':Planning(),
						'e':Exploration()}
		self.n_ex = len(self.experts.keys())
		self.n_lc = self.experts['t'].parameters['nlc'] # NUmber of landmarks cells
		self.w_lc = np.random.rand(self.n_ex, self.n_lc)
		self.n_nodes = len(self.experts['p'].nodes)
		self.actions = dict.fromkeys(self.experts.keys())
		self.g = dict(izip(self.experts.keys(),np.zeros(3)))

	def setPosition(self, position, direction, distance):		
		self.experts['t'].computeLandmarkActivity(direction,distance)
		self.experts['p'].computePlaceCellActivity(position)
		self.n_nodes = len(self.experts['p'].nodes)
		self.tmp = np.dot(self.w_lc, self.experts['t'].lc)


	def getAction(self):
		super(Dolle, self).getAction()
		for i in self.actions.iterkeys():
			self.actions[i] = self.experts[i].computeNextAction()




		


