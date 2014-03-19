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

	def __init__(self, model, parameters):
		self.model = model
		self.parameters = parameters
		self.model.setAllParameters(self.parameters)
		

	def step(self):
		self.model.move()		
		self.model.learn(0.0)