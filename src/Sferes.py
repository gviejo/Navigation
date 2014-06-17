#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sferes.py    

    class for multi-objective optimization
    to interface with sferes2 : see
    http://sferes2.isir.upmc.fr/
    TODO : define fitness function

	
Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

import numpy as np
import os

class EA():
	"""
	Black-box optimization
	"""
	def __init__(self, data, model):
		self.model = model
		self.data = data
		self.speed = 18.0 #cm/second
		self.time_step = 0.33 # second
		self.time_limit = 200.0 # second

	def getFitness(self):
		np.seterr(all='ignore')
		


	def leastSqares(self):
		pass