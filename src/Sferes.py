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
import sys

class EA_pearce():
	"""
	Black-box optimization
	"""
	def __init__(self, data, agent, agent_lesion):
		self.agent = agent
		self.agent_lesion = agent_lesion
		self.data = data		
		self.speed = 18.0 # cm/second
		self.time_step = 0.33 # second
		self.time_limit = 10.0 # second
		self.n_steps_max = int(self.time_limit/self.time_step)
		self.n_trials = 4
		self.n_sessions = 11
		self.n_repet = 3
		self.time = np.zeros((self.n_sessions, 4, self.n_repet)) # The time to escape to compare with pearce

	def runSession(self, agent, repet, session, k):		
		agent.world.startSession()
		for i in xrange(self.n_trials):
			print k, repet, session
			agent.world.startTrial()			
			agent.start()			
			n_step = 0
			while n_step <= self.n_steps_max and not agent.world.reward_found:
				agent.step()
				n_step+=1
				if i == 0 or i == 3:
					self.time[session, k+i%2, repet] += self.time_step			
			if not agent.world.reward_found:
				agent.guide()			

	def getFitness(self):
		np.seterr(all='ignore')	
		for i in xrange(self.n_repet):
			self.agent.model.reset()
			self.agent_lesion.model.reset()			
			for j in xrange(self.n_sessions):
				self.runSession(self.agent,i,j,0)		
				self.runSession(self.agent_lesion,i,j,2)
		
		return None


	def leastSqares(self):
		pass