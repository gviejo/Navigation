#!/usr/bin/python
# encoding: utf-8
"""
taxon_test.py

    

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""
import sys
sys.path.append("../src")
import numpy as np
from matplotlib import *
from pylab import *
from Models import Dolle
from setup import *
from time import time
import matplotlib.patches as mpatches

parameters = { 'nlc': 100,		 		    # Number of landmarks cells
				'sigma_lc': 0.1,			# Normalized landmark width
				'sigma_vc': 0.04,    	    # Visual cell width
				'sigma':0.1,				# Action cell width
				'nac': 36,					# Standard deviation of the generalization profile
				'eta': 0.1,				    # Learning rate
				'lambda': 0.1,				# Eligibility trace decay factor
				'gamma' : 0.8,
				'theta_pc': 0.2,			# Activity threshold for place cells node linking
				'theta_node': 0.3,			# Activity threshold for node creation
				'alpha': 0.7, 				# Decay factor of the goal value
				'npc': 1681,				# Number of simulated Place cells
				'sigma_pc': 0.2,
				'epsilon': 0.01,
				'speed':0.1 				# Max Speed
				 }	

agent = Agent(Dolle(('t', 'e'), parameters), World(), parameters, stats = True)

def learn():
	t1 = time()
	for i in xrange(10):
		print "Trial :" + str(i)
		agent.start()
		#while not agent.world.reward_found and agent.n_steps[-1] < 1000:
		while agent.n_steps[-1] < 1000:
			agent.step()
	t2 = time()
	print t2 - t1
	PLOT_LEARN()


def test():
	t1 = time()	
	while not agent.world.reward_found and agent.n_steps[-1] < 1000:
		agent.step()		
	t2 = time()
	print t2-t1
	PLOT()

def PLOT_LEARN():
	position = map(np.array, agent.positions)
	trace = agent.model.experts['t'].trace
	W = agent.model.experts['t'].W
	rewards = np.array(agent.rewards)*1.0
	figure(figsize = (17, 11))

	ax1 = subplot2grid((3,3),(0,0), colspan = 1)
	[plot(position[i][:,0], position[i][:,1], '.-', alpha = 0.1) for i in xrange(len(position))]
	xlim(-1,1);ylim(-1,1)
	scatter(agent.landmark_position[0], agent.landmark_position[1], s = 100, c = 'red')
	c = mpatches.Circle(agent.world.reward_position, agent.world.reward_size, fc = 'g')
	ax1.add_patch(c)

	subplot2grid((3,3),(1,0), colspan = 1)	
	imshow(trace, interpolation='nearest')
	title("trace")

	subplot2grid((3,3),(2,0), colspan = 1)
	imshow(W, interpolation = 'nearest')
	title("Weight")

	subplot2grid((3,3),(0,1))
	plot(agent.n_steps, 'o-')

	legend()
	#ylim(-np.pi, np.pi)
	show()

def PLOT():
	position = np.array(agent.positions)	
	distance = np.array(agent.distances)	
	actions = np.array(agent.actions)
	trace = np.array(agent.model.experts['t'].ltrace)
	W = np.array(agent.model.experts['t'].lW)	
	gates = np.array(agent.gates)
	figure(figsize = (17, 11))

	ax1 = subplot2grid((3,3),(0,0), colspan = 1)
	plot(position[0][:,0], position[0][:,1], '.-', alpha = 0.5);xlim(-1,1);ylim(-1,1)
	scatter(agent.landmark_position[0], agent.landmark_position[1], s = 100, c = 'red')
	c = mpatches.Circle(agent.world.reward_position, agent.world.reward_size, fc = 'g')
	ax1.add_patch(c)

	subplot2grid((3,3),(1,0), colspan = 1)	
	imshow(trace[-1], interpolation='nearest')
	title("trace")

	subplot2grid((3,3),(2,0), colspan = 1)
	imshow(W[-1], interpolation = 'nearest')
	title("Weight")

	subplot2grid((3,3),(0,1), colspan = 2)
	plot(distance[:,0], 'o-', label = "Distance to landmark", color = 'red')
	plot(distance[:,1], 'o-', label = "Distance to reward", color = 'green')
	legend()

	subplot2grid((3,3),(1,1), colspan = 2)
	#[plot(agent.model.experts['t'].lc_direction, lcs[:,i], 'o-', label = str(i)) for i in xrange(lcs.shape[1])]
	#plot(agent.rewards)
	[plot(gates[:,i], 'o-', color = agent.colors[k], alpha = 0.9) for k,i in zip(agent.model.k_ex, range(agent.model.n_ex))]
	line = [Line2D(range(1),range(1), marker = 'o', alpha =1.0, color = agent.colors[e], label = str(e)) for e in agent.colors.keys()]
	legend(line, tuple(agent.colors.keys()))
	
	# xlim(0,len(agent.actions))
	#ylim(0, 2*np.pi)
	#plot(wall[:,1], 'o-', label = 'distance to wall')
	#[plot(agent.model.experts['t'].ac_direction, lvc[:,i], 'o-', label = str(i)) for i in xrange(lvc.shape[1])]
	legend()

	subplot2grid((3,3),(2,1), colspan = 2)
	plot(actions[:,0], label = 'angle')
	plot(actions[:,1], label = 'speed')
	
	#hist(agent.winners)
	#plot(agent.speeds)
	#plot(lac)
	#plot(agent.model.experts['t'].ldirec, 'o-', label = 'direction')
	#plot(actions[:,0], 'o-', label = 'action direction')
	#[plot(agent.model.experts['t'].ac_direction, ldelta[:,i], 'o-', label = str(i)) for i in xrange(ldelta.shape[1])]
	#plot(wall[:,0], 'o-', label = 'wall angle')

	legend()
	#ylim(-np.pi, np.pi)
	show()

learn()