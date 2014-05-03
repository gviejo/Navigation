#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dolle.py

    

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""
import sys
sys.path.append("../src")
import numpy as np
from matplotlib import *
from pylab import *
from setup import *
from time import time
import matplotlib.patches as mpatches

parameters = {  'nlc': 100,		 		    # Number of landmarks cells
				'sigma_lc': 0.47,			# Normalized landmark width
				'nac': 36, 					# Number of action cells				
				'sigma':0.39,				# Standard deviation of the generalization profile
				'eta': 0.001,				# Learning rate
				'lambda': 0.76,				# Eligibility trace decay factor
				'gamma' : 0.8,				# Future reward discount factor
				'epsilon': 0.01, 			# Learning rate of the gating network
				'theta_pc': 0.3,			# Activity threshold for place cells node linking
				'theta_node': 0.3,			# Activity threshold for node creation
				'alpha': 0.7, 				# Decay factor of the goal value
				'npc': 1681,				# Number of simulated Place cells
				'sigma_pc': 0.1, 			# Place field size
				
				'speed':0.1, 				# Max Speed
				'sigma_vc': 0.04    	    # Visual cell width
				 }	


name = {'p':'Planning','t':'Taxon','e':'Exploratory'}

agent = Agent(('p','t','e'), Maze(), parameters)

def learn():
	t1 = time()
	for i in xrange(10):
		print "Trial :" + str(i)
		agent.start()
		while not agent.world.reward_found and agent.n_steps[-1] < 200:		
			agent.step()
	t2 = time()
	print t2 - t1
	PLOT_LEARN()


def test():
	t1 = time()	
	agent.stats = 'test'
	agent.start()
	while not agent.world.reward_found and agent.n_steps[-1] < 200:		
		agent.step()
	t2 = time()
	print t2-t1
	PLOT_TEST()

def PLOT_LEARN():
	position = map(np.array, agent.positions)
	trace = agent.model.experts['t'].trace
	W = agent.model.experts['t'].W
	experts = np.array(agent.experts)
	experts = experts/experts.sum(1, keepdims = True, dtype = np.double)
	gating = np.array(agent.gating)

	figure(figsize = (17, 11))

	ax1 = subplot2grid((3,3),(0,0), colspan = 1, rowspan = 1)
	[plot(position[i][:,0], position[i][:,1], '.-', alpha = 0.1) for i in xrange(len(position))]
	[plot(agent.world.walls[i][:,0], agent.world.walls[i][:,1], color = 'black', linewidth = 5) for i in agent.world.walls.iterkeys()]
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

	subplot2grid((3,3),(0,1), colspan = 1)
	scatter(agent.model.experts['p'].pc_position[:,0], agent.model.experts['p'].pc_position[:,1], marker = '+', s = 150, c = agent.model.experts['p'].pc, cmap = cm.coolwarm)
	for i in agent.model.experts['p'].nodes.keys():
		a = agent.model.experts['p'].pc_position[agent.model.experts['p'].pc_nodes[i].keys()]
		plot(a[:,0], a[:,1], 'o', alpha = 0.8)	
		plot(agent.model.experts['p'].nodes_position[i][0], agent.model.experts['p'].nodes_position[i][1], 'o')
		for j in agent.model.experts['p'].edges[i]:
			if j:
				tmp = np.array([agent.model.experts['p'].nodes_position[i], agent.model.experts['p'].nodes_position[j]])
				plot(tmp[:,0], tmp[:,1], color = 'black', alpha  = 0.9)
	xlim(-1,1);ylim(-1,1)	
	title("Graph")

	subplot2grid((3,3),(0,2), colspan = 1)
	plot(agent.n_steps, 'o-')
	title("Escape latency")

	subplot2grid((3,3),(1,1), colspan = 2)
	[plot(gating[:,i], '-+', label = name[agent.model.k_ex[i]]) for i in xrange(agent.model.n_ex)]
	legend()
	title("Gating value")

	subplot2grid((3,3),(2,1), colspan = 2)
	[plot(experts[:,i], 'o-', label = name[agent.model.k_ex[i]]) for i in xrange(agent.model.n_ex)]
	ylim(0,1)
	legend()
	title("p(Experts)")

	show()


def PLOT_TEST():
	position = np.array(agent.positions[-1])
	distance = np.array(agent.distances)
	trace = agent.model.experts['t'].trace
	W = agent.model.experts['t'].W
	experts = np.array(agent.experts)
	experts = experts/experts.sum(1, keepdims = True, dtype = np.double)
	pgates = np.array(agent.pgates)

	figure(figsize = (17, 11))

	ax1 = subplot2grid((3,3),(0,0), colspan = 1, rowspan = 1)
	plot(position[:,0], position[:,1], '.-')
	[plot(agent.world.walls[i][:,0], agent.world.walls[i][:,1], color = 'black', linewidth = 5) for i in agent.world.walls.iterkeys()]
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

	subplot2grid((3,3),(0,1), colspan = 1)
	scatter(agent.model.experts['p'].pc_position[:,0], agent.model.experts['p'].pc_position[:,1], marker = '+', s = 150, c = agent.model.experts['p'].pc, cmap = cm.coolwarm)
	for i in agent.model.experts['p'].nodes.keys():
		a = agent.model.experts['p'].pc_position[agent.model.experts['p'].pc_nodes[i].keys()]
		plot(a[:,0], a[:,1], 'o', alpha = 0.8)	
		plot(agent.model.experts['p'].nodes_position[i][0], agent.model.experts['p'].nodes_position[i][1], 'o')
		for j in agent.model.experts['p'].edges[i]:
			if j:
				tmp = np.array([agent.model.experts['p'].nodes_position[i], agent.model.experts['p'].nodes_position[j]])
				plot(tmp[:,0], tmp[:,1], color = 'black', alpha  = 0.9)
	xlim(-1,1);ylim(-1,1)	
	title("Graph")

	subplot2grid((3,3),(0,2), colspan = 1)
	plot(distance[:,0], 'o-', label = "Distance to landmark", color = 'red')
	plot(distance[:,1], 'o-', label = "Distance to reward", color = 'green')

	subplot2grid((3,3),(1,1), colspan = 2)
	[plot(pgates[:,i], 'o-', label = name[agent.model.k_ex[i]]) for i in xrange(agent.model.n_ex)]
	legend()
	title("Gating value")


	show()
	
	


learn()
