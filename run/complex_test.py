#!/usr/bin/python
# encoding: utf-8
"""
dolle.py

    

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
				'theta_pc': 0.3,			# Activity threshold for place cells node linking
				'theta_node': 0.3,			# Activity threshold for node creation
				'alpha': 0.7, 				# Decay factor of the goal value
				'npc': 1681,				# Number of simulated Place cells
				'sigma_pc': 0.2,
				'epsilon': 0.01
				 }	

agent = Agent(Dolle(('p'), parameters), Maze(), parameters, stats = True)
plan = agent.model.experts['p']

def test():
	t1 = time()
	agent.start()
	while not agent.world.reward_found and agent.n_steps[-1] < 5000:		
		agent.step()
		#print plan.path
		#print plan.action, plan.speed
	t2 = time()

	print t2-t1
	
	PLOT()

def PLOT():
	position = map(np.array, agent.positions)
	#direction = np.array(agent.directions)
	distance = np.array(agent.distances)
	# actions = np.array(agent.actions)
	# wall = np.array(agent.walls)
	figure(figsize = (17, 11))

	ax1 = subplot2grid((2,3),(0,0), colspan = 1, rowspan = 1)
	scatter(agent.landmark_position[0], agent.landmark_position[1], s = 100, c = 'red')
	c = mpatches.Circle(agent.world.reward_position, agent.world.reward_size, fc = 'g')
	ax1.add_patch(c)

	[plot(agent.world.walls[i][:,0], agent.world.walls[i][:,1], color = 'black', linewidth = 5) for i in agent.world.walls.iterkeys()]

	#[ax1.plot(position[i][:,0], position[i][:,1], '.-', alpha = 0.5, label = str(i)) for i in xrange(len(position))]
	ax1.plot(position[-1][:,0], position[-1][:,1], '.-')
	ax1.set_xlim(-1,1);ax1.set_ylim(-1,1);legend()
	
	subplot2grid((2,3),(1,0), colspan = 1)
	scatter(agent.model.experts['p'].pc_position[:,0], agent.model.experts['p'].pc_position[:,1], marker = '+', s = 150, c = agent.model.experts['p'].pc, cmap = cm.coolwarm)
	for i in agent.model.experts['p'].nodes.keys():
		a = agent.model.experts['p'].pc_position[agent.model.experts['p'].pc_nodes[i].keys()]
		plot(a[:,0], a[:,1], 'o', alpha = 0.8)	
		plot(agent.model.experts['p'].nodes_position[i][0], agent.model.experts['p'].nodes_position[i][1], 'o', label = str(i))
		for j in agent.model.experts['p'].edges[i]:
			if j:
				tmp = np.array([agent.model.experts['p'].nodes_position[i], agent.model.experts['p'].nodes_position[j]])
				plot(tmp[:,0], tmp[:,1], color = 'black', alpha  = 0.9)
	xlim(-1,1);ylim(-1,1)	
	#legend()

	subplot2grid((2,3),(0,1), colspan = 2)
	plot(distance[:,0], 'o-', label = "Distance to landmark", color = 'red')
	plot(distance[:,1], 'o-', label = "Distance to reward", color = 'green')
	
	subplot2grid((2,3),(1,1), colspan = 2)
	plot(agent.n_steps, 'o-')

	legend()
	# #ylim(-np.pi, np.pi)
	show()


test()
