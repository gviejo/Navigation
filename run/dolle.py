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


parameters = { 'nlc': 100,		 		    # Number of landmarks cells
				'sigma_lc': 0.1,			# Normalized landmark width
				'sigma':0.392,				# Number of action cells
				'nac': 36,					# Standard deviation of the generalization profile
				'eta': 0.001,				# Learning rate
				'lambda': 0.76,				# Eligibility trace decay factor
				'gamma' : 0.8,
				'theta_pc': 0.2,			# Activity threshold for place cells node linking
				'theta_node': 0.3,			# Activity threshold for node creation
				'alpha': 0.7, 				# Decay factor of the goal value
				'npc': 1681,				# Number of simulated Place cells
				'sigma_pc': 0.2,
				'epsilon': 0.01 }	


agent = Agent(Dolle(parameters), World(), parameters, stats = True)


t1 = time()
for i in xrange(100):
	agent.step()	
t2 = time()

print "\n"
print t2-t1



position = np.array(agent.positions)
direction = np.array(agent.directions)
distance = np.array(agent.distances)
lcs = np.array(agent.model.experts['t'].lcs).T
lac = np.array(agent.model.experts['t'].lac).T

figure(figsize = (17, 11))

subplot2grid((3,3),(0,0), colspan = 1)
plot(position[:,0], position[:,1], '.-');xlim(-1,1);ylim(-1,1)
scatter(agent.landmark_position[0], agent.landmark_position[1], s = 100, c = 'red')

subplot2grid((3,3),(1,0), colspan = 1)
#scatter(agent.model.experts['p'].pc_position[:,0], agent.model.experts['p'].pc_position[:,1], marker = '+', s = 150, c = agent.model.experts['p'].pc, cmap = cm.coolwarm)
#for i in agent.model.experts['p'].nodes.keys():#
#	a = agent.model.experts['p'].pc_position[agent.model.experts['p'].pc_nodes[i].keys()]
#	plot(a[:,0], a[:,1], 'o', alpha = 0.8)	
#xlim(-1,1);ylim(-1,1)

subplot2grid((3,3),(0,1), colspan = 2)
plot(distance, 'o-', label = "Distance to landmark")
legend()

subplot2grid((3,3),(1,1), colspan = 2)
[plot(agent.gates[k], 'o-', color = agent.colors[k], alpha = 0.9) for k in agent.gates.keys()]
line = [Line2D(range(1),range(1), marker = 'o', alpha =1.0, color = agent.colors[e], label = str(e)) for e in agent.colors.keys()]
legend(line, tuple(agent.colors.keys()))
xlim(0,len(agent.actions))
#ylim(0, 2*np.pi)

subplot2grid((3,3),(2,1), colspan = 2)
#plot(agent.actions)
#[plot(agent.model.experts['t'].lc_direction, lcs[:,i]) for i in xrange(lcs.shape[1])]
plot(lac)
#plot(agent.model.experts['t'].ldirec, 'o-', label = 'direction')
#plot(direction)
legend()
#ylim(0, parameters['nlc'])
show()

