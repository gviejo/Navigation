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
				'sigma_pc': 0.2,
				'epsilon': 0.01
				 }	


agent = Agent(Dolle(parameters), World(), parameters, stats = True)

t1 = time()
for i in xrange(10):
	print i
	if i%30:
		agent.stats = False
	else:
		agent.stats = True
	agent.start()		
	while not agent.world.reward_found and agent.n_steps[-1] < 1000:			
		agent.step()

t2 = time()

print "\n"
print t2-t1

position = map(np.array, agent.positions)


# direction = np.array(agent.directions)
# distance = np.array(agent.distances)
# lcs = np.array(agent.model.experts['t'].lcs).T
# lac = np.array(agent.model.experts['t'].lac).T
# ldelta = np.array(agent.model.experts['t'].ldelta).T
# actions = np.array(agent.actions)
# lvc = np.array(agent.model.experts['t'].lvc).T
# wall = np.array(agent.walls)
# trace = np.array(agent.model.experts['t'].ltrace)
# W = np.array(agent.model.experts['t'].lW)

figure(figsize = (17, 11))

ax1 = subplot2grid((2,3),(0,0), colspan = 2, rowspan = 2)
scatter(agent.landmark_position[0], agent.landmark_position[1], s = 100, c = 'red')
c = mpatches.Circle(agent.world.reward_position, agent.world.reward_size, fc = 'g')
ax1.add_patch(c)

[plot(position[i][:,0], position[i][:,1], '.-', alpha = 0.5) for i in xrange(len(position))]
xlim(-1,1);ylim(-1,1)

subplot2grid((2,3),(0,2))
plot(agent.n_steps)

# subplot2grid((3,3),(2,0), colspan = 1)
# imshow(W[-1], interpolation = 'nearest')
# title("Weight")


# subplot2grid((3,3),(0,1), colspan = 2)
# plot(distance[:,0], 'o-', label = "Distance to landmark", color = 'red')
# plot(distance[:,1], 'o-', label = "Distance to reward", color = 'green')
# legend()

# subplot2grid((3,3),(1,1), colspan = 2)
# #[plot(agent.model.experts['t'].lc_direction, lcs[:,i], 'o-', label = str(i)) for i in xrange(lcs.shape[1])]
# plot(agent.rewards)

# # xlim(0,len(agent.actions))
# #ylim(0, 2*np.pi)
# #plot(wall[:,1], 'o-', label = 'distance to wall')
# #[plot(agent.model.experts['t'].ac_direction, lvc[:,i], 'o-', label = str(i)) for i in xrange(lvc.shape[1])]
# legend()

# subplot2grid((3,3),(2,1), colspan = 2)
# #plot(actions[:,0], label = 'angle')
# #plot(actions[:,1], label = 'speed')
# #plot(ldelta)
# #[plot(agent.gates[k], 'o-', color = agent.colors[k], alpha = 0.9) for k in agent.gates.keys()]
# #line = [Line2D(range(1),range(1), marker = 'o', alpha =1.0, color = agent.colors[e], label = str(e)) for e in agent.colors.keys()]
# #legend(line, tuple(agent.colors.keys()))
# #hist(agent.winners)
# #plot(agent.speeds)
# #plot(lac)
# #plot(agent.model.experts['t'].ldirec, 'o-', label = 'direction')
# #plot(actions[:,0], 'o-', label = 'action direction')
# #[plot(agent.model.experts['t'].ac_direction, ldelta[:,i], 'o-', label = str(i)) for i in xrange(ldelta.shape[1])]
# #plot(wall[:,0], 'o-', label = 'wall angle')

legend()
# #ylim(-np.pi, np.pi)
show()

