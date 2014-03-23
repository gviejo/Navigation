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




def generatePosition(position):
	position += np.random.normal(0,0.3, 2)
	position = np.tanh(position)
	return position

parameters = { 'nlc': 100,		 		    # Number of landmarks cells
				'sigma_lc': 0.475,			# Normalized landmark width
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


agent = Agent(Dolle(), parameters)
agent.landmark_position = np.array([-0.5, 0.5])
agent.position = np.array([0.5, 0.0])
agent.direction = np.pi/4.
agent.update()
position = []
distance = []
direction = []
for i in xrange(9):	
	position.append(list(agent.position))	
	distance.append(agent.distance)
	direction.append([agent.agent_direction, agent.landmark_direction])
	agent.action = np.pi/4.
	agent.step()

	



position = np.array(position)
direction = np.array(direction)
figure()
subplot(3,2,1)
plot(position[:,0], position[:,1]);xlim(-1,1);ylim(-1,1)
scatter(agent.landmark_position[0], agent.landmark_position[1])

subplot(3,2,2)
scatter(agent.model.experts['p'].pc_position[:,0], agent.model.experts['p'].pc_position[:,1], marker = '+', s = 150, c = agent.model.experts['p'].pc, cmap = cm.coolwarm)
for i in agent.model.experts['p'].nodes.keys():
	a = agent.model.experts['p'].pc_position[agent.model.experts['p'].pc_nodes[i].keys()]
	plot(a[:,0], a[:,1], 'o', alpha = 0.8, label = str(i))	
xlim(-1,1);ylim(-1,1);legend()

subplot(3,1,2)
plot(np.array(distance), 'o-')
title("Distance to landmark")

subplot(3,1,3)
#plot(direction[:,0], 'o-', label = 'agent')
plot(direction[:,1], 'o-', label = 'mark')
legend()
title("Direction of the agent")

show()

