#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sferes_call.py

"""

import sys
from optparse import OptionParser
sys.path.append("../src")
from Sferes import EA_pearce
from setup import Agent, Pearce
import numpy as np

# TO REMOVE
from matplotlib import *
from pylab import *
####

import warnings
warnings.simplefilter('error')

parser = OptionParser()
parser.add_option("--nlc", action = "store", type = 'float')
parser.add_option("--sigma_lc", action = "store", type = 'float')
parser.add_option("--nac", action = "store", type = 'float')
parser.add_option("--sigma", action = "store", type = 'float')
parser.add_option("--eta", action = "store", type = 'float')
parser.add_option("--lambda", action = "store", type = 'float')
parser.add_option("--gamma", action = "store", type = 'float')
parser.add_option("--epsilon", action = "store", type = 'float')
parser.add_option("--theta_pc", action = "store", type = 'float')
parser.add_option("--theta_node", action = "store", type = 'float')
parser.add_option("--alpha", action = "store", type = 'float')
parser.add_option("--npc", action = "store", type = 'float')
parser.add_option("--sigma_pc", action = "store", type = 'float')
(options, args) = parser.parse_args()

parameters = dict()

# FUCKING UGLY BUT NO FUCKING CHOICE
bounds = dict({'epsilon' : [0.0001, 1.0], 	# Dolle
				'gamma' : [0.0001, 1.0],  	# Dolle, Taxon
				'lambda': [0.0001, 1.0],  	# Dolle, Taxon
				'nlc' : [5, 1000],	      	# Dolle, Taxon			
				'sigma_lc' : [0.0001, 1.0],	# Taxon
				'sigma' : [0.0001, 1.0],   	# Taxon
				'nac' : [5,1000], 			# Taxon
				'eta' : [0.0001, 1.0], 		# Taxon
				'theta_pc': [0.0001, 1.0],   # Planning
				'theta_node': [0.0001, 1.0], # Planning
				'alpha' : [0.0001, 1.0],     # Planning
				'npc' : [5, 5000], 			# Planning
				'sigma_pc' : [0.0001, 1.0]  # Planning				
				})

# rescale the parameters
# parameters = vars(options)
# for p in parameters.iterkeys(): 	
# 	parameters[p] = bounds[p][0]+parameters[p]*(bounds[p][1]-bounds[p][0])

### TO REMOVE #########################################################
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
				 }	
#######################################################################

# Set the speed, should be evolved at some point
parameters['speed'] = 6.0

# Setup the agent
agent = Agent(('p', 't', 'e'), Pearce(), parameters, stats = False)
agent_lesion = Agent(('t', 'e'), Pearce(), parameters, stats = False)

# Load the data
data = np.load("../pearce/pearce.npy")

# Setup the optimization
opt = EA_pearce(data, agent, agent_lesion)

# Evaluate the agent
fit = opt.getFitness()
print fit

pos = map(np.array, agent.positions)
figure()
plot(agent.world.pool_circle[:,0], agent.world.pool_circle[:,1])
plot(agent.world.reward_circle[:,0], agent.world.reward_circle[:,1])
plot(agent.world.landmark_position[0], agent.world.landmark_position[1], '*', markersize = 20)
plot(agent.world.start_position[0], agent.world.start_position[1], 'o', markersize = 10)

for i in xrange(4):
	plot(pos[i][:,0], pos[i][:,1], 'o-', label = str(i+1))
legend()

figure()	
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


show()
sys.exit()
