#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
sferes_call.py

"""

import sys
from optparse import OptionParser
sys.path.append("../src")
from Sferes import EA
from setup import Agent, World
import numpy as np

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
parser.add_option("--speed", action = "store", type = 'float')
parser.add_option("--sigma_vc", action = "store", type = 'float')
(options, args) = parser.parse_args()

parameters = dict()

# Setup the agent
model = Agent(('p', 't', 'e'), parameters)

# rescale the parameters
parameters = vars(options)
for p in parameters.iterkeys():
	if parameters[p]:
		parameters[p] = model.bounds[p][0]+parameters[p]*(model.bounds[p][1]-model.bounds[p][0])
model.setAllParameters(parameters)

# Load the data
data = np.load("../pearce/pearce.npy")

# Setup the optimization
opt = EA(data, model)

# Evaluate the agent
fit = opt.getFitness()
print fit