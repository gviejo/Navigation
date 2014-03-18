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
from Expert import Taxon, Planning



tax = Taxon()

tax.computeLandmarkActivity(np.pi, 0.5)

plan = Planning()
position = np.random.uniform(-1,1,2)
def generatePosition(position):
	position =+ np.random.normal(0,0.3, 2)
	position = np.tanh(position)
	return position


tmp = []
for i in xrange(10):
	position = generatePosition(position)		
	plan.computeNextAction(position)
	tmp.append(position)

print plan.edges

tmp = np.array(tmp)
figure()
subplot(1,2,1)
plot(tmp[:,0], tmp[:,1]);xlim(-1,1);ylim(-1,1)

subplot(1,2,2)
scatter(plan.pc_position[:,0], plan.pc_position[:,1], marker = '+', s = 150, c = plan.pc, cmap = cm.coolwarm)
for i in plan.nodes.keys():
	a = plan.pc_position[plan.pc_nodes[i].keys()]
	plot(a[:,0], a[:,1], 'o', alpha = 0.8, label = str(i))	
xlim(-1,1);ylim(-1,1);legend()
show()

