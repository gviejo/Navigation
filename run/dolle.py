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
position = np.array([0.2,0.4])
plan.computePlaceCellActivity(position)
plan.computeGraphNodeActivity()
# plan.computePlaceCellActivity([0.0, 0.0])
# plan.computeGraphNodeActivity()



plot(position[0], position[1], 'o', c = 'black')
scatter(plan.pc_position[:,0], plan.pc_position[:,1], marker = '+', s = 150, c = plan.pc, cmap = cm.coolwarm)
a = plan.pc_position[plan.pc_nodes[1].keys()]
#b = plan.pc_position[plan.pc_nodes[2].keys()]
plot(a[:,0], a[:,1], 'o', alpha = 0.6)
# plot(b[:,0], b[:,1], 'o', alpha = 0.6)
show()

