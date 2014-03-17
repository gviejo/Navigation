#!/usr/bin/python
# encoding: utf-8
"""
Models.py    

Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""

from Expert import Taxon, Planning, Exploration

class Dolle(object):

	def __init__(self):
		self.taxon = Taxon()
		self.planning = Planning()
		self.exploration = Exploration()

		


