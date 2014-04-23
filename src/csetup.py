#!/usr/bin/python
# encoding: utf-8
"""
configuration file for cython optimization


Copyright (c) 2014 Guillaume Viejo. All rights reserved.
"""

from distutils.core import setup
from Cython.Distutils import build_ext
from distutils.extension import Extension

ext_modules = [Extension("cExpert", ["Expert.pyx"]), 
			   Extension("cModels", ["Models.pyx"]),
			   Extension("cAgents", ["setup.pyx"])]

setup(
	name = 'test',
	cmdclass = {'build_ext': build_ext},
	ext_modules = ext_modules,
	)
