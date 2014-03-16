#!/usr/bin/python
# encoding: utf-8
"""
testSimulator.py


Copyright (c) 2013 Guillaume VIEJO. All rights reserved.
"""
from __future__ import division

from pyglet.window import key
import pyglet as gt
import resources
import numpy as np

def tex_coord(x, y, n=4):
    """ Return the bounding vertices of the texture square.

    """
    m = 1.0 / n
    dx = x * m
    dy = y * m
    return dx, dy, dx + m, dy, dx + m, dy + m, dx, dy + m



def tex_coords(top, bottom, side):
    """ Return a list of the texture squares for the top, bottom and side.

    """
    top = tex_coord(*top)
    bottom = tex_coord(*bottom)
    side = tex_coord(*side)
    result = []
    result.extend(top)
    result.extend(bottom)
    result.extend(side * 4)
    return result

def cube_vertices(x, y, z, n):
    """ Return the vertices of the cube at position x, y, z with size 2*n.

    """
    return [
        x-n,y+n,z-n, x-n,y+n,z+n, x+n,y+n,z+n, x+n,y+n,z-n,  # top
        x-n,y-n,z-n, x+n,y-n,z-n, x+n,y-n,z+n, x-n,y-n,z+n,  # bottom
        x-n,y-n,z-n, x-n,y-n,z+n, x-n,y+n,z+n, x-n,y+n,z-n,  # left
        x+n,y-n,z+n, x+n,y-n,z-n, x+n,y+n,z-n, x+n,y+n,z+n,  # right
        x-n,y-n,z+n, x+n,y-n,z+n, x+n,y+n,z+n, x-n,y+n,z+n,  # front
        x+n,y-n,z-n, x-n,y-n,z-n, x-n,y+n,z-n, x+n,y+n,z-n,  # back
    ]


GRASS = tex_coords((1, 0), (0, 1), (0, 0))
SAND = tex_coords((1, 1), (1, 1), (1, 1))
BRICK = tex_coords((2, 0), (2, 0), (2, 0))
STONE = tex_coords((2, 1), (2, 1), (2, 1))


class World(object):
	def __init__(self):
		self.batch = gt.graphics.Batch()
		self.TEXTURE_PATH = '../texture.png'
		self.image = gt.image.load(self.TEXTURE_PATH)		
		self.group  = gt.graphics.TextureGroup(self.image.get_texture())
		self.texture = self.image.get_texture()
		self.world = {}
		self.shown = {}
		self._shown = {}
		self.sectors = {}		
		self._initialize()

	def _initialize(self):
		n = 80
		s = 1
		y = 0
		for x in xrange(-n, n+1, s):
			for z in xrange(-n,n+1,s):
				self.add_block((x,y-2,z),STONE, immediate=False)

	def add_block(self, position, texture, immediate=True):
		self.show_block(position, texture)

	def show_block(self, position, texture):
		x, y, z = position
		vertex_data = cube_vertices(x, y, z, 2.0)
		texture_data = list(texture)
		shown = self.batch.add(24, gt.graphics.GL_QUADS, self.group, 
			('v3f/static', vertex_data),
			('t2f/static', texture_data))



class Window(gt.window.Window):
	def __init__(self, *args, **kwargs):
		super(Window, self).__init__(*args, **kwargs)		
		self.width, self.height = self.get_size()
		self.rotation = (0,0)
		self.position = (-1,0,0)
		self.sector = (0,0,0)
		self.SECTOR_SIZE = 16
		self.world = World()
		gt.clock.schedule_interval(self.update, 0.5)
		self.FLYING_SPEED = 15
		self.GRAVITY = 15
		self.TERMINAL_VELOCITY = 50		
		self.PLAYER_HEIGHT = 2

	def update(self, dt):
		sector = self.sectorize(self.position)
		if sector != self.sector
			self.model.change_sectors(self.sector, sector)
			self.sector = sector
		m = 8
		dt = min(dt, 0.2)
		for _ in xrange(m):
			self._update(dt/m)

	def _update(self, dt):
		speed = self.FLYING_SPEED
		d = dt*speed
		dx,dy,dz = self.get_motion_vector()
		dx

	def on_draw(self):
		self.clear()
		self.set_3d()
		gt.gl.glColor3d(1,1,1)
		self.world.batch.draw()


	def set_3d(self):
		gt.gl.glEnable(gt.graphics.GL_DEPTH_TEST)
		gt.gl.glViewport(0, 0, self.width, self.height)
		gt.gl.glMatrixMode(gt.graphics.GL_PROJECTION)
		gt.gl.glLoadIdentity()
		gt.gl.gluPerspective(65.0, self.width/float(self.height), 0.1, 60.0)
		gt.gl.glMatrixMode(gt.graphics.GL_MODELVIEW)
		gt.gl.glLoadIdentity()
		x, y = self.rotation
		gt.gl.glRotatef(x, 0, 1, 0)
		gt.gl.glRotatef(-y, np.cos(np.radians(x)), 0, np.sin(np.radians(x)))
		x, y, z = self.position
		gt.gl.glTranslatef(-x, -y, -z)

	def normalize(self, position):
		x,y,z = position
		z,y,z = (int(round(x)), int(round(y)), int(round(z)))
		return (x,y,z)

	def sectorize(self, position):
		x,y,z = self.normalize(position)
		x,y,z = x/self.SECTOR_SIZE, y/self.SECTOR_SIZE, z/self.SECTOR_SIZE
		return (x, 0, z)


def setup():
	gt.gl.glClearColor(0.5, 0.69, 1.0, 1.0)
	gt.gl.glEnable(gt.graphics.GL_CULL_FACE)
	gt.gl.glTexParameteri(gt.graphics.GL_TEXTURE_2D, gt.graphics.GL_TEXTURE_MIN_FILTER,gt.graphics.GL_NEAREST)
	gt.gl.glTexParameteri(gt.graphics.GL_TEXTURE_2D, gt.graphics.GL_TEXTURE_MAG_FILTER,gt.graphics.GL_NEAREST)

def main():
	window = Window(width=800, height=600, caption='Pyglet', resizable=True)	
	window.set_exclusive_mouse(False)
	setup()
	gt.app.run()		


main()

