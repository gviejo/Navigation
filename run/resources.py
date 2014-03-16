import pyglet as gt

gt.resource.path = ["../resources"]
gt.resource.reindex()

player = gt.resource.image("player.png")
enemy = gt.resource.image("player.png")