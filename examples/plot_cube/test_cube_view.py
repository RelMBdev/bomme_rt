# axes.py
from pymol.cgo import *
from pymol import cmd

#cmd.set('connect_cutoff', 1.7)
cmd.load('geom.xyz')
#cmd.bond('id 1', 'id 2')
#cmd.bond('id 1', 'id 3')
#cmd.bond('id 1', 'id 4')

#cmd.set_bond ('stick_color', 'white', 'all', 'all')
#cmd.set_bond ('stick_radius', -0.14, 'all', 'all')
#cmd.set ('stick_ball', 1)
#cmd.set ('stick_ball_ratio', -1)
#cmd.set ('stick_ball_color', 'atomic')
#cmd.show ('sticks', 'all')

#cmd.color('black', 'id 1')
#cmd.color('gray', '(name Au*)')

cmd.set_bond ('stick_radius', 0.04, 'all', 'all')
cmd.set ('sphere_scale', 0.15, 'all')
cmd.show ('sticks', 'all')
cmd.show ('spheres', 'all')
cmd.set ('stick_ball_color', 'atomic')


cmd.color('gray20', '(name C*)')

cmd.set ('transparency_mode', 1)
 
#w = 0.01 # cylinder width 
#l = 0.5 # cylinder length
#h = 0.15 # cone hight
#d = w * 1.618 # cone base diameter
w = 0.03 # cylinder width 
l = 1.65/1.5 # cylinder length
h = 0.2 # cone hight
d = w * 1.618 # cone base diameter
 
obj = [CYLINDER, 0.0, 0.0, 0.0,   l, 0.0, 0.0, w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       CYLINDER, 0.0, 0.0, 0.0, 0.0,   l, 0.0, w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       CYLINDER, 0.0, 0.0, 0.0, 0.0, 0.0,   1*l, w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
       CONE,   l, 0.0, 0.0, h+l, 0.0, 0.0, d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 
       CONE, 0.0,   l, 0.0, 0.0, h+l, 0.0, d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 
       CONE, 0.0, 0.0,   1*l, 0.0, 0.0, h+1*l, d, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]

#cmd.load_cgo(obj, 'axes')

cmd.bg_color('white')


#cmd.set_bond ('stick_color', 'white', 'clauc2h2', 'clauc2h2')
#cmd.set_bond ('stick_radius', -0.14, 'clauc2h2', 'clauc2h2')
#cmd.set ('stick_ball', 1)
#cmd.set ('stick_ball_ratio', -0.5)
#cmd.set ('stick_ball_color', 'atomic')
#cmd.show ('sticks', 'clauc2h2')
##cmd.set ('sphere_scale', 0.25, 'clauc2h2')
##cmd.show ('spheres', 'clauc2h2')

cmd.load('./td.0002000/diff.cube')
cmd.isosurface('sp', 'this', _iso)
cmd.color('_colorp', 'sp')
cmd.isosurface('sm', 'this', -_iso)
cmd.color('_colorm', 'sm')

#cmd.load('025_density.dx')
#cmd.isosurface('sp', '025_density', 2.3e-5)
##cmd.color('lightblue', 'sp')
#cmd.color('blue', 'sp')
#cmd.isosurface('sm', '025_density', -2.3e-5)
#cmd.color('red', 'sm')

#cmd.set ('label_font_id', 16)
#cmd.set ('label_size', 24)
#cmd.pseudoatom('xatom', pos=[1,0,0], label="x")
#cmd.pseudoatom('yatom', pos=[0,1,0], label="y")
#cmd.pseudoatom('zatom', pos=[0,0,2], label="z")
cmd.set ('label_font_id', 16)
cmd.set ('label_size', 20)

#cmd.pseudoatom('xatom', pos=[1.5,0,0], label="x")
#cmd.pseudoatom('yatom', pos=[0,1.5,0], label="y")
#cmd.pseudoatom('zatom', pos=[0,0,1.5], label="z")

cmd.set('ray_opaque_background', 'on')
cmd.set('transparency', 0.40)

#cmd.set_view ('\
#     0.031727057,    0.785191953,    0.618473470,\
#    -0.490347385,    0.551440895,   -0.674938798,\
#    -0.870980203,   -0.281856358,    0.402507514,\
#     0.000011624,    0.000000090,  -15.328842163,\
#     1.082860231,   -0.718130112,    0.259954333,\
#     4.162666798,   26.531684875,  -20.000000000' )
cmd.set_view ('\
     0.031727057,    0.785191953,    0.618473470,\
    -0.490347385,    0.551440895,   -0.674938798,\
    -0.870980203,   -0.281856358,    0.402507514,\
     0.000038814,    0.000021812,  -12.181253433,\
     0.391719162,   -0.228159577,   -0.641706526,\
     0.997265995,   23.366279602,  -20.000000000')
cmd.pseudoatom('latom', pos=[3.5,-2.5,0], label=" ")

cmd.ray(600,600)

