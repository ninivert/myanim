from pathlib import Path
import math

import sys
sys.path.append('..')
from anim import *

disk = Disk(
	x=Var(300), y=Var(200), r=Var(50),
	rgba=(0/255, 181/255, 204/255, 1)
)
rect = Rect(
	x=Var(100), y=Var(380), w=Var(40), h=Var(60),
	rgba=(249/255, 180/255, 45/255, 1)
)
disk2 = Disk(
	x=Var(30), y=Var(250), r=Var(10),
	rgba=(50/255, 205/255, 50/255, 1)
)
rect2 = Rect(
	x=Var(120), y=Var(350), w=Var(30), h=Var(10),
	rgba=(255/255, 20/255, 147/255, 1)
)

w, h = 500, 500
background = Rect(Var(0), Var(0), Var(w), Var(h), rgba=(0.05, 0.05, 0.05, 1))

grow_radius = Transition(var=disk.r,
	initial=disk.r.value, final=disk.r.value+100,
	start=1.05, length=1,
	tween=ease_in_out_quad
)
translate_x = Transition(var=rect.x,
	initial=rect.x.value, final=rect.x.value+300,
	start=0.5, length=2,
	tween=ease_out_back
)
oscillate_y = Transition(var=disk2.y,
	initial=disk2.y.value-100, final=disk2.y.value+100,
	start=1, length=4,
	tween=lambda t: (math.sin(t * 3 *2*math.pi)+1)/2
)
# TODO : utility function to move along a line ?
move_line = [
	Transition(var=rect2.x,
		initial=rect2.x.value, final=rect2.x.value+250,
		start=1.4, length=2,
		tween=ease_in_out_trig,
	),
	Transition(var=rect2.y,
		initial=rect2.y.value, final=rect2.y.value-300,
		start=1.4, length=2,
		tween=ease_in_out_trig,
	)
]
# TODO : utility function to move along a bezier
# i.e. we projet the bezier curve on the x, y

anim = Animation(
	actors=[background, disk, rect, disk2, rect2],
	transitions=[grow_radius, translate_x, oscillate_y] + move_line,
	framerate=60,
	width=w, height=h
)
anim.render(0, 5, Path('../frames'))

stitch(
	framerate=anim.framerate,
	framedir=Path('../frames'),
	outpath=Path('../renders') / 'out.mp4',
	overwrite=True
)