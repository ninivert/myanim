"""Basic demo"""

from pathlib import Path
import math

import sys
sys.path.append('..')
from anim import *

disc = Disc(
	xy=Var(Vec([300, 200])), r=Var(50),
	rgb=Var(Vec((0/255, 181/255, 204/255)))
)
rect = Rect(
	xy=Var(Vec([100, 380])), w=Var(40), h=Var(60),
	rgb=Var(Vec((249/255, 180/255, 45/255)))
)
disc2 = Disc(
	xy=Var(Vec([30, 250])), r=Var(10),
	rgb=Var(Vec((50/255, 205/255, 50/255)))
)
rect2 = Rect(
	xy=Var(Vec([120, 350])), w=Var(30), h=Var(10),
	rgb=Var(Vec((255/255, 20/255, 147/255)))
)
rect3 = Rect(
	xy=Var(Vec([460, 140])), w=Var(50), h=Var(50),
	rgb=Var(Vec((0.1, 0.9, 0.2)))
)

w, h = 500, 500
background = Rect(Var(Vec([0.0, 0.0])), Var(w), Var(h), rgb=Var((0.05, 0.05, 0.05)))

grow_radius = Transition(var=disc.r,
	initial=disc.r.value, final=disc.r.value+100,
	start=1.05, length=1,
	tween=ease_in_out_quad
)
translate_x = Transition(var=rect.xy,
	initial=rect.xy.value, final=Vec([rect.xy.value[0]+300, rect.xy.value[1]]),
	start=0.5, length=2,
	tween=ease_out_back
)
oscillate_y = Transition(var=disc2.xy,
	initial=Vec([disc2.xy.value[0], disc2.xy.value[1]-100]), final=Vec([disc2.xy.value[0], disc2.xy.value[1]+100]),
	start=1, length=4,
	tween=lambda s: (math.sin(s * 3 *2*math.pi)+1)/2
)
translate_line = Transition(var=rect2.xy,
	initial=rect2.xy.value, final=Vec([rect2.xy.value[0]+250, rect2.xy.value[1]-300]),
	start=1.4, length=2,
	tween=ease_in_out_trig,
)
green_to_pink = Transition(var=disc2.rgb,
	initial=disc2.rgb.value, final=Vec((255/255, 20/255, 147/255)),
	start=1, length=4
)
translate_bquad = Transition(var=rect3.xy,
	initial=rect3.xy.value, final=Vec((100, 200)),
	start=1.8, length=1,
	interp=QuadraticBezier(Vec([230, -50]))
)
translate_bcub = Transition(var=rect3.xy,
	initial=translate_bquad.final, final=Vec((470, 430)),
	start=translate_bquad.end, length=1.4,
	interp=CubicBezier(Vec((20, 600)), Vec((570, 25)))
)

anim = Animation(
	actors=[background, disc, rect, disc2, rect2, rect3],
	transitions=[grow_radius, translate_x, oscillate_y, translate_line, green_to_pink, translate_bquad, translate_bcub],
	framerate=60,
	width=w, height=h
)
anim.render(0, 5, Path('../frames'))

stitch(
	framerate=anim.framerate,
	framedir=Path('../frames'),
	outpath=Path('../renders') / 'example1.mp4',
	overwrite=True
)