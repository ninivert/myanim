import os
os.chdir(os.path.dirname(__file__))
import sys
sys.path.append('../..')

from pathlib import Path
from cairo import Context
import cairo
import networkx as nx, numpy as np
from nptyping import NDArray, Shape, UInt

from anim import *

from helpers import *
from bp import BP, Params

## Custom Actors

class BarPlot(Actor):
	__slots__ = 'w', 'h', 'ydata', 'ymin', 'ymax', 'bar_width', 'cycler'

	def __init__(self, xy: Var[Vec[float]], w: Var[float], h: Var[float], ydata: Var[Vec[float]], ymin: Var[float], ymax: Var[float], bar_width: float = 0.7, cycler: Var[ColorCycler] = Var(ColorCyclerMPL), alpha: Var[float] = Var(1.0)) -> None:
		assert ymin.value <= ymax.value
		super().__init__(xy, alpha=alpha)
		self.w = w
		self.h = h
		self.ydata = ydata
		self.ymin = ymin
		self.ymax = ymax
		self.bar_width = bar_width
		self.cycler = cycler

	def render(self, ctx: Context) -> None:
		if len(self.ydata.value) == 0:
			return

		xspan = len(self.ydata.value)
		yspan = self.ymax.value - self.ymin.value
		ctx.save()
		ctx.translate(*self.xy.value)
		ctx.scale(self.w.value/xspan, self.h.value/yspan)  # normalize to data coordinates
		# set the y axis pointing up instead of down
		ctx.transform(cairo.Matrix(
			1.0, 0.0,
			0.0, -1.0,
			0.0, 1.0
		))

		s = 1/(xspan-1)  # space allowed for each bar
		w = self.bar_width*s  # bar width

		for i, y in enumerate(self.ydata.value):
			h = linterp(y, self.ymin.value, self.ymax.value)
			ctx.set_source_rgba(*self.cycler.value[i], self.alpha.value)
			ctx.rectangle((i+0.5)*s-w/2, 0, w, h) 
			ctx.fill()

		ctx.restore()

class Graph(Actor):
	__slots__ = 'w', 'h', 'xynodes', 'edgelist', 'edgewidth', 'edgecolor'

	def __init__(self, xy: Var[Vec[float]], w: Var[float], h: Var[float], xynodes: Var[np.ndarray], edgelist: list[tuple[int, int]], edgewidth: Var[float], edgecolor: Var[RGB], alpha: Var[float] = Var(1.0)) -> None:
		# xynodes are given in normalized coordinates in [0,1]²
		super().__init__(xy, edgecolor, alpha)
		self.w = w
		self.h = h
		self.xynodes = xynodes
		self.edgelist = edgelist
		self.edgewidth = edgewidth
		self.edgecolor = edgecolor

	def render(self, ctx: Context) -> None:
		ctx.save()
		ctx.scale(self.w.value, self.h.value)  # normalized [0,1]² coordinates

		for i, j in self.edgelist:
			ctx.set_line_width(self.edgewidth.value)
			ctx.set_source_rgba(*self.edgecolor.value, self.alpha.value)
			ctx.move_to(*self.xynodes.value[i])
			ctx.line_to(*self.xynodes.value[j])
			ctx.stroke()

		ctx.restore()

## Setup animation

W, H = 1280, 720

bp = BP(Params.new_bimodal_params(q=2, c_out=1, c_in=10, N=32, init='paramagnetic', seed=1, damping=0.4))
bp.params.N

g = nx.from_edgelist(bp.ord_edges)
pos = nx.fruchterman_reingold_layout(g, seed=0)
xynodes = np.vstack([v for k, v in sorted(pos.items())])
# rescale the layout so it fits in a [0, 1] x [0, 1] box
from helpers import linterp
xynodes[:, 0] = linterp(xynodes[:, 0], xynodes[:, 0].min(), xynodes[:, 0].max(), 0.05, 0.95)
xynodes[:, 1] = linterp(xynodes[:, 1], xynodes[:, 1].min(), xynodes[:, 1].max(), 0.05, 0.95)
xynodes_screen = xynodes * np.array([W, H])

BARPLOT_W, BARPLOT_H = 60, 80
graph = Graph(xy=Var(Vec((0,0))), w=Var(W), h=Var(H), xynodes=Var(xynodes.copy()), edgelist=bp.ord_edges.copy(), edgewidth=Var(0.002), edgecolor=Var(RGB((0.5, 0.5, 0.5))))
barplots_mu = [ 
	BarPlot(xy=Var(Vec(xy.tolist()) - Vec((BARPLOT_W/2, BARPLOT_H/2))), w=Var(BARPLOT_W), h=Var(BARPLOT_H), ydata=Var(Vec(mu.tolist())), ymin=Var(0.0), ymax=Var(1.0))
	for mu, xy in zip(bp.mu, xynodes_screen)
]
barplots_chi = [
	BarPlot(xy=Var(Vec(xynodes_screen[i].tolist()) - Vec((BARPLOT_W/2, BARPLOT_H/2))), w=Var(BARPLOT_W), h=Var(BARPLOT_H), ydata=Var(Vec(chi.tolist())), ymin=Var(0.0), ymax=Var(1.0), cycler=Var(ColorCyclerMPL), alpha=Var(0.5))
	for chi, (i, _) in zip(bp.chi, bp.full_edges)
]

text = Text(xy=Var(Vec((20, 40))), scalar=Var(0), formatter=lambda n: f'n={int(n)}', font_size=Var(36), rgb=Var(RGB((1,1,1))))

background = Rect(Var(Vec([0.0, 0.0])), Var(W), Var(H), rgb=Var((0.05, 0.05, 0.05)))

## Run the BP algorithm

nsteps = 10
transitions_mu = []
transitions_chi = []
transitions_text = []
t = 0
it = iter(bp)

for n in range(nsteps):
	mus_prev = bp.mu.copy()
	chis_prev = bp.chi.copy()
	next(bp)
	# message passing
	# TODO : maybe the message should be weighed by connectivity ? also how to show messages from non-connected ?
	for k, ((i, j), chi_prev, chi) in enumerate(zip(bp.full_edges, chis_prev, bp.chi)):
		# update the ydata
		transitions_chi.append(Transition(
			var=barplots_chi[k].ydata,
			initial=Vec(chi_prev.tolist()),
			final=Vec(chi.tolist()),
			start=t, length=0.01,
			tween=one  # the transition happens instantly
		))
		# transition opacity
		transitions_chi.append(Transition(
			var=barplots_chi[k].alpha,
			initial=0.0, final=0.0,
			start=t, length=2.2,
			tween=linear,
			interp=QuadraticBezier(1.0)
		))
		# transition the positions
		transitions_chi.append(Transition(
			var=barplots_chi[k].xy,
			initial=Vec(xynodes_screen[i].tolist()) - Vec((BARPLOT_W/2, BARPLOT_H/2)),
			final=Vec(xynodes_screen[j].tolist()) - Vec((BARPLOT_W/2, BARPLOT_H/2)),
			start=t, length=1.8,
			tween=ease_in_out_quad
		))
	# update marginals
	for k, (mu_prev, mu) in enumerate(zip(mus_prev, bp.mu)):
		transitions_mu.append(Transition(
			var=barplots_mu[k].ydata,
			initial=Vec(mu_prev.tolist()), final=Vec(mu.tolist()),
			start=t+2, length=0.8,
			tween=ease_in_out_quad
		))
	transitions_text.append(Transition(
		var=text.scalar,
		initial=n, final=n+1,
		start=t, length=0.01,
		tween=one,
	))
	t += 3

## Render

anim = Animation(
	actors=[background, graph, text, *barplots_chi, *barplots_mu],
	transitions=[*transitions_chi, *transitions_mu, *transitions_text],
	framerate=60,
	width=W, height=H
)

outdir = Path('../../frames/bp')
# clear previous run
outdir.touch(exist_ok=True)
for f in outdir.iterdir():
	f.unlink()

anim.render(0, t, outdir)

stitch(
	framerate=anim.framerate,
	framedir=outdir,
	outpath=Path('../../renders') / 'bp.mp4',
	overwrite=True
)