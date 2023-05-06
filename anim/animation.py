import cairo
from pathlib import Path
import os
from tqdm import tqdm
from .utils import *
from .var import *
from .actor import *
from .transition import *

__all__ = ['Timeline', 'Animation']

class Timeline:
	def __init__(self, transitions: list[Transition]) -> None:
		# sort transitions in order of appearance
		self.transitions = list(sorted(transitions, key=lambda trans: trans.start))

	def transitions_at(self, t: float) -> list[Transition]:
		"""Return a list of transitions that are currently playing"""
		# TODO : if this is too slow with many transitions,
		# it could be possible to track which transitions have already finished
		return [ trans for trans in self.transitions if trans.start <= t <= trans.end ]

class Animation:
	def __init__(self, actors: list[Actor] = [], transitions: list[Transition] = [], framerate: int = 60, width: int = 500, height: int = 500) -> None:
		self.actors = actors
		self.framerate = framerate
		self.width = width
		self.height = height
		self.timeline = Timeline(transitions)

	def render_current(self, outpath: Path = Path('./frame.png')):
		"""Renders the current scene to the given path"""
		surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
		ctx = cairo.Context(surface)

		for actor in self.actors:
			actor.render(ctx)

		with open(outpath, 'wb') as file:
			surface.write_to_png(file)

	def render(self, start: float, end: float, outdir: Path = Path('frames')) -> list[Path]:
		nframes = self.framerate*(end-start)
		prev_trans: set[Transition] = set()
		framepaths: list[Path] = []

		os.makedirs(outdir, exist_ok=True)  # ensure output directory exists

		# for n in range(nframes+1):
		for n in tqdm(range(nframes+1), total=nframes+1, desc='frame'):
			t = linterp(n, 0, nframes, start, end)

			# update the current transitions
			curr_trans = set(self.timeline.transitions_at(t))
			for trans in curr_trans:
				trans.update(t)
			# make sure the animations properly finish
			# (we want the properties to animate until the end, even if it lies between two frames)
			finishing_trans = prev_trans - curr_trans
			for trans in finishing_trans:
				trans.update(t)
			prev_trans = curr_trans

			framepath = outdir / f'frame_{n:04d}.png'
			self.render_current(framepath)
			framepaths.append(framepath)
			
			# print('--')
			# print(f'{curr_trans=}', f'{finishing_trans=}')
			# print(self)
			# print(f'rendered frame #{n:04d} at {t=}')

		return framepaths

	def __str__(self) -> str:
		return f'''Scene(actors={list(map(str, self.actors))}, framerate={self.framerate})'''