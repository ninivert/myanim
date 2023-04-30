from typing import Callable
from .var import *
from .utils import *
from .interpolator import *
from .easings import linear

__all__ = ['Transition']

class Transition:
	"""Animate a property smoothly"""

	__slots__ = 'var', 'initial', 'final', 'start', 'length', 'end', 'tween', 'interp'
	
	def __init__(self, var: Var[T], initial: T, final: T, start: float, length: float, tween: Callable[[float], float] = linear, interp: Interpolator = InterpLinear()):
		self.var = var
		self.initial = initial
		self.final = final
		self.start = start
		self.length = length
		self.end = self.start + self.length
		self.tween = tween  # the way that t changes from 0 to 1
		self.interp = interp  # the path the value takes

	def update(self, t: float) -> None:
		s = linterp(t, self.start, self.end)  # normalize time
		s = clamp(s, 0, 1)  # clamp (t < 0 if even is a little early, t > 0 if event is late)
		s = self.tween(s)  # apply tweening function
		self.var.value = self.interp(s, self.initial, self.final)  # interpolate between initial and final position

	def __str__(self) -> str:
		return f'Transition(var @ 0x{id(self.var):x}, from {self.initial} (t={self.start}) to {self.final} (t={self.end}) with tweening {self.tween.__name__})'
