import math
from cairo import Context
from .var import Var
from .vec import Vec

# TODO : a better color class, which implements from_hsl, default alpha=1, etc
RGBA = Vec[float]  # (r, g, b, a)

__all__ = ['Actor', 'RGBA', 'Disc', 'Rect']

class Actor:
	__slots__ = 'xy', 'rgba'

	def __init__(self, xy: Var[Vec[float]], rgba: Var[RGBA] = (0, 0, 0, 1)) -> None:
		self.xy = xy
		self.rgba = rgba

	def render(self, ctx: Context) -> None:
		pass


class Disc(Actor):
	__slots__ = 'r'

	def __init__(self, xy: Var[Vec[float]], r: Var[float], rgba: Var[RGBA] = (0, 0, 0, 1)) -> None:
		super().__init__(xy, rgba)
		self.r = r

	def __str__(self) -> str:
		return f'Disk(xy={self.xy}, r={self.r}, rgba={self.rgba})'

	def render(self, ctx: Context) -> None:
		ctx.set_source_rgba(*self.rgba.value)
		ctx.arc(self.xy.value[0], self.xy.value[1], self.r.value, 0, 2*math.pi)
		ctx.fill()


class Rect(Actor):
	__slots__ = 'w', 'h'

	def __init__(self, xy: Var[Vec[float]], w: Var[float], h: Var[float], rgba: Var[RGBA] = (0, 0, 0, 1)) -> None:
		super().__init__(xy, rgba)
		self.w = w
		self.h = h

	def __str__(self) -> str:
		return f'Rect(xy={self.xy}, w={self.w}, h={self.h}, rgba={self.rgba})'

	def render(self, ctx: Context) -> None:
		ctx.set_source_rgba(*self.rgba.value)
		ctx.rectangle(self.xy.value[0], self.xy.value[1], self.w.value, self.h.value)
		ctx.fill()