import math
from cairo import Context
from .var import Var

# TODO : support of arithmetic operations on the tuple
RGBA = tuple[float, float, float, float]

__all__ = ['Actor', 'RGBA', 'Disk', 'Rect']

class Actor:
	__slots__ = 'x', 'y', 'rgba'

	def __init__(self, x: Var[float], y: Var[float], rgba: Var[RGBA] = (0, 0, 0, 1)) -> None:
		self.x = x
		self.y = y
		self.rgba = rgba

	def render(self, ctx: Context) -> None:
		pass


class Disk(Actor):
	__slots__ = 'r'

	def __init__(self, x: Var[float], y: Var[float], r: Var[float], rgba: Var[RGBA] = (0, 0, 0, 1)) -> None:
		super().__init__(x, y, rgba)
		self.r = r

	def __str__(self) -> str:
		return f'Disk(x={self.x}, y={self.y}, r={self.r}, rgba={self.rgba})'

	def render(self, ctx: Context) -> None:
		ctx.set_source_rgba(*self.rgba)
		ctx.arc(self.x.value, self.y.value, self.r.value, 0, 2*math.pi)
		ctx.fill()


class Rect(Actor):
	__slots__ = 'w', 'h'

	def __init__(self, x: Var[float], y: Var[float], w: Var[float], h: Var[float], rgba: Var[RGBA] = (0, 0, 0, 1)) -> None:
		super().__init__(x, y, rgba)
		self.w = w
		self.h = h

	def __str__(self) -> str:
		return f'Rect(x={self.x}, y={self.y}, w={self.w}, h={self.h}, rgba={self.rgba})'

	def render(self, ctx: Context) -> None:
		ctx.set_source_rgba(*self.rgba)
		ctx.rectangle(self.x.value, self.y.value, self.w.value, self.h.value)
		ctx.fill()