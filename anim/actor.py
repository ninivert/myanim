import math
from typing import Callable
from cairo import Context
import cairo
from .var import T, Var
from .vec import Vec

# TODO : a better color class, which implements from_hsl, converting, etc
RGB = Vec[float]  # (r, g, b)

__all__ = ['Actor', 'RGB', 'Disc', 'Rect', 'Text']

class Actor:
	__slots__ = 'xy', 'rgb', 'alpha'

	def __init__(self, xy: Var[Vec[float]], rgb: Var[RGB] = Var((0, 0, 0)), alpha: Var[float] = Var(1.0)) -> None:
		self.xy = xy
		self.rgb = rgb
		self.alpha = alpha

	def render(self, ctx: Context) -> None:
		pass


class Disc(Actor):
	__slots__ = 'r'

	def __init__(self, xy: Var[Vec[float]], r: Var[float], rgb: Var[RGB] = Var((0, 0, 0)), alpha: Var[float] = Var(1.0)) -> None:
		super().__init__(xy, rgb, alpha)
		self.r = r

	def __str__(self) -> str:
		return f'Disk(xy={self.xy}, r={self.r}, rgb={self.rgb}, alpha={self.alpha})'

	def render(self, ctx: Context) -> None:
		ctx.set_source_rgba(*self.rgb.value, self.alpha.value)
		ctx.arc(self.xy.value[0], self.xy.value[1], self.r.value, 0, 2*math.pi)
		ctx.fill()


class Rect(Actor):
	__slots__ = 'w', 'h'

	def __init__(self, xy: Var[Vec[float]], w: Var[float], h: Var[float], rgb: Var[RGB] = Var((0, 0, 0)), alpha: Var[float] = Var(1.0)) -> None:
		super().__init__(xy, rgb, alpha)
		self.w = w
		self.h = h

	def __str__(self) -> str:
		return f'Rect(xy={self.xy}, w={self.w}, h={self.h}, rgb={self.rgb}, alpha={self.alpha})'

	def render(self, ctx: Context) -> None:
		ctx.set_source_rgba(*self.rgb.value, self.alpha.value)
		ctx.rectangle(self.xy.value[0], self.xy.value[1], self.w.value, self.h.value)
		ctx.fill()


class Text(Actor):
	"""Animated text
	
	``scalar`` is animated through the animation, and the ``formatter`` function maps ``scalar -> text``
	"""

	__slots__ = 'formatter', 'scalar', 'font_size'
	# TODO : font options

	def __init__(self, xy: Var[Vec[float]], scalar: Var[T], formatter: Callable[[T], str] = str, font_size: Var[float] = Var(16), rgb: Var[RGB] = Var((0, 0, 0)), alpha: Var[float] = Var(1.0)) -> None:
		super().__init__(xy, rgb, alpha)
		self.formatter = formatter
		self.scalar = scalar
		self.font_size = font_size

	def render(self, ctx: Context) -> None:
		ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
		ctx.set_font_size(self.font_size.value)

		ctx.move_to(*self.xy.value)
		ctx.set_source_rgba(*self.rgb.value, self.alpha.value)
		ctx.show_text(self.formatter(self.scalar.value))