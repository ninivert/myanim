from dataclasses import dataclass
from anim import Vec, RGB

__all__ = [
	'ColorCycler',
	'ColorCyclerMPL'
]


def linterp(t: float, a: float, b: float, x: float = 0, y: float = 1):
	"""[a, b] : original range, [x, y] : destination range"""
	return x+(t-a)/(b-a)*(y-x)

def hex_to_rgb(c: str) -> tuple[int, int, int]:
	"""'#RRGGBBAA' -> (RR, GG, BB)'"""
	return tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def rgb_normalize(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
	"""Convert the 0-255 range to 0-1"""
	return tuple(c/255 for c in rgb)


class ColorCycler(Vec[RGB]):
	def __getitem__(self, index: int) -> RGB:
		return super().__getitem__(index % len(self))

ColorCyclerMPL = ColorCycler(tuple(rgb_normalize(hex_to_rgb(c)) for c in ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']))