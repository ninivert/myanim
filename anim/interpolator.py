from .var import T

__all__ = [
	'Interpolator',
	'LinearBezier', 'QuadraticBezier', 'CubicBezier',
	'InterpLinear', 'InterpQuadatric', 'InterpCubic',
]

class Interpolator:
	"""Base class for interpolation functions"""

	slots = ()

	def __call__(self, s: float, initial: T, final: T) -> T:
		pass


class LinearBezier(Interpolator):
	"""Interpolate linearly"""

	slots = ()

	def __call__(self, s: float, initial: T, final: T) -> T:
		# p0 = initial
		# p1 = final
		# return (1-t)*self.p0 + t*self.p1
		return (1-s)*initial + s*final


class QuadraticBezier(Interpolator):
	"""Interpolate using a quadratic Bezier curve (one control point)"""

	slots = ('p1',)

	def __init__(self, p1: T):
		self.p1 = p1  # first control point

	def __call__(self, s: float, initial: T, final: T) -> T:
		return (1-s)**2*initial + 2*(1-s)*s*self.p1 + s**2*final


class CubicBezier(Interpolator):
	"""Interpolate using a cubic Bezier curve (two control points)"""

	slots = ('p1', 'p2')

	def __init__(self, p1: T, p2: T):
		self.p1 = p1  # first control point
		self.p2 = p2  # second control point

	def __call__(self, s: float, initial: T, final: T) -> T:
		return (1-s)**3*initial + 3*(1-s)**2*s*self.p1 + 3*(1-s)*s**2*self.p2 + s**3*final


# define some aliases
InterpLinear = LinearBezier
InterpQuadatric = QuadraticBezier
InterpCubic = CubicBezier