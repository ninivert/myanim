from math import sin, cos, pi, sqrt

__all__ = [
	'linear',
	'ease_in_quad', 'ease_out_quad', 'ease_in_out_quad',
	'ease_in_cubic', 'ease_out_cubic', 'ease_in_out_cubic',
	'ease_in_quart', 'ease_out_quart', 'ease_in_out_quart',
	'ease_in_trig', 'ease_out_trig', 'ease_in_out_trig',
	'ease_in_circ', 'ease_out_circ', 'ease_in_out_circ',
	'ease_in_back', 'ease_out_back', 'ease_in_out_back',
	'ease_out_bounce', 'ease_in_bounce', 'ease_in_out_bounce'
]

def linear(t: float) -> float:
	return t

def ease_in_quad(t: float) -> float:
	return t**2

def ease_out_quad(t: float) -> float:
	return -t * (t-2)

def ease_in_out_quad(t: float) -> float:
	return 2*t**2 if t < 0.5 else -2*t**2 + 4*t - 1

def ease_in_cubic(t: float) -> float:
	return t**3

def ease_out_cubic(t: float) -> float:
	return (t-1)**3 + 1

def ease_in_out_cubic(t: float) -> float:
	return 4*t**3 if t < 0.5 else 1 - (-2*t+2)**3 / 2

def ease_in_quart(t: float) -> float:
	return t**4

def ease_out_quart(t: float) -> float:
	return 1 - (t-1)**4

def ease_in_out_quart(t: float) -> float:
	return 8*t**4 if t < 0.5 else 1 - 8*(t-1)**4

def ease_in_trig(t: float) -> float:
	return 1 - cos(t*pi/2)

def ease_out_trig(t: float) -> float:
	return sin(t*pi/2)

def ease_in_out_trig(t: float) -> float:
	return (1-cos(t*pi))/2

def ease_in_circ(t: float) -> float:
	return 1 - sqrt(1-t**2)

def ease_out_circ(t: float) -> float:
	return sqrt(1-(t-1)**2)

def ease_in_out_circ(t: float) -> float:
	return (1-sqrt(1-4*t**2))/2 if t < 0.5 else (1+sqrt(1-4*(t-1)**2))/2

def ease_in_back(t: float) -> float:
	a = 1.70158
	b = 2.70158
	return b*t**3 - a*t**2

def ease_out_back(t: float) -> float:
	a = 1.70158
	b = 2.70158
	return 1 + b*(t-1)**3 + a*(t-1)**2

def ease_in_out_back(t: float) -> float:
	a = 1.70158
	b = a*1.525
	return 4*t**2 * ((b+1)*t - 0.5*b) if t < 0.5 else (2*t-2)**2 * ((b+1)*(t-1)+0.5*b) + 1

def ease_out_bounce(t: float) -> float:
	# FIXME : this doesn't work
	a = 7.5625
	b = 2.75
	if t < 1/b:
		return a*t**2
	elif t < 2/b:
		return a*((t-1.5)/b)*t + 0.75
	elif t < 2.5/b:
		return a*((t-2.25)/b)*t + 0.9375
	else:
		return a*((t-2.625)/b)*t + 0.984375

def ease_in_bounce(t: float) -> float:
	return 1 - ease_out_bounce(t)

def ease_in_out_bounce(t: float) -> float:
	return (1-ease_out_bounce(1-2*t))/2 if t < 0.5 else (1+ease_out_bounce(2*t-1))/2