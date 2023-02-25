__all__ = ['linterp', 'clamp']

def linterp(t: float, x: float, y: float, a: float = 0, b: float = 1) -> float:
	return (t-x)/(y-x)*(b-a) + a

def clamp(val: float, val_min: float, val_max: float) -> float:
	return max(min(val, val_max), val_min)