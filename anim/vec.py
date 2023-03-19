from typing import TypeVar, Union
import operator as op

__all__ = ['Vec']

T = TypeVar('T')

class Vec(tuple[T]):
	def __add__(self, other: 'Vec[T]') -> 'Vec[T]':
		if isinstance(other, (float, int)):
			other = Vec.full(len(self), other)
		assert len(self) == len(other), 'shape mistmatch'
		return Vec(map(op.add, self, other))

	def __radd__(self, other: 'Vec[T]') -> 'Vec[T]':
		if isinstance(other, (float, int)):
			other = Vec.full(len(self), other)
		assert len(self) == len(other), 'shape mistmatch'
		return Vec(map(op.add, other, self))

	def __sub__(self, other: 'Vec[T]') -> 'Vec[T]':
		if isinstance(other, (float, int)):
			other = Vec.full(len(self), other)
		assert len(self) == len(other), 'shape mistmatch'
		return Vec(map(op.sub, self, other))

	def __rsub__(self, other: 'Vec[T]') -> 'Vec[T]':
		if isinstance(other, (float, int)):
			other = Vec.full(len(self), other)
		assert len(self) == len(other), 'shape mistmatch'
		return Vec(map(op.sub, other, self))

	def __mul__(self, other: Union['Vec[T]', float, int]) -> 'Vec[T]':
		if isinstance(other, (float, int)):
			other = Vec.full(len(self), other)
		assert len(self) == len(other), 'shape mistmatch'
		return Vec(map(op.mul, self, other))

	def __rmul__(self, other: Union['Vec[T]', float, int]) -> 'Vec[T]':
		if isinstance(other, (float, int)):
			other = Vec.full(len(self), other)
		assert len(self) == len(other), 'shape mistmatch'
		return Vec(map(op.mul, other, self))

	def __truediv__(self, other: 'Vec[T]') -> 'Vec[T]':
		if isinstance(other, (float, int)):
			other = Vec.full(len(self), other)
		assert len(self) == len(other), 'shape mistmatch'
		return Vec(map(op.truediv, self, other))

	def __rtruediv__(self, other: 'Vec[T]') -> 'Vec[T]':
		if isinstance(other, (float, int)):
			other = Vec.full(len(self), other)
		assert len(self) == len(other), 'shape mistmatch'
		return Vec(map(op.truediv, other, self))

	def __neg__(self) -> 'Vec[T]':
		return Vec(map(op.neg, self))

	def dot(self, other: 'Vec[T]') -> T:
		return sum(self * other)

	@staticmethod
	def full(n: int, v: T = 0) -> 'Vec[T]':
		return Vec(v for _ in range(n))