from typing import Generic, TypeVar

T = TypeVar('T')

__all__ = ['Var', 'T']

class Var(Generic[T]):
	"""A reference to a value. ``T`` needs to support basic arithmetic operations"""

	__slots__ = 'value',

	def __init__(self, value: T) -> None:
		self.value: T = value

	def __str__(self) -> str:
		return str(self.value)

	def __repr__(self) -> str:
		return f'Var({self.value})'