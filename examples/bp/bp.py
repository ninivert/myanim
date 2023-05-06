import numpy as np, numpy.typing as npt
from scipy.special import logsumexp
from dataclasses import dataclass
from collections import defaultdict
import itertools
from typing import Literal
import warnings

__all__ = ['Params', 'BPSave', 'BP']

@dataclass
class Params:
	c_ab: npt.NDArray[np.float64]  # connectivity between classes, p_ab = c_ab/N
	q: int  # number of classes
	N: int  # number of nodes
	damping: float = 0  # updated = (1-damping)*new + damping*current
	seed: int = 0  # random seed
	init: Literal['random', 'planted', 'paramagnetic'] = 'random'  # initialization mode

	@staticmethod
	def new_coloring_params(q: int, c: int, **kwargs) -> 'Params':
		"""Create a parameter set for planted graph coloring, i.e. `c_in = 0`

		Parameters
		----------
		q : int
			number of classes
		c : int
			average degree
		**kwargs :
			remaining arguments to ``Params`` constructor

		Returns
		-------
		Params
		"""
		return Params.new_bimodal_params(q, c_in=0, c_out=c*q/(q-1), **kwargs)

	@staticmethod
	def new_bimodal_params(q: int, c_in: float, c_out: float, **kwargs) -> 'Params':
		"""Create a parameter set for bimodal SBM, i.e. `c_ab = c_in if a == b else c_out`

		Parameters
		----------
		q : int
			number of classes
		c_in : int
			average degree inside the same class
		c_out : int
			average degree between two classes
		**kwargs :
			remaining arguments to ``Params`` constructor

		Returns
		-------
		Params
		"""
		c_ab = np.full((q, q), c_out, dtype=float)
		c_ab[np.diag_indices_from(c_ab)] = c_in
		return Params(c_ab=c_ab, q=q, **kwargs)


def log_logA_dot_B(logA: npt.NDArray[np.float64], B: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
	"""Computes log(A@B), given log(A) and B
	
	Typically useful if A has large mean, but small fluctuations. This method prevent overflowing the exp
	"""
	# maxlogA = logA.max()
	# return maxlogA + np.log(np.exp(logA-maxlogA) @ B)
	logA_max = np.amax(logA, axis=1, keepdims=True)
	return logA_max + np.log(np.exp(logA-logA_max) @ B)


def log_normalize(logA: npt.NDArray[np.float64], eps: float = 2e-40) -> npt.NDArray[np.float64]:
	"""Renormalize such that exp(logA).sum() == 1, in a numerically safe way"""
	logA_thresh = np.maximum(np.log(eps), logA)  # threshold to prevent taking log(0)
	z = logsumexp(logA_thresh, axis=1, keepdims=True)  # normalization constant
	logA_thresh -= z
	return logA_thresh


# saving the entire BP class is expensive in memory, so we only save these
@dataclass
class BPSave:
	logchi: npt.NDArray[np.float64]
	logmu: npt.NDArray[np.float64]
	h: npt.NDArray[np.float64]
	t: int
	params: Params


class BP:
	LOGEPS: float = 2e-40

	def __init__(self, params: Params):
		self.params = params

		# TODO : specify the distribution n in the params
		# generate the class for each node
		# uniform sampling : n_a -> 1/q as N -> infty
		# NOTE : each node is uniquely identified by its index in the node_to_class array
		# i.e. i = 0 .. N-1
		rng = np.random.default_rng(self.params.seed)
		self.node_to_class: npt.NDArray[np.int64] = rng.integers(0, self.params.q, self.params.N)
		self.s_star = self.node_to_class  # alias

		# precompute 1hot encoding of the node classes
		self.nodes_1hot = np.zeros((self.params.N, self.params.q), dtype=np.int8)  # NOTE : in numpy, bools are stored on 8 bits (1 byte)
		self.nodes_1hot[range(self.params.N), self.node_to_class] = 1

		# precompute mappings from class to its nodes, and the number of nodes
		self.class_to_nodes = { a: np.nonzero(self.nodes_1hot[:, a])[0] for a in range(self.nodes_1hot.shape[1]) }
		self.class_to_numnodes = { a: len(nodes) for a, nodes in self.class_to_nodes.items() }

		# precompute n vector
		self.n: npt.NDArray[np.float64] = np.array([ self.class_to_numnodes[a]/self.params.N for a in range(self.params.q) ], dtype=float)
		self.logn: npt.NDArray[np.float64] = np.log(self.n)

		# precompute average degree
		self.c: float = np.einsum('ab,a,b', self.params.c_ab, self.n, self.n)

		# generate the edges
		rng = np.random.default_rng(self.params.seed)
		edges = []
		for i, j in itertools.combinations(range(self.params.N), 2):
			if rng.random() < self.params.c_ab[self.node_to_class[i], self.node_to_class[j]] / self.params.N:
				edges.append((i, j))
		# NOTE : to recover c_ab from the edges, we can do this :
		# p_ab = np.zeros((self.params.q, self.params.q))
		# for i, j in self.ord_edges:
		# 	p_ab[self.node_to_class[i], self.node_to_class[j]] += 1
		# # normalize by number of spawn attempts in each group
		# p_ab / (self.params.N*(self.params.N-1)/2 * np.einsum('i,j->ij', self.n, self.n))

		self.ord_edges: npt.NDArray[np.int64] = np.unique(np.sort(edges, axis=1), axis=0)  # ordered edges (i, j), where i <= j, with no duplicates
		if len(self.ord_edges) == 0:
			raise RuntimeError(f'zero ordered edges have been generated, {edges}')
		self.full_edges: npt.NDArray[np.int64] = np.vstack((self.ord_edges, self.ord_edges[:, [1,0]]))  # all edges, in both directions : (i, j) and (j, i) are included
		self.E = len(self.ord_edges)
		# ord_edges = [
		#   (i1 -> j1)
		#   (i2 -> j2)
		#   ...
		#   (iE -> jE)
		# ]
		# full_edges = [
		#   (i1 -> j1)  ordered edges
		#   (i2 -> j2)  |
		#   ...         |
		#   (iE -> jE)  |
		#   (j1 -> i1)  flipped ordered edges
		#   (j2 -> i2)  |
		#   ...         |
		#   (jE -> iE)  |
		# ]

		# precompute neighbourhood of nodes
		# i -> neighbours of i
		self.neigh_node: dict[int, npt.NDArray[np.int64]] = { node: self.full_edges[self.full_edges[:, 0] == node][:, 1] for node in range(self.params.N) }
		# ij -> index of ij in the edge array
		self.edge_index: dict[tuple[int, int], int] = { (i, j): e for e, (i, j) in enumerate(self.full_edges) }
		# i -> index of ji in the edge array, for each j neighbor of i
		self.neigh_incoming_edge_index: dict[int, npt.NDArray[np.int64]] = {
			i: np.array([ self.edge_index[(j, i)] for j in self.neigh_node[i] ], dtype=np.int64)
			for i in range(self.params.N) }

		self.init()

	def init(self):
		# initialize messages
		# chi = [
		#   chi(i1 -> j1)
		#   chi(i2 -> j2)
		#   ... all ordered edges
		#   chi(j1 -> i1)
		#   chi(j2 -> i2)
		#   ... all reverse ordered edges
		# ]
		# -> (2E, q) matrix

		# initialize marginals
		# mu = [
		#   mu(1)
		#   mu(2)
		#   ... all nodes (even those not participating in graph, as they contribute to the h term !)
		#   mu(N-1)
		#   mu(N)
		# ]
		# -> (N, q) matrix

		if self.params.init == 'planted':
			chi = np.zeros((2*self.E, self.params.q), dtype=float)
			for e in range(len(chi)):
				i = self.full_edges[e, 0]  # extract i from directed edge (i, j)
				chi[e, :] = self.nodes_1hot[i]  # χi→j = vec δ(s*ᵢ, sᵢ)
			mu = np.zeros((self.params.N, self.params.q), dtype=float)
			for i in range(len(mu)):
				mu[i, :] = self.nodes_1hot[i]

		elif self.params.init == 'paramagnetic':
			chi = np.zeros((2*self.E, self.params.q), dtype=float)
			for e in range(len(chi)):
				i = self.full_edges[e, 0]  # extract i from directed edge (i, j)
				chi[e, :] = self.n  # χi→j = n
			mu = np.zeros((self.params.N, self.params.q), dtype=float)
			for i in range(len(mu)):
				mu[i, :] = self.n

		elif self.params.init == 'random':
			rng = np.random.default_rng(self.params.seed)
			chi = rng.random((2*self.E, self.params.q))
			mu = rng.random((self.params.N, self.params.q))

		else:
			raise RuntimeError(f'unknown initialization {self.params.init}')

		self.logchi: npt.NDArray[np.float64] = np.log(np.maximum(chi, BP.LOGEPS))  # take the log
		self.unnorm_logchi: npt.NDArray[np.float64] = self.logchi.copy()
		self.logmu: npt.NDArray[np.float64] = np.log(np.maximum(mu, BP.LOGEPS))  # take the log
		self.unnorm_logmu: npt.NDArray[np.float64] = self.logmu.copy()
		self.logchi, self.logmu = log_normalize(self.logchi), log_normalize(self.logmu)

		# initialize h
		# NOTE : h will get overwritten on the first iteration of BP, so we can init with whatever
		self.h = np.zeros(self.params.q)

		# initialize the delta
		# delta is not defined at the first iteration
		self.delta = np.nan

		# initialize time
		self.t = 0

	def __iter__(self):
		# comment out to reset on every run.
		# to be able to run where it left off from BPSave, leave commented out
		# self.init()
		return self

	def __next__(self):
		"""Performs one BP step"""
		# update h, numerically stable way
		self.h = 1/self.params.N * np.exp(log_logA_dot_B(self.logmu, self.params.c_ab)).sum(axis=0)

		# NOTE : we use the current h for the computations of logmu and logchi
		logmu, logchi = self._message_step()
		self.unnorm_logmu = logmu  # we do not need copies because log_normalize already takes a copy
		self.unnorm_logchi = logchi
		logmu, logchi = log_normalize(logmu), log_normalize(logchi)

		# compute delta
		self.delta = np.abs(np.exp(logmu) - np.exp(self.logmu)).max() + np.abs(np.exp(logchi) - np.exp(self.logchi)).max()
		
		# assign new arrays
		if self.params.damping > 0:
			self.logmu = (1-self.params.damping)*logmu + self.params.damping*self.logmu
			self.logchi = (1-self.params.damping)*logchi + self.params.damping*self.logchi
			self.logmu, self.logchi = log_normalize(self.logmu), log_normalize(self.logchi)  # renormalize after the damping
		else:
			self.logmu = logmu
			self.logchi = logchi
		
		self.t += 1

	def _message_step(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
		"""Returns unnormalized logmu and logchi"""
		# precompute the matrix of sums over spins
		# numerically stable way
		logy_edge = log_logA_dot_B(self.logchi, self.params.c_ab)

		# compute mu messages
		logmu = np.zeros_like(self.logmu)
		logmu += (self.logn - self.h)[None, :]
		# NOTE : we need this for loop, because neigh_incoming_edge_index would create a ragged index array
		for i in range(len(logmu)):
			logmu[i] += logy_edge[self.neigh_incoming_edge_index[i]].sum(axis=0)

		# compute chi messages
		logchi = np.zeros_like(self.logchi)
		for e, (i, j) in enumerate(self.full_edges):
			logchi[e, :] = logmu[i, :] - logy_edge[self.edge_index[(j, i)], :]

		return logmu, logchi

	def phi_bethe(self) -> float:
		"""Computes Bethe free entropy"""
		log_Zi: npt.NDArray[np.float64] = logsumexp(np.maximum(self.unnorm_logmu, np.log(BP.LOGEPS)), axis=1)  # has length N
		chi = np.exp(self.logchi)
		chi_ij, chi_ji = chi[:self.E], chi[self.E:]
		Zij = np.einsum('ab,ea,eb->e', self.params.c_ab, chi_ij, chi_ji)
		log_Zij: npt.NDArray[np.float64] = np.log(np.maximum(Zij, BP.LOGEPS))  # has length E
		return (log_Zi.sum() - log_Zij.sum())/self.params.N + self.c/2

	def overlap(self) -> tuple[npt.NDArray[np.int64], float]:
		"""Computed overlap metric Q. Returns the s hat of largest overlap, and the corresponding overlap"""
		# convert s_hat to 1-hot matrix with no permutations
		s_hat0 = self.s_hat
		nodes_1hot_hat0 = np.zeros((self.params.N, self.params.q), dtype=np.int8)
		nodes_1hot_hat0[range(self.params.N), s_hat0] = 1

		def compute_Q(pi: tuple[int, ...]):
			tp = ((nodes_1hot_hat0[:, pi] == 1) & (self.nodes_1hot == 1)).sum()  # true positives
			return pi, (tp/self.params.N - 1/self.params.q) / (1 - 1/self.params.q)

		pi_opt, Q = max(( compute_Q(pi) for pi in itertools.permutations(range(self.params.q)) ), key=lambda x: x[1] )
		return nodes_1hot_hat0[:, pi_opt].nonzero()[1], Q

	def run(self, tolerance: float = 1e-5, max_iter: int = 100, accumulate: list[Literal['logmu', 'logchi', 'h', 'delta', 'phi_bethe', 'overlap']] = [], init=True) -> dict[str, list]:
		accs = defaultdict(list)
		it = iter(self)
		
		if init:
			self.init()

		def log():
			accs['t'].append(self.t)
			accs['delta'].append(self.delta)
			if 'logmu' in accumulate: accs['logmu'].append(self.logmu.copy())
			if 'logchi' in accumulate: accs['logchi'].append(self.logchi.copy())
			if 'delta' in accumulate: accs['delta'].append(self.delta)
			if 'phi_bethe' in accumulate: accs['phi_bethe'].append(self.phi_bethe())
			if 'overlap' in accumulate: accs['overlap'].append(self.overlap()[1])

		log()

		for n in range(max_iter):
			next(it)
			log()

			if self.delta < tolerance:
				break
		
		if n == max_iter-1:
			warnings.warn('Max iteration limit')

		return accs

	@property
	def mu(self) -> npt.NDArray[np.float64]:
		"""Compute marginals from log marginals"""
		return np.exp(self.logmu)

	@property
	def chi(self) -> npt.NDArray[np.float64]:
		"""Compute messages from log messages"""
		return np.exp(self.logchi)

	@property
	def s_hat(self) -> npt.NDArray[np.int64]:
		"""Return the pointwise maximum posterior solution"""
		return np.argmax(self.logmu, axis=1)

	def __str__(self):
		edgerepr = np.array2string(self.ord_edges).replace("\n", "")
		r = f'BP at t={self.t} with'
		r += f'\n* {self.params}'
		r += f'\n* ordered edges={edgerepr} (E={self.E})'
		r += f'\n* n (paramagnetic fixed point)={self.n}'
		# r += f'\n* connected nodes={np.array2string(self.connected_nodes, threshold=6)} (N={self.N})'
		r += f'\n* s star         ={np.array2string(self.s_star, threshold=20)}'
		s_max_overlap, Q = self.overlap()
		r += f'\n* s max overlap  ={np.array2string(s_max_overlap, threshold=20)}'
		r += f'\n* overlap={Q:.3f}'
		r += f'\n* phi_bethe={self.phi_bethe():.3f}'
		r += '\n-'
		logchirepr = np.array2string(np.exp(self.logchi), precision=3)
		r += f'\n* chi=\n{logchirepr}'
		logmurepr = np.array2string(np.exp(self.logmu), precision=3)
		r += f'\n* mu=\n{logmurepr}'
		hrepr = np.array2string(self.h, precision=3)
		r += f'\n* h={hrepr}'
		return r

	@staticmethod
	def from_save(save: BPSave) -> 'BP':
		bp = BP(save.params)
		bp.logmu = save.logmu.copy()
		bp.logchi = save.logchi.copy()
		bp.h = save.h.copy()
		bp.t = save.t
		return bp

	def save(self) -> BPSave:
		return BPSave(
			logchi=self.logchi.copy(),
			logmu=self.logmu.copy(),
			h=self.h.copy(),
			t=self.t,
			params=self.params
		)