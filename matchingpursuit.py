import numpy as np
import sys
import copy

class MatchingPursuit:
	"""Simple implementation of the Matching Pursuit (MP) algorithm
	
	Parameters
	----------
	
	smax: int
		Maximum number of MP iterations
	pmax: int, optional
		Maximum sparsity level of x (default: pmax = smax)
	tol: float, optional
		Threshold on approximation quality ||Ax - y|| (default: 0.0)
		
	Attributes
	----------
	
	coef_: array, shape (n_samples)
		coefficient vector (solution)
	
	
	Note: Stops after smax iterations, or when approximation quality specified by tol
	      is attained, or when the sparsity level of the coefficient vector is pmax
	
	"""
	
	def __init__(self,smax,pmax=None,tol=None):
		self._smax = smax if smax != None else sys.maxsize
		self._pmax = pmax if pmax != None else smax
		self._tol = 0.0 if tol == None else tol
		self.coef_ = None
	
	def fit(self,A,y):
		"""
		Finds a sparse (approximate) solution x to Ax = y
		
		Parameters
		----------
		
		X: dictionary, array, shape (n_features, n_samples)
		y: target, array, shape (n_features)
		
		"""
		
		assert(len(A.shape) == 2)
		assert(len(y.shape) == 1 and A.shape[0] == y.shape[0])
		
		x = np.zeros(A.shape[1])
		r = copy.deepcopy(y)
		nit = 0
		while np.linalg.norm(r) > self._tol \
			and nit < self._smax \
			and np.sum(np.abs(x) > 0) < self._pmax:
			
			idx = np.argmax(np.dot(r.T,A))
			dx = np.dot(r.T,A[:,idx])/np.dot(A[:,idx].T,A[:,idx])
			x[idx] += dx
			r -= dx*A[:,idx]
			nit += 1
		
		self.coef_ = x