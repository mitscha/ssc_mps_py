import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.cluster import SpectralClustering
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import normalize

from matchingpursuit import MatchingPursuit


def ssc_mps(X,smax,L,tol=None,alg_name='OMP',pmax=None):
	"""
	Implements Sparse Subspace Clustering-Orthogonal Matching Pursuit (SSC-OMP) and 
	SSC-Matching Pursuit (SSC-MP)
	
	Parameters
	----------
	
	X: array, shape (n_features, n_samples)
		data matrix
	smax: int
		Maximum number of OMP/MP iterations
	L: int
		Number of clusters
	tol: float, optional
		Threshold on approximation quality
	alg_name: string, optional
		'OMP' (default) or 'MP'
	pmax:
		Maximum sparsity level for MP
	
	
	Note: 
	
	- Stopping behavior:
	  SSC-OMP: Stop after smax iterations if tol=None, stop when approximation quality
	           specified by tol is attained otherwise
	  SSC-MP:  Stop after smax iterations, or when approximation quality specified by tol
	           is attained, or when the sparsity level of the coefficient vector is pmax
	- See https://arxiv.org/abs/1612.03450 for a discussion of the stopping criteria
	
	"""
	
	
	XX = np.array(X)
	
	assert(len(XX.shape) == 2)
	
	m = XX.shape[0]
	N = XX.shape[1]
	
	
	alg = None
	if alg_name == 'MP':
		alg = MatchingPursuit(smax, pmax, tol)
	else:
		alg = OrthogonalMatchingPursuit(
			n_nonzero_coefs=smax, 
			tol=tol, 
			fit_intercept=False, 
			normalize=False)
	
	
	C = np.zeros((N,N))
	
	for i in range(N):
		data_idx = [j for j in range(i)]
		data_idx.extend([j for j in range(i+1,N)])
		alg.fit(XX[:,data_idx],np.squeeze(XX[:,i]))
		c = np.zeros(N)
		c[data_idx] = alg.coef_
		C[:,i] = c
	
	sc = SpectralClustering(n_clusters=L, affinity='precomputed', n_init=50, n_jobs=-1)
	sc.fit(np.abs(C) + np.abs(C.T))
	
	return sc.labels_



"""

Example: Cluster face images of 3 randomly selected individuals (10 images each)
from the Olivetti face data base.

"""
if __name__ == '__main__':
	
	ids = np.random.choice([i for i in range(40)], size=3, replace=False)
	faces = fetch_olivetti_faces()
	X = np.hstack((
			faces.data[faces.target==ids[0],:].T,
			faces.data[faces.target==ids[1],:].T,
			faces.data[faces.target==ids[2],:].T))
	
	Xn = normalize(X - np.outer(np.mean(X, axis=1), np.ones(X.shape[1])), axis=0)
	
	# L=3, smax=3
	labels_omp = ssc_mps(Xn,3,3)
	labels_mp = ssc_mps(Xn,3,3,alg_name='MP')
	
	print('Image labels (SSC-OMP): %s\nImage labels (SSC-MP) : %s' 
		% (str(labels_omp), str(labels_mp)))