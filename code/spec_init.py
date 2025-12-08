import numpy as np
import numpy.linalg as nlg
from scipy.optimize import minimize


def sort_cols_lex(mat):
	"""
	mat: np.ndarray of shape (m, n) - binary matrix
	sort columns by lexicographical order, e.g. in order of columns [1, 1], [1, 0], [0, 1], [0, 0]
	--------------------------------------------------
	return sorted_mat: np.ndarray of shape (m, n)
	order: np.ndarray of length n - order of mat column indices in sorted_mat
	"""
	m, n = mat.shape
	order = np.argsort(["".join(list(mat[:, j].astype(str))) for j in range(n)])[::-1]
	sorted_mat = mat[:, order]
	return sorted_mat, order


def sort_perm_A(A):
	"""
	A: list of A_k (A_0 is None) - each a np.ndarray of shape (p_k, p_{k - 1})
	--------------------------------------------------
	layerwise from bottom to top, sort A_k columns by lexicographical order and permute rows of A_{k - 1} accordingly
	return sorted_A: list of sorted A_k (A_0 is None) - each a np.ndarray of shape (p_k, p_{k - 1})
	"""
	sorted_A_rev = []
	order = np.arange(A[-1].shape[0])
	for Ak in A[:0:-1]:
		sorted_Ak, order = sort_cols_lex(Ak[order])
		sorted_A_rev.append(sorted_Ak)
	return [None] + sorted_A_rev[::-1]


def average_A_list(A_list):
	"""
	A_list: list of A
	--------------------------------------------------
	return: their average, i.e. [None, average of A_1, ..., average of A_K]
	"""
	return [None] + [np.stack(Ak_list).mean(axis = 0) for Ak_list in list(zip(*A_list))[1:]]


def sort_perm_A_pair(A, AA):
	"""
	A: list of A_k (A_0 is None) - each a np.ndarray of shape (p_k, p_{k - 1})
	AA: list of AA_k - each a np.ndarray of same shape as A_k
	--------------------------------------------------
	sort and permute A as in func <sort_perm_A>, while using the same order to permute AA
	return sorted_A, sorted_AA
	"""
	sorted_A_rev, sorted_AA_rev = [], []
	order = np.arange(A[-1].shape[0])
	for Ak, AAk in zip(A[:0:-1], AA[:0:-1]):
		sorted_Ak, new_order = sort_cols_lex(Ak[order])
		sorted_A_rev.append(sorted_Ak)
		sorted_AA_rev.append(AAk[order][:, new_order])
		order = new_order
	return [None] + sorted_A_rev[::-1], [None] + sorted_AA_rev[::-1]


def successive_projection(R):
	"""
	R: np.ndarray of shape (n, d) - coordinates of n points in R^d
	(d + 1) points among them form a simplex that contains all other points
	successive projection algorithm for finding these (d + 1) simplex vertices
	--------------------------------------------------
	return V: np.ndarray of shape (d + 1, d) - coordinates of (d + 1) vertices
	indices: np.ndarray of length (d + 1) - indices of vertices among these 1, ..., n points
	"""
	n, d = R.shape
	indices = []
	Y = np.hstack([np.ones((n, 1)), R])
	for t in range(d + 1):
		i = np.argmax(nlg.norm(Y, axis = 1))
		indices.append(i)
		Y -= Y @ np.outer(Y[i], Y[i]) / nlg.norm(Y[i]) ** 2
	return R[indices], np.array(indices)


def mean_shift(X, bandwidth_ratio = 0.5):
	"""
	X: np.ndarray of shape (n, q) - q-dimensional coordinates of n points
	bandwidth_ratio: ratio of Gaussian kernel bandwidth (for weighted averaging) to average pairwise distance
	--------------------------------------------------
	return mean shifted X: np.ndarray of shape (n, q) - used as denoised version of X
	"""
	n, q = X.shape
	dist_mat = ((X.reshape(n, 1, q) - X.reshape(1, n, q)) ** 2).sum(axis = 2) ** 0.5
	kernel_bandwidth = dist_mat.mean() * bandwidth_ratio
	weights = np.exp(- dist_mat ** 2 / kernel_bandwidth ** 2)
	weights /= weights.sum(axis = 1).reshape(n, 1)
	return weights @ X


def sketched_vertex_search(R):
	"""
	R: np.ndarray of shape (n, d) - coordinates of n noisy points in R^d
	(d + 1) points among them form a simplex that contains all other points
	sketched vertex search algorithm for finding these (d + 1) simplex vertices
	step 1: denoise R by mean shifting
	step 2: use successive projection algorithm to find the (d + 1) simplex vertices among the k medoids
	--------------------------------------------------
	return V: np.ndarray of shape (d + 1, d) - coordinates of (d + 1) vertices
	"""
	V = successive_projection(mean_shift(R))[0]
	return V


def simplex_mixture_weights(R, V):
	"""
	R: np.ndarray of shape (n, d) - coordinates of n points in R^d
	V: np.ndarray of shape (d + 1, d) - coordinates of (d + 1) simplex vertices
	--------------------------------------------------
	project each point to the simplex and express the projection as a linear combination of vertices
	return W: np.ndarray of shape (n, d + 1) - linear combination mixture weights satisfying each entry >= 0, each row sums to 1
	"""
	n, d = R.shape
	W = []
	for r in R:
		objective = lambda weights: ((weights @ V - r) ** 2).sum()
		constraints = (
			{'type': 'eq', 'fun': lambda weights: weights.sum() - 1},
			{'type': 'ineq', 'fun': lambda weights: weights}
		)
		W.append(minimize(objective, x0 = np.ones(d + 1) / (d + 1), constraints = constraints, method = 'SLSQP').x)
	return np.array(W)


def mixed_score(X, q):
	"""
	X: np.ndarray of shape (n, n) - adjacency/affinity matrix
	q: number of communities
	modified Mixed-SCORE algorithm from "Mixed Membership Estimation for Social Networks" by Jin, Ke, Luo
	--------------------------------------------------
	return M: np.ndarray of shape (n, q) - mixed memberships under Mixed-SCORE algorithm
	indices: np.ndarray of length q - indices of the q community medoids (i.e. points closest to simplex vertices)
	"""
	eigval, eigvec = nlg.eig(X)
	top_eigvec, fiedler_eigvec = eigvec[:, np.argsort(eigval)[-1]], eigvec[:, np.argsort(eigval)[-2:-(q + 1):-1]]
	R = fiedler_eigvec / top_eigvec.reshape(-1, 1)
	V = sketched_vertex_search(R)
	W = simplex_mixture_weights(R, V)
	b1 = (eigval[np.argsort(eigval)[-1]] + (eigval[np.argsort(eigval)[-2:-(q + 1):-1]].reshape(1, -1) * V ** 2).sum(axis = 1)) ** (-0.5)
	M = np.maximum(0, W / b1.reshape(1, -1))
	M /= M.sum(axis = 1).reshape(-1, 1)
	L = []
	for r in R:
		try:
			l_r = ((r - V[0]) @ nlg.inv(V[1:] - V[0].reshape(1, -1))).reshape(-1)
		except nlg.LinAlgError as error:
			print('Matrix is singular, switching nlg.inv to nlg.pinv:', error)
			l_r = ((r - V[0]) @ nlg.pinv(V[1:] - V[0].reshape(1, -1))).reshape(-1)
		L.append(np.append(1 - l_r.sum(), l_r))
	L = np.array(L)
	indices = [np.argmax(L[:, i]) for i in range(q)]
	return M, np.array(indices)


def convert_mixed_to_overlap(M, sparsity, min_n_nodes = 2, threshold = 0.2):
	"""
	M: np.ndarray of shape (n, q) - mixed memberships
	sparsity: sparsity parameter S, i.e. maximum number of ones in each row of connection matrices A
	min_n_nodes: minimum number of nodes required for each community, i.e. minimum number of ones in each column of A
	this implicitly requires n >= min_n_nodes * q
	threshold: threshold for a mixed membership to be candidate for overlapping membership
	--------------------------------------------------
	convert mixed membership matrix to overlapping membership matrix under the sparsity and min_n_nodes constraints
	return A: np.ndarray of shape (n, q) - binary matrix of overlapping memberships
	"""
	n, q = M.shape
	A = np.zeros((n, q))
	rows_filled = np.zeros(n)
	for _ in range(min_n_nodes):
		cols_filled = np.zeros(q)
		for _ in range(q):
			ij = np.argmax((M - 2 * rows_filled.reshape(n, 1) - 2 * cols_filled.reshape(1, q)).reshape(-1))
			i, j = ij // q, ij % q
			A[i, j], rows_filled[i], cols_filled[j] = 1, 1, 1
	for i in range(n):
		M_i_left = M[i] * (1 - A[i])
		top_entries = np.argsort(M_i_left)[::-1][:int(min(sparsity - rows_filled[i], (M_i_left > threshold).sum()))]
		A[i, top_entries] = 1
	return A


def spectral_initialization(X_ave, p, sparsity, min_n_nodes = 2, threshold = 0.2):
	"""
	X_ave: np.ndarray of shape (p_K, p_K) - sample average of observed adjacency matrices
	p: list of number of nodes at each layer [p_0, p_1, ..., p_K]
	sparsity: sparsity parameter S, i.e. maximum number of ones in each row of connection matrices A
	min_n_nodes: minimum number of nodes required for each community, i.e. minimum number of ones in each column of A
	this implicitly requires n >= min_n_nodes * q
	threshold: threshold for a mixed membership to be candidate for overlapping membership
	--------------------------------------------------
	apply Mixed SCORE algorithm layerwise from bottom to top to initialize connection matrices A
	truncation is applied to convert mixed memberships to overlapping memberships under sparsity constraint
	return: spectral initialization of A
	"""
	M_rev = []
	for p_k_1 in p[-2::-1]:
		M_k, indices = mixed_score(X_ave, p_k_1)
		order = sort_cols_lex((M_k > threshold).astype(int))[1]
		M_rev.append(M_k[:, order])
		X_ave = X_ave[indices[order]][:, indices[order]]
	M = [None] + M_rev[::-1]
	A = [None] + [convert_mixed_to_overlap(M_k, sparsity, min_n_nodes, threshold) for M_k in M[1:]]
	return A
