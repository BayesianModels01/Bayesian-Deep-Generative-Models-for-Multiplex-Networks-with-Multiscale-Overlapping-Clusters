import numpy as np
import numpy.linalg as nlg
from polyagamma import random_polyagamma
from scipy.stats import norm, truncnorm
from numba import jit
from copy import *
import pickle
from utils import *
from spec_init import *


class Parameter():
	"""
	parameter class for saving all parameters and latent variables associated with one Gibbs sampling iteration
	e.g. self.A, self.theta, ...
	"""
	def __init__(self):
		pass

	def __str__(self):
		return str(self.__dict__)


@jit(nopython = True)
def update_Z_kappa_k(pk1, pk, N, Xk1, Xk, Ak, Zk, kappak, mask):
	"""
	pk1: number of nodes p_{k - 1} at layer k - 1
	pk: number of nodes p_k at layer k
	N: number of subjects, i.e. sample size
	Xk1: adjacency matrices X_{k - 1} at layer k - 1 of shape N x p_{k - 1} x p_{k - 1}
	Xk: adjacency matrices X_k at layer k of shape N x p_k x p_k
	Ak: parameter A_k
	Zk: array of shape (N p_k (p_k - 1) / 2) x (1 + p_{k - 1} (p_{k - 1} + 1) / 2)
		Z_k is an expression of A_k, X_{k - 1} satisfyting <Z_{k, i, j}^{(n)}, theta_k> = C_k + a_{k, i}^T (Gamma_k * X_{k - 1}^{(n)}) a_{k, j}
		the indices n, i, j are collapsed into the first dimension of Zk
	kappak: array of length N p_k (p_k - 1) / 2
		kappa_{k, i, j}^{(n)} = X_{k, i, j}^{(n)} - 1 / 2
		the indices n, i, j are collapsed into one dimension in kappa_k
	--------------------------------------------------
	after sampling new adjacency matrices X and parameters A, use this function to update Z and kappa accordingly
	updates are made in place of Z and kappa
	"""
	l = 0
	for n in range(N):
		for i in range(pk):
			for j in range(i):
				if mask[i, j] == 0:
					continue
				kappak[l] = Xk[n, i, j] - 1 / 2
				l_ = 1
				for i_ in range(pk1):
					for j_ in range(i_ + 1):
						if j_ == i_:
							Zk[l, l_] = Ak[i, i_] * Ak[j, j_] * Xk1[n, i_, j_]
						else:
							Zk[l, l_] = Ak[i, i_] * Ak[j, j_] * Xk1[n, i_, j_] + Ak[i, j_] * Ak[j, i_] * Xk1[n, i_, j_]
						l_ += 1
				l += 1


@jit(nopython = True)
def encode_am(mat, p):
	"""
	mat: p x p adjacency matrix with all-ones diagonal
	p: mat dimension
	--------------------------------------------------
	encode the p(p - 1) / 2 lower-triangular binary entries as a value between 0 ~ 2^{p(p - 1) / 2} - 1
	return: encoded value
	"""
	m = 2 ** (p * (p - 1) / 2 - 1)
	val = 0
	for i in range(p):
		for j in range(i):
			val += mat[i, j] * m
			m /= 2
	return val


@jit(nopython = True)
def decode_am(val, p):
	"""
	val: value between 0 ~ 2^{p(p - 1) / 2} - 1
	p: mat dimension
	--------------------------------------------------
	decode the encoded value between 0 ~ 2^{p(p - 1) / 2} to the p(p - 1) / 2 lower-triangular binary entries of p x p adjacency matrix
	return: decoded adjacency matrix
	"""
	mat = np.identity(p)
	for i in range(p - 1, -1, -1):
		for j in range(i - 1, -1, -1):
			mat[i, j] = val % 2
			mat[j, i] = mat[i, j]
			val //= 2
	return mat


@jit(nopython = True)
def count_X0(p0, X0_arr):
	"""
	p0: number of nodes at top layer, i.e. dimension of X_0
	X0_arr: N x p_0 x p_0 - top layer adjacency matrices of all subjects
	--------------------------------------------------
	count the frequency of each X_0
	return: an array of length 2^{p_0(p_0 - 1)} with ith entry being the count number of decode_am(i, p_0) among N subjects
	"""
	count_arr = np.zeros(int(2 ** (p0 * (p0 - 1) / 2)))
	for n in range(X0_arr.shape[0]):
		count_arr[int(encode_am(X0_arr[n], p0))] += 1
	return count_arr


@jit(nopython = True)
def log_posterior_Xn0(p1, C1, Xn0_Gamma1, A1, Xn1, mask):
	"""
	p1: number of nodes at layer 1
	C1: parameter C_1
	Xn0_Gamma1: entrywise product of X_0^{(n)} and Gamma_1
	A1: parameter A_1
	Xn1: p_1 x p_1 adjacency matrix X_1^{(n)} at layer 1 of subject n
	mask: p_1 x p_1 mask matrix on observed adjacency matrix (i.e. only applicable if K = 1) - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	--------------------------------------------------
	return: log posterior full conditional of X_0^{(n)}
	"""
	log_post = 0
	for i in range(p1):
		for j in range(i):
			if mask[i, j] < 0.5:
				continue
			inner_prod = C1
			for p, e in enumerate(A1[i]):
				if e:
					for q, f in enumerate(A1[j]):
						if f:
							inner_prod += Xn0_Gamma1[p, q]
			log_post += Xn1[i, j] * inner_prod - np.log(1 + np.exp(inner_prod))
	return log_post


@jit(nopython = True)
def log_posterior_Xnkij(pk2, Ck1, Ck2, Xnk0_Gammak1, Xnk1_Gammak2, Ak1, Ak2, Xnk1, Xnk2, i, j, mask):
	"""
	k0 represents layer k - 1, k1 represents layer k, k2 represents layer k + 1
	pk2: number of nodes p_{k + 1} at layer k + 1
	Ck1: parameter C_k
	Ck2: parameter C_{k + 1}
	Xnk0_Gammak1: entrywise product of X_{k - 1}^{(n)} and Gamma_k
	Xnk1_Gammak2: entrywise product of X_k^{(n)} and Gamma_{k + 1}
	Ak1: parameter A_k
	Ak2: parameter A_{k + 1}
	Xnk1: p_k x p_k adjacency matrix X_k^{(n)} at layer k of subject n
	Xnk2: p_{k + 1} x p_{k + 1} adjacency matrix X_{k + 1}^{(n)} at layer k + 1 of subject n
	i: row index of X_k^{(n)} - integer between 0 ~ p_k - 1
	j: column index of X_k^{(n)} - integer between 0 ~ p_k - 1
	mask: p_{k + 1} x p_{k + 1} mask matrix on observed adjacency matrix (i.e. only applicable if k = K - 1) - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	--------------------------------------------------
	return: log posterior full conditional of X_{k, i, j}^{(n)}, i.e. (i, j)th entry of adjacency matrix X_k^{(n)} at layer k for subject n
	"""
	inner_prod = Ck1
	for p, e in enumerate(Ak1[i]):
		if e:
			for q, f in enumerate(Ak1[j]):
				if f:
					inner_prod += Xnk0_Gammak1[p, q]
	log_post = Xnk1[i, j] * inner_prod - np.log(1 + np.exp(inner_prod))
	for i_ in range(pk2):
		for j_ in range(i_):
			if mask[i_, j_] < 0.5:
				continue
			inner_prod = Ck2
			for p, e in enumerate(Ak2[i_]):
				if e:
					for q, f in enumerate(Ak2[j_]):
						if f:
							inner_prod += Xnk1_Gammak2[p, q]
			log_post += Xnk2[i_, j_] * inner_prod - np.log(1 + np.exp(inner_prod))
	return log_post


@jit(nopython = True)
def sample_X_Gibbs(p, nu, C, Gamma, A, X, alpha, mask, seed):
	"""
	p: number of nodes [p_0, p_1, ..., p_K]
	nu: parameter nu
	C: parameter [None, C_1, ..., C_K]
	Gamma: parameter [None, Gamma_1, ..., Gamma_K]
	A: parameter [None, A_1, ..., A_K]
	X: list of adjacency matrices at each layer: [X_0, X_1, ..., X_K], each X_k of shape N x p_k x p_k
	alpha: raising full conditional to power alpha, can be used for simulated annearling or for subsampling Gibbs sampler
	mask: p_K x p_K mask matrix on observed adjacency matrix - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	seed: random state
	--------------------------------------------------
	sample in the order:
	- top layer k = 0: for each subject n draw X_0^{(n)} blockwise
	- for layer k = 1, 2, ..., K:
		for entry i, j of X_k:
			for each subject n draw X_{k, i, j}^{(n)} entrywise
	new sample of X is updated in place of X
	"""
	np.random.seed(seed)
	for k in range(p.size - 1):
		mask_k = np.ones((p[k + 1], p[k + 1])) if k < p.size - 2 else mask
		# sample top layer k = 0
		if k == 0:
			p0, p1 = p[0], p[1]
			C1 = C[0]
			Gamma1 = Gamma[:(p0 ** 2)].copy().reshape(p0, p0)
			A1 = A[:(p0 * p1)].copy().reshape(p1, p0)
			iX1, iX2, iX3 = 0, p0 ** 2, p0 ** 2 + p1 ** 2
			X0, X1 = X[:, iX1:iX2].copy().reshape(-1, p0, p0), X[:, iX2:iX3].copy().reshape(-1, p1, p1)
			n_count = int(2 ** (p0 * (p0 - 1) / 2))
			for n in range(X.shape[0]):
				log_Xn0_post = np.zeros(n_count)
				for val in range(n_count):
					Xn0 = decode_am(val, p0)
					log_Xn0_post[val] = np.log(nu[val]) + log_posterior_Xn0(p1, C1, Xn0 * Gamma1, A1, X1[n], mask_k)
				log_Xn0_post -= log_Xn0_post.max()
				Xn0_post = np.exp(log_Xn0_post * alpha)
				Xn0_post_cum = np.cumsum(Xn0_post / Xn0_post.sum())
				r = np.random.rand()
				for val in range(n_count):
					if Xn0_post_cum[val] > r:
						break
				Xn0 = decode_am(val, p0)
				X[n, iX1:iX2] = Xn0.reshape(1, -1).copy()
		# sample layers k = 1, 2, ..., K
		else:
			pk0, pk1, pk2 = p[k - 1], p[k], p[k + 1]
			Ck1, Ck2 = C[k - 1], C[k]
			iG1, iG2, iG3 = np.sum(p[:(k - 1)] ** 2), np.sum(p[:k] ** 2), np.sum(p[:(k + 1)] ** 2)
			Gammak1, Gammak2 = Gamma[iG1:iG2].copy().reshape(pk0, pk0), Gamma[iG2:iG3].copy().reshape(pk1, pk1)
			iA1, iA2, iA3 = np.sum(p[:(k - 1)] * p[1:k]), np.sum(p[:k] * p[1:(k + 1)]), np.sum(p[:(k + 1)] * p[1:(k + 2)])
			Ak1, Ak2 = A[iA1:iA2].copy().reshape(pk1, pk0), A[iA2:iA3].copy().reshape(pk2, pk1)
			iX1, iX2, iX3, iX4 = np.sum(p[:(k - 1)] ** 2), np.sum(p[:k] ** 2), np.sum(p[:(k + 1)] ** 2), np.sum(p[:(k + 2)] ** 2)
			Xk0, Xk1, Xk2 = X[:, iX1:iX2].copy().reshape(-1, pk0, pk0), X[:, iX2:iX3].copy().reshape(-1, pk1, pk1), X[:, iX3:iX4].copy().reshape(-1, pk2, pk2)
			Xk0_Gammak1, Xk1_Gammak2 = Xk0 * Gammak1.reshape(1, pk0, pk0), Xk1 * Gammak2.reshape(1, pk1, pk1)
			for i in range(p[k]):
				for j in range(i):
					for n in range(X.shape[0]):
						log_Xnkij_post = np.zeros(2)
						for Xnkij in range(2):
							Xk1[n, i, j], Xk1[n, j, i], Xk1_Gammak2[n, i, j], Xk1_Gammak2[n, j, i] = Xnkij, Xnkij, Xnkij * Gammak2[i, j], Xnkij * Gammak2[i, j]
							log_Xnkij_post[Xnkij] = log_posterior_Xnkij(pk2, Ck1, Ck2, Xk0_Gammak1[n], Xk1_Gammak2[n], Ak1, Ak2, Xk1[n], Xk2[n], i, j, mask_k)
						log_Xnkij_post -= np.max(log_Xnkij_post)
						Xnkij_post = np.exp(log_Xnkij_post * alpha)
						Xnkij_post /= np.sum(Xnkij_post)
						if np.random.rand() <= Xnkij_post[0]:
							Xk1[n, i, j], Xk1[n, j, i], Xk1_Gammak2[n, i, j], Xk1_Gammak2[n, j, i] = 0, 0, 0, 0
						else:
							Xk1[n, i, j], Xk1[n, j, i], Xk1_Gammak2[n, i, j], Xk1_Gammak2[n, j, i] = 1, 1, Gammak2[i, j], Gammak2[i, j]
			X[:, iX2:iX3] = Xk1.reshape(-1, pk1 ** 2)


@jit(nopython = True)
def Ak_row_cardinality(n, sparsity):
	"""
	n: p_{k - 1}, i.e. number of columns in A_k
	sparsity: maximum number of ones in each row of A_k, i.e. sparsity
	--------------------------------------------------
	count the cardinality of the possible set of A_k rows, i.e. {p_{k - 1} choose 1} + {p_{k - 1} choose 2} + ... + {p_{k - 1} choose sparsity} <- corresponding to row sum = 1, 2, ..., sparsity
	return: cardinality
	"""
	count, count_sum = 1, 0
	for i in range(sparsity):
		count *= (n - i) / (i + 1)
		count_sum += count
	return int(count_sum)


@jit(nopython = True)
def encode(arr, n, sparsity):
	"""
	arr: row of A_k, i.e. array of length n
	n: p_{k - 1}, i.e. number of columns in A_k
	sparsity: maximum number of ones in each row of A_k, i.e. sparsity
	--------------------------------------------------
	encode arr into a value between 0 ~ Ak_row_cardinality(n, sparsity) - 1
	return: encoded value
	"""
	ones_arr = np.arange(1, sparsity + 1)
	count_arr = np.ones(ones_arr.size)
	count = 1
	for i, ones in enumerate(ones_arr):
		count *= (n - i) / (i + 1)
		count_arr[i] = count
	n_ones = int(arr.sum())
	val = count_arr[:(n_ones - 1)].sum()
	comb = 1
	for i in range(n_ones - 1):
		comb *= (n - 1 - i) / (i + 1)
	n_ones_left = n_ones
	for i, a in enumerate(arr[:-1]):
		if a:
			comb *= (n_ones_left - 1) / (n - i - 1)
			n_ones_left -= 1
		else:
			val += comb
			comb *= (n - i - n_ones_left) / (n - i - 1)
	return val


@jit(nopython = True)
def decode(val, n, sparsity):
	"""
	val: encoded value of arr between 0 ~ Ak_row_cardinality(n, sparsity) - 1
	n: p_{k - 1}, i.e. number of columns in A_k
	sparsity: maximum number of ones in each row of A_k, i.e. sparsity
	--------------------------------------------------
	decode val back to arr: row of A_k, i.e. array of length n
	return: arr
	"""
	ones_arr = np.arange(1, sparsity + 1)
	count_arr = np.ones(ones_arr.size)
	count = 1
	for i, ones in enumerate(ones_arr):
		count *= (n - i) / (i + 1)
		count_arr[i] = count
	for n_ones in range(ones_arr.size):
		if val < count_arr[n_ones]:
			break
		val -= count_arr[n_ones]
	n_ones += 1
	comb = 1
	for i in range(n_ones - 1):
		comb *= (n - 1 - i) / (i + 1)
	n_ones_left = n_ones
	arr = np.zeros(n)
	for i in range(n - 1):
		if val < comb:
			arr[i] = 1
			comb *= (n_ones_left - 1) / (n - i - 1)
			n_ones_left -= 1
		else:
			val -= comb
			comb *= (n - i - n_ones_left) / (n - i - 1)
	arr[-1] = n_ones_left
	return arr


@jit(nopython = True)
def log_posterior_Aki(pk, Ck, Xk1_Gammak, Ak, Xk, i, mask):
	"""
	pk: number of nodes p_k at layer k
	Ck: parameter C_k
	Xk1_Gammak: entrywise product of X_{k - 1} and Gamma_k - array of shape N x p_{k - 1} x p_{k - 1}
	Ak: parameter A_k
	Xk: adjacency matrices at layer k - array of shape N x p_k x p_k
	i: row i between 0 ~ p_k - 1 in A_k
	mask: p_k x p_k mask matrix on observed adjacency matrix (i.e. only applicable if k = K) - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	--------------------------------------------------
	return: log posterior full conditional of a_{k, i}, i.e. ith row of A_k
	"""
	log_post = 0
	for j in range(pk):
		if mask[i, j] < 0.5:
			continue
		if j < i:
			l = i * (i - 1) // 2 + j
		elif i == j:
			continue
		else:
			l = j * (j - 1) // 2 + i
		inner_prod = np.ones(Xk1_Gammak.shape[0]) * Ck
		for p in np.where(Ak[i])[0]:
			for q in np.where(Ak[j])[0]:
				inner_prod += Xk1_Gammak[:, p, q]
		log_post += np.sum(Xk[:, i, j] * inner_prod - np.log(1 + np.exp(inner_prod)))
	return log_post


@jit(nopython = True)
def sample_Aki_Gibbs(pk1, pk, Ck, Gammak, Ak, Xk1, Xk, i, sparsity, lower_bound, alpha, mask, seed):
	"""
	pk1: number of nodes p_{k - 1} at layer k - 1
	pk: number of nodes p_k at layer k
	Ck: parameter C_k
	Gammak: parameter Gamma_k
	Ak: parameter A_k
	Xk1: adjacency matrices at layer k - 1 -> array of shape N x p_{k - 1} x p_{k - 1}
	Xk: adjacency matrices at layer k - array of shape N x p_k x p_k
	i: row i between 0 ~ p_k - 1 in A_k
	sparsity: maximum number of ones in each row of A_k, i.e. sparsity
	lower_bound: binary array of length p_{k - 1}, all its one entries are enforced in a_{k, i}, i.e. we only consider a_{k, i} >= lower_bound entrywise
	alpha: raising full conditional to power alpha, can be used for simulated annearling or for subsampling Gibbs sampler
	mask: p_k x p_k mask matrix on observed adjacency matrix (i.e. only applicable if k = K) - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	seed: random state
	--------------------------------------------------
	sample a_{k, i} blockwise, i.e. ith row of A_k
	new sample of a_{k, i} is updated in place of A_k
	"""
	np.random.seed(seed)
	Xk1_Gammak = Xk1 * Gammak.reshape(1, *Gammak.shape)
	n_count = Ak_row_cardinality(pk1, sparsity)
	log_Ak_post = np.zeros(n_count)
	for val in range(n_count):
		arr = decode(val, pk1, sparsity)
		if (arr < lower_bound).any():
			log_Ak_post[val] = -np.inf
		else:
			Ak_ = Ak.copy()
			Ak_[i, :] = arr.copy()
			log_Ak_post[val] = log_posterior_Aki(pk, Ck, Xk1_Gammak, Ak_, Xk, i, mask) * alpha
	log_Ak_post -= np.max(log_Ak_post)
	Ak_post = np.exp(log_Ak_post)
	Ak_post_cum = np.cumsum(Ak_post / np.sum(Ak_post))
	r = np.random.rand()
	for val in range(n_count):
		if Ak_post_cum[val] > r:
			break
	arr = decode(val, pk1, sparsity)
	Ak[i, :] = arr.copy()


@jit(nopython = True)
def sample_A_Gibbs(p, C, Gamma, A, X, sparsity, min_n_nodes, alpha, mask, seed_arr):
	"""
	p: number of nodes [p_0, p_1, ..., p_K]
	C: parameter [None, C_1, ..., C_K]
	Gamma: parameter [None, Gamma_1, ..., Gamma_K]
	A: parameter [None, A_1, ..., A_K]
	X: list of adjacency matrices at each layer: [X_0, X_1, ..., X_K], each X_k of shape N x p_k x p_k
	sparsity: maximum number of ones in each row of A_k, i.e. sparsity
	min_n_nodes: minimum number of nodes required for each community, i.e. minimum number of ones in each column of A
	alpha: raising full conditional to power alpha, can be used for simulated annearling or for subsampling Gibbs sampler
	mask: p_K x p_K mask matrix on observed adjacency matrix - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	seed: an array of random state
	--------------------------------------------------
	for each layer k = 1, ..., K
		sample each row a_{k, i} blockwise using sample_Aki
	new sample of A is updated in place of A
	"""
	i_seed = 0
	for k in range(1, p.size):
		Ck =  C[k - 1]
		iG1, iG2 = np.sum(p[:(k - 1)] ** 2), np.sum(p[:k] ** 2)
		Gammak = Gamma[iG1:iG2].reshape(p[k - 1], p[k - 1])
		iA1, iA2 = np.sum(p[1:k] * p[:(k - 1)]), np.sum(p[1:(k + 1)] * p[:k])
		Ak = A[iA1:iA2].reshape(p[k], p[k - 1])
		iX1, iX2, iX3 = np.sum(p[:(k - 1)] ** 2), np.sum(p[:k] ** 2), np.sum(p[:(k + 1)] ** 2)
		Xk1, Xk = X[:, iX1:iX2].copy().reshape(-1, p[k - 1], p[k - 1]), X[:, iX2:iX3].copy().reshape(-1, p[k], p[k])
		mask_k = np.ones((p[k], p[k])) if k < p.size - 1 else mask
		for i in range(p[k]):
			lower_bound = ((Ak.sum(axis = 0) - Ak[i]) < min_n_nodes - 0.5).astype(np.float64)
			sample_Aki_Gibbs(p[k - 1], p[k], Ck, Gammak, Ak, Xk1, Xk, i, sparsity, lower_bound, alpha, mask_k, seed_arr[i_seed])
			i_seed += 1


@jit(nopython = True)
def log_likelihood_0(p0, nu, X0):
	"""
	p0: number of nodes p_0 at layer 0
	nu: parameter nu
	X0: adjacency matrices X_0 at layer 0 of shape N x p_0 x p_0
	--------------------------------------------------
	return log likelihood of X_0, i.e. log P(X_0 | nu)
	"""
	count_arr = np.zeros(int(2 ** (p0 * (p0 - 1) / 2)))
	for n in range(X0.shape[0]):
		count_arr[int(encode_am(X0[n], p0))] += 1
	return np.sum(np.log(nu) * count_arr)


@jit(nopython = True)
def log_likelihood_k(pk1, pk, Ck, Gammak, Ak, Xk1, Xk, mask):
	"""
	pk1: number of nodes p_{k - 1} at layer k - 1
	pk: number of nodes p_k at layer k
	Ck: parameter C_k
	Gammak: parameter Gamma_k
	Ak: parameter A_k
	Xk1: adjacency matrices X_{k - 1} at layer k - 1 of shape N x p_{k - 1} x p_{k - 1}
	Xk: adjacency matrices X_k at layer k of shape N x p_k x p_k
	mask: p_k x p_k mask matrix on observed adjacency matrix (i.e. only applicable if k = K) - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	--------------------------------------------------
	return log likelihood of X_k, i.e. log P(X_k | X_{k - 1}, A_k, C_k, Gamma_k) <- note: only for entries after masking
	"""
	log_lik = 0
	Xk1_Gammak = Xk1 * Gammak.reshape(1, pk1, pk1)
	for i in range(pk):
		for j in range(i):
			if mask[i, j] < 0.5:
				continue
			inner_prod = np.ones(Xk.shape[0]) * Ck
			for p in np.where(Ak[i])[0]:
				for q in np.where(Ak[j])[0]:
					inner_prod += Xk1_Gammak[:, p, q]
			log_lik += (Xk[:, i, j] * inner_prod - np.log(1 + np.exp(inner_prod))).sum()
	return log_lik


@jit(nopython = True)
def log_likelihood_0_each(p0, nu, X0):
	"""
	p0: number of nodes p_0 at layer 0
	nu: parameter nu
	X0: adjacency matrices X_0 at layer 0 of shape N x p_0 x p_0
	--------------------------------------------------
	return an array of length N containing log likelihood of each X_0^{(n)}, i.e. array [log P(X_0^{(1)} | nu), ..., log P(X_0^{(N)} | nu)]
	"""	
	log_lik_0_arr = np.zeros(X0.shape[0])
	for n in range(X0.shape[0]):
		log_lik_0_arr[n] = np.log(nu[int(encode_am(X0[n], p0))])
	return log_lik_0_arr


@jit(nopython = True)
def log_likelihood_k_each(pk1, pk, Ck, Gammak, Ak, Xk1, Xk, mask):
	"""
	pk1: number of nodes p_{k - 1} at layer k - 1
	pk: number of nodes p_k at layer k
	Ck: parameter C_k
	Gammak: parameter Gamma_k
	Ak: parameter A_k
	Xk1: adjacency matrices X_{k - 1} at layer k - 1 of shape N x p_{k - 1} x p_{k - 1}
	Xk: adjacency matrices X_k at layer k of shape N x p_k x p_k
	mask: p_k x p_k mask matrix on observed adjacency matrix (i.e. only applicable if k = K) - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	--------------------------------------------------
	return an array of length N containing log likelihood of each X_k^{(n)}, i.e. array [log P(X_k^{(1)} | X_{k - 1}, A_k, C_k, Gamma_k), ..., log P(X_k^{(N)} | X_{k - 1}, A_k, C_k, Gamma_k)] <- note: only for entries after masking
	"""
	log_lik_k_arr = np.zeros(Xk.shape[0])
	Xk1_Gammak = Xk1 * Gammak.reshape(1, pk1, pk1)
	for i in range(pk):
		for j in range(i):
			if mask[i, j] < 0.5:
				continue
			inner_prod = np.ones(Xk.shape[0]) * Ck
			for p in np.where(Ak[i])[0]:
				for q in np.where(Ak[j])[0]:
					inner_prod += Xk1_Gammak[:, p, q]
			log_lik_k_arr += Xk[:, i, j] * inner_prod - np.log(1 + np.exp(inner_prod))
	return log_lik_k_arr


def convert_theta_to_C_Gamma(p, theta):
	"""
	p: list of number of nodes at each layer [p_0, p_1, ..., p_K]
	theta: list of arrays [None, theta_1, ..., theta_k]
	each theta_k of length (1 + p_{k - 1} (p_{k - 1} + 1) / 2) containing continuous parameters at layer k, i.e. [C_k, Gamma_{k, 1, 1}, Gamma_{k, 2, 1}, Gamma_{k, 2, 2}, Gamma_{k, 3, 1}, ..., Gamma_{k, p_{k - 1}, p_{k - 1}}]
	--------------------------------------------------
	decompress theta back to C, Gamma
	return C, Gamma
	C: list [None, C_1, ..., C_K]
	Gamma: list [None, Gamma_1, ..., Gamma_K]
	"""
	C = [None]
	Gamma = [None]
	for k in range(1, len(p)):
		C.append(theta[k][0])
		Gamma_k = np.zeros((p[k - 1], p[k - 1]))
		l = 1
		for i in range(p[k - 1]):
			for j in range(i + 1):
				Gamma_k[i][j] = theta[k][l]
				Gamma_k[j][i] = Gamma_k[i][j]
				l += 1
		Gamma.append(Gamma_k)
	return C, Gamma


class GibbsSampler():
	def __init__(self, X_K, p, sparsity = 2, min_n_nodes = 2, mask = None, seed = None):
		"""
		X_K: np.ndarray of shape N x p_K x p_K - the observed p_K x p_K adjacency matrices of all N subjects
		p: list of number of nodes at each layer [p_0, p_1, ..., p_K]
		sparsity: sparsity parameter, i.e. maximum number of ones allowed in each row of connection matrix A_k
		min_n_nodes: minimum number of nodes required for each community, i.e. minimum number of ones in each column of A, default to 2 by generic identifiability condition
		mask: p_K x p_K mask matrix on observed adjacency matrix - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
		seed: random state
		--------------------------------------------------
		initialize gibbs sampler
			- model structure
			- spectral initialization of connection matrix A
			- random initialization of parameter theta
			- fixed initialization of parameter nu
			- random initialization of latent adjacency matrices X_0, ..., X_{K - 1}
			- list of gibbs samples
		"""
		# initialize model structure
		self.X_K = X_K
		self.N = len(X_K)
		self.K = len(p) - 1
		self.p = p
		self.sparsity = sparsity
		self.min_n_nodes = min_n_nodes
		self.mask = np.ones((p[-1], p[-1])) if mask is None else mask
		self.seed_gen = random_seed_generator(np.random.randint(0, 1e8) if seed is None else seed)

		# initialize connection matrix A
		self.A = spectral_initialization(X_K.mean(axis = 0), p, sparsity, min_n_nodes)

		# initialize parameter theta from its prior
		self.theta_min = [None] + [np.array([-np.inf] + [(2.0 if i == j else 1.0) for i in range(p[k - 1]) for j in range(i + 1)]) for k in range(1, self.K + 1)]
		self.theta_max = [None] + [np.array([-2.0] + [np.inf for i in range(p[k - 1]) for j in range(i + 1)]) for k in range(1, self.K + 1)]
		self.theta_prior_mean = [None] + [np.array([-7.0] + [(10.0 if i == j else 4.0) for i in range(p[k - 1]) for j in range(i + 1)]) for k in range(1, self.K + 1)]
		self.theta_prior_precision = [None] + [np.array([0.25] + [0.25 for i in range(p[k - 1]) for j in range(i + 1)]) for k in range(1, self.K + 1)]
		self.theta = [None] + [np.array([truncnorm(loc = m, scale = 1 / c ** 0.5, a = (a - m) * c ** 0.5, b = (b - m) * c ** 0.5).rvs(random_state = next(self.seed_gen)) for (m, c, a, b) in zip(self.theta_prior_mean[k], self.theta_prior_precision[k], self.theta_min[k], self.theta_max[k])]) for k in range(1, self.K + 1)]

		# placeholder for augmented variable omega
		self.omega = None

		# initialize parameter nu
		self.nu_prior = np.ones(2 ** (p[0] * (p[0] - 1) // 2))
		self.nu = np.ones(2 ** (p[0] * (p[0] - 1) // 2)) / 2 ** (p[0] * (p[0] - 1) // 2)

		# initialize latent adjacency matrices X_0, ..., X_{K - 1}
		np.random.seed(next(self.seed_gen))
		self.X = []
		for k in range(self.K):
			self.X.append(np.stack([np.identity(p[k]) for _ in range(self.N)]))
			X_k_ber = np.random.binomial(1, 0.5 ** 0.5, size = self.N * p[k] ** 2).reshape(self.N, p[k], p[k])
			X_k_ber *= np.transpose(X_k_ber, (0, 2, 1))
			self.X[k].reshape(-1)[X_k_ber.reshape(-1).astype(bool)] = 1
		self.X.append(X_K)

		# list of gibbs samples - posterior sample at each iteration will be saved here
		self.samples = []

	@timeit
	def update_log_posterior(self, subset):
		"""
		subset: an array subset of [1, ..., N] containing the samples used in this iteration of subsampling Gibbs sampler
		--------------------------------------------------
		compute log posterior, i.e. log joint density of everything (parameters & adjacency matrices)
		log likelihood of subset is rescaled by factor of N / subset size
		update is made in place on self.log_post
		"""
		self.update_C_Gamma()
		log_post = - 0.5 * np.sum([np.sum((self.theta[k] - self.theta_prior_mean[k]) ** 2 * self.theta_prior_precision[k]) for k in range(1, self.K + 1)]) + log_likelihood_0(self.p[0], self.nu, self.X[0][subset]) * self.N / subset.size
		for k in range(1, self.K + 1):
			mask_k = np.ones((self.p[k], self.p[k])) if k < self.K else self.mask
			log_post += log_likelihood_k(self.p[k - 1], self.p[k], self.C[k], self.Gamma[k], self.A[k], self.X[k - 1][subset], self.X[k][subset], mask_k) * self.N / subset.size
		self.log_post = log_post

	@timeit
	def return_log_posterior(self, A = None, theta = None, nu = None, X = None):
		"""
		A: parameter A
		theta: parameter theta
		nu: parameter nu
		X: adjacency matrices at all layers [X_0, ..., X_K]
		--------------------------------------------------
		compute log posterior as in function self,update_log_posterior, but allow A, theta, nu, X be inputted from exterior, and return log posterior
		"""
		if A is None:
			A = self.A
		if theta is None:
			theta = self.theta
		if nu is None:
			nu = self.nu
		if X is None:
			X = self.X
		C, Gamma = convert_theta_to_C_Gamma(self.p, theta)
		log_post = - 0.5 * np.sum([np.sum((theta[k] - self.theta_prior_mean[k]) ** 2 * self.theta_prior_precision[k]) for k in range(1, self.K + 1)]) + log_likelihood_0(self.p[0], nu, X[0])
		for k in range(1, self.K + 1):
			mask_k = np.ones((self.p[k], self.p[k])) if k < self.K else self.mask
			log_post += log_likelihood_k(self.p[k - 1], self.p[k], C[k], Gamma[k], A[k], X[k - 1], X[k], mask_k)
		return log_post

	@timeit
	def sample_A(self, subset, alpha):
		"""
		subset: an array subset of [1, ..., N] containing the samples used in this iteration of subsampling Gibbs sampler
		alpha: raising full conditional to power alpha, can be used for simulated annearling or for subsampling Gibbs sampler
		--------------------------------------------------
		sample parameter A using function sample_A_Gibbs
		update A in place on self.A
		"""
		p_arr = np.array(self.p)
		C_arr = np.array(self.C[1:])
		Gamma_arr = np.hstack([Gammak.reshape(-1) for Gammak in self.Gamma[1:]])
		A_arr = np.hstack([Ak.reshape(-1) for Ak in self.A[1:]])
		X_arr = np.hstack([Xk.reshape(self.N, -1) for Xk in self.X])
		X_arr_subset = X_arr[subset, :]
		np.random.seed(next(self.seed_gen))
		sample_A_Gibbs(p_arr, C_arr, Gamma_arr, A_arr, X_arr_subset, self.sparsity, self.min_n_nodes, alpha, self.mask, np.random.randint(0, 1e8, 10000))
		self.A = [None] + [A_arr[np.sum(p_arr[1:k] * p_arr[:(k - 1)]):np.sum(p_arr[1:(k + 1)] * p_arr[:k])].reshape(p_arr[k], p_arr[k - 1]) for k in range(1, self.K + 1)]

	@timeit
	def sample_omega(self, subset, alpha):
		"""
		subset: an array subset of [1, ..., N] containing the samples used in this iteration of subsampling Gibbs sampler
		alpha: raising full conditional to power alpha, can be used for simulated annearling or for subsampling Gibbs sampler
		--------------------------------------------------
		sample augmented variable omega
		update omega in place on self.omega
		"""
		self.omega = [None]
		for k in range(1, self.K + 1):
			self.omega.append(random_polyagamma(alpha, (self.Z[k] @ self.theta[k]).flatten(), random_state = next(self.seed_gen)))

	@timeit
	def sample_nu(self, subset, alpha):
		"""
		subset: an array subset of [1, ..., N] containing the samples used in this iteration of subsampling Gibbs sampler
		alpha: raising full conditional to power alpha, can be used for simulated annearling or for subsampling Gibbs sampler
		--------------------------------------------------
		sample parameter nu
		update nu in place on self.nu
		"""
		np.random.seed(next(self.seed_gen))
		nu_post = self.nu_prior + count_X0(self.p[0], self.X[0][subset]) * alpha
		nu = np.random.dirichlet(nu_post)
		self.nu = deepcopy(nu)

	@timeit
	def sample_theta(self, subset, alpha):
		"""
		subset: an array subset of [1, ..., N] containing the samples used in this iteration of subsampling Gibbs sampler
		alpha: raising full conditional to power alpha, can be used for simulated annearling or for subsampling Gibbs sampler
		--------------------------------------------------
		sample parameter theta
		update theta in place on self.theta
		"""
		np.random.seed(next(self.seed_gen))
		theta = [None]
		for k in range(1, self.K + 1):
			theta_k = self.theta[k].copy()
			WZTheta_k = self.omega[k] * (self.Z[k] @ theta_k)
			for l in range(theta_k.size):
				u = np.random.rand()
				# C_k
				if l == 0:
					var = 1 / (self.theta_prior_precision[k][l] + np.sum(self.omega[k]))
					mean = var * (self.theta_prior_precision[k][l] * self.theta_prior_mean[k][l] + alpha * np.sum(self.kappa[k]) - np.sum(WZTheta_k - self.omega[k] * self.Z[k][:, 0] * theta_k[0]))
				# Gamma_{k, i, j}
				else:
					var = 1 / (self.theta_prior_precision[k][l] + np.sum(self.Z[k][:, l] ** 2 * self.omega[k]))
					mean = var * (self.theta_prior_precision[k][l] * self.theta_prior_mean[k][l] + np.dot(self.Z[k][:, l], alpha * self.kappa[k] - (WZTheta_k - self.omega[k] * self.Z[k][:, l] * theta_k[l])))
				rescaled_a, rescaled_b = (self.theta_min[k][l] - mean) / var ** 0.5, (self.theta_max[k][l] - mean) / var ** 0.5
				# for truncated normal distribution with standard lower/upper bounds, use inverse cdf for sampling
				if rescaled_b >= -6 and rescaled_a <= 6:
					u = np.random.rand()
					Fa, Fb = norm.cdf(rescaled_a), norm.cdf(rescaled_b)
					theta_k[l] = mean + var ** 0.5 * norm.ppf(Fa + u * (Fb - Fa))
				# for truncated normal distribution with non-standard lower/upper bounds, i.e. higher tail probability accuracy required, use function truncnorm for sampling
				else:
					theta_k[l] = truncnorm(loc = mean, scale = var ** 0.5, a = rescaled_a, b = rescaled_b).rvs(random_state = next(self.seed_gen))
				WZTheta_k += self.omega[k] * self.Z[k][:, l] * (theta_k[l] - self.theta[k][l])
			theta.append(theta_k)
		self.theta = deepcopy(theta)

	@timeit
	def sample_X(self, subset, alpha):
		"""
		subset: an array subset of [1, ..., N] containing the samples used in this iteration of subsampling Gibbs sampler
		alpha: raising full conditional to power alpha, can be used for simulated annearling or for subsampling Gibbs sampler
		--------------------------------------------------
		sample latent adjacency matrices X using function sample_X_Gibbs
		update X in place on self.X
		"""
		p_arr = np.array(self.p)
		nu_arr = self.nu
		C_arr = np.array(self.C[1:])
		Gamma_arr = np.hstack([Gammak.reshape(-1) for Gammak in self.Gamma[1:]])
		A_arr = np.hstack([Ak.reshape(-1) for Ak in self.A[1:]])
		X_arr = np.hstack([Xk.reshape(self.N, -1) for Xk in self.X])
		X_arr_subset = X_arr[subset, :]
		sample_X_Gibbs(p_arr, nu_arr, C_arr, Gamma_arr, A_arr, X_arr_subset, alpha, self.mask, next(self.seed_gen))
		X_arr[subset, :] = X_arr_subset
		for k in range(self.K + 1):
			self.X[k] = X_arr[:, np.sum(p_arr[:k] ** 2):np.sum(p_arr[:(k + 1)] ** 2)].reshape(self.N, self.p[k], self.p[k])

	@timeit
	def save_param(self, subset):
		"""
		subset: an array subset of [1, ..., N] containing the samples used in this iteration of subsampling Gibbs sampler
		--------------------------------------------------
		save everything sampled into Parameter class
		append param into self.samples
		"""
		param = Parameter()
		param.A = deepcopy(self.A)
		param.theta = deepcopy(self.theta)
		param.nu = deepcopy(self.nu)
		param.subset = subset.copy()
		param.X_subset = deepcopy([Xk[subset] for Xk in self.X[:-1]])
		param.omega = deepcopy(self.omega)
		param.log_post = self.log_post
		self.samples.append(param)

	@timeit
	def update_C_Gamma(self):
		"""
		decompress self.theta back to self.C, self.Gamma
		self.C: list [None, C_1, ..., C_K]
		self.Gamma: list [None, Gamma_1, ..., Gamma_K]
		update in place
		"""
		C = [None]
		Gamma = [None]
		for k in range(1, self.K + 1):
			C.append(self.theta[k][0])
			Gamma_k = np.zeros((self.p[k - 1], self.p[k - 1]))
			l = 1
			for i in range(self.p[k - 1]):
				for j in range(i + 1):
					Gamma_k[i][j] = self.theta[k][l]
					Gamma_k[j][i] = Gamma_k[i][j]
					l += 1
			Gamma.append(Gamma_k)
		self.C = C
		self.Gamma = Gamma

	@timeit
	def update_Z_kappa(self, subset):
		"""
		subset: an array subset of [1, ..., N] containing the samples used in this iteration of subsampling Gibbs sampler
		--------------------------------------------------
		after sampling new adjacency matrices X and parameters A, use this function to update Z and kappa accordingly
		Z: list [None, Z_1, ..., Z_K]
			each Z_k: array of shape (N p_k (p_k - 1) / 2) x (1 + p_{k - 1} (p_{k - 1} + 1) / 2)
			Z_k is an expression of A_k, X_{k - 1} satisfyting <Z_{k, i, j}^{(n)}, theta_k> = C_k + a_{k, i}^T (Gamma_k * X_{k - 1}^{(n)}) a_{k, j}
			the indices n, i, j are collapsed into the first dimension of Zk
		kappa: list [None, kappa_1, ..., kappa_K]
			kappa_k: array of length N p_k (p_k - 1) / 2
			kappa_{k, i, j}^{(n)} = X_{k, i, j}^{(n)} - 1 / 2
			the indices n, i, j are collapsed into one dimension in kappa_k
		use function update_Z_kappa_k
		updates are made in place on self.Z and self.kappa
		"""
		self.Z = [None]
		self.kappa = [None]
		for k in range(1, self.K + 1):
			mask_k = np.ones((self.p[k], self.p[k])) if k < self.K else self.mask
			length_k = int((np.sum(mask_k) - np.trace(mask_k)) // 2)
			Zk = np.ones((length_k * subset.size, 1 + self.p[k - 1] * (self.p[k - 1] + 1) // 2))
			kappak = np.zeros(length_k * subset.size)
			update_Z_kappa_k(self.p[k - 1], self.p[k], subset.size, self.X[k - 1][subset], self.X[k][subset], self.A[k], Zk, kappak, mask_k)
			self.Z.append(Zk)
			self.kappa.append(kappak)

	def sample(self, subset_proportion = None, fix = [], alpha = 1):
		"""
		subset_proportion: ratio of subsampling subset size to full sample size - None represents using full sample size, values between (0, 1) are handled by sampling without replacement
		fix: a list potentially containing ['X', 'nu', 'theta', 'A'] for which the contained parameter/latent variable will be fixed (i.e. skipped in Gibbs sampling) - typically used for debugging
		alpha: raising posterior full conditional to power alpha - can be used for simulated annealing, default to 1
		--------------------------------------------------
		conduct on iteration of Gibbs sampling, sequentially
			- sample subset of data
			- sample latent adjacency matrices X
			- sample parameter nu
			- sample augmented variable omega and parameter theta
			- sample parameter A
			- compute log posterior
			- save everything into self.samples
		return: an instance of Parameter class containing everything sampled
		"""
		np.random.seed(next(self.seed_gen))
		if subset_proportion is None:
			subset = np.arange(self.N)
		else:
			subset = np.random.choice(self.N, int(subset_proportion * self.N), replace = False)
		self.update_C_Gamma()
		if "X" not in fix:
			self.sample_X(subset, alpha)
		alpha *= (self.N / subset.size)
		self.update_Z_kappa(subset)
		if "nu" not in fix:
			self.sample_nu(subset, alpha)
		if "theta" not in fix:
			self.sample_omega(subset, alpha)
			self.sample_theta(subset, alpha)
		if "A" not in fix:
			self.sample_A(subset, alpha)
		self.update_log_posterior(subset)
		self.save_param(subset)
		return self.samples[-1]

	def write(self, param, path):
		"""
		param: instance of Parameter class - containing everything sampled in one iteration
		path: a file path
		--------------------------------------------------
		save param to path as a pickle file
		"""
		with open(path, "wb") as hf:
			pickle.dump(param, hf)
