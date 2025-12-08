import numpy as np
from utils import *


def transform_Gamma(K, p, gamma, delta):
	"""
	(K + 1): number of layers
	p: list of number of nodes p_k at each layer
	gamma: list of diagonal entry values of each Gamma_k
	delta: list of off-diagonal entry values of each Gamma_k
	--------------------------------------------------
	transform diag, off-diag values into Gamma: list of Gamma_k (Gamma_0 is None) - each a np.ndarray of shape (p_{k - 1}, p_{k - 1})
	"""
	Gamma = [None]
	for k in range(1, K + 1):
		# transform GAmma_k of layer k
		Gammak = np.identity(p[k - 1]) * (gamma[k] - delta[k]) + np.ones((p[k - 1], p[k - 1])) * delta[k]
		Gamma.append(Gammak)
	return Gamma


def simulate_A(K, p, A_row_sum_max, seed = None):
	"""
	(K + 1): number of layers
	p: list of number of nodes p_k at each layer
	A_row_sum_max: max of the sum of each row in A_k, i.e. the sparsity parameter S
	seed: an integer / None - random state
	--------------------------------------------------
	simulate A: list of A_k (A_0 is None) - each a np.ndarray of shape (p_k, p_{k - 1})
	"""
	if seed is not None:
		np.random.seed(seed)
	A = [None]
	for k in range(1, K + 1):
		# simulate A_k of layer k, ensuring top p_{k - 1} x p_{k - 1} block is identity and next p_{k - 1} x p_{k - 1} block has all-ones diagonal
		Ak = np.vstack([np.identity(p[k - 1]), np.identity(p[k - 1]), np.zeros((p[k] - 2 * p[k - 1], p[k - 1]))])
		for i in range(p[k - 1], p[k - 1] * 2):
			Ak[i, np.random.choice(np.append(np.arange(i - p[k - 1]), np.arange(i - p[k - 1] + 1, p[k - 1])), np.random.randint(0, A_row_sum_max))] = 1
		for i in range(p[k - 1] * 2, p[k]):
			Ak[i, np.random.choice(np.arange(p[k - 1]), np.random.randint(1, A_row_sum_max + 1))] = 1
		A.append(Ak)
	return A


def simulate_X(K, p, N, C, Gamma, A, PX0, seed = None):
	"""
	(K + 1): number of layers
	p: list of number of nodes p_k at each layer
	N: number of samples
	Gamma: list of Gamma_k
	A: list of A_k
	C: list of C_k
	PX0: P(X_{0, i, j} = 1) for off-diagonal X_0 entries i != j
	seed: an integer / None - random state
	--------------------------------------------------
	simulate X: list of X_k - each a np.ndarray of shape (N, p_k, p_k)
	"""
	if seed is not None:
		np.random.seed(seed)
	X = [np.stack([np.identity(p[k]) for _ in range(N)]) for k in range(K + 1)]
	for k in range(K + 1):
		# simulate X_k of layer k from top to bottom
		PXk = (1 / (1 + np.exp(- C[k] - A[k] @ (X[k - 1] * Gamma[k]) @ A[k].T))).reshape(-1) if k > 0 else PX0
		X[k].reshape(-1)[np.random.rand(N * p[k] ** 2) < PXk ** 0.5] = 1
		X[k] *= np.transpose(X[k], (0, 2, 1))
	return X


@timeit
def simulate_all(K, p, gamma, delta, C, PX0, N, A_row_sum_max = 2, seed = None):
	"""
	(K + 1): number of layers
	p: list of number of nodes p_k at each layer
	gamma: list of diagonal entry values of each Gamma_k
	delta: list of off-diagonal entry values of each Gamma_k
	C: list of C_k
	PX0: P(X_{0, i, j} = 1) for off-diagonal X_0 entries i != j
	N: number of samples
	A_row_sum_max: max of the sum of each row in A_k, i.e. the sparsity parameter S
	seed: an integer / None - random state
	--------------------------------------------------
	simulate all, including parameters A, theta and adjacency matrices X
	theta: list of theta_k (theta_0 is None) - each a np.ndarray of size 1 + p_{k - 1}(p_{k - 1} + 1) / 2
	"""
	# simulate Gamma, A, X
	seed_gen = random_seed_generator(seed)
	Gamma = transform_Gamma(K, p, gamma, delta)
	A = simulate_A(K, p, A_row_sum_max, seed = next(seed_gen))
	X = simulate_X(K, p, N, C, Gamma, A, PX0, seed = next(seed_gen))
	# merge parameters C, Gamma into theta - in the sequence C_k, Gamma_{k, 1, 1}, Gamma_{k, 2, 1}, Gamma_{k, 2, 2}, Gamma_{k, 3, 1}, ... for theta_k of layer k
	theta = [None]
	for k in range(1, K + 1):
		theta_k = [C[k]]
		for i in range(p[k - 1]):
			for j in range(i + 1):
				theta_k.append(delta[k] if i != j else gamma[k])
		theta.append(np.array(theta_k))
	return A, theta, X
