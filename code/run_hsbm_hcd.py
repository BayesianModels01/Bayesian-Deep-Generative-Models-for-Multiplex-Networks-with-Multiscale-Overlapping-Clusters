import numpy as np
import numpy.linalg as nlg
import scipy.linalg as slg
from sklearn.cluster import KMeans
from collections import deque
import os
from tqdm import tqdm
import warnings


def eigenvector_sign_check_partition(A):
	"""
	A: np.ndarray of shape p x p - symmetric, binary adjacency matrix
	partition p nodes into two communities by eigenvector sign check
	see Algo 1 in https://arxiv.org/pdf/1810.01509
	"""
	eigval, eigvec = nlg.eig(A)
	fiedler_vec = eigvec[:, np.argsort(eigval.real)[-2]].real
	return (fiedler_vec < 0).astype(int)


def regularized_spectral_clustering_partition(A):
	"""
	A: np.ndarray of shape p x p - symmetric, binary adjacency matrix
	partition p nodes into two communities by regularized spectral clustering
	see Algo 2 in https://arxiv.org/pdf/1810.01509
	"""
	p = A.shape[0]
	d_ave = A.sum() / p
	A_reg = A + 0.1 * d_ave / p * np.ones((p, p))
	D_reg = A_reg.sum(axis = 1)
	Laplacian_reg = A_reg * np.outer(D_reg, D_reg) ** (-0.5)
	eigval, eigvec = nlg.eig(Laplacian_reg)
	top2_vecs = eigvec[:, np.argsort(eigval.real)[-2:]].real
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		km = KMeans(n_clusters = 2, random_state = 0).fit(top2_vecs)
	return km.labels_


def non_backtracking_stopping_criterion(A, **kwargs):
	"""
	A: np.ndarray of shape p x p - symmetric, binary adjacency matrix
	non-backtracking stopping criterion for partitioning the hierarchical community detection tree
	see Eq.(1) in https://arxiv.org/pdf/1810.01509
	"""
	if A.shape[0] <= 1:
		return True
	p = A.shape[0]
	D = A.sum(axis = 1)
	B_nb = np.block([[np.zeros((p, p)), np.diag(D) - np.identity(p)], [- np.identity(p), A]])
	B_nb_norm = (D ** 2).sum() / D.sum() - 1
	eigval = nlg.eigvals(B_nb)
	return (np.abs(eigval.real) > B_nb_norm ** 0.5 + 1e-6).sum() <= 1


def depth_stopping_criterion(A, **kwargs):
	"""
	A: np.ndarray of shape p x p - symmetric, binary adjacency matrix
	c: an interger with binary bit length representing the current depth
	stop when depth reaches max depth 5
	"""
	if A.shape[0] <= 1:
		return True
	c = kwargs['c']
	return len(bin(c)[2:]) >= 5


def hierarchical_community_detection(A, algo, max_communities = None):
	"""
	A: np.ndarray of shape p x p - symmetric, binary adjacency matrix
	algo: 'sign' or 'spec' - specify the partition function - eigenvector_sign_check_partition or regularized_spectral_clustering_partition
	recursively partition each community until stopping criterion is met
	max_communities: int - maximum number of communities allowed
	see https://arxiv.org/pdf/1810.01509
	return the community memberships of each node as a np.ndarray of size p
	"""
	p = A.shape[0]
	if algo == 'sign':
		partition_func = eigenvector_sign_check_partition
	elif algo == 'spec':
		partition_func = regularized_spectral_clustering_partition
	# alternatively, we can also use stopping_criterion = non_backtracking_stopping_criterion
	stopping_criterion = depth_stopping_criterion
	if max_communities is None:
		max_communities = p

	n_communities = 1
	communities = np.ones(p).astype(int)
	queue = deque([np.arange(p)])
	while len(queue) > 0:
		nodes = queue.popleft()
		c = communities[nodes[0]]
		A_sub = A[nodes, :][:, nodes]
		if not stopping_criterion(A_sub, c = c):
			partition_labels = partition_func(A_sub)
			nodes_0, nodes_1 = nodes[partition_labels == 0], nodes[partition_labels == 1]
			if len(nodes_0) < len(nodes_1):
				nodes_0, nodes_1 = nodes_1, nodes_0
			communities[nodes_0] = c * 2
			communities[nodes_1] = c * 2 + 1
			queue.extend([nodes_0, nodes_1])
			n_communities += 1
			if n_communities >= max_communities:
				break
	return communities


def simulate_hierarchical_stochastic_block_model(N, seed = None):
	"""
	simulate N adjacency matrices sampled from a hierarchical stochastic block model with community tree structure of 1 - 3 - 9 - 27 nodes
	"""
	if seed is not None:
		np.random.seed(seed)
	shared_tree_depth = slg.block_diag(*[np.ones((9, 9))] * 3) + slg.block_diag(*[np.ones((3, 3))] * 9) + np.identity(27)
	prob_mat = np.identity(27)
	for i in range(27):
		for j in range(i):
			if shared_tree_depth[i, j] == 2:
				prob_mat[i, j] = np.random.uniform(0.7, 0.8)
			elif shared_tree_depth[i, j] == 1:
				prob_mat[i, j] = np.random.uniform(0.4, 0.5)
			else:
				prob_mat[i, j] = np.random.uniform(0, 0.1)
			prob_mat[j, i] = prob_mat[i, j]
	rand_arr = np.random.rand(N, 27, 27)
	for i in range(27):
		for j in range(i):
			rand_arr[:, i, j] = rand_arr[:, j, i].copy()
	return (rand_arr <= prob_mat.reshape(1, 27, 27)).astype(int)


# simulation settings
directory = f"hsbm_hcd"
os.mkdir(directory)
for N in [10, 20, 30, 40, 50]:
	# simulate data
	hsbm_networks = simulate_hierarchical_stochastic_block_model(N, seed = 0)
	print(f'Adjacency matrices with N = {N} simulated from hierarchical stochastic block model.')

	# apply HCD-Sign to each simulated adjacency matrix to obtain 3 or 9 communities
	sign_communities_3 = np.column_stack([
		hierarchical_community_detection(A, algo = 'sign', max_communities = 3)
		for A in tqdm(hsbm_networks)
	])
	sign_communities_9 = np.column_stack([
		hierarchical_community_detection(A, algo = 'sign', max_communities = 9)
		for A in tqdm(hsbm_networks)
	])

	# apply HCD-Spec to each simulated adjacency matrix to obtain 3 or 9 communities
	spec_communities_3 = np.column_stack([
		hierarchical_community_detection(A, algo = 'spec', max_communities = 3)
		for A in tqdm(hsbm_networks)
	])
	spec_communities_9 = np.column_stack([
		hierarchical_community_detection(A, algo = 'spec', max_communities = 9)
		for A in tqdm(hsbm_networks)
	])

	np.savez(os.path.join(directory, f'N_{N}.npz'), sign_algo_n_communities_3 = sign_communities_3, sign_algo_n_communities_9 = sign_communities_9, spec_algo_n_communities_3 = spec_communities_3, spec_algo_n_communities_9 = spec_communities_9)
	print(f"All communities saved to folder {directory}/N_{N}.npz.")
