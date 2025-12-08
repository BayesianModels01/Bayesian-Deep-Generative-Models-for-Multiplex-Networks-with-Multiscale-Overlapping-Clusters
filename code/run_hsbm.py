import numpy as np
import scipy.linalg as slg
import os
from simulate import *
from sample import *


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
K = 2
p = [3, 9, 27]
sparsity = 1

for N in [10, 20, 30, 40, 50]:
	# simulate data
	hsbm_networks = simulate_hierarchical_stochastic_block_model(N, seed = 0)
	print(f'Adjacency matrices with N = {N} simulated from hierarchical stochastic block model.')

	# set up gibbs sampler and directory for saving posterior samples
	gibbssampler = GibbsSampler(hsbm_networks, p, sparsity, seed = 0)
	directory = f"hsbm_samples_N_{N}"
	os.mkdir(directory)

	# save initialization values to hsbm_samples/init_values.p
	init_values = {"A": gibbssampler.A, "theta": gibbssampler.theta, "nu": gibbssampler.nu, "X": gibbssampler.X}
	gibbssampler.write(init_values, os.path.join(directory, "init_values.p"))

	# run gibbs sampler
	T = 1000
	for t in range(T):
		param = gibbssampler.sample()
		gibbssampler.write(param, os.path.join(directory, f"iter_{t + 1}.p"))
		print("Iteration {} log-posterior: {}".format(t + 1, param.log_post))
	print(f"All samples saved to folder {directory}.")
