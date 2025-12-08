import numpy as np
import os
from simulate import *
from sample import *
from tqdm import tqdm


# simulation settings
K = 1
true_p = [4, 68]
gamma = [None, 10.0]
delta = [None, 4.0]
C = [None, -7.0]
PX0 = 0.5
N = 10
sparsity = 1

# simulate data
A, theta, X = simulate_all(K, true_p, gamma, delta, C, PX0, N, A_row_sum_max = sparsity, seed = 0)

# save true values to large_p_small_n_samples/true_values.p
os.mkdir('large_p_small_n_samples')
true_values = {"K": K, "p": true_p, "N": N, "sparsity": sparsity, "A": A, "theta": theta, "nu": np.ones(2 ** (true_p[0] * (true_p[0] - 1) // 2)) / 2 ** (true_p[0] * (true_p[0] - 1) // 2), "X": X}
with open("large_p_small_n_samples/true_values.p", "wb") as hf:
	pickle.dump(true_values, hf)

for pK in range(8, 69):
	p = true_p[:K] + [pK]
	print(f'Working on p = {p}.')

	# set up gibbs sampler and directory for saving posterior samples
	gibbssampler = GibbsSampler(X[-1][:, :pK, :pK], p, sparsity, seed = 0)
	directory = f"large_p_small_n_samples/observed_dimension_{pK}"
	os.mkdir(directory)

	# save initialization values to sim_samples/init_values.p
	init_values = {"A": gibbssampler.A, "theta": gibbssampler.theta, "nu": gibbssampler.nu, "X": gibbssampler.X}
	gibbssampler.write(init_values, os.path.join(directory, "init_values.p"))

	# run standard gibbs sampler
	T = 2000
	for t in tqdm(range(T)):
		param = gibbssampler.sample()
		gibbssampler.write(param, os.path.join(directory, f"iter_{t + 1}.p"))
	print(f"All samples saved to folder {directory}.")
