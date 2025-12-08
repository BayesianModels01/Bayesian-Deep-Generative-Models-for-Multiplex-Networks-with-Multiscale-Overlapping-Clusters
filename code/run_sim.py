import numpy as np
import os
from simulate import *
from sample import *


# simulation settings
K = 2
p = [4, 16, 68]
gamma = [None, 10.0, 10.0]
delta = [None, 4.0, 4.0]
C = [None, -7.0, -7.0]
PX0 = 0.5
N = 1000
sparsity = 2

# simulate data
A, theta, X = simulate_all(K, p, gamma, delta, C, PX0, N, seed = 0)

# set up gibbs sampler and directory for saving posterior samples
gibbssampler = GibbsSampler(X[-1], p, sparsity, seed = 0)
directory = "sim_samples"
os.mkdir(directory)

# save true values to sim_samples/true_values.p
true_values = {"K": K, "p": p, "N": N, "sparsity": sparsity, "A": A, "theta": theta, "nu": np.ones(2 ** (p[0] * (p[0] - 1) // 2)) / 2 ** (p[0] * (p[0] - 1) // 2), "X": X}
true_values["log_post"] = gibbssampler.return_log_posterior(true_values["A"], true_values["theta"], true_values["nu"], true_values["X"])
print("True log_posterior: {}".format(true_values["log_post"]))
gibbssampler.write(true_values, os.path.join(directory, "true_values.p"))

# save initialization values to sim_samples/init_values.p
init_values = {"A": gibbssampler.A, "theta": gibbssampler.theta, "nu": gibbssampler.nu, "X": gibbssampler.X}
gibbssampler.write(init_values, os.path.join(directory, "init_values.p"))

# run subsampling and standard gibbs sampler
T1, T2 = 10000, 100
for t in range(T1 + T2):
	if t < T1:
		param = gibbssampler.sample(subset_proportion = 0.01)
	else:
		param = gibbssampler.sample()
	gibbssampler.write(param, os.path.join(directory, f"iter_{t + 1}.p"))
	print("Iteration {} log-posterior: {}".format(t + 1, param.log_post))
print(f"All samples saved to folder {directory}.")
