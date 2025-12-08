import numpy as np
import os
from scipy.io import loadmat
from simulate import *
from sample import *


# load brain connectivity network data
dic = loadmat("data/brain_connectivity_network_data.mat")
brain_networks = dic["loaded_bd_network"]
brain_networks += np.transpose(brain_networks, (1, 0, 2))
brain_networks = np.transpose(brain_networks, (2, 0, 1)) + np.identity(68).reshape(1, 68, 68)
print("Brain connectivity network data loaded from data/brain_connectivity_network_data.mat.")

# model structure from model selection
K = 2
for p, sparsity in [
	[[4, 21, 68], 1],
	[[4, 18, 68], 2]
]:
	# set up gibbs sampler and directory for saving posterior samples
	gibbssampler = GibbsSampler(brain_networks, p, sparsity, seed = 0)
	directory = f"app_S{sparsity}_samples"
	os.mkdir(directory)

	# save initialization values to app_S{sparsity}_samples/init_values.p
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
