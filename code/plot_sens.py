import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm
from spec_init import *
from sample import *


# function for Wasserstein-1 distance between two empirical distributions
def W1(arr1, arr2):
	arr2 = np.repeat(arr2, arr1.size // arr2.size)
	arr1 = arr1[:arr2.size]
	return np.abs(np.sort(arr1) - np.sort(arr2)).mean()


# load gibbs samples under each prior distribution
prior_names = ['default', 'larger_mu_c', 'smaller_mu_c', 'larger_sigma_c', 'smaller_sigma_c', 'larger_mu_gamma', 'smaller_mu_gamma', 'larger_sigma_gamma', 'smaller_sigma_gamma', 'larger_mu_delta', 'smaller_mu_delta', 'larger_sigma_delta', 'smaller_sigma_delta', 'larger_alpha', 'smaller_alpha']
samples_dict = {prior: [] for prior in prior_names}
directory = "sens_samples"
for m, prior in enumerate(prior_names):
	for t in tqdm(range(10000 + 100 * m, 10000 + 100 * (m + 1))):
		with open(os.path.join(directory, f"iter_{t + 1}.p"), 'rb') as hf:
			sample = pickle.load(hf)
			samples_dict[prior].append(np.hstack([
				sample.A[1].reshape(-1),
				sample.A[2].reshape(-1),
				sample.theta[1].reshape(-1),
				sample.theta[2].reshape(-1),
				sample.nu.reshape(-1)
			]))
	samples_dict[prior] = np.array(samples_dict[prior])


post_std = samples_dict['default'].std(axis = 0).mean()
# compute average difference in posterior means of each coordinate
post_mean_diff = [
	np.abs(samples_dict[prior].mean(axis = 0) - samples_dict['default'].mean(axis = 0)).mean()
	for prior in prior_names[1:]
]
# compute average difference in posterior standard deviations of each coordinate
post_std_diff = [
	np.abs(samples_dict[prior].std(axis = 0) - samples_dict['default'].std(axis = 0)).mean()
	for prior in prior_names[1:]
]
# compute average Wasserstein-1 distance between posterior distributions of each coordinate
post_w1_dist = [
	np.mean([W1(samples_dict[prior][:, i], samples_dict['default'][:, i]) for i in range(samples_dict['default'].shape[1])])
	for prior in prior_names[1:]
]
# save results to csv
df = pd.DataFrame({
	'posterior mean difference': post_mean_diff,
	'posterior standard deviation difference': post_std_diff,
	'posterior Wasserstein-1 distance': post_w1_dist,
	'ratio of posterior mean difference over posterior standard deviation': post_mean_diff / post_std,
	'ratio of posterior standard deviation difference over posterior standard deviation': post_std_diff / post_std,
	'ratio of posterior Wasserstein-1 distance over posterior standard deviation': post_w1_dist / post_std,
}, index = prior_names[1:])
df.to_csv('output/simulation/sensitivity_analysis.csv')
