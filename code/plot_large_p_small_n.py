import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm
from spec_init import *
from sample import *


def equal_number(arr1, arr2):
	"""
	arr1, arr2: n x p x p
	return: number of i = 1, 2, ..., n such that arr1[i] = arr2[i] (up to floating point error)
	"""
	return sum([np.abs(mat1 - mat2).max() < 1e-8 for mat1, mat2 in zip(arr1, arr2)])


# load true_values
N, K = 10, 1
with open("large_p_small_n_samples/true_values.p", "rb") as hf:
	true_values = pickle.load(hf)
	XK1_true_values, AK_true_values = true_values['X'][K - 1], true_values['A'][K]


def post_process(AK_sample, AK_true_values):
	"""
	AK_sample: pK x 4
	AK_true_values: 68 x 4
	return: an order of [0, 1, 2, 3] that permutes the pK1 = 4 nodes at layer 1 to handle label switching
	"""
	pK = AK_sample.shape[0]
	possible_orders = [
		[0, 1, 2, 3], [0, 1, 3, 2], [0, 2, 1, 3], [0, 2, 3, 1], [0, 3, 1, 2], [0, 3, 2, 1],
		[1, 0, 2, 3], [1, 0, 3, 2], [1, 2, 0, 3], [1, 2, 3, 0], [1, 3, 0, 2], [1, 3, 2, 0],
		[2, 0, 1, 3], [2, 0, 3, 1], [2, 1, 0, 3], [2, 1, 3, 0], [2, 3, 0, 1], [2, 3, 1, 0],
		[3, 0, 1, 2], [3, 0, 2, 1], [3, 1, 0, 2], [3, 1, 2, 0], [3, 2, 0, 1], [3, 2, 1, 0],
	]
	min_diff, best_order = np.inf, None
	for order in possible_orders:
		diff = np.abs(AK_sample[:, order] - AK_true_values[:pK]).sum()
		if diff < min_diff:
			min_diff, best_order = diff, order
	return best_order


# load samples of X
if not os.path.exists('output/simulation/large_p_small_n.csv'):
	accuracy_results = []
	for pK in range(8, 69):
		print(f'Working on pK = {pK}.')
		directory = f"large_p_small_n_samples/observed_dimension_{pK}"
		XK1_samples, accuracy_by_sample = [], []
		for t in range(1000, 2000):
			with open(os.path.join(directory, f"iter_{t + 1}.p"), 'rb') as hf:
				sample = pickle.load(hf)
				sample_order = post_process(sample.A[K], AK_true_values)
				XK1_sample = sample.X_subset[K - 1][:, sample_order, :][:, :, sample_order]
				XK1_samples.append(XK1_sample)
				accuracy_by_sample.append([
					equal_number(XK1_sample, XK1_true_values),
					np.abs(XK1_sample - XK1_true_values).mean(),
				])
		XK1_samples, accuracy_by_sample = np.array(XK1_samples), np.array(accuracy_by_sample)
		accuracy_results.append([
			*list(accuracy_by_sample.mean(axis = 0)),
			np.abs(XK1_samples.mean(axis = 0) - XK1_true_values).mean()
		])
	accuracy_df = pd.DataFrame(accuracy_results, columns = ['average_XK1_sample_equal_number', 'average_XK1_sample_distance', 'XK1_posterior_mean_distance'], index = pd.Series(np.arange(8, 69), name = 'observed_dimension_pK'))
	accuracy_df.to_csv('output/simulation/large_p_small_n.csv')
else:
	accuracy_df = pd.read_csv('output/simulation/large_p_small_n.csv', header = 0, index_col = 0)
	plt.figure(figsize = (6, 5))
	plt.plot(accuracy_df.index, accuracy_df['average_XK1_sample_equal_number'], color = 'black')
	plt.xlabel(r'observed dimension $p_K$')
	plt.ylabel(r'recovery accuracy of latent adjacency matrix $X_{K - 1}$')
	plt.grid(ls = ':')
	plt.tight_layout()
	plt.savefig('output/simulation/large_p_small_n_latent_recovery_accuracy.png', dpi = 300, bbox_inches = "tight")
