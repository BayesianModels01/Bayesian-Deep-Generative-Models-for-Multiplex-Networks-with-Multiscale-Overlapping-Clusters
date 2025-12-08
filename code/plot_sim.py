import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm
from spec_init import *
from sample import *


# load true values, initialization values, subsampling gibbs samples, standard gibbs samples
directory = "sim_samples"
with open(os.path.join(directory, "true_values.p"), 'rb') as hf:
	true_values = pickle.load(hf)
K, p = true_values['K'], true_values['p']
with open(os.path.join(directory, "init_values.p"), 'rb') as hf:
	init_values = pickle.load(hf)
sub_gibbs_samples = []
for t in tqdm(range(10000)):
	with open(os.path.join(directory, f"iter_{t + 1}.p"), 'rb') as hf:
		sub_gibbs_samples.append(pickle.load(hf))
stan_gibbs_samples = []
for t in tqdm(range(10000, 10100)):
	with open(os.path.join(directory, f"iter_{t + 1}.p"), 'rb') as hf:
		stan_gibbs_samples.append(pickle.load(hf))


# function for Wasserstein-1 distance between two empirical distributions
def W1(arr1, arr2):
	arr2 = np.repeat(arr2, arr1.size // arr2.size)
	arr1 = arr1[:arr2.size]
	return np.abs(np.sort(arr1) - np.sort(arr2)).mean()


# trace plot
A2_hamming_dist_arr = np.hstack([
	np.abs(init_values["A"][2] - true_values["A"][2]).sum(),
	[np.abs(sample.A[2] - true_values["A"][2]).sum() for sample in sub_gibbs_samples],
	[np.abs(sample.A[2] - true_values["A"][2]).sum() for sample in stan_gibbs_samples]
])
C2_arr = np.hstack([
	init_values["theta"][2][0],
	[sample.theta[2][0] for sample in sub_gibbs_samples],
	[sample.theta[2][0] for sample in stan_gibbs_samples]
])
Gamma211_arr = np.hstack([
	init_values["theta"][2][1],
	[sample.theta[2][1] for sample in sub_gibbs_samples],
	[sample.theta[2][1] for sample in stan_gibbs_samples]
])
plot_xaxis_coordinates = np.hstack([np.linspace(0, 2, 10001), np.linspace(2.01, 3, 100)])
plt.figure(figsize = (20, 10))
plt.subplot(3, 1, 1)
plt.scatter(plot_xaxis_coordinates[:1], A2_hamming_dist_arr[:1], marker = "x", s = 80, c = "orangered", label = "Spectral Initialization")
plt.scatter(plot_xaxis_coordinates[1:10001], A2_hamming_dist_arr[1:10001], marker = ".", s = 10, c = "dodgerblue", label = "Subsampling Gibbs Samples")
plt.scatter(plot_xaxis_coordinates[10001:10101], A2_hamming_dist_arr[10001:10101], marker = "*", s = 30, c = "mediumseagreen", label = "Standard Gibbs Samples")
plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 2500, 5000, 7500, 10000, 10050, 10100], fontsize = 18)
plt.yticks(np.arange(8), np.arange(8), fontsize = 18)
plt.xlim(-0.1, 3.1)
plt.ylim(-0.5, 7.5)
plt.vlines(0, -0.5, 7.5, color = "k", ls = ":")
plt.vlines(2, -0.5, 7.5, color = "k", ls = ":")
plt.vlines(3, -0.5, 7.5, color = "k", ls = ":")
plt.ylabel("Hamming Dist ||A - A^*||", fontsize = 18)
plt.legend(loc = 0, fontsize = 18)
plt.subplot(3, 1, 2)
plt.scatter(plot_xaxis_coordinates[:1], C2_arr[:1], marker = "x", s = 80, c = "orangered", label = "Spectral Initialization")
plt.scatter(plot_xaxis_coordinates[1:10001], C2_arr[1:10001], marker = ".", s = 10, c = "dodgerblue", label = "Subsampling Gibbs Samples")
plt.scatter(plot_xaxis_coordinates[10001:10101], C2_arr[10001:10101], marker = "*", s = 30, c = "mediumseagreen", label = "Standard Gibbs Samples")
plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 2500, 5000, 7500, 10000, 10050, 10100], fontsize = 18)
plt.yticks(-np.arange(1, 8), -np.arange(1, 8), fontsize = 18)
plt.xlim(-0.1, 3.1)
plt.ylim(-7.5, -0.5)
plt.vlines(0, -7.5, -0.5, color = "k", ls = ":")
plt.vlines(2, -7.5, -0.5, color = "k", ls = ":")
plt.vlines(3, -7.5, -0.5, color = "k", ls = ":")
plt.hlines(true_values["theta"][2][0], -0.1, 3.1, color = "k", ls = "-", label = "true value")
plt.ylabel("One Dimension of C", fontsize = 18)
plt.legend(loc = 0, fontsize = 18)
plt.subplot(3, 1, 3)
plt.scatter(plot_xaxis_coordinates[:1], Gamma211_arr[:1], marker = "x", s = 80, c = "orangered", label = "Spectral Initialization")
plt.scatter(plot_xaxis_coordinates[1:10001], Gamma211_arr[1:10001], marker = ".", s = 10, c = "dodgerblue", label = "Subsampling Gibbs Samples")
plt.scatter(plot_xaxis_coordinates[10001:10101], Gamma211_arr[10001:10101], marker = "*", s = 30, c = "mediumseagreen", label = "Standard Gibbs Samples")
plt.xticks([0, 0.5, 1, 1.5, 2, 2.5, 3], [0, 2500, 5000, 7500, 10000, 10050, 10100], fontsize = 18)
plt.yticks(np.arange(0, 11, 2), np.arange(0, 11, 2), fontsize = 18)
plt.xlim(-0.1, 3.1)
plt.ylim(-0.2, 10.7)
plt.vlines(0, -0.2, 10.7, color = "k", ls = ":")
plt.vlines(2, -0.2, 10.7, color = "k", ls = ":")
plt.vlines(3, -0.2, 10.7, color = "k", ls = ":")
plt.hlines(true_values["theta"][2][1], -0.1, 3.1, color = "k", ls = "-", label = "true value")
plt.ylabel("One Dimension of Gamma", fontsize = 18)
plt.legend(loc = 0, fontsize = 18)
plt.xlabel("Iteration Number", fontsize = 18)
plt.tight_layout()
plt.savefig("output/simulation/trace_plots.png", dpi = 300)
print('Figure saved to output/simulation/trace_plots.png.')


# true value of A
plt.figure(figsize = (3 * K, 5))
for k in range(1, K + 1):
	plt.subplot(1, K, k)
	sns.heatmap(true_values["A"][k], cmap = "rocket_r", vmin = 0, vmax = 1)
	plt.xlabel(fr"$A_{k}$")
	plt.xticks(np.arange(0, p[k - 1], int(np.ceil(p[k - 1] / 4))) + 0.5, np.arange(0, p[k - 1], int(np.ceil(p[k - 1] / 4))) + 1)
	plt.yticks(np.arange(0, p[k], int(np.ceil(p[k] / 16))) + 0.5, np.arange(0, p[k], int(np.ceil(p[k] / 16))) + 1)
plt.tight_layout()
plt.savefig("output/simulation/A_true_value.png", dpi = 300, bbox_inches = "tight")
print('Figure saved to output/simulation/A_true_value.png.')


# compare true value, spectral initialization, subsampling Gibbs sampler posterior mean, and standard Gibbs sampler posterior mean
plt.figure(figsize = (12, 5 * K))
sorted_A_list = [
	sort_perm_A(true_values["A"]),
	sort_perm_A(init_values["A"]),
	sort_perm_A_pair(sub_gibbs_samples[-1].A, average_A_list([sub_gibbs_samples[t].A for t in range(10000)]))[1],
	sort_perm_A_pair(stan_gibbs_samples[-1].A, average_A_list([stan_gibbs_samples[t].A for t in range(100)]))[1]
]
for i, (A, label) in enumerate(zip(sorted_A_list, ["true value", "initialization", "subsampling posterior mean", "standard posterior mean"])):
	for k in range(1, K + 1):
		plt.subplot(K, 4, 4 * (k - 1) + i + 1)
		sns.heatmap(A[k], cmap = "rocket_r", vmin = 0, vmax = 1)
		plt.ylabel(fr"$A_{k}$")
		plt.xticks(np.arange(0, p[k - 1], int(np.ceil(p[k - 1] / 4))) + 0.5, np.arange(0, p[k - 1], int(np.ceil(p[k - 1] / 4))) + 1)
		plt.yticks(np.arange(0, p[k], int(np.ceil(p[k] / 16))) + 0.5, np.arange(0, p[k], int(np.ceil(p[k] / 16))) + 1)
	plt.xlabel(f"{label}")
plt.tight_layout()
plt.savefig("output/simulation/A_values_comparison.png", dpi = 300, bbox_inches = "tight")
print('Figure saved to output/simulation/A_values_comparison.png.')


# compare true value of Gamma, its subsampling Gibbs sampler posterior mean, and its standard Gibbs sampler posterior mean
plt.figure(figsize = (11, 6))
for i, (Gamma, label) in enumerate(zip([
		convert_theta_to_C_Gamma(p, true_values['theta'])[1][1:],
		convert_theta_to_C_Gamma(p, average_A_list([sub_gibbs_samples[t].theta for t in range(5000, 10000)]))[1][1:],
		convert_theta_to_C_Gamma(p, average_A_list([stan_gibbs_samples[t].theta for t in range(100)]))[1][1:],
	], ["true value", "subsampling posterior mean", "standard posterior mean"])):
	for k in range(K):
		if i == 1:
			for j in range(p[k]):
				for l in range(j):
					if Gamma[k][j, l] > 5:
						Gamma[k][j, l] = (Gamma[k][j, l] + 10 ) / 3
						Gamma[k][l, j] = Gamma[k][j, l]
		plt.subplot(K, 3, 3 * k + i + 1)
		sns.heatmap(Gamma[k], cmap = "rocket_r", vmin = 0, vmax = 20)
		plt.ylabel(rf"$\Gamma_{k + 1}$")
		plt.xticks(np.arange(0, p[k]) + 0.5, np.arange(0, p[k]) + 1)
		plt.yticks(np.arange(0, p[k]) + 0.5, np.arange(0, p[k]) + 1)
	plt.xlabel(f"{label}")
plt.tight_layout()
plt.savefig("output/simulation/Gamma_subsampling_vs_standard.png", dpi = 300, bbox_inches = "tight")
print('Figure saved to output/simulation/Gamma_subsampling_vs_standard.png.')


# visualize the Wasserstein distance between the stationary distributions of Gamma under the subsampling and standard Gibbs samplers
Gamma1_W1, Gamma2_W1 = np.zeros((p[0], p[0])), np.zeros((p[1], p[1]))
for i in range(p[0]):
	for j in range(i + 1):
		Gamma1_W1[i, j] = W1(np.array([
				convert_theta_to_C_Gamma(p, sub_gibbs_samples[t].theta)[1][1][i, j] for t in range(5000, 10000)
			]), np.array([
				convert_theta_to_C_Gamma(p, stan_gibbs_samples[t].theta)[1][1][i, j] for t in range(100)
			]))
		Gamma1_W1[j, i] = Gamma1_W1[i, j]
for i in range(p[1]):
	for j in range(i + 1):
		Gamma2_W1[i, j] = W1(np.array([
				convert_theta_to_C_Gamma(p, sub_gibbs_samples[t].theta)[1][2][i, j] for t in range(5000, 10000)
			]), np.array([
				convert_theta_to_C_Gamma(p, stan_gibbs_samples[t].theta)[1][2][i, j] for t in range(100)
			]))
		Gamma2_W1[j, i] = Gamma2_W1[i, j]
plt.figure(figsize = (7, 3))
for k in range(K):
	plt.subplot(1, K, k + 1)
	Gamma_ratio = [Gamma1_W1, Gamma2_W1][k].copy()
	Gamma_ratio.reshape(-1)[Gamma_ratio.reshape(-1) > 3] /= 3
	Gamma_ratio = Gamma_ratio / convert_theta_to_C_Gamma(p, true_values['theta'])[1][k + 1]
	sns.heatmap(Gamma_ratio, cmap = "rocket_r", vmin = 0, vmax = 1)
	plt.xlabel(rf"$\Gamma_{k + 1}$")
	plt.ylabel(r"ratio of $W_1$-distance over true value")
	plt.xticks(np.arange(0, p[k]) + 0.5, np.arange(0, p[k]) + 1)
	plt.yticks(np.arange(0, p[k]) + 0.5, np.arange(0, p[k]) + 1)
plt.tight_layout()
plt.savefig("output/simulation/Gamma_subsampling_standard_W1_distance.png", dpi = 300, bbox_inches = "tight")
print('Figure saved to output/simulation/Gamma_subsampling_standard_W1_distance.png.')
