import numpy as np
import numpy.linalg as nlg
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from nilearn.image import new_img_like
from nilearn.plotting import plot_glass_brain
import nibabel as nib
from PIL import Image
import os
import pickle
from tqdm import tqdm
from collections import Counter
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_decomposition import PLSCanonical
import multiprocessing as mp
from sample import *


# load atlas of brain regions on the brain cortex from data/atlas_desikan_killiany.nii
brain_region_indices = np.hstack([np.arange(1001, 1004), np.arange(1005, 1036), np.arange(2001, 2004), np.arange(2005, 2036)])
desim = nib.load("data/atlas_desikan_killiany.nii")
atlas_data = np.asanyarray(desim.dataobj)


# function for plotting a cluster of brain regions on the brain cortex
def plot_cluster_on_cortex(cluster_arr, display_mode = 'ortho', cmap = 'Reds'):
	"""
	cluster_arr: np.ndarray of length 68 corresponding to the 68 brain reginons in brain_region_indices, each entry >= 0
	display_mode: 'ortho' plots brain cortex from all three axes, 'z' plots brain cortex from top-down view only
	cmap: color map, e.g. 'Reds', 'Purples', 'Greens'
	"""
	if cluster_arr.max() > 1:
		cluster_arr -= min(np.median(cluster_arr), 1)
	cluster_arr = np.maximum(0, cluster_arr)
	each_roi_data = np.zeros(atlas_data.shape)
	for i in range(cluster_arr.size):
		each_roi_data += (atlas_data == brain_region_indices[i]).astype(float) * cluster_arr[i]
	each_roi_img = new_img_like(desim, each_roi_data, affine = desim.affine)
	plot_glass_brain(each_roi_img, display_mode = display_mode, cmap = cmap, vmin = 0, vmax = 1)


# function for merging png files horizontally
def merge_images_horizontally(image_paths, output_path):
	images = [Image.open(path) for path in image_paths]
	total_width = sum(img.width for img in images)
	max_height = max(img.height for img in images)
	merged_img = Image.new('RGB', (total_width, max_height))
	x_offset = 0
	for img in images:
		merged_img.paste(img, (x_offset, 0))
		x_offset += img.width
		img.close()
	merged_img.save(output_path)
	merged_img.close()
	for path in image_paths:
		os.remove(path)


# function for merging png files vertically
def merge_images_vertically(image_paths, output_path):
	images = [Image.open(path) for path in image_paths]
	max_width = max(img.width for img in images)
	total_height = sum(img.height for img in images)
	merged_img = Image.new('RGB', (max_width, total_height), color = (255, 255, 255))
	y_offset = 0
	for img in images:
		x_centered = (max_width - img.width) // 2
		merged_img.paste(img, (x_centered, y_offset))
		y_offset += img.height
		img.close()
	merged_img.save(output_path)
	merged_img.close()
	for path in image_paths:
		os.remove(path)


# load standard gibbs samples for sparsity = 1 and 2
stan_gibbs_samples_sparsity_1 = []
for t in tqdm(range(10000, 10100)):
	with open(os.path.join("app_S1_samples", f"iter_{t + 1}.p"), 'rb') as hf:
		stan_gibbs_samples_sparsity_1.append(pickle.load(hf))
stan_gibbs_samples_sparsity_2 = []
for t in tqdm(range(10000, 10100)):
	with open(os.path.join("app_S2_samples", f"iter_{t + 1}.p"), 'rb') as hf:
		stan_gibbs_samples_sparsity_2.append(pickle.load(hf))


# plot multi-resolution partitioning of brain regions
# higher-level clusters, sparsity = 1
higher_level_clusters_sparsity_1 = np.mean([sample.A[2] @ sample.A[1] for sample in stan_gibbs_samples_sparsity_1], axis = 0)
for i in range(4):
	plot_cluster_on_cortex(higher_level_clusters_sparsity_1[:, i], display_mode = 'z', cmap = 'Blues')
	plt.savefig(f"output/application/higher_level_brain_cluster_{i + 1}_sparsity_1.png")
merge_images_horizontally([f"output/application/higher_level_brain_cluster_{i + 1}_sparsity_1.png" for i in range(4)], "output/application/higher_level_brain_cluster_sparsity_1.png")
# lower-level clusters, sparsity = 1
lower_level_clusters_sparsity_1 = np.mean([sample.A[2] for sample in stan_gibbs_samples_sparsity_1], axis = 0)
for i in range(21):
	plot_cluster_on_cortex(lower_level_clusters_sparsity_1[:, i], display_mode = 'z', cmap = 'Purples')
	plt.savefig(f"output/application/lower_level_brain_cluster_{i + 1}_sparsity_1.png")
merge_images_horizontally([f"output/application/lower_level_brain_cluster_{i + 1}_sparsity_1.png" for i in range(10)], "output/application/lower_level_brain_cluster_sparsity_1_row1.png")
merge_images_horizontally([f"output/application/lower_level_brain_cluster_{i + 1}_sparsity_1.png" for i in range(10, 21)], "output/application/lower_level_brain_cluster_sparsity_1_row2.png")
merge_images_vertically([f"output/application/lower_level_brain_cluster_sparsity_1_row{r + 1}.png" for r in range(2)], "output/application/lower_level_brain_cluster_sparsity_1.png")
# higher-level clusters, sparsity = 2
higher_level_clusters_sparsity_2 = np.mean([sample.A[2] @ sample.A[1] for sample in stan_gibbs_samples_sparsity_2], axis = 0)
for i in range(4):
	plot_cluster_on_cortex(higher_level_clusters_sparsity_2[:, i], display_mode = 'z', cmap = 'Blues')
	plt.savefig(f"output/application/higher_level_brain_cluster_{i + 1}_sparsity_2.png")
merge_images_horizontally([f"output/application/higher_level_brain_cluster_{i + 1}_sparsity_2.png" for i in range(4)], "output/application/higher_level_brain_cluster_sparsity_2.png")
# lower-level clusters, sparsity = 2
lower_level_clusters_sparsity_2 = np.mean([sample.A[2] for sample in stan_gibbs_samples_sparsity_2], axis = 0)
for i in range(18):
	plot_cluster_on_cortex(lower_level_clusters_sparsity_2[:, i], display_mode = 'z', cmap = 'Purples')
	plt.savefig(f"output/application/lower_level_brain_cluster_{i + 1}_sparsity_2.png")
merge_images_horizontally([f"output/application/lower_level_brain_cluster_{i + 1}_sparsity_2.png" for i in range(9)], "output/application/lower_level_brain_cluster_sparsity_2_row1.png")
merge_images_horizontally([f"output/application/lower_level_brain_cluster_{i + 1}_sparsity_2.png" for i in range(9, 18)], "output/application/lower_level_brain_cluster_sparsity_2_row2.png")
merge_images_vertically([f"output/application/lower_level_brain_cluster_sparsity_2_row{r + 1}.png" for r in range(2)], "output/application/lower_level_brain_cluster_sparsity_2.png")


# function for computing Hotelling's two-sample T^2 test p-value
def hotelling_two_sample_t2_test(arr1, arr2):
	"""
	arr1: np.ndarray of shape n1 x d
	arr2: np.ndarray of shape n2 x d
	return p-value of Hotelling's two-sample T^2 test - for testing whether the n1 d-dimensional samples in arr1 and n2 d-dimensional samples in arr2 come from different groups
	"""
	n1, n2, d = arr1.shape[0], arr2.shape[0], arr1.shape[1]
	arr1, arr2 = np.nan_to_num(arr1) + np.nanmean(arr1).reshape(1, -1) * np.isnan(arr1), np.nan_to_num(arr2) + np.nanmean(arr2).reshape(1, -1) * np.isnan(arr2)
	mu_diff = arr1.mean(axis = 0) - arr2.mean(axis = 0)
	Sigma = ((n1 - 1) * np.cov(arr1.T) + (n2 - 1) * np.cov(arr2.T)) / (n1 + n2 - 2)
	t2_stat = n1 * n2 / (n1 + n2) * mu_diff @ nlg.pinv(Sigma) @ mu_diff
	f_stat = (n1 + n2 - d - 1) / (n1 + n2 - 2) / d * t2_stat
	return stats.f.sf(f_stat, d, n1 + n2 - 1 - d)


# load cognitive measure data and names
cognitive_measure_data = np.load("data/cognitive_measure_data.npy")
with open("data/cognitive_measure_names.txt", "r") as hf:
	cognitive_measure_names = [line.strip() for line in hf.readlines()]
left_bin, right_bin = lambda entry: np.argsort(entry)[:(len(entry) // 3)], lambda entry: np.argsort(entry)[-(len(entry) // 3):]


# plot histograms for Hotelling's p-values of cognitive measures of each subject binned by their latent adjacency matrix active entry posterior means
# for sparsity = 1
X1_sparsity_1 = np.mean([sample.X_subset[1][sample.subset] for sample in stan_gibbs_samples_sparsity_1], axis = 0)
active_entries_sparsity_1 = np.column_stack([X1_sparsity_1[:, i, j] for i in range(21) for j in range(i)])
active_entries_sparsity_1 = active_entries_sparsity_1[:, (active_entries_sparsity_1.mean(axis = 0) >= 0.01) & (active_entries_sparsity_1.mean(axis = 0) <= 0.99)]
p_vals_sparsity_1 = [hotelling_two_sample_t2_test(cognitive_measure_data[left_bin(active_entry)], cognitive_measure_data[right_bin(active_entry)]) for active_entry in active_entries_sparsity_1.T]
plt.figure(figsize = (5, 4))
sns.histplot(p_vals_sparsity_1, bins = np.linspace(0, 1, 11), stat = 'probability')
plt.xlabel(r"p-value of Hotelling's $T^2$ test")
plt.ylabel("proportion")
plt.savefig("output/application/hotelling_test_p_vals_sparsity_1.png", dpi = 300, bbox_inches = "tight")
# for sparsity = 2
X1_sparsity_2 = np.mean([sample.X_subset[1][sample.subset] for sample in stan_gibbs_samples_sparsity_2], axis = 0)
active_entries_sparsity_2 = np.column_stack([X1_sparsity_2[:, i, j] for i in range(18) for j in range(i)])
active_entries_sparsity_2 = active_entries_sparsity_2[:, (active_entries_sparsity_2.mean(axis = 0) >= 0.01) & (active_entries_sparsity_2.mean(axis = 0) <= 0.99)]
p_vals_sparsity_2 = [hotelling_two_sample_t2_test(cognitive_measure_data[left_bin(active_entry)], cognitive_measure_data[right_bin(active_entry)]) for active_entry in active_entries_sparsity_2.T]
plt.figure(figsize = (5, 4))
sns.histplot(p_vals_sparsity_2, bins = np.linspace(0, 1, 11), stat = 'probability')
plt.xlabel(r"p-value of Hotelling's $T^2$ test")
plt.ylabel("proportion")
plt.savefig("output/application/hotelling_test_p_vals_sparsity_2.png", dpi = 300, bbox_inches = "tight")


# plot example histograms of cognitive measures compared across clusters of individuals binned by latent adjacency matrix active entry posterior means
# example for sparsity = 1
i_cog_sparsity_1 = 67
i_sparsity_1, j_sparsity_1 = 17, 6
print('Cognitive measure name:', cognitive_measure_names[i_cog_sparsity_1])
left_bin_first_cigarette_age, right_bin_first_cigarette_age = cognitive_measure_data[left_bin(X1_sparsity_1[:, i_sparsity_1, j_sparsity_1]), i_cog_sparsity_1], cognitive_measure_data[right_bin(X1_sparsity_1[:, i_sparsity_1, j_sparsity_1]), i_cog_sparsity_1]
left_bin_counter, right_bin_counter = Counter(left_bin_first_cigarette_age[np.isnan(left_bin_first_cigarette_age) == 0]), Counter(right_bin_first_cigarette_age[np.isnan(right_bin_first_cigarette_age) == 0])
first_cigarette_age_df = pd.DataFrame({
	'age': ['<=14', '15-17', '18-20', '>=21', '<=14', '15-17', '18-20', '>=21'],
	'cluster': ['x < m', 'x < m', 'x < m', 'x < m', 'x > m', 'x > m', 'x > m', 'x > m'],
	'proportion': [left_bin_counter[14] / sum(left_bin_counter.values()), left_bin_counter[17] / sum(left_bin_counter.values()), left_bin_counter[20] / sum(left_bin_counter.values()), left_bin_counter[21] / sum(left_bin_counter.values()),
				   right_bin_counter[14] / sum(right_bin_counter.values()), right_bin_counter[17] / sum(right_bin_counter.values()), right_bin_counter[20] / sum(right_bin_counter.values()), right_bin_counter[21] / sum(right_bin_counter.values())]
})
sns.set_theme(style = "whitegrid")
g = sns.catplot(data = first_cigarette_age_df, kind = "bar", x = "age", y = "proportion", hue = "cluster", palette = "dark", alpha = 0.6, ci = None, height = 3, aspect = 1.5)
g.set_axis_labels("Age of First Cigarette", "proportion")
g.legend.set_title("Cluster")
plt.savefig("output/application/individual_clusters_age_first_cigarette_sparsity_1.png", dpi = 300, bbox_inches = "tight")
# example for sparsity = 2
i_cog_sparsity_2 = 51
i_sparsity_2, j_sparsity_2 = 13, 9
print('Cognitive measure name:', cognitive_measure_names[i_cog_sparsity_2])
left_bin_age_adj_grip_strength, right_bin_age_adj_grip_strength = cognitive_measure_data[left_bin(X1_sparsity_2[:, i_sparsity_2, j_sparsity_2]), i_cog_sparsity_2] // 20 * 20, cognitive_measure_data[right_bin(X1_sparsity_2[:, i_sparsity_2, j_sparsity_2]), i_cog_sparsity_2] // 20 * 20
left_bin_counter, right_bin_counter = Counter(left_bin_age_adj_grip_strength[np.isnan(left_bin_age_adj_grip_strength) == 0]), Counter(right_bin_age_adj_grip_strength[np.isnan(right_bin_age_adj_grip_strength) == 0])
age_adj_grip_strength_df = pd.DataFrame({
	'strength': ['60~79', '80~99', '100~119', '120~139', '140~159', '60~79', '80~99', '100~119', '120~139', '140~159'],
	'cluster': ['x < m', 'x < m', 'x < m', 'x < m', 'x < m', 'x > m', 'x > m', 'x > m', 'x > m', 'x > m'],
	'proportion': [left_bin_counter[v] / sum(left_bin_counter.values()) for v in [60, 80, 100, 120, 140]] + [right_bin_counter[v] / sum(right_bin_counter.values()) for v in [60, 80, 100, 120, 140]]
})
sns.set_theme(style = "whitegrid")
g = sns.catplot(data = age_adj_grip_strength_df, kind = "bar", x = "strength", y = "proportion", hue = "cluster", palette = "dark", alpha = 0.6, ci = None, height = 3, aspect = 1.5)
g.set_axis_labels("Age-adjusted Grip Strength", "proportion")
g.ax.tick_params(axis = 'x', labelsize = 10)
g.legend.set_title("Cluster")
plt.savefig("output/application/individual_clusters_age_adj_grip_strength_sparsity_2.png", dpi = 300, bbox_inches = "tight")


# load brain connectivity networks
dic = loadmat("data/brain_connectivity_network_data.mat")
brain_networks = dic["loaded_bd_network"]
brain_networks += np.transpose(brain_networks, (1, 0, 2))
brain_networks = np.transpose(brain_networks, (2, 0, 1)) + np.identity(68).reshape(1, 68, 68)


# latent probability matrix inferred by degree corrected mixed membership model (DCMM)
def DCMM_latent_matrix(X, q):
	"""
	X: np.ndarray of shape p x p - observed adjacency matrix
	q: number of communities
	return: latent probability matrix: np.ndarray of shape q x q
	"""
	eigval, eigvec = nlg.eig(X)
	top_eigvec, fiedler_eigvec = eigvec[:, np.argsort(eigval)[-1]], eigvec[:, np.argsort(eigval)[-2:-(q + 1):-1]]
	R = fiedler_eigvec / top_eigvec.reshape(-1, 1)
	V = sketched_vertex_search(R)
	Lambda = np.diag(eigval[np.argsort(eigval)[-1:-(q + 1):-1]])
	b1 = (eigval[np.argsort(eigval)[-1]] + (eigval[np.argsort(eigval)[-2:-(q + 1):-1]].reshape(1, -1) * V ** 2).sum(axis = 1)) ** (-0.5)
	B = np.diag(b1) @ np.hstack([np.ones((q, 1)), V])
	return np.nan_to_num(B @ Lambda @ B.T)
dcmm_latent_P = np.stack([DCMM_latent_matrix(network, 18) for network in brain_networks])
dcmm_latent_P = np.column_stack([dcmm_latent_P[:, i, j] for i in range(18) for j in range(i)])


# function for partial least square canonical correlation
def partial_least_square_canonical_correlation(X, Y):
	"""
	X: array of shape n x p
	Y: array of shape n x q, Y may contain some np.NaN's
	"""
	X, Y = X[np.isnan(Y).any(axis = 1) == 0], Y[np.isnan(Y).any(axis = 1) == 0]
	X_transformed, Y_transformed = PLSCanonical(n_components = 1).fit_transform(X, Y)
	return np.corrcoef(X_transformed.reshape(-1), Y_transformed.reshape(-1))[0, 1]


# compute partial least square canonical correlation between latent X/P and each group of cognitive measures
pls_corr_df = []
for cognitive_group_name in np.unique([name.split(" @ ")[1] for name in cognitive_measure_names]):
	cognitive_group = cognitive_measure_data[:, [i for i, name in enumerate(cognitive_measure_names) if name.split(" @ ")[1] == cognitive_group_name]]
	pls_corr_df.append([cognitive_group_name, partial_least_square_canonical_correlation(np.column_stack([X1_sparsity_2[:, i, j] for i in range(18) for j in range(i)]), cognitive_group), partial_least_square_canonical_correlation(dcmm_latent_P, cognitive_group)])
pls_corr_df = pd.DataFrame(pls_corr_df, columns = ['cognitive_group_name', 'latent_X_pls_corr', 'latent_P_pls_corr'])
pls_corr_df.to_csv('output/application/partial_least_square_canonical_correlations.csv', index = False)


# function for cross validated ridge regression
def cross_validated_ridge_regression_mse(X, y, n_splits, alpha, seed = None):
	"""
	X: np.ndarray of shape n x p
	y: np.ndarray of length n
	n_splits: number of folds in cross validation
	alpha: ridge regression hyperparameter
	seed: random state
	return: mean squared error (MSE) of y vs. y_pred on the test fold
	"""
	kf = KFold(n_splits = n_splits, shuffle = True, random_state = seed)
	test_rmse_list = []
	for train_idx, test_idx in kf.split(X):
		X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
		model = Ridge(alpha = alpha).fit(X_train, y_train)
		test_rmse_list.append(mean_squared_error(y_test, model.predict(X_test)) ** 0.5)
	return np.mean(test_rmse_list)


# function for nested cross validated ridge regression
def nested_cross_validated_ridge_regression_corr(X, y, n_splits = 10, alpha_grid = np.logspace(-5, 5, 21), seed = None):
	"""
	X: np.ndarray of shape n x p
	y: np.ndarray of length n
	n_splits: number of folds in cross validation
	alpha_grid: np.ndarray - grid of ridge regression hyperparameters to choose from (automatically using the validation fold)
	seed: random state
	return: correlation between y and y_pred
	"""
	if seed is None:
		seed = np.random.randint(0, 1e8)
	seed_gen = random_seed_generator(seed)
	X, y = X[np.isnan(y) == 0], y[np.isnan(y) == 0]
	kf = KFold(n_splits = n_splits, shuffle = True, random_state = next(seed_gen))
	y_pred = np.zeros(y.size)
	for train_idx, test_idx in kf.split(X):
		X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
		vali_rmse_list = []
		for alpha in alpha_grid:
			vali_rmse_list.append(cross_validated_ridge_regression_mse(X_train, y_train, n_splits - 1, alpha, next(seed_gen)))
		best_alpha = alpha_grid[np.argmin(vali_rmse_list)]
		model = Ridge(alpha = best_alpha).fit(X_train, y_train)
		y_pred[test_idx] = model.predict(X_test)
	return np.corrcoef(y, y_pred)[0, 1]


# compute correlation in using latent X/P to predict each cognitive measure with nested cross validated ridge regression
def run_nested_cv(name, y):
	return [name, nested_cross_validated_ridge_regression_corr(active_entries_sparsity_2, y, seed = 0), nested_cross_validated_ridge_regression_corr(dcmm_latent_P, y, seed = 0)]
if os.path.exists('output/application/nested_cv_ridge_prediction_correlations.csv'):
	ncv_ridge_corr_df = pd.read_csv('output/application/nested_cv_ridge_prediction_correlations.csv', header = 0)
else:
	with mp.Pool(processes = 10) as pool:
		ncv_ridge_corr_df = pool.starmap(run_nested_cv, list(zip(cognitive_measure_names, cognitive_measure_data.T)))
	ncv_ridge_corr_df = pd.DataFrame(ncv_ridge_corr_df, columns = ['cognitive_measure_name', 'latent_X_ridge_corr', 'latent_P_ridge_corr'])
	ncv_ridge_corr_df.to_csv('output/application/nested_cv_ridge_prediction_correlations.csv', index = False)


# plot correlation barplots for each predictable cognitive measure
predictable_ncv_ridge_corr_df = ncv_ridge_corr_df[(ncv_ridge_corr_df['latent_X_ridge_corr'] > 0) & (ncv_ridge_corr_df['latent_P_ridge_corr'] > 0)]
ncv_ridge_corr_df_melted = predictable_ncv_ridge_corr_df.melt(id_vars = "cognitive_measure_name", value_vars = ["latent_X_ridge_corr", "latent_P_ridge_corr"], var_name = "model", value_name = "ridge_corr")
ncv_ridge_corr_df_melted['cognitive_measure_name'] = ncv_ridge_corr_df_melted['cognitive_measure_name'].apply(lambda x: x.split(' (')[0].replace('Instrument: ', ''))
ncv_ridge_corr_df_melted['model'] = ncv_ridge_corr_df_melted['model'].apply(lambda x: {'latent_X_ridge_corr': 'BDGM', 'latent_P_ridge_corr': 'DCMM'}[x])
g = sns.catplot(data = ncv_ridge_corr_df_melted, kind = "bar", y = "cognitive_measure_name", x = "ridge_corr", hue = "model", height = 10, aspect = 1.5)
g.set_xlabels('Forecast Correlation', fontsize = 20)
g.set_ylabels('Cognitive Measure', fontsize = 20)
g.ax.legend(title = "Model", fontsize = 20, title_fontsize = 20, loc = 'best')
g._legend.remove()
g.ax.tick_params(axis = 'x', labelsize = 16)
g.ax.tick_params(axis = 'y', labelsize = 16)
plt.tight_layout()
plt.show()
plt.savefig("output/application/latent_variables_forecast_correlation_with_cognitive_measures.png", dpi = 300, bbox_inches = "tight")
