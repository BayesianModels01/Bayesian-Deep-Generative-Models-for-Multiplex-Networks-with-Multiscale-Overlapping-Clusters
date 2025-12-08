import numpy as np
import scipy.linalg as slg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm
from sample import *
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score


def greedy_label_match(true_labels, pred_labels):
	"""
	true_labels, pred_labels: 1-d np.ndarray of same size
	true_labels takes values 0, 1, 2, ...
	pred_labels has the same number of unique values
	return permuted pred_labels by matching it with true_labels to maximize their label overlaps ina greedy way
	"""
	n_labels = np.unique(true_labels).size

	preprocess_dict = {v: i for i, v in enumerate(np.unique(pred_labels))}
	pred_labels = np.array([preprocess_dict[v] for v in pred_labels])
	cm = confusion_matrix(true_labels, pred_labels, labels = np.arange(n_labels))

	mapping, used_true, used_pred = {}, set(), set()
	for _ in range(max(true_labels) + 1):
		max_val, max_pair = -1, None
		for i in range(n_labels):
			if i in used_true:
				continue
			for j in range(n_labels):
				if j in used_pred:
					continue
				if cm[i, j] > max_val:
					max_val, max_pair = cm[i, j], (i, j)
		if max_pair is None:
			break
		i, j = max_pair
		mapping[j] = i
		used_true.add(i)
		used_pred.add(j)

	return np.array([mapping[v] for v in pred_labels])


true_3_communities = (np.arange(27) // 9).astype(int)
true_9_communities = (np.arange(27) // 3).astype(int)
true_3_pairwise_similarities = (1 + slg.block_diag(*[np.ones((9, 9))] * 3)) / 2
true_9_pairwise_similarities = (1 + slg.block_diag(*[np.ones((9, 9))] * 3) + slg.block_diag(*[np.ones((3, 3))] * 9)) / 3


def get_metrics(true_labels, pred_labels, true_pair_sims, pred_pair_sims):
	"""
	true_labels, pred_labels: 1-d np.ndarray of same size
	true_labels takes values 0, 1, 2, ...
	pred_labels has the same number of unique values
	true_pair_sims, pred_pair_sims: symmetric matrices representing the similarity measure between each pair of tree nodes, i.e. the normalized shared tree branch length, for true and pred hierarchical community trees
	see Sec 2.4 and Sec 4 bullet point 2 in https://arxiv.org/pdf/1810.01509
	return: under the greedily found best match of labels, normalized mutual information, label matching accuracy, and tree distance
	"""
	best_match_pred_labels = greedy_label_match(true_labels, pred_labels)
	nmi = normalized_mutual_info_score(true_labels, best_match_pred_labels)
	acc = np.mean(true_labels == best_match_pred_labels)
	tree_dist = ((true_pair_sims - pred_pair_sims) ** 2).sum() / (true_pair_sims ** 2).sum()
	return [nmi, acc, tree_dist]


def count_shared_head_length(m1, m2):
	"""
	m1, m2: intergers
	convert m1, m2 to binary values and count the number of digits shared in their fronts
	e.g. binary integers 10 and 111 have shared head length 1, while 1100 and 1101 has shared head length 3
	return shared head length
	"""
	bin_str_1, bin_str_2 = bin(m1)[2:], bin(m2)[2:]
	count = 0
	for i in range(min(len(bin_str_1), len(bin_str_2))):
		if bin_str_1[i] == bin_str_2[i]:
			count += 1
		else:
			break
	return count

result_df_columns = ['sample size N', 'method', 'layer of community tree structure', 'true number of communities', 'normalized mutual information', 'label matching accuracy', 'hierarchical tree structure distance']
result_df = pd.DataFrame([], columns = result_df_columns)

for N in [10, 20, 30, 40, 50]:
	# load BDGM 3/9-community memberships
	bdgm_3_communities, bdgm_9_communities, bdgm_3_pairwise_similarities, bdgm_9_pairwise_similarities = [], [], [], []
	for t in tqdm(range(500, 1000)):
		with open(f"hsbm_samples_N_{N}/iter_{t + 1}.p", 'rb') as hf:
			sample = pickle.load(hf)
		A1, A2 = sample.A[1], sample.A[2]
		bdgm_3_communities.append(A2 @ A1 @ np.arange(3))
		bdgm_9_communities.append(A2 @ np.arange(9))
		pairwise_similarities_3, pairwise_similarities_9 = np.ones((27, 27)), np.ones((27, 27))
		for i in range(27):
			for j in range(27):
				if bdgm_3_communities[-1][i] == bdgm_3_communities[-1][j]:
					pairwise_similarities_3[i, j] += 1
					pairwise_similarities_9[i, j] += 1
				if bdgm_9_communities[-1][i] == bdgm_9_communities[-1][j]:
					pairwise_similarities_9[i, j] += 1
		bdgm_3_pairwise_similarities.append(pairwise_similarities_3)
		bdgm_9_pairwise_similarities.append(pairwise_similarities_9)
	bdgm_3_communities, bdgm_9_communities, bdgm_3_pairwise_similarities, bdgm_9_pairwise_similarities = np.stack(bdgm_3_communities), np.stack(bdgm_9_communities), np.stack(bdgm_3_pairwise_similarities), np.stack(bdgm_9_pairwise_similarities)
	bdgm_3_pairwise_similarities, bdgm_9_pairwise_similarities = bdgm_3_pairwise_similarities / bdgm_3_pairwise_similarities.max(), bdgm_9_pairwise_similarities / bdgm_9_pairwise_similarities.max()

	# load HCD-Sign/Spec 3/9-community memberships
	hcd_dict = np.load(f'hsbm_hcd/N_{N}.npz')
	hcd_sign_3_communities = hcd_dict['sign_algo_n_communities_3'].T
	hcd_sign_9_communities = hcd_dict['sign_algo_n_communities_9'].T
	hcd_spec_3_communities = hcd_dict['spec_algo_n_communities_3'].T
	hcd_spec_9_communities = hcd_dict['spec_algo_n_communities_9'].T
	hcd_sign_3_pairwise_similarities, hcd_spec_3_pairwise_similarities, hcd_sign_9_pairwise_similarities, hcd_spec_9_pairwise_similarities = [], [], [], []
	for n in tqdm(range(N)):
		pairwise_similarities_sign_3, pairwise_similarities_spec_3, pairwise_similarities_sign_9, pairwise_similarities_spec_9 = np.ones((27, 27)), np.ones((27, 27)), np.ones((27, 27)), np.ones((27, 27))
		for i in range(27):
			for j in range(27):
				pairwise_similarities_sign_3[i, j] = count_shared_head_length(hcd_sign_3_communities[n, i], hcd_sign_3_communities[n, j])
				pairwise_similarities_spec_3[i, j] = count_shared_head_length(hcd_spec_3_communities[n, i], hcd_spec_3_communities[n, j])
				pairwise_similarities_sign_9[i, j] = count_shared_head_length(hcd_sign_9_communities[n, i], hcd_sign_9_communities[n, j])
				pairwise_similarities_spec_9[i, j] = count_shared_head_length(hcd_spec_9_communities[n, i], hcd_spec_9_communities[n, j])
		hcd_sign_3_pairwise_similarities.append(pairwise_similarities_sign_3)
		hcd_spec_3_pairwise_similarities.append(pairwise_similarities_spec_3)
		hcd_sign_9_pairwise_similarities.append(pairwise_similarities_sign_9)
		hcd_spec_9_pairwise_similarities.append(pairwise_similarities_spec_9)
	hcd_sign_3_pairwise_similarities, hcd_spec_3_pairwise_similarities, hcd_sign_9_pairwise_similarities, hcd_spec_9_pairwise_similarities = np.stack(hcd_sign_3_pairwise_similarities), np.stack(hcd_spec_3_pairwise_similarities), np.stack(hcd_sign_9_pairwise_similarities), np.stack(hcd_spec_9_pairwise_similarities)
	hcd_sign_3_pairwise_similarities, hcd_spec_3_pairwise_similarities, hcd_sign_9_pairwise_similarities, hcd_spec_9_pairwise_similarities = hcd_sign_3_pairwise_similarities / hcd_sign_3_pairwise_similarities.max(), hcd_spec_3_pairwise_similarities / hcd_spec_3_pairwise_similarities.max(), hcd_sign_9_pairwise_similarities / hcd_sign_9_pairwise_similarities.max(), hcd_spec_9_pairwise_similarities / hcd_spec_9_pairwise_similarities.max()

	# evaluate metrics
	bdgm_3_metrics = np.array([get_metrics(true_3_communities, pred_3_communities, true_3_pairwise_similarities, pred_3_pairwise_similarities) for pred_3_communities, pred_3_pairwise_similarities in zip(bdgm_3_communities, bdgm_3_pairwise_similarities)]).mean(axis = 0)
	hcd_sign_3_metrics = np.array([get_metrics(true_3_communities, pred_3_communities, true_3_pairwise_similarities, pred_3_pairwise_similarities) for pred_3_communities, pred_3_pairwise_similarities in zip(hcd_sign_3_communities, hcd_sign_3_pairwise_similarities)]).mean(axis = 0)
	hcd_spec_3_metrics = np.array([get_metrics(true_3_communities, pred_3_communities, true_3_pairwise_similarities, pred_3_pairwise_similarities) for pred_3_communities, pred_3_pairwise_similarities in zip(hcd_spec_3_communities, hcd_spec_3_pairwise_similarities)]).mean(axis = 0)
	bdgm_9_metrics = np.array([get_metrics(true_9_communities, pred_9_communities, true_9_pairwise_similarities, pred_9_pairwise_similarities) for pred_9_communities, pred_9_pairwise_similarities in zip(bdgm_9_communities, bdgm_9_pairwise_similarities)]).mean(axis = 0)
	hcd_sign_9_metrics = np.array([get_metrics(true_9_communities, pred_9_communities, true_9_pairwise_similarities, pred_9_pairwise_similarities) for pred_9_communities, pred_9_pairwise_similarities in zip(hcd_sign_9_communities, hcd_sign_9_pairwise_similarities)]).mean(axis = 0)
	hcd_spec_9_metrics = np.array([get_metrics(true_9_communities, pred_9_communities, true_9_pairwise_similarities, pred_9_pairwise_similarities) for pred_9_communities, pred_9_pairwise_similarities in zip(hcd_spec_9_communities, hcd_spec_9_pairwise_similarities)]).mean(axis = 0)

	result_df = pd.concat([result_df, pd.DataFrame([
		[N, 'BDGM', 1, 3, *list(bdgm_3_metrics)],
		[N, 'HCD-Sign', 1, 3, *list(hcd_sign_3_metrics)],
		[N, 'HCD-Spec', 1, 3, *list(hcd_spec_3_metrics)],
		[N, 'BDGM', 2, 9, *list(bdgm_9_metrics)],
		[N, 'HCD-Sign', 2, 9, *list(hcd_sign_9_metrics)],
		[N, 'HCD-Spec', 2, 9, *list(hcd_spec_9_metrics)],
	], columns = result_df_columns)], ignore_index = True)
	print(f'Metrics evaluated for sample size N = {N}.')

result_df.to_csv('output/simulation/bdgm_vs_hcd.csv', index = False)
