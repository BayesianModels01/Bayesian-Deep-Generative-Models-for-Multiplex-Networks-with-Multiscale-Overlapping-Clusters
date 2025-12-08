import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm
from scipy.io import loadmat
from spec_init import *
from sample import *
from sklearn.metrics import roc_curve, auc


# load brain connectivity network data
dic = loadmat("data/brain_connectivity_network_data.mat")
brain_networks = dic["loaded_bd_network"]
brain_networks += np.transpose(brain_networks, (1, 0, 2))
brain_networks = np.transpose(brain_networks, (2, 0, 1)) + np.identity(68).reshape(1, 68, 68)
print("Brain connectivity network data loaded from data/brain_connectivity_network_data.mat.")

# mask out a block of brain connectivity networks
entries = [1,  2,  3,  4, 13, 15, 16, 18, 19, 24, 27, 30, 31, 37, 38, 48, 50, 53, 65, 66]
onehot_entries = np.array([1 if i in entries else 0 for i in range(68)])
mask = (1 - np.outer(onehot_entries, onehot_entries)).astype(np.float64)
print("Mask matrix generated.")

# load masked gibbs samples
K, p, sparsity = 2, [4, 18, 68], 2
gibbs_samples = []
for t in tqdm(range(10000, 10100)):
	with open(os.path.join("mask_samples", f"iter_{t + 1}.p"), 'rb') as hf:
		gibbs_samples.append(pickle.load(hf))

# generate masked prediction of our Bayesian deep generative model using gibbs samples
bdgm_pred_samples = []
for sample in gibbs_samples:
	C, Gamma = convert_theta_to_C_Gamma(p, sample.theta)
	C_K, Gamma_K, A_K_mask = C[K], Gamma[K], sample.A[K][entries]
	exponent = C_K + A_K_mask @ (sample.X_subset[K - 1] * Gamma_K.reshape(1, p[K - 1], p[K - 1])) @ A_K_mask.T
	bdgm_pred_samples.append(1 / (1 + np.exp(- exponent)))
bdgm_pred = np.stack(bdgm_pred_samples).mean(axis = 0)

# load masked dcmm predictions
dcmm_pred = []
for i in range(brain_networks.shape[0]):
	dcmm_pred.append(np.load(f'mask_em_dcmm/subject_{i + 1}.npy'))
dcmm_pred = np.stack(dcmm_pred)

# subtract off-diagonal entries of masked submatrices
true_vec = np.column_stack([brain_networks[:, entries[i], entries[j]] for i in range(20) for j in range(i)]).reshape(-1)
bdgm_vec = np.column_stack([bdgm_pred[:, i, j] for i in range(20) for j in range(i)]).reshape(-1)
dcmm_vec = np.column_stack([dcmm_pred[:, i, j] for i in range(20) for j in range(i)]).reshape(-1)

# evaluate ROC and AUC
bdgm_fpr, bdgm_tpr, bdgm_thresholds = roc_curve(true_vec, bdgm_vec)
bdgm_auc = auc(bdgm_fpr, bdgm_tpr)
dcmm_fpr, dcmm_tpr, dcmm_thresholds = roc_curve(true_vec, dcmm_vec)
dcmm_auc = auc(dcmm_fpr, dcmm_tpr)
plt.figure(figsize = (6, 5))
plt.plot(bdgm_fpr, bdgm_tpr, color = '#1f77b4', label = f'BDGM (AUC = {bdgm_auc:.3f})')
plt.plot(dcmm_fpr, dcmm_tpr, color = '#ff7f0e', label = f'DCMM (AUC = {dcmm_auc:.3f})')
plt.plot([0, 1], [0, 1], color = 'black', lw = 1, linestyle = '--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.title('ROC curves')
plt.legend(loc = 0)
plt.grid(ls = ':')
plt.tight_layout()
plt.savefig('output/application/mask_pred_roc.png', dpi = 300, bbox_inches = "tight")
