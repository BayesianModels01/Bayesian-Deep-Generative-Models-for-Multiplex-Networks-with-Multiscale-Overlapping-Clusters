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

# mask out a block of brain connectivity networks
entries = [1,  2,  3,  4, 13, 15, 16, 18, 19, 24, 27, 30, 31, 37, 38, 48, 50, 53, 65, 66]
onehot_entries = np.array([1 if i in entries else 0 for i in range(68)])
mask = (1 - np.outer(onehot_entries, onehot_entries)).astype(np.float64)
print("Mask matrix generated.")

# degree corrected mixed membership model (DCMM)
def DCMM(X, q):
	"""
	X: np.ndarray of shape p x p - observed adjacency matrix
	q: number of communities
	DCMM model has E[X] = Theta Pi P Pi^T Theta
	return:
		- mixed membership matrix Pi: np.ndarray of shape p x q
		- diagonal degree matrix Theta: np.ndarray of shape p x p
		- latent probability matrix P: np.ndarray of shape q x q
	"""
	eigval, eigvec = nlg.eig(X)
	top_eigvec, fiedler_eigvec = eigvec[:, np.argsort(eigval)[-1]], eigvec[:, np.argsort(eigval)[-2:-(q + 1):-1]]
	R = np.nan_to_num(fiedler_eigvec / top_eigvec.reshape(-1, 1), posinf = 0, neginf = 0)
	V = sketched_vertex_search(R)
	W = simplex_mixture_weights(R, V)
	b1 = (eigval[np.argsort(eigval)[-1]] + (eigval[np.argsort(eigval)[-2:-(q + 1):-1]].reshape(1, -1) * V ** 2).sum(axis = 1)) ** (-0.5)
	Pi = np.maximum(0, W / b1.reshape(1, -1))
	Pi /= Pi.sum(axis = 1).reshape(-1, 1)
	Theta = np.diag(top_eigvec / (Pi @ b1))
	Lambda = np.diag(eigval[np.argsort(eigval)[-1:-(q + 1):-1]])
	B = np.diag(b1) @ np.hstack([np.ones((q, 1)), V])
	P = B @ Lambda @ B.T
	return Pi, Theta, P

# DCMM for prediction of masked missing entries
def DCMM_EM(X, mask, q):
	"""
	X: np.ndarray of shape p x p - observed adjacency matrix
	mask: p x p mask matrix on observed adjacency matrix - mask entry 1 indicates observed, mask entry 0 indicates entry masked out (unobserved)
	q: number of communities
	use EM algorithm with DCMM model to fill in the masked missing entries in X
	return X_filled: np.ndarray of shape p x p with mask = 1 entries same as X, mask = 0 entries filled in
	"""
	X *= mask
	for t in range(30):
		Pi, Theta, P = DCMM(X, q)
		X_filled = Theta @ Pi @ P @ Pi.T @ Theta * (1 - mask) + X * mask
		if np.sum(X_filled - X) / np.sum(1 - mask) < 0.01:
			break
		X = X_filled.copy()
	return X_filled


# fill in missing entries for brain connectivity networks
directory = f"mask_em_dcmm"
os.mkdir(directory)
for i in range(brain_networks.shape[0]):
	X_filled = DCMM_EM(brain_networks[i], mask, 18)
	X_filled_mask_entries = X_filled[entries, :][:, entries]
	np.save(os.path.join(directory, f'subject_{i + 1}.npy'), X_filled_mask_entries)
	print(f'Subject {i + 1} saved.')
