import numpy as np
import numpy.linalg as nlg


def is_class_M(mat):
	"""
	mat: np.ndarray of shape d x d
	check if binary square matrix mat belongs to class M_d
	return: True/False
	"""
	# condition (i)
	if (np.diag(mat) != 1).any():
		return False
	# condition (ii)
	d = mat.shape[0]
	ij_pairs = [(i, j) for i in range(d) for j in range(i)]
	subset_sum_val_set = set()
	for subset in range(2 ** len(ij_pairs)):
		subset_sum = np.zeros((d, d))
		for i, j in ij_pairs:
			if subset % 2:
				subset_sum += np.outer(mat[:, i], mat[:, j]) + np.outer(mat[:, j], mat[:, i])
			subset //= 2
		subset_sum_val = tuple(list((subset_sum - np.diag(np.diag(subset_sum))).astype(int).reshape(-1)))
		if subset_sum_val in subset_sum_val_set:
			return False
		subset_sum_val_set.add(subset_sum_val)
	return True


if __name__ == "__main__":
	print(is_class_M(np.array([
		[1, 1, 0, 0, 1],
		[0, 1, 1, 1, 0],
		[1, 0, 1, 0, 0],
		[1, 0, 1, 1, 0],
		[1, 1, 0, 0, 1]
	])))
