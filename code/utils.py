import numpy as np
import time


# global variable controling on/off of timeit wrapper
TIMEIT_ON = False


def timeit(func):
	"""
	creates wrapper for timing the running time of func
	"""
	def wrapper(*args, **kwargs):
		t0 = time.time()
		result = func(*args,  **kwargs)
		dt = time.time() - t0
		print("{} uses {} seconds.".format(str(func)[:str(func).find(" at ")], np.round(dt, 4)))
		return result
	return wrapper if TIMEIT_ON else func


def random_seed_generator(seed):
	"""
	creates a reproducible generator of an infinite sequence of random seeds using a given seed
	"""
	np.random.seed(seed)
	seed_arr, i_seed = np.random.randint(0, 1e8, 10000), -1
	while 1:
		i_seed += 1
		if i_seed == 9999:
			np.random.seed(seed_arr[-1])
			seed_arr, i_seed = np.random.randint(0, 1e8, 10000), 0
		yield seed_arr[i_seed]
