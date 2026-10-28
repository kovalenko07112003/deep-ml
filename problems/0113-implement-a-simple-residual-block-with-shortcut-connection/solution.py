import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	def RELU(x: np.ndarray) -> np.ndarray:
		return np.maximum(0, x)
	first_layer = RELU(x @ w1)
	second_layer = RELU(w2 @ first_layer + x)
	return second_layer