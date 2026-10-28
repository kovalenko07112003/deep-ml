import math
import numpy as np 

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# make predictions 
	probabilities = []
	for x in features: 
		predictions = sum(xi * wi for xi, wi in zip(x, weights)) + bias
		p = 1 / (1 + np.exp(-predictions))
		probabilities.append(p)

	mse = sum(((p - y) ** 2 for p, y in zip(probabilities,labels))) / len(labels)
	return ([round(p ,4) for p in probabilities], round(mse,4))