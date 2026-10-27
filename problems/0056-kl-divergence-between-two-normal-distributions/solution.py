import numpy as np

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
	elem1 = (mu_p + mu_q) ** 2/ sigma_q ** 2
	elem2 = sigma_p ** 2 / sigma_q ** 2
	elem3 = np.log(elem2)
	result = 1 / 2 * (elem1 + elem2 - elem3 - 1)
	return result  
