import numpy as np
from numba import njit
import matplotlib.pyplot as plt

@njit
def b_maxL_2(m, t, C):
    first = C[0]+C[1]-(m[0]+m[1])*(t[0]+t[1])
    root = (C[0]+C[1]+(m[0]-m[1])*(t[0]+t[1]))**2-4*C[0]*(m[0]-m[1])*(t[0]+t[1])
    res = (first+np.sqrt(root))/(2*(t[0]+t[1]))
    # if res < 0:
    #     print()
    #     print("WARNING: Maximum Likelihood Background is less than 0!")
    #     print(res)
    return res


@njit
def calc_b_distribution(b, s1, t1, s2, t2, num_samples=1000000):
    b_max_L = np.zeros(num_samples)
    for n_i in range(num_samples):
        s1m = np.random.poisson(t1 * s1)
        s2m = np.random.poisson(t2 * s2)
        b1m = np.random.poisson(t1 * b)
        b2m = np.random.poisson(t2 * b)
        
        c1m = s1m + b1m
        c2m = s2m + b2m
        b_max_L[n_i] = b_maxL_2(np.array([s1, s2]), np.array([t1, t2]), np.array([c1m, c2m]))
        
        
    return b_max_L
                        
fig, axes = plt.subplots()

s1 = 10
s2 = 1
b = 1000
t1 = 100000
t2 = 100000

bmaxL = calc_b_distribution(b, s1, t1, s2, t2)

axes.hist(bmaxL, 30)
axes.axvline(b, label="True", c="C1")
axes.axvline(np.mean(bmaxL), label="Average", c="C2")

axes.legend()


plt.savefig("./main_files/max_L_background_distribution_test/test.pdf")