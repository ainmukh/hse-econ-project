import cdf_module_np
import matplotlib.pyplot as plt
import numpy as np
import sys

alphas = np.array([1, 0.3, -0.5], dtype=np.float64)
theta = np.array([1], dtype=np.float64)
x = np.array([1.6], dtype=np.float64)

print(cdf_module_np.CDF(2.2, 3.5, alphas, theta, x))
# 0.26169482155955937
# 0.4572475058826808 fx без C1
sys.exit(0)
