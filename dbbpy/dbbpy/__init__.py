import bindings
import numpy as np

# Example input data
n = 4
alpha = 1.3
lambda_val = 1.0
omega_l = np.array([0, 3])
sp = np.array([1200, 1250, 1300, 1350])
strike = np.array([1290, 1295, 1295, 1300])
bid = np.array([27.7, 27.4, 29.4, 25.0])
ask = np.array([29.3, 29.7, 31.4, 26.9])
pFlag = np.array([True, False, True, False])

p, q = bindings.performOptimization(n, alpha, lambda_val, omega_l, sp, strike, bid, ask, pFlag)

print(p)
print(q)