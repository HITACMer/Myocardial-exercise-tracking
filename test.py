import torch
import numpy as np
import copy
a = np.arange(9).reshape((3, 3))
#b = a
b = copy.deepcopy(a)
b[1][1] = 99
b[0][1] = 131
print(a)
print(a-b)
print(abs(a-b))
print(np.sum(a, axis=None))
