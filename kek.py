import numpy as np


def kek():
    k = np.vstack([[1, 2], [3, 4], [5, 6]])
    l = np.hstack([1, 10])
    print(k / l)

kek()