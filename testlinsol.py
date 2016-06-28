import numpy as np

#y = np.array([1,1,1])
#y = np.array([3,5,7]).T
y = np.array([[1, 3], [1, 5], [1, 7]])
a = np.array([[3, 0, 0], [0, 4, 0], [0, 0, 5]])
print y
print a
v = np.linalg.solve(a, y)
print v
