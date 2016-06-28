import numpy as np

#y = np.array([1,1,1])
#y = np.array([3,5,7]).T
y = np.array([[1, 3], [1, 5], [1, 7]])
a = np.array([[3, 0, 0], [0, 4, 0], [0, 0, 5]])
print y
print a
v = np.linalg.solve(a, y)
print v

#b = np.array([3,4,20])
b = np.array([[3, 4], [4, 4], [20, 4]]).T
c = np.array([[2, 5], [3, 6], [4, 5]])
d = np.dot(b, c)
print b
print c
print d


