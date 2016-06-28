import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

# Maximum likelihood estimation function
def lnlike(theta,x,y,ye):
    m, b, lnf = theta
    model = m*x+b
    inv_sigma2 = 1.0/(ye**2+model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# True parameters
mt = -0.9594
bt = 4.294
ft = 0.534
xt = np.arange(0,10.5)
yt = mt*xt+bt
print "True values:"
print "m =",mt
print "b =",bt

# Generating fake data with randomization
N = 50
x = np.sort(10*np.random.rand(N))
ye = 0.1 + 0.5*np.random.rand(N)
y = mt*x+bt
y += np.abs(ft*y) * np.random.randn(N)
y += ye * np.random.randn(N)

# Least squares fit
A = np.vstack((np.ones_like(x),x)).T
C = np.diag(ye*ye)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
bls, mls = np.dot(cov, np.dot(A.T, np.linalg.solve(C,y)))
yls = mls*xt+bls
print "Least squares fit:"
print "m =",mls
print "b =",bls

# Maximum likelihood estimation
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [mt, bt, np.log(ft)], args=(x,y,ye))
mml, bml, lnfml = result["x"]
yml = mml*xt+bml
print "Maximum likelihood fit:"
print "m = ",mml
print "b = ",bml

# Plots
plt.plot(xt,yt,'-',xt,yls,'--',xt,yml,':')
plt.errorbar(x,y,yerr=ye,fmt='.')
plt.show()
