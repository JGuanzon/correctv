import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner


# Maximum likelihood estimation function
def lnlike(theta, x, y, yerr):
    m, b, lnf = theta
    model = m * x + b
    inv_sigma2 = 1.0 / (yerr ** 2 + model ** 2 * np.exp(2 * lnf))
    return -0.5 * (np.sum((y - model) ** 2 * inv_sigma2 - np.log(
        inv_sigma2)))  # For some reason missing 2*pi? Should be + np.log(2*np.pi*inv_sigma2**(-1))


# MCMC: Log prior
def lnprior(theta):
    m, b, lnf = theta
    if -5.0 < m < 0.5 and 0 < b < 10 and -10 < lnf < 1.0:
        return 0.0
    return -np.inf


# MCMC: Log probability function
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


# True parameters
mt = -0.9594
bt = 4.294
ft = 0.534
xt = np.arange(0, 10.5)
yt = mt * xt + bt
print "True values:"
print "m =", mt
print "b =", bt
print "f = ", ft

# Generating fake data with randomization
N = 50
x = np.sort(10 * np.random.rand(N))
ye = 0.1 + 0.5 * np.random.rand(N)
y = mt * x + bt
y += np.abs(ft * y) * np.random.randn(N)
y += ye * np.random.randn(N)

# Least squares fit
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(ye * ye)
T = np.dot(A.T, np.linalg.solve(C, A))
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
bls, mls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))
yls = mls * xt + bls
print "Least squares fit:"
print "m =", mls
print "b =", bls
# Notes for LS
# Minimises the sum of the squared of the differences between the data and a changing model (weighted for uncertainty).
# A: ones_like creates a matrix of 1's same size as x, then vstack puts this matrix ontop of x, then .T takes the transpose.
# C: diag creates a 50x50 matrix where ye*ye vector is along the axis.
# cov: Firstly, linalg.solve solves for x using Cx=A, where A is two columns of solutions thus it will be (ye*ye)^(-1)|a column of answers. Next, dot product between to matrices. Then finally take the inverse of the matrix.

# Maximum likelihood estimation
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [mt, bt, np.log(ft)], args=(x, y, ye))
mml, bml, lnfml = result["x"]
yml = mml * xt + bml
print "Maximum likelihood fit:"
print "m = ", mml
print "b = ", bml
print "f = ", np.exp(lnfml)
# Notes for ML
# For a fixed data set, finds the set of parameters which maximises the likelihood function.
# nll: lambda creates an anonymous function equivalent to the negative lnlike function, passes the arguments to it.
# then passes it through to the minimise function (as its the negative of the function).

# Marginalization & uncertainty estimation
ndim, nwalkers = 3, 100
pos = [result["x"] + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, ye))
sampler.run_mcmc(pos, 500)
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
# Notes for MCMC
# Uses Markov Chain Monte Carlo process to sample paramaters from the lnprob distribution function.
# Initial positions of 100 walkers are a Gaussian sphere of 1e^-4*normal distribution centered at the ML results.
# Then run over 500 steps of the sampler. Finally, removes the first 50 steps and combines the rest together.
samples[:, 2] = np.exp(samples[:, 2])
mmcmc, bmcmc, fmcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                          zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print "MCMC + Uncertainty:"
print "m = ", mmcmc
print "b = ", bmcmc
print "f = ", fmcmc

# Corner Plots
fig = corner.corner(samples, labels=["$m$", "$b$", "$\ln\,f$"], truths=[mt, bt, np.log(ft)])
# fig.savefig("triangle.png")

# Plot Comparing Data vs Actual vs LS vs ML
plt.figure()
plt.plot(xt, yt, '-', xt, yls, '--', xt, yml, ':')
plt.errorbar(x, y, yerr=ye, fmt='.')

# Another Plot
plt.figure()
x1 = np.array([0, 10])
for m, b, lnf in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(x1, m * x1 + b, color="k", alpha=0.1)
plt.plot(x1, mt * x1 + bt, color="r", lw=2, alpha=0.8)
plt.errorbar(x, y, yerr=ye, fmt=".k")
plt.show()
