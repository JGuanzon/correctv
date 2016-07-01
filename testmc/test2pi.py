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

# Save and load system
x = np.array([0.47309812, 0.62203426, 0.81436239, 1.03622309, 1.13577798, 1.18144536,
     1.44947567, 1.78214638, 1.87481509, 1.95159825, 2.09900717, 2.31826688,
     2.42992207, 2.51350519, 2.95665482, 2.95829245, 3.3058118,  3.30769273,
     3.54615659, 3.99723336, 4.53070362, 4.54004583, 4.59971426, 4.99015477,
     5.60822604, 5.76024738, 5.81712598, 5.87262993, 6.0147759,  6.02561861,
     6.07655375, 6.89289927, 7.1403128,  7.33723313, 7.44183504, 7.56534686,
     7.74972617, 8.47383004, 8.50475187, 8.7184224,  8.90476975, 8.91907784,
     9.08023877, 9.34042794, 9.42524555, 9.69072934, 9.79268412, 9.97083039,
     9.98336229, 9.98748293])
ye = np.array([0.21368209, 0.22905571, 0.35481151, 0.31619723, 0.40755549, 0.30152046,
      0.37083179, 0.22387475, 0.18361903, 0.11049344, 0.45224703, 0.24848633,
      0.46626268, 0.45452195, 0.30438812, 0.49890995, 0.11846827, 0.3787327,
      0.30189437, 0.23784413, 0.32563293, 0.11767763, 0.45401808, 0.59533977,
      0.42762569, 0.14524736, 0.24591079, 0.58109867, 0.16595405, 0.58711162,
      0.43594151, 0.22919483, 0.54059551, 0.13780241, 0.54699614, 0.49917144,
      0.19228373, 0.29461836, 0.58239004, 0.33973395, 0.25460515, 0.44566806,
      0.56090841, 0.25801077, 0.450436, 0.15616435, 0.1760339, 0.22378059,
      0.11086911, 0.47701686])
y = np.array([2.97640906, 5.78642737, 5.45388087, 1.72594738, 0.99823832, 3.91505512,
     -0.0300223, 4.84328516, 2.06494953, 0.77506309, 3.03613988, 0.7877017,
     2.8343958, 1.65918707, 1.6070071, 2.17631714, 0.67930218, 2.31395568,
     0.15495991, 0.47322884, 0.14053288, -0.08975287, 0.11908669, -0.7186827,
     -1.55504861, -1.03848986, -2.40433136, -0.52385319, -1.89203263, -1.9752534,
     -0.09652228, -2.12487223, -2.56230105, -2.46358637, -1.73648344, -3.39276234,
     -1.97904063, -4.22596076, -6.58292155, -1.53024399, -3.20455249, -0.98158909,
     -2.56254208, -3.78768435, -6.257571, -3.77123499, -2.02665781, -7.4580868,
    -8.06534201, -3.40989655])
print x
print ye
print y

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
