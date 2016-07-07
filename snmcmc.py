import numpy as np
import emcee
import pyfits
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt

# ****** Distance Calculation Functions ******

def iEz(z, Omega_M, Omega_L, w_o, w_a):
	# scale factor
	a = 1/(1+z)
	# equation of state of Dark Energy w(z)
	w_z = w_o + w_a * ( z / a )
	# E(z)
	Ez = Omega_L*math.pow(1+z,3*(1+w_z)) + Omega_M*math.pow(1+z,3)
	Ez = math.sqrt(Ez)
	invEz = 1.0 / Ez
	return invEz

def Comoving_Distance(z, Omega_M, Omega_L, w_o, w_a):
	# Speed of light, in km / s
	cLight = 299792.458
	# Hubble's constant, in (km / s) / Mpc
	H_o = 70.0
	# integrate E(z) to get comoving distance
	Dc, error = quad(iEz, 0, z, args=(Omega_M, Omega_L, w_o, w_a))
	# factor in units
	Dc = Dc* (cLight/H_o)
	return Dc

def Luminosity_Distance(z_hel, z_cmb, Omega_M, Omega_L, w_o, w_a):
	# factor in units
	Dc = Comoving_Distance(z_cmb, Omega_M, Omega_L, w_o, w_a)
	# calculate the luminosity distance
	Dl = Dc * (1.0+z_hel)
	return Dl

def distance_modulus(z_hel, z_cmb, Omega_M, Omega_L, w_o, w_a):
	# get the luminosity distance
	d_L = Luminosity_Distance(z_hel, z_cmb, Omega_M, Omega_L, w_o, w_a)
	# convert to distance modulus
	mu = 25 + 5.0 *  np.log10(d_L)
	return mu


# ****** EMCEE Functions ******

# Defines a prior.  just sets acceptable ranges
def lnprior(theta):
    my_Om0, my_w0, alpha, beta, M_1_B, Delta_M = theta
    if  0.0 < my_Om0 < 1.0 and -2.0 < my_w0 < -0.0:
        return 0.0
    return -np.inf

# Defines likelihood.  has to be ln likelihood
def lnlike(theta, zhel, zcmb, mb, x1, color, thirdvar, Ceta):
    my_Om0, my_w0, alpha, beta, M_1_B, Delta_M = theta

    # Assemble covariance matrix
    #Cmu = np.zeros_like(Ceta[::3, ::3])
    #for i, coef1 in enumerate([1., alpha, -beta]):
    #    for j, coef2 in enumerate([1., alpha, -beta]):
    #        Cmu += (coef1 * coef2) * Ceta[i::3, j::3]

    # Add diagonal term from Eq. 13
    #sigma = np.loadtxt('covmat/sigma_mu.txt')
    #sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
    #Cmu[np.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2

    # Observation
    mod = mb - (M_1_B - alpha * x1 + beta * color)
    for i in range(0, len(zcmb)):
        if thirdvar[i] > 10:
            mod[i] = mod[i] - Delta_M

    # Theory
    mod_theory = []
    for i in range(0, len(zcmb)):
        mod_i = distance_modulus(zhel[i], zcmb[i], my_Om0, (1.0-my_Om0), my_w0, 0.0)
        mod_theory = np.append(mod_theory,mod_i)

    # ChSq
    Delta = mod - mod_theory
    #inv_CM = np.linalg.pinv(Cmu)
    ChSq = np.dot(Delta, (np.dot(inv_CM, Delta)))

    # Write parameters
    f_handle = open('Chains/my_params_JLA_FlatwCDM_20160610c.txt', 'a')
    stringOut = str(my_Om0) +  ',' + str(my_w0)  + ',' + str(
        alpha) + ',' + str(beta) + ',' + str(M_1_B) + ',' + str(Delta_M) + '\n'
    f_handle.write(stringOut)
    f_handle.close()

    f_handle = open('ChSq_Chains/ChSqFile_JLA_FlatwCDM_20160610c.txt', 'a')
    stringOut = str(ChSq) + '\n'
    f_handle.write(stringOut)
    f_handle.close()

    return -0.5 * ChSq

# lnprob - this just combines prior with likelihood
def lnprob(theta, zhel, zcmb, mb, x1, color, thirdvar, Ceta):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, zhel, zcmb, mb, x1, color, thirdvar, Ceta)

# ****** load eta covariance matrix ******
#Ceta = pyfits.getdata('C_total_20160610.fits')
Ceta = 0

# ****** load JLA ******
FileName = 'jla_lcparams-header.txt'
DataBlock = np.genfromtxt(FileName, skip_header=1, delimiter=' ')

zcmb = DataBlock[:, 1]
zhel = DataBlock[:, 2]
mb = DataBlock[:, 4]
x1 = DataBlock[:, 6]
color = DataBlock[:, 8]
thirdvar = DataBlock[:, 10]
ra = DataBlock[:, 18]
dec = DataBlock[:, 19]

# best fit values from Betoule paper
alpha = 0.141
beta = 3.101
M_1_B = -19.05
Delta_M = -0.07

H0 = 70.0
my_Om0 = 0.3
Ode0 = 0.7
my_w0 = -1.0
wa = 0.0

startValues = [my_Om0, my_w0, alpha, beta, M_1_B, Delta_M]

# how many parameters to fit
ndim = len(startValues)

# how many walkers
nwalkers = 100
nSteps = 1000
pos = [startValues + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

# setup the sampler
#sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(zhel, zcmb, mb, x1, color, thirdvar, Ceta), threads=16)
# run the sampler
# how many steps (will have nSteps*nwalkers of samples)
#sampler.run_mcmc(pos, nSteps)


# ****** Calculating Observational Comoving Distance ******

# Calculate Modulus Distance mod
mod = mb - (M_1_B - alpha * x1 + beta * color)
for i in range(0, len(zcmb)):
    if thirdvar[i] > 10:
        mod[i] = mod[i] - Delta_M

# Calculate Luminosity Distance d_L
d_L = np.power(10.0,(mod-25.0)/5)

# Calculate Comoving Distance d_c
d_c = d_L/(1.0+zhel)


# ****** Calculating Observational Comoving Distance Covariance Matrix ******
dmod = np.genfromtxt('Cvm.txt', delimiter=' ') #This covariance of distance modulus
dmodi = np.sqrt(dmod.diagonal()) #independent sqrt

dmodir = dmodi/mod
dd_c = dmodir*d_c

# ****** Calculating Observational Hubble's Law ******
cLight = 299792.458
cz = cLight*zcmb
Hobs = cz/d_c
Hav = np.average(Hobs)

plt.figure()
plt.errorbar(cz, d_c, yerr=dd_c, fmt='.')
#plt.plot(d_c, cz, '.')
plt.show()
