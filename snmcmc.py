import numpy as np
import emcee
import pyfits
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt
import corner
import time
t = time.time()


# ****** Distance Calculation Functions ******

def iEz(z, Omega_M, Omega_L, w_o, w_a):
	# scale factor
	a = 1/(1+z)
	# equation of state of Dark Energy w(z)
	w_z = w_o + w_a * ( z * a )
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
    my_Om0, my_w0, my_wa = theta
    if  0.0 < my_Om0 < 1.0 and -2.0 < my_w0 < -0.0:
        return 0.0
    return -np.inf

# Defines likelihood.  has to be ln likelihood
def lnlike(theta, zhel, zcmb, dmodm, mod):
    my_Om0, my_w0, my_wa = theta

    # Assemble covariance matrix
    #Cmu = np.zeros_like(Ceta[::3, ::3])
    #for i, coef1 in enumerate([1., alpha, -beta]):
    #    for j, coef2 in enumerate([1., alpha, -beta]):
    #        Cmu += (coef1 * coef2) * Ceta[i::3, j::3]

    # Add diagonal term from Eq. 13
    #sigma = np.loadtxt('covmat/sigma_mu.txt')
    #sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
    #Cmu[np.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2

    # Theory
    mod_theory = []
    for i in range(0, len(zcmb)):
        mod_i = distance_modulus(zhel[i], zcmb[i], my_Om0, (1.0-my_Om0), my_w0, my_wa)
        mod_theory = np.append(mod_theory,mod_i)

    # ChSq
    Delta = mod - mod_theory
    inv_CM = np.linalg.pinv(dmodm)
    ChSq = np.dot(Delta, (np.dot(inv_CM, Delta)))

    # Write parameters
    #f_handle = open('Chains/my_params_JLA_FlatwCDM_20160610c.txt', 'a')
    #stringOut = str(my_Om0) +  ',' + str(my_w0)  + ',' + str(
    #    alpha) + ',' + str(beta) + ',' + str(M_1_B) + ',' + str(Delta_M) + '\n'
    #f_handle.write(stringOut)
    #f_handle.close()

    #f_handle = open('ChSq_Chains/ChSqFile_JLA_FlatwCDM_20160610c.txt', 'a')
    #stringOut = str(ChSq) + '\n'
    #f_handle.write(stringOut)
    #f_handle.close()
    print ChSq
    return -0.5 * ChSq

# lnprob - this just combines prior with likelihood
def lnprob(theta, zhel, zcmb, dmodm, mod):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, zhel, zcmb, dmodm, mod)


# ****** load eta covariance matrix ******

#Ceta = pyfits.getdata('C_total_20160610.fits')


# ****** load JLA ******

FileName = 'jla_lcparams-header.txt'
DataBlock = np.genfromtxt(FileName, skip_header=1, delimiter=' ')

zcmb = DataBlock[:, 1]
zhel = DataBlock[:, 2]
mb = DataBlock[:, 4]
x1 = DataBlock[:, 6]
color = DataBlock[:, 8]
thirdvar = DataBlock[:, 10]
set = DataBlock[:, 17]
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
my_wa = 0.0


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
dmodm = np.diag(dmodi)

dmodir = dmodi/mod
dd_c = dmodir*d_c


# ****** Start emcee ******

startValues = [my_Om0, my_w0, my_wa]

# how many parameters to fit
ndim = len(startValues)

# how many walkers
nwalkers = 10
nSteps = 1
pos = [startValues + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]

if __name__ == '__main__':
    # setup the sampler
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(zhel, zcmb, dmodm, mod), threads=2)
    # run the sampler
    # how many steps (will have nSteps*nwalkers of samples)
    #sampler.run_mcmc(pos, nSteps)
    #samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
    #samples = sampler.chain.reshape((-1, ndim))
    #fig = corner.corner(samples, labels=["$Omega_0$", "$w_0$", "$w_a$"], truths=[0.3, -1, 0])

    elapsed = time.time() - t
    print elapsed

    # ****** Calculating Observational Hubble's Law ******
    cLight = 299792.458
    cz = cLight*zcmb
    Hobs = cz/d_c
    Hav = np.average(Hobs)

    plt.figure()
    plt.errorbar(d_c, cz, xerr=dd_c, fmt='.')
    plt.title('JLA Hubbles Diagram ($cz = H_0d$)')
    plt.xlabel('Comoving Distance')
    plt.ylabel('Velocity cz')

    #Theoretical Mod Values
    modt = []
    for i in range(0, len(zcmb)):
        modi = distance_modulus(zhel[i], zcmb[i], my_Om0, (1.0-my_Om0), my_w0, my_wa)
        modt = np.append(modt,modi)

    setval = ['SNLS', 'SDSS', 'Low-z', 'HST']
    orderval = [3, 2, 1, 4]
    colorval = ['DarkGoldenRod', 'SeaGreen', 'blue', 'red']
    plt.figure()
    for i in orderval:
        ind = [n for n, s in enumerate(set == i) if s]
        plt.errorbar(zcmb[ind], mod[ind], yerr=dmodi[ind], fmt='.', color=colorval[i-1], label=setval[i-1])
    plt.plot(sorted(zcmb),sorted(modt),color='black',label='Theoretical')
    plt.legend(loc='best')
    plt.title('JLA Hubbles Diagram')
    plt.xlabel('Red shift $z_{cmb}$')
    plt.ylabel('Modulus Distance $\mu$')
    plt.axis([0, 1.3, 32, 46])


# ****** Create Fake Data ******
plt.figure()
plt.plot(zcmb,zhel)
plt.title('JLA Comparing the CMB frame redshift and the Heliocentric redshift')
plt.xlabel('$z_{cmb}$')
plt.ylabel('$z_{hel}$')

plt.figure()
for i in orderval:
    ind = [n for n, s in enumerate(set == i) if s]
    plt.hist(zcmb[ind]-zhel[ind], color=colorval[i - 1], label=setval[i - 1], alpha=0.7)
plt.legend(loc='best')
plt.title('JLA Histogram $z_{cmb}-z_{hel}$ (seperated by SN survey)')
plt.xlabel('$z_{cmb}-z_{hel}$')
plt.ylabel('Frequency')
zstd = np.std(zcmb-zhel)

plt.figure()
plt.hist(zcmb-zhel)
plt.title('JLA Histogram $z_{cmb}-z_{hel}$ (total)')
plt.xlabel('$z_{cmb}-z_{hel}$')
plt.ylabel('Frequency')

# Create fake zcmb and zhel
nf = 10000 #number of fake data points
zcmbf = np.random.rand(n)
zhelf = zcmbf + np.random.randn(n)*zstd

plt.figure()
plt.hist(zcmbf-zhelf)
plt.title('Fake Histogram $z_{cmb}-z_{hel}$ (total)')
plt.xlabel('Fake $z_{cmb}-z_{hel}$')
plt.ylabel('Frequency')

# Other fake variables
OMf = 0.3 #Omega_M fake, matter density parameter
OLf = 1-OMf #Omega_Lambda fake, cosmological constant/dark energy density parameter
w0f = -1.0 #w_0 fake, first dark energy equation of state variable, should be approx -1
waf = 0.0 #w_a fake, second dark energy equation of state variable, around 0?

# Create fake distance modulus
dmft = []
for i in range(0, len(zcmbf)):
    dmfi = distance_modulus(zhelf[i], zcmbf[i], OMf, OLf, w0f, waf)
    dmft = np.append(dmft, dmfi)
dmf = dmft + np.random.randn(n)*0.14 # fake distance modulus
ddmf = np.ones(n)*0.14 # uncertainty dm

plt.figure()
plt.errorbar(zcmbf, dmf, yerr=ddmf, fmt='.', label='Fake Data')
plt.plot(sorted(zcmbf), sorted(dmft), color='black', label='Theoretical')
plt.legend(loc='best')
plt.title('Fake Hubbles Diagram')
plt.xlabel('Fake Red shift $z_{cmb}$')
plt.ylabel('Fake Modulus Distance $\mu$')
plt.axis([0, 1.3, 32, 46])

# 

plt.show()