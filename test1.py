# Testing the probability function

import math
import numpy as np
from scipy.integrate import quad
import scipy.optimize as op


# Distance Calculation Functions
def Ez(z, Omega_M, Omega_L, w_o, w_a):
    # scale factor
    a = 1 / (1 + z)

    # equation of state of Dark Energy w(z)
    w_z = w_o + w_a * (z / a)

    # E(z)
    Ez = Omega_L * math.pow((1 + z), (3 * (1 + w_z))) + Omega_M * math.pow((1 + z), 3)
    print z, Omega_M, Omega_L, w_o, w_a, Ez
    Ez = math.sqrt(Ez)
    Ez = 1.0 / Ez
    return Ez

def distance_modulus(z_hel, z_cmb, Omega_M, Omega_L, w_o, w_a):
    # Inverted E(z)
    #iEz = lambda z: 1.0 / math.sqrt(Omega_L * math.pow(1 + z, 3 * (1 + w_o + w_a * (z / (1 + z)))) + Omega_M * math.pow(1 + z, 3))
    # integrate E(z) to get comoving distance with factor in units
    Dc = []
    for i in range(0, len(z_cmb)):
        Dci, error = quad( Ez, 0, z_cmb[i], args=(Omega_M, Omega_L, w_o, w_a)  )
        Dc = np.append(Dc, Dci)
    Dc = Dc*(299792.458/70.0)
    # calculate the luminosity distance
    Dl = Dc*(1.0+z_hel)
    # convert to distance modulus
    mu = 25 + 5.0 *  np.log10(Dl)
    return mu

# Defines likelihood.  has to be ln likelihood
def lnlike(theta, zhel, zcmb, invdmodm, mod):
    my_Om0, my_w0, my_wa = theta

    # Theory
    modt = distance_modulus(zhel, zcmb, my_Om0, (1.0-my_Om0), my_w0, my_wa)

    # ChSq
    Delta = mod - modt
    ChSq = np.dot(Delta, (np.dot(invdmodm, Delta)))
    print ChSq
    return -0.5 * ChSq

# Load data
FileName = 'jla_lcparams-header.txt'
DataBlock = np.genfromtxt(FileName, skip_header=1, delimiter=' ')
zcmb = DataBlock[:, 1]
zhel = DataBlock[:, 2]
mb = DataBlock[:, 4]
x1 = DataBlock[:, 6]
color = DataBlock[:, 8]
thirdvar = DataBlock[:, 10]
# best fit values from Betoule paper
alpha = 0.141
beta = 3.101
M_1_B = -19.05
Delta_M = -0.07
# Calculate Modulus Distance mod
mod = mb - (M_1_B - alpha * x1 + beta * color)
for i in range(0, len(zcmb)):
    if thirdvar[i] > 10:
        mod[i] = mod[i] - Delta_M
#Independent variance matrix
dmod = np.genfromtxt('Cvm.txt', delimiter=' ') #This covariance of distance modulus
dmodi = np.sqrt(dmod.diagonal()) #independent sqrt
dmodm = np.diag(dmodi)
invdmodm = np.linalg.pinv(dmodm)

nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [0.3, -1, 0], args=(zhel, zcmb, invdmodm, mod))
Om0, w0, wa = result["x"]
print "Optimised Values:"
print "$\Omega_M$ =", Om0
print "$\omega_0$ =", w0
print "$\omega_a$ =", wa