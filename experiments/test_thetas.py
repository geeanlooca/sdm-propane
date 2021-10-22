import sys
import os
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))
import math
import numpy as np
import matplotlib.pyplot as plt 
from perturbation_angles import generate_perturbation_angles

generate_thetas = generate_perturbation_angles

fiber_length = 1e3
Lf = 10

runs = 10000

def get_acorr(Lf, dz, fiber_length):
    step_count = math.ceil(fiber_length/dz)
    psd = np.zeros((step_count, ))
    for i in range(runs):
        thetas = generate_thetas(Lf, dz, fiber_length)
        cosTheta = np.cos(thetas)
        psd = (psd * i  + np.abs(np.fft.fft(cosTheta)) ** 2) / (i+1)

    acorr = np.fft.ifft(psd)
    acorr /= acorr[0]
    z = np.arange(step_count) * dz

    return z, acorr


def verify_stationarity(Lf, dz, fiber_length):
    step_count = math.ceil(fiber_length/dz)
    N = int(step_count/2)
    psdA = np.zeros((N, ))
    psdB = np.zeros((N, ))
    for i in range(runs):
        thetas = generate_thetas(Lf, dz, fiber_length)
        cosTheta = np.cos(thetas)
        cosThetaA = cosTheta[:N]
        cosThetaB = cosTheta[N:]

        psdA = (psdA * i  + np.abs(np.fft.fft(cosThetaA)) ** 2) / (i+1)
        psdB = (psdB * i  + np.abs(np.fft.fft(cosThetaB)) ** 2) / (i+1)

    acorrA = np.fft.ifft(psdA)
    acorrA /= acorrA[0]

    acorrB = np.fft.ifft(psdB)
    acorrB /= acorrB[0]

    z1 = np.arange(N) * dz
    z2 = np.arange(N) * dz
    return z1, z2, acorrA, acorrB


dz = Lf/10
z1, acorr1 = get_acorr(Lf, dz, fiber_length)

dz = Lf/20
step_count = int(fiber_length / dz)
z2, acorr2 = get_acorr(Lf, dz, fiber_length)

dz = Lf/30
step_count = int(fiber_length / dz)
z3, acorr3 = get_acorr(Lf, dz, fiber_length)

plt.figure()
plt.plot(z1, np.log(acorr1))
plt.plot(z2, np.log(acorr2))
plt.plot(z3, np.log(acorr3))

z1,z2, acorr1, acorr2 = verify_stationarity(Lf, dz, fiber_length)


plt.figure()
plt.plot(z1, acorr1)
plt.plot(z2, acorr2)
plt.show()