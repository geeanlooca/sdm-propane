import numpy as np
import matplotlib.pyplot as plt

def stokes_to_jones(sop):


    ndim = sop.ndim

    if ndim == 1:
        sop_ = np.reshape(sop, (sop.size,1))
    else:
        sop_ = sop

    # sops concatenated along second dimension
    points = sop_.shape[1]

    output = np.zeros((2, points), dtype=np.complex)

    Q=sop_[0]
    U=sop_[1]
    V=sop_[2]

    for x in range(points):
        A=np.sqrt((1+Q[x])/2)

        if A == 0:
            B=1
        else:
            B = U[x]/(2*A)+1j*V[x]/(2*A)
        
        output[0, x] = A
        output[1, x] = B
    
    return output.squeeze()


def compute_stokes(E):
    ndim = E.ndim
    if ndim == 1:
        S = np.zeros((3,))
    else:
        S = np.zeros((3, E.shape[1]))

    I = np.abs(E[0] ** 2) + np.abs(E[1]) ** 2
    S[0] = (np.abs(E[0] ** 2) - np.abs(E[1]) ** 2) / I
    S[1] = (2 * np.real(E[0] * np.conj(E[1]))) / I
    S[2] = (-2 * np.imag(E[0] * np.conj(E[1]))) / I

    return S

def plot_sphere(pts=30):
    ax = plt.gca()

    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = 1 * np.outer(np.cos(u), np.sin(v))
    y = 1 * np.outer(np.sin(u), np.sin(v))
    z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', edgecolor="k", linewidth=.0, alpha=0.1)
    ax.plot(np.sin(u),np.cos(u),0,color='k', linestyle = 'dashed', linewidth=0.5)
    ax.plot(np.sin(u),np.zeros_like(u),np.cos(u),color='k', linestyle = 'dashed', linewidth=0.5)
    ax.plot(np.zeros_like(u),np.sin(u),np.cos(u),color='k', linestyle = 'dashed', linewidth=0.5)

    ax.plot([0,1],[0,0],[0,0],'k--',lw=1.5, alpha=0.5)
    ax.plot([0,0],[0,1],[0,0],'k--',lw=1.5, alpha=0.5)
    ax.plot([0,0],[0,0],[0,1],'k--',lw=1.5, alpha=0.5)
    ax.set_xlabel(r"$S_1$")
    ax.set_ylabel(r"$S_2$")
    ax.set_zlabel(r"$S_3$")
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15


def random_sop(npts=1, astuple=False):
    azimuth = np.random.uniform(low=0, high=2*np.pi, size=(npts,))
    inclination = np.arccos(1 - 2*np.random.uniform(low=0, high=1, size=(npts,)))
    radius = 1.
    x = radius * np.sin(inclination) * np.sin(azimuth)
    y = radius * np.sin(inclination) * np.cos(azimuth)
    z = radius * np.cos(inclination)

    if astuple:
        return x, y, z
    else:
        return np.array([x,y,z])

def plot_stokes(sop, **kwargs):
    ax = plt.gca()
    ax.scatter(sop[0],  sop[1], sop[2], **kwargs)

def plot_stokes_trajectory(sop, plot_sphere=False, jones=False, plot_kw={}, scatter_kw={}):
    ax = plt.gca()
        
    if plot_sphere:
        plot_sphere()

    lines = ax.plot3D(sop[0], sop[1], sop[2], linewidth=1.5, **plot_kw)
    color = lines[-1].get_color()
    ax.scatter3D(sop[0, 0], sop[1, 0], sop[2, 0], color=color, marker="o", **scatter_kw)
    ax.scatter3D(sop[0, -1], sop[1, -1], sop[2, -1], color=color, marker="x", **scatter_kw)
    ax.set_xlabel(r"$S_1$")
    ax.set_ylabel(r"$S_2$")
    ax.set_zlabel(r"$S_3$")
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    ax.zaxis.labelpad = 15
