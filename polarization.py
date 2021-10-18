import numpy as np
import matplotlib.pyplot as plt

def stokes_to_jones(sop):

    sop_ = np.atleast_2d(sop)

    # sops concatenated along second dimension
    ndim = sop_.shape[0]

    output = np.zeros((ndim, 2), dtype=np.complex)

    Q=sop_[:, 0]
    U=sop_[:, 1]
    V=sop_[:, 2]

    for x in range(ndim):
        A=np.sqrt((1+Q[x])/2)

        if A == 0:
            B=1
        else:
            B = U[x]/(2*A)+1j*V[x]/(2*A)
        
        output[x, 0] = A
        output[x, 1] = B
    
    return output.squeeze()

def linear_stokes(direction="x", angle=None, num=1):
    if angle:
        x = compute_stokes(np.array([np.cos(angle), np.sin(angle)]))
    else:
        if direction == "x":
            x = np.array([1.0, 0.0, 0.0])
        elif direction == "y":
            x = np.array([-1.0, 0.0, 0.0])
        else:
            raise ValueError("Invalid direction")

    return np.tile(x, (num,1)).squeeze()
    
def linear_hyperstokes(modes, direction="x", angle=None, num=1):
    x = linear_stokes(direction=direction, angle=angle, num=num)
    return np.tile(x, (1, modes)).squeeze()

def compute_stokes(E):
    E_ = np.atleast_2d(E)
    ndim = E.shape[0]
    S = np.zeros((ndim, 3))

    I = np.abs(E_[:, 0] ** 2) + np.abs(E_[:, 1]) ** 2
    S[:, 0] = (np.abs(E_[:, 0] ** 2) - np.abs(E_[:, 1]) ** 2) / I
    S[:, 1] = (2 * np.real(E_[:, 0] * np.conj(E_[:, 1]))) / I
    S[:, 2] = (-2 * np.imag(E_[:, 0] * np.conj(E_[:, 1]))) / I

    return S.squeeze()

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

def random_sop(num=1, astuple=False):
    azimuth = np.random.uniform(low=0, high=2*np.pi, size=(num,))
    inclination = np.arccos(1 - 2*np.random.uniform(low=0, high=1, size=(num,)))
    radius = 1.
    
    x = radius * np.sin(inclination) * np.sin(azimuth)
    y = radius * np.sin(inclination) * np.cos(azimuth)
    z = radius * np.cos(inclination)

    if astuple:
        return x, y, z
    else:
        sop = np.zeros((num, 3))
        sop[:, 0] = x
        sop[:, 1] = y
        sop[:, 2] = z
        return sop.squeeze()

def random_hypersop(nmodes, num=1):
    """Generate a vector of states of polarization of the number of specified modes."""

    sops = [random_sop(num=num).squeeze() for _ in range(nmodes)]

    return np.hstack(sops).squeeze()

def random_hyperjones(nmodes, num=1):
    """Generate a generalized Jones vector with random polarization for each mode."""
    hsop = random_hypersop(nmodes, num=num)
    hjones = hyperstokes_to_jones(hsop)
    return hjones.squeeze()

def hyperstokes_to_jones(hsop):
    """Convert an hyper-state of polarization to the equivalent Jones vector."""
    sop = np.atleast_2d(hsop)
    nmodes = sop.shape[1] // 3
    num = sop.shape[0]
    jones = np.zeros((num, 2 * nmodes), dtype=np.complex128)

    for i in range(num):

            s = sop[i]
            s = np.reshape(s, (nmodes, 3))
            j = [stokes_to_jones(x) for x in s]
            jones[i] = np.concatenate(j)


    return jones.squeeze()

def plot_stokes(sop, **kwargs):
    ax = plt.gca()
    ax.scatter(sop.T[0],  sop.T[1], sop.T[2], **kwargs)

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
