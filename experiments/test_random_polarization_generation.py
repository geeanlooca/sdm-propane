import sys
import os
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(cwd))

import matplotlib.pyplot as plt
from polarization import plot_sphere, random_sop, stokes_to_jones, compute_stokes, plot_stokes



sops = random_sop(num=30)
jones = stokes_to_jones(sops)
sops_reconstructed = compute_stokes(jones)

fig = plt.figure(figsize=(7, 5))
ax = plt.axes(projection='3d')

plot_sphere()   
plot_stokes(sops,  marker="x", label="Original")
plot_stokes(sops_reconstructed,  marker="o", label="Reconstructed")


ax.set_xlabel(r"$S_1$")
ax.set_ylabel(r"$S_2$")
ax.set_zlabel(r"$S_3$")
ax.xaxis.labelpad=15
ax.yaxis.labelpad=15
ax.zaxis.labelpad=15
ax.legend()
plt.show()
