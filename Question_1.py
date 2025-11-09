import numpy as np
import matplotlib.pyplot as plt

c = 1.0                
q = 1.0                  
t0 = 5.0                  
beta = 0.3
gamma = 1.0 / np.sqrt(1 - beta**2)


def field_components(x, y, z, beta, gamma, t):
    """Return E and B field components for uniform motion along x-axis."""
    R = np.sqrt((gamma * (x - beta * c * t))**2 + y**2 + z**2)
    Ex = q * gamma * (x - beta * c * t) / (R**3)
    Ey = q * gamma * y / (R**3)
    Ez = q * gamma * z / (R**3)
    Bx = np.zeros_like(Ex)
    By = -q * gamma * beta * z / (R**3)
    Bz = q * gamma * beta * y / (R**3)
    return Ex, Ey, Ez, Bx, By, Bz


npts = 25
x = np.linspace(-30, 30, npts)
y = np.linspace(-30, 30, npts)
z = np.linspace(-30, 30, npts)

X, Y = np.meshgrid(x, y)

#  Case 1: z = 0 plane 
Ex_xy, Ey_xy, _, Bx_xy, By_xy, Bz_xy = field_components(X, Y, 0, beta, gamma, t0)

#  Case 2: y = 0 plane 
X2, Z = np.meshgrid(x, z)
Ex_xz, _, Ez_xz, Bx_xz, By_xz, Bz_xz = field_components(X2, 0, Z, beta, gamma, t0)

#  Case 3: x = 0 plane 
Y3, Z3 = np.meshgrid(y, z)
_, Ey_yz, Ez_yz, Bx_yz, By_yz, Bz_yz = field_components(0, Y3, Z3, beta, gamma, t0)


def normalize(vx, vy):
    mag = np.sqrt(vx**2 + vy**2)
    mag[mag == 0] = 1.0
    return vx/mag, vy/mag, mag

Exn_xy, Eyn_xy, magE_xy = normalize(Ex_xy, Ey_xy)
Exn_xz, Ezn_xz, magE_xz = normalize(Ex_xz, Ez_xz)
Eyn_yz, Ezn_yz, magE_yz = normalize(Ey_yz, Ez_yz)

Bxn_xy, Byn_xy, magB_xy = normalize(Bx_xy, By_xy)
Bxn_xz, Bzn_xz, magB_xz = normalize(Bx_xz, Bz_xz)
Byn_yz, Bzn_yz, magB_yz = normalize(By_yz, Bz_yz)


fig, axs = plt.subplots(2, 3, figsize=(12, 10))

# Electric fields
axs[0, 0].streamplot(x, y, Exn_xy, Eyn_xy, color=np.log10(magE_xy), cmap='magma', density=1.2)
axs[0, 0].set_title("E-field (x-y plane)")
axs[0, 1].streamplot(x, z, Exn_xz, Ezn_xz, color=np.log10(magE_xz), cmap='plasma', density=1.2)
axs[0, 1].set_title("E-field (x-z plane)")
axs[0, 2].streamplot(y, z, Eyn_yz, Ezn_yz, color=np.log10(magE_yz), cmap='inferno', density=1.2)
axs[0, 2].set_title("E-field (y-z plane)")

# Magnetic fields
axs[1, 0].streamplot(x, y, Bxn_xy, Byn_xy, color=np.log10(magB_xy), cmap='cividis', density=1.2)
axs[1, 0].set_title("B-field (x-y plane)")
axs[1, 1].streamplot(x, z, Bxn_xz, Bzn_xz, color=np.log10(magB_xz), cmap='cividis', density=1.2)
axs[1, 1].set_title("B-field (x-z plane)")
axs[1, 2].streamplot(y, z, Byn_yz, Bzn_yz, color=np.log10(magB_yz), cmap='cividis', density=1.2)
axs[1, 2].set_title("B-field (y-z plane)")

for ax in axs.flat:
    ax.set_aspect('equal')
    ax.set_xlabel('axis-1')
    ax.set_ylabel('axis-2')

plt.tight_layout()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

X3, Y3, Z3 = np.meshgrid(np.linspace(-20, 20, 10),
                         np.linspace(-20, 20, 10),
                         np.linspace(-20, 20, 10))

Ex3, Ey3, Ez3, Bx3, By3, Bz3 = field_components(X3, Y3, Z3, beta, gamma, t0)

figE = plt.figure(figsize=(7, 6))
axE = figE.add_subplot(projection='3d')
axE.quiver(X3, Y3, Z3, Ex3, Ey3, Ez3, length=4, normalize=True, color='orange')
axE.set_title("3D Electric Field")
axE.set_xlabel("x")
axE.set_ylabel("y")
axE.set_zlabel("z")
plt.tight_layout()
plt.show()

figB = plt.figure(figsize=(7, 6))
axB = figB.add_subplot(projection='3d')
axB.quiver(X3, Y3, Z3, Bx3, By3, Bz3, length=4, normalize=True, color='teal')
axB.set_title("3D Magnetic Field")
axB.set_xlabel("x")
axB.set_ylabel("y")
axB.set_zlabel("z")
plt.tight_layout()
plt.show()
