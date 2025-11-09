
import numpy as np
import matplotlib.pyplot as plt


q = 1.0       
c = 1.0       
t_ret = 5.0   
grid_points = 22
grid_range = 12


def radiation_fields(r_obs, r_p, v_p, a_p):
    """Return the radiation part of E and B fields at observation point."""
    R_vec = r_obs - r_p
    R = np.linalg.norm(R_vec)
    n_hat = R_vec / R
    beta_vec = v_p / c
    beta_dot = a_p / c
    kappa = 1.0 - np.dot(n_hat, beta_vec)
    numerator = np.cross(n_hat, np.cross(n_hat - beta_vec, beta_dot))
    E = numerator / (R * (kappa ** 3))
    B = np.cross(n_hat, E)
    return E, B


#  Case (A): β = 0.3, acceleration along z-axis  

beta = 0.3
a0 = 0.3

def r_particle_z(t):
    return np.array([beta * c * t, 0.0, 0.5 * a0 * t**2])

def v_particle_z(t):
    return np.array([beta * c, 0.0, a0 * t])

def a_particle_z(t):
    return np.array([0.0, 0.0, a0])


x = np.linspace(-grid_range, grid_range, grid_points)
y = np.linspace(-grid_range, grid_range, grid_points)
X, Y = np.meshgrid(x, y)


Ex_xy, Ey_xy, Ex_xz, Ez_xz, Ey_yz, Ez_yz = [np.zeros_like(X) for _ in range(6)]
Bx_xy, By_xy, Bx_xz, Bz_xz, By_yz, Bz_yz = [np.zeros_like(X) for _ in range(6)]

for j in range(grid_points):
    for i in range(grid_points):
        # (x,y) plane
        r = np.array([X[j,i], Y[j,i], 0])
        E, B = radiation_fields(r, r_particle_z(t_ret), v_particle_z(t_ret), a_particle_z(t_ret))
        Ex_xy[j,i], Ey_xy[j,i] = E[0], E[1]
        Bx_xy[j,i], By_xy[j,i] = B[0], B[1]

        # (x,z) plane
        r = np.array([X[j,i], 0, Y[j,i]])
        E, B = radiation_fields(r, r_particle_z(t_ret), v_particle_z(t_ret), a_particle_z(t_ret))
        Ex_xz[j,i], Ez_xz[j,i] = E[0], E[2]
        Bx_xz[j,i], Bz_xz[j,i] = B[0], B[2]

        # (y,z) plane
        r = np.array([0, X[j,i], Y[j,i]])
        E, B = radiation_fields(r, r_particle_z(t_ret), v_particle_z(t_ret), a_particle_z(t_ret))
        Ey_yz[j,i], Ez_yz[j,i] = E[1], E[2]
        By_yz[j,i], Bz_yz[j,i] = B[1], B[2]

#  Plot Case (A) 
plt.figure(figsize=(12,8))
plt.suptitle("Case (A): β = 0.3 along x, ẋ = 0.3 along z", fontsize=14, y=0.94)

def plot_field(ax, X, Y, U, V, title, xlabel, ylabel):
    magnitude = np.hypot(U, V)
    magnitude[magnitude == 0] = 1.0
    ax.quiver(X, Y, U/magnitude, V/magnitude, magnitude, cmap='coolwarm', scale=30)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

ax1 = plt.subplot(2,3,1); plot_field(ax1, X, Y, Ex_xy, Ey_xy, "E(x,y) at z=0", "x", "y")
ax2 = plt.subplot(2,3,2); plot_field(ax2, X, Y, Ex_xz, Ez_xz, "E(x,z) at y=0", "x", "z")
ax3 = plt.subplot(2,3,3); plot_field(ax3, X, Y, Ey_yz, Ez_yz, "E(y,z) at x=0", "y", "z")
ax4 = plt.subplot(2,3,4); plot_field(ax4, X, Y, Bx_xy, By_xy, "B(x,y) at z=0", "x", "y")
ax5 = plt.subplot(2,3,5); plot_field(ax5, X, Y, Bx_xz, Bz_xz, "B(x,z) at y=0", "x", "z")
ax6 = plt.subplot(2,3,6); plot_field(ax6, X, Y, By_yz, Bz_yz, "B(y,z) at x=0", "y", "z")
plt.tight_layout()
plt.show()


#   Case (B): β = 0.9, acceleration along x-axis  

beta = 0.9
a0 = 0.3

def r_particle_x(t):
    return np.array([beta * c * t + 0.5 * a0 * t**2, 0.0, 0.0])

def v_particle_x(t):
    return np.array([beta * c + a0 * t, 0.0, 0.0])

def a_particle_x(t):
    return np.array([a0, 0.0, 0.0])


Ex_xy, Ey_xy, Ex_xz, Ez_xz, Ey_yz, Ez_yz = [np.zeros_like(X) for _ in range(6)]
Bx_xy, By_xy, Bx_xz, Bz_xz, By_yz, Bz_yz = [np.zeros_like(X) for _ in range(6)]

for j in range(grid_points):
    for i in range(grid_points):
        r = np.array([X[j,i], Y[j,i], 0])
        E, B = radiation_fields(r, r_particle_x(t_ret), v_particle_x(t_ret), a_particle_x(t_ret))
        Ex_xy[j,i], Ey_xy[j,i] = E[0], E[1]
        Bx_xy[j,i], By_xy[j,i] = B[0], B[1]

        r = np.array([X[j,i], 0, Y[j,i]])
        E, B = radiation_fields(r, r_particle_x(t_ret), v_particle_x(t_ret), a_particle_x(t_ret))
        Ex_xz[j,i], Ez_xz[j,i] = E[0], E[2]
        Bx_xz[j,i], Bz_xz[j,i] = B[0], B[2]

        r = np.array([0, X[j,i], Y[j,i]])
        E, B = radiation_fields(r, r_particle_x(t_ret), v_particle_x(t_ret), a_particle_x(t_ret))
        Ey_yz[j,i], Ez_yz[j,i] = E[1], E[2]
        By_yz[j,i], Bz_yz[j,i] = B[1], B[2]

# Plot Case (B) 
plt.figure(figsize=(12,8))
plt.suptitle("Case (B): β = 0.9 along x, ẋ = 0.3 along x", fontsize=14, y=0.94)

ax1 = plt.subplot(2,3,1); plot_field(ax1, X, Y, Ex_xy, Ey_xy, "E(x,y) at z=0", "x", "y")
ax2 = plt.subplot(2,3,2); plot_field(ax2, X, Y, Ex_xz, Ez_xz, "E(x,z) at y=0", "x", "z")
ax3 = plt.subplot(2,3,3); plot_field(ax3, X, Y, Ey_yz, Ez_yz, "E(y,z) at x=0", "y", "z")
ax4 = plt.subplot(2,3,4); plot_field(ax4, X, Y, Bx_xy, By_xy, "B(x,y) at z=0", "x", "y")
ax5 = plt.subplot(2,3,5); plot_field(ax5, X, Y, Bx_xz, Bz_xz, "B(x,z) at y=0", "x", "z")
ax6 = plt.subplot(2,3,6); plot_field(ax6, X, Y, By_yz, Bz_yz, "B(y,z) at x=0", "y", "z")
plt.tight_layout()
plt.show()
