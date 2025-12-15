import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------- Utilities ----------
def initial_star(R=1.0, eps=0.9, n=6, N=8192):
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    r0 = R*(1 + eps*np.cos(n*theta))
    return theta, r0

def deriv_theta_periodic(f, theta):
    # 2nd-order central differences on a periodic grid
    dtheta = theta[1] - theta[0]
    return np.roll(f, -1) - np.roll(f, 1) / (2*dtheta)  # WRONG grouping corrected below

def deriv_theta_central(f, theta):
    # Correct central difference implementation
    dtheta = theta[1] - theta[0]
    return (np.roll(f, -1) - np.roll(f, 1)) / (2*dtheta)

def area_from_r(theta, r):
    # A = 1/2 ∫ r^2 dθ
    return 0.5 * np.trapz(r*r, theta)

def perimeter_from_r(theta, r):
    # P = ∫ sqrt(r^2 + (dr/dθ)^2) dθ
    dr = deriv_theta_central(r, theta)
    integrand = np.sqrt(r*r + dr*dr)
    return np.trapz(integrand, theta)

# ---------- Evolution PDE ----------
def step_r_explicit(r, theta, Vn, dt):
    # r_t = - Vn * r / sqrt(r^2 + r_theta^2)
    r_theta = deriv_theta_central(r, theta)
    denom = np.sqrt(r*r + r_theta*r_theta)
    geom = r / denom
    drdt =  Vn * geom
    r_new = r + dt * drdt
    # ensure positivity
    r_new = np.maximum(r_new, 1e-6)
    return r_new

# ---------- Simulation + Animation ----------
def simulate_and_animate(R=1.0, eps=0.9, n=6, V0=0.02, dt=0.5, steps=500, N=400):
    theta, r = initial_star(R, eps, n, N)
    # constant normal velocity inward
    Vn_array = V0 * np.ones_like(theta)

    fig, ax = plt.subplots(figsize=(6,6))
    line, = ax.plot([], [], lw=1.2)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3*R, 3*R)
    ax.set_ylim(-3*R, 3*R)
    ax.grid(True)

    texts = {
        'A': ax.text(0.02, 0.95, '', transform=ax.transAxes),
        'P': ax.text(0.02, 0.90, '', transform=ax.transAxes),
        'AP': ax.text(0.02, 0.85, '', transform=ax.transAxes),
        't': ax.text(0.02, 0.80, '', transform=ax.transAxes),
    }

    def init():
        line.set_data([], [])
        for t in texts.values():
            t.set_text('')
        return (line, *texts.values())

    def update(frame):
        nonlocal r
        # small substeps for stability
        substeps = 5
        dt_sub = dt / substeps
        for _ in range(substeps):
            r = step_r_explicit(r, theta, Vn_array, dt_sub)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        line.set_data(x, y)

        A = area_from_r(theta, r)
        P = perimeter_from_r(theta, r)
        texts['A'].set_text(f"Area = {A:.6f}")
        texts['P'].set_text(f"Perim = {P:.6f}")
        texts['AP'].set_text(f"A/P = {A/P:.6f}")
        texts['t'].set_text(f"t = {frame*dt:.4f} s")
        return (line, *texts.values())

    anim = FuncAnimation(fig, update, frames=range(steps+1), init_func=init,
                         blit=True, interval=50, repeat=False)
    plt.show()
    return anim

if __name__ == "__main__":
    # Example run
    simulate_and_animate(R=1.0, eps=0.9, n=6, V0=0.2, dt=0.05, steps=50, N=400)
