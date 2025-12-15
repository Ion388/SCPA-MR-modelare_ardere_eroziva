import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.interpolate import interp1d

# ----------------- Utilities -----------------
def make_star(R=1.0, eps=0.4, n=7, Npts=800):
    theta = np.linspace(0, 2*np.pi, Npts, endpoint=False)
    r = R * (1 + eps * np.cos(n * theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y], axis=1)
    return pts

def contour_length(pts):
    d = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seglen = np.hypot(d[:,0], d[:,1])
    return seglen.sum()

def area_shoelace(pts):
    # polygon area (positive)
    x = pts[:,0]
    y = pts[:,1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def perimeter(pts):
    d = np.diff(np.vstack([pts, pts[0]]), axis=0)
    return np.hypot(d[:,0], d[:,1]).sum()

def unit_tangents_and_outward_normals(pts):
    """
    Returns (t_unit, normals) where normals are guaranteed to point OUTWARD
    from the polygon (away from its centroid).
    pts: array (N,2) ordered counterclockwise OR clockwise (we fix normals).
    """
    # central difference approximation for tangent (periodic)
    prev = np.roll(pts, 1, axis=0)
    nxt  = np.roll(pts, -1, axis=0)
    tang = nxt - prev                     # approx 2*ds * tangent
    tang_norm = np.hypot(tang[:,0], tang[:,1])
    tang_norm[tang_norm == 0] = 1.0
    t_unit = (tang.T / tang_norm).T       # unit tangent (signed by point order)

    # rotate tangent by +90 deg to get an "outward"-candidate normal:
    normals = np.stack([-t_unit[:,1], t_unit[:,0]], axis=1)

    # ensure outward: compare normal direction with vector from centroid to point
    centroid = pts.mean(axis=0)
    vec_from_centroid = pts - centroid   # points from center -> boundary
    dots = (normals * vec_from_centroid).sum(axis=1)  # positive if normal points outward
    # if dot < 0 then normal points inward: flip it
    inward_mask = dots < 0
    if np.any(inward_mask):
        normals[inward_mask] *= -1.0

    return t_unit, normals

def resample_by_arclength(pts, Nnew):
    # compute cumulative arclength
    closed = np.vstack([pts, pts[0]])
    d = np.diff(closed, axis=0)
    seglen = np.hypot(d[:,0], d[:,1])
    s = np.concatenate(([0.0], np.cumsum(seglen)))  # length N+1
    L = s[-1]
    # make periodic by dropping final repeated point for interpolation
    s_nodes = s[:-1]
    x_nodes = closed[:-1,0]
    y_nodes = closed[:-1,1]
    s_uniform = np.linspace(0.0, L, Nnew, endpoint=False)

    # use periodic cubic interpolation: extend one extra period for smoothness
    # add the first node at s=L
    s_ext = np.concatenate([s_nodes, [L]])
    x_ext = np.concatenate([x_nodes, [x_nodes[0]]])
    y_ext = np.concatenate([y_nodes, [y_nodes[0]]])

    fx = interp1d(s_ext, x_ext, kind='cubic')
    fy = interp1d(s_ext, y_ext, kind='cubic')
    new_x = fx(s_uniform)
    new_y = fy(s_uniform)
    return np.stack([new_x, new_y], axis=1)

# ----------------- Evolution -----------------
def step_normal_move_corrected(pts, Vn, dt):
    """
    Move points along OUTWARD normal by distance Vn*dt.
    Vn: scalar or array(len=N). Positive Vn --> port GROWS (outward).
    """
    t_unit, normals = unit_tangents_and_outward_normals(pts)
    if np.isscalar(Vn):
        move = normals * (Vn * dt)
    else:
        move = normals * ((Vn * dt)[:, None])
    pts_new = pts + move
    return pts_new

# ----------------- Demo / Animation -----------------
def simulate_front_tracking(R=1.0, eps=0.45, n=7, Npts=800,
                            V0=0.02, dt=0.5, steps=300, resample_N=800,
                            show_interval=30):
    """
    V0 : outward normal speed (units of length / time)
    dt : time step size (chosen to keep V0*dt small relative to segment length)
    steps : number of frames to animate (or iterations)
    resample_N : number of points to maintain by resampling
    show_interval : ms between frames for animation
    """
    pts = make_star(R=R, eps=eps, n=n, Npts=Npts)

    fig, ax = plt.subplots(figsize=(6,6))
    line, = ax.plot([], [], lw=1.2)
    circ, = ax.plot([], [], lw=0.6, ls='--')  # showing best-fit circle optionally
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-3.0*R, 3.0*R)
    ax.set_ylim(-3.0*R, 3.0*R)
    ax.grid(True)

    txtA = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    txtP = ax.text(0.02, 0.90, '', transform=ax.transAxes)
    txtAP = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    txtt = ax.text(0.02, 0.80, '', transform=ax.transAxes)

    def init():
        line.set_data([], [])
        circ.set_data([], [])
        txtA.set_text('')
        txtP.set_text('')
        txtAP.set_text('')
        txtt.set_text('')
        return line, circ, txtA, txtP, txtAP, txtt

    def update(frame):
        nonlocal pts
        # stability: take small internal steps
        n_sub = 5
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            pts = step_normal_move_corrected(pts, V0, dt_sub)
            # optionally stop outward movement if points go too far from center
            # pts = np.maximum(pts, 1e-12)  # not needed here

        # critical: re-sample to uniform arclength (this implements tangential motion)
        pts = resample_by_arclength(pts, resample_N)

        x = np.append(pts[:,0], pts[0,0])
        y = np.append(pts[:,1], pts[0,1])
        line.set_data(x, y)

        A = area_shoelace(pts)
        P = perimeter(pts)
        txtA.set_text(f"A = {A:.6f}")
        txtP.set_text(f"P = {P:.6f}")
        txtAP.set_text(f"A/P = {A/P:.6f}")
        txtt.set_text(f"t = {frame * dt:.4f}")

        # optional: show best-fit circle of same area
        Rfit = np.sqrt(A / np.pi)
        th = np.linspace(0, 2*np.pi, 200)
        circ.set_data(Rfit * np.cos(th), Rfit * np.sin(th))

        return line, circ, txtA, txtP, txtAP, txtt

    anim = FuncAnimation(fig, update, frames=range(steps+1), init_func=init,
                         blit=True, interval=show_interval, repeat=False)
    plt.show()
    return anim

if __name__ == "__main__":
    # Choose dt so that V0*dt is << average segment length (for stability)
    # Rough heuristic: avg segment ~ perimeter/Npts, so dt < 0.2*(perim/Npts)/V0
    simulate_front_tracking(R=1.0, eps=0.9, n=6, Npts=900,
                            V0=0.2, dt=0.02, steps=100, resample_N=900,
                            show_interval=30)
