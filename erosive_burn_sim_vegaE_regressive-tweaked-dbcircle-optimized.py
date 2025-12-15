# Erosive burning SRM simulator (0D internal ballistics, single cylindrical port)
# - Base burning law: r0 = a * p^n
# - Erosive increment (Lenoir–Robillard): r = r0 + α G^0.8 D^-0.2 exp(-β ρ_p r / G)
# - Quasi-steady nozzle: m_dot_noz = p * At / Cstar(T0, gamma, R)
# - Chamber pressure ODE: dp/dt = (R T0 / V)(m_dot_gen - m_dot_noz) - (p/V) dV/dt
# - Geometry: cylindrical port in a grain of length L; S_burn = 2*pi*R*L, V = V_free + pi*R^2*L
#
# Notes:
# - All units SI. Time in seconds, Pa, m, kg.
# - This script uses only the Python stdlib; plotting is optional if matplotlib is installed.
# 1 = chamber
# 2 = nozzle exit
# 3/t = throat
# amb = ambient

from __future__ import annotations
import math
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional


def Cstar(T1: float, k: float, Rgas: float) -> float:
    """Characteristic velocity c* [m/s] for ideal gas at chamber T0.
    c* = sqrt(R*T1/k) * ((k+1)/2)**((k+1)/(2*(k-1)))
    """
    return math.sqrt(Rgas * T1 / k) * ((k + 1.0) * 0.5) ** ((k + 1.0) / (2.0 * (k - 1.0)))

def nozzle_mdot_from_p(p1: float, At: float, cstar: float) -> float:
    """Choked mass flow through nozzle: m_dot = p1 * At / c*"""
    return p1 * At / cstar


def area_mach_function(M: float, k: float) -> float:
    """A/A* as a function of Mach number (isentropic)."""
    return (1.0 / M) * (((1.0 + 0.5 * (k - 1.0) * M * M)/(0.5 * (k + 1))) ** ((k + 1.0) / (2.0 * (k - 1.0))))


def solve_exit_mach(area_ratio: float, k: float) -> float:
    """Solve for supersonic exit Mach given area ratio Ae/At >= 1."""
    # Simple Newton-Raphson with clamp; start at M=2.5
    M = max(1.5, 0.5 * area_ratio + 1.0)
    # M = 3
    for _ in range(50):
        f = area_mach_function(M, k) - area_ratio
        # Numerical derivative
        dM = 1e-5 * max(1.0, M)
        df = (area_mach_function(M + dM, k) - area_mach_function(M - dM, k)) / (2.0 * dM)
        step = -f / df
        M = max(1.01, M + step)
        if abs(step) < 1e-8:
            break
    return M


def nozzle_perf(p1: float, At: float, eps: float, k: float, Rgas: float, T1: float, p_amb: float) -> Tuple[float, float, float]:
    """Compute thrust and exit conditions assuming isentropic nozzle with given Ae/At.
    Returns (F, mdot_noz, v_e). Pressure thrust uses actual exit pressure from isentropic flow; may over/under expand vs ambient.
    """
    cstar = Cstar(T1, k, Rgas)
    mdot = nozzle_mdot_from_p(p1, At, cstar)
    A2 = eps * At
    M2 = solve_exit_mach(eps, k)
    # print(f"Mach number {M2}")
    # Exit static conditions relative to chamber total
    T2 = T1 / (1.0 + 0.5 * (k - 1.0) * M2 * M2)
    p2 = p1 * (T2 / T1) ** (k / (k - 1.0))
    # p2 = p1 * 8.184e-3  # Precomputed for M2 and k=1.14
    #compute t2 from p2
    # T2 = T1 * (p2 / p1) ** ((k - 1.0) / k)
    a_2 = math.sqrt(k * Rgas * T2)
    v_e = M2 * a_2
    F = mdot * v_e + (p2 - p_amb) * A2
    return F, mdot, v_e, p2/p1


def alpha_er_from_properties(
    cp_gas: float,
    mu_gas: float,
    K: float,
    cs_solid: float,
    rho_p: float,
    T1: float,
    Ts: float,
    T2: float,
    Tp: float,
) -> float:
    """Compute Lenoir–Robillard coefficient alpha_er from thermophysical properties.

    Formula (from Lenoir & Robillard heat-transfer model):
        alpha = 0.0288 * cp * mu^0.2 * Pr^(-2/3) / (rho_p * cs) * (T1 - Ts) / (T2 - Tp)

    Units:
    - Use consistent units for cp and cs (both J/kg-K or both kcal/kg-K); their ratio cancels units.
    - mu in kg/(m·s), rho_p in kg/m^3, temperatures in K, Pr dimensionless.

    Args
    - cp_gas: average specific heat of combustion gases
    - mu_gas: gas viscosity
    - K: gas thermal conductivity
    - cs_solid: heat capacity of the solid propellant
    - rho_p: density of unburned propellant
    - T1: gas reaction/combustion temperature (K)
    - Ts: propellant surface temperature (K)
    - T2: near-wall gas temperature (K)
    - Tp: initial propellant temperature (K)
    """
    Pr = mu_gas * cp_gas * 4184 / K
    if (T2 - Tp) == 0:
        return 0.0
    return 0.0288 * cp_gas * (mu_gas ** 0.2) * (Pr ** (-2.0 / 3.0)) / (rho_p * cs_solid) * ((T1 - Ts) / (T2 - Tp))


def simulate(params: Dict) -> Dict[str, List[float]]:
    # Unpack
    L_case = params.get("L_case")  # m
    R_case = params.get("R_case")  # m inner case radius (burnout when R>=R_case)
    V_free = params.get("V_free")  # m^3 free volume not tied to port (voids, plenum)

    A2 = params.get("A2")  # m^2 throat
    eps = params.get("eps")  # Area ratio Ae/At
    p_amb = params.get("p_amb")  # Pa

    rho_p = params.get("rho_p")  # kg/m^3
    # Typical composite propellant: r ≈ 5 mm/s at ~10 MPa with n≈0.35
    # In SI with p in Pa => a ≈ 2e-5 m/s/Pa^n (order of magnitude)
    a = params.get("a")  # m/s/Pa^n (with p in Pa)
    n = params.get("n")

    k = params.get("k")
    Rgas = params.get("Rgas")  # J/kg-K
    T1 = params.get("T1")  # K

    # Erosive model parameters (Lenoir–Robillard):
    #   r = r0 + re,   re = alpha_er * G^0.8 * D^-0.2 * exp(-beta_er * rho_p * r / G)
    # Default alpha_er chosen to give modest increments for G~200 kg/m^2/s and D~0.02 m
    alpha_er = params.get("alpha_er")
    beta_er = params.get("beta_er")

    # Integration
    t_end = params.get("t_end")  # s safeguard; true stop is web burnout
    dt = params.get("dt")
    p0 = params.get("p0")  # Pa initial guess

    # Pre-compute cstar
    cstar = Cstar(T1, k, Rgas)

    # State
    t = 0.0
    p = max(1.01 * p_amb, p0)

    # Outputs
    T_list: List[float] = []
    P_list: List[float] = []
    V_list: List[float] = []
    r0_list: List[float] = []
    rtot_list: List[float] = []
    G_list: List[float] = []
    mdot_g_list: List[float] = []
    mdot_n_list: List[float] = []
    F_list: List[float] = []
    expo_list: List[float] = []
    port_to_throat_list: List[float] = []
    r_ratio_list: List[float] = []

    # Helper lambdas
    def port_area(R: float) -> float:
        return math.pi * R * R

    def burn_area(R: float) -> float:
        return math.pi * R**2

    def chamber_volume(A: float, L: float) -> float:
        return A * L

    def hydraulic_diam(R: float) -> float:
        # Circular port => Dh = 2R
        return 2.0 * R
    
    def throat_area(eps:float, A2: float) -> float:
        return A2/eps

    # Integrate until burnout or t_end
    # while t <= t_end and R < (R_case - 1e-6):
    # Start values

    # pts = make_double_anchor()
    # pts1 = make_circle(R=0.83337, Npts=400)
    # pts2 = make_circle(R=0.61663, Npts=400)
    A_port = V_free / L_case
    # A_port = 0.5
    # Initial port radii for given area
    R2 = (A_port / math.pi + R_case**2)/(2*R_case)
    R1 = R_case - R2
    # A_port = area_shoelace(pts1) - area_shoelace(pts2)
    # P = perimeter(pts1, R_max=R_case) + perimeter(pts2, R_max=R_case)
    P = 2.0 * math.pi * (R2 + R1)
    Dh = 4*A_port/P
    S_burn = L_case*P
    At = throat_area(eps, A2)
    V = chamber_volume(A_port, L_case)

    while t <= t_end:
        # A_port = port_area(R)
        # S_burn = burn_area(R)
        # V = chamber_volume(R, L)
        # Dh = hydraulic_diam(R)
        # At = throat_area(eps, A2)

        # print(f"Port area: {A_port:.6f} m^2, Burn area: {S_burn:.6f} m^2, Vol: {V:.6f} m^3")

        # Previous step nozzle flow as pass-through estimate for core G
        # First, compute nozzle mass flow and thrust from current p
        F_now, mdot_noz, v_e, port_to_throat = nozzle_perf(p, At, eps, k, Rgas, T1, p_amb)

        # Mass flux at port entrance (engineering shortcut)
        G = (mdot_noz / A_port) if A_port > 0.0 else 0.0

        # Burning rate: base + Lenoir–Robillard erosive addition
        r0 = a * (p ** n)
        # print(f"Base burn rate r0: {r0*1000:.3f} mm/s at p={p/1e6:.3f} MPa, G={G:.1f} kg/m^2/s")
        if G > 1e-9 and Dh > 1e-9:
            # Fixed-point iterate r = r0 + alpha * G^0.8 * D^-0.2 * exp(-beta * rho_p * r / G)
            D_char = Dh  # For circular port, D = 4Ap/S = 2R = Dh
            re_scale = np.power(G, 0.8) * np.power(D_char, -0.2)
            r_iter = r0
            for i in range(100):
                expo = -beta_er * rho_p * r_iter / G
                # print(f"expo: {expo:.3f}")
                # Guard extreme negatives to avoid underflow
                if expo < -60.0:
                    re = 0.0
                else:
                    re = alpha_er * re_scale * np.exp(expo)
                    if i == 3:
                        print(f"re: {re*1000:.3f} mm/s")
                r_new = r0 + re
                if abs(r_new - r_iter) <= 1e-6 * max(1e-4, r_new):
                    r_iter = r_new
                    break
                r_iter = r_new
            # print(-beta_er * rho_p * r_iter / G)
            expo_list.append(expo)
            # print(f"expo: {expo:.3f}")
            r = max(0.0, r_iter)
            # print(f"Erosive increment re: {r - r0:.3f} m/s, total r: {r:.3f} m/s")
        else:
            r = r0
        # r = r0

        # Generation and nozzle outflow
        mdot_gen = rho_p * r * S_burn
        mdot_noz = nozzle_mdot_from_p(p, At, cstar)  # use c* for stability in ODE


        # Pressure ODE (0D)
        dVdt = S_burn * r
        dpdt = (Rgas * T1 / V) * (mdot_gen - mdot_noz) - (p / V) * dVdt

        # print(f"mdot_gen: {mdot_gen:.3f} kg/s, mdot_noz: {mdot_noz:.3f} kg/s, dV/dt: {dVdt:.6f} m^3/s, dp/dt: {dpdt/1e6:.3f} MPa/s")

        # Advance (semi-implicit for geometry)
        p_new = p + dpdt * dt
        p = max(p_amb * 1.0001, p_new)
        t += dt

        # pts1, pts2, A_port, P, Dh = area_perimeter_helper(pts1, pts2, r, resample_N=3000, n_sub=200, R_max=R_case)
        if np.isclose(R2, R_case, atol=1e-4) or np.isclose(R1, 0.0, atol=1e-4) or R2>=R_case or R1<=0.0:
            R2 = 0.0
            R1 = 0.0
            P = 0.0
            A_port = R_case**2 * math.pi  # full volume
        else:
            R2 += r * dt
            R1 -= r * dt
            A_port = math.pi * (R2*R2 - R1*R1)
            P = 2.0 * math.pi * (R2 + R1)
            Dh = 4*A_port/P
            
        S_burn = L_case*P
        V = chamber_volume(A_port, L_case)

        # Store
        T_list.append(t)
        P_list.append(p)
        V_list.append(V)
        r0_list.append(1000*r0)
        rtot_list.append(1000*r)
        G_list.append(G)
        mdot_g_list.append(mdot_gen)
        mdot_n_list.append(mdot_noz)
        F_list.append(F_now)
        port_to_throat_list.append(port_to_throat)
        r_ratio_list.append(r/r0 if r0>0 else 0.0)

    # Summaries
    I_tot = 0.0
    if T_list:
        # Trapezoidal impulse
        for i in range(1, len(T_list)):
            I_tot += 0.5 * (F_list[i] + F_list[i - 1]) * (T_list[i] - T_list[i - 1])

    # dists = np.hypot(pts1[:,0], pts1[:,1])

    return {
        "t": T_list,
        "p": P_list,
        "r0": r0_list,
        "r": rtot_list,
        "V": V_list,
        "G": G_list,
        "mdot_gen": mdot_g_list,
        "mdot_noz": mdot_n_list,
        "F": F_list,
        "I_tot": [I_tot],
        "expo": expo_list,
        # "burnout": [1.0 if np.isclose(dists, R_case, atol=1e-6).all() else 0.0],
        "burnout": [1.0 if np.isclose(R2, R_case, atol=1e-6) or np.isclose(R1, 0.0, atol=1e-6) else 0.0],
        "port_to_throat": port_to_throat_list,
        "r_ratio": r_ratio_list,
    }


def _try_plot(res: Dict[str, List[float]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        print("matplotlib not found; skipping plots. To enable, install matplotlib.")
        return
    t = res["t"]
    if not t:
        print("No data to plot.")
        return
    
    path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])))
    ar = 'auto'
    height = 16
    width = 9
    dpi = 100
    title_size = 28
    label_size = 28
    tick_size = 20
    legend_size = 28
    line_color = 'orangered'

    # ax.plot(dt_exact_list, X_exact[0], color=err_color, label="Exact Solution X1")
    # ax.plot(dt_list, X_approx[0], color=line_color, label="Approximate Solution X1")
    # ax.set_xlabel("Time", fontsize=label_size)
    # ax.set_ylabel("X1", fontsize=label_size)
    # ax.tick_params(axis='both', which='major', labelsize=tick_size)
    # ax.set_title(f"Exact Solution vs Approximate Solution for X1, dt={dt} - a={a}, b={b}, c={c}, T={T}", fontsize=title_size)
    # ax.legend(fontsize=legend_size)
    # ax.grid()
    # ax.set_aspect(ar)
    # plt.savefig(path + f"/Exact_vs_Approx_X1_{dt}.png")
    # plt.clf()

 
    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.plot(t, [x * 1e-6 for x in res["p"]], linewidth=4, color=line_color)
    ax.set_ylabel("Presiunea în camera arderii $p_1$ [MPa]", fontsize=label_size)
    ax.set_xlabel("Timp [s]", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid(True)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_aspect(ar)
    plt.savefig(path + f"/p_circle.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.plot(t, res["G"], linewidth=4, color=line_color)
    ax.set_ylabel("Fluxul $G$ $\mathrm{[kg/(m^2\cdot s)]}$", fontsize=label_size)
    ax.set_xlabel("Timp [s]", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid(True)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    plt.tight_layout()
    ax.set_aspect(ar)
    plt.savefig(path + f"/G_circle.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.plot(t, res["r0"], label="$r_0$", linewidth=4, color=line_color)
    ax.plot(t, res["r"], label="$r_{\mathrm{total}}$", linewidth=4, color="slateblue")
    ax.set_ylabel("Rata arderii $r$ [mm/s]", fontsize=label_size)
    ax.set_xlabel("Timp [s]", fontsize=label_size)
    ax.legend(fontsize=legend_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid(True)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_aspect(ar)
    plt.savefig(path + f"/r_circle.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.plot(t, res["F"], linewidth=4, color=line_color)
    ax.set_ylabel("Forța de tracțiune $F$ [N]", fontsize=label_size)
    ax.set_xlabel("Timp [s]", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid(True)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_aspect(ar)
    plt.savefig(path + f"/F_circle.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.plot(t, res["V"], linewidth=4, color=line_color)
    ax.set_ylabel("Volumul liber $V$ [m^3]", fontsize=label_size)
    ax.set_xlabel("Timp [s]", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid(True)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.set_aspect(ar)
    plt.savefig(path + f"/V_circle.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.plot(t, res["expo"], linewidth=4, color=line_color)
    ax.set_ylabel(r"$-\beta \rho_p r / G$", fontsize=label_size)
    ax.set_xlabel("Timp [s]", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid(True)
    ax.set_xlim(left=0.0)
    ax.set_ylim(top=0.0)
    ax.set_aspect(ar)
    plt.savefig(path + f"/expo_circle.png")
    plt.clf()

    fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    ax.plot(t, res["r_ratio"], linewidth=4, color=line_color)
    ax.set_ylabel("Raportul $r_{\mathrm{total}}/r_0$", fontsize=label_size)
    ax.set_xlabel("Timp [s]", fontsize=label_size)
    ax.tick_params(axis='both', which='major', labelsize=tick_size)
    ax.grid(True)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.9)
    ax.set_aspect(ar)
    plt.savefig(path + f"/r_ratio_circle.png")
    plt.clf()

    # fig, ax = plt.subplots(figsize=(height, width), dpi=dpi)
    # ax.plot(t, res["port_to_throat"], linewidth=4)
    # ax.set_ylabel("A_port / A_throat")
    # ax.set_xlabel("t [s]")
    # ax.grid(True)
    # ax.set_xlim(left=0.0)
    # ax.set_ylim(bottom=0.0)
    # ax.set_aspect(ar)
    # plt.savefig(path + f"/p_star.png")
    # plt.clf()

    # plt.tight_layout()
    # plt.show()


def example_params() -> Dict:
    return {
        # Geometry
        "L_case": 8.69,  # m #gasit
        "R_case": 1.45, # gasit
        "V_free": 8.58, #gasit
        # Nozzle
        "A2": math.pi*1.76*1.76/4, #gasit
        "eps": 16.0, 
        "p_amb": 101325.0,
        # "p_amb": 0.0,
        # Propellant
        "rho_p": 1810.0, #gasit
        "a": 0.93e-5,
        "n": 0.4,
        # Combustion gas
        "k": 1.1383,
        "Rgas": 274.7249,
        "T1": 3260.0,
        # Erosive (Lenoir–Robillard)
        # "alpha_er": 2.0e-4,     # or compute from properties using alpha_er_from_properties(...)
        "alpha_er": alpha_er_from_properties(
            cp_gas=3007.3*0.239/1000,    # kJ/kg-K to kcal/kg-K
            mu_gas=9.6392e-5, # kg/m-s
            K=0.76604,  # W/m-K+
            cs_solid=1230.0*0.239/1000,  # kJ/kg-K to kcal/kg-K
            rho_p=1810.0, # kg/m^3
            T1=3260.0, # K
            Ts=1060.0, # K
            T2=1825.0, # K
            Tp=300.0, # K
        ),
        "beta_er": 53.0,
        # Integration
        "t_end": 110,
        "dt": 1e-1,
        "p0": 0.1e6,
    }


def main() -> None:
    params = example_params()
    res = simulate(params)

    # Basic summaries
    p_max = max(res["p"]) if res["p"] else 0.0
    F_max = max(res["F"]) if res["F"] else 0.0
    G_max = max(res["G"]) if res["G"] else 0.0
    I_tot = res["I_tot"][0] if res["I_tot"] else 0.0

    print("Results:")
    print(f"  Burnout reached: {'yes' if res['burnout'][0] > 0.5 else 'no'}")
    print(f"  Max pressure: {p_max/1e6:.3f} MPa")
    print(f"  Max thrust (ideal): {F_max:.1f} N")
    print(f"  Max mass flux G: {G_max:.1f} kg/m^2/s")
    print(f"  Total impulse: {I_tot:.1f} N·s")
    print(f"  Specific impulse (ideal): {I_tot / (88365 * 9.81):.1f} s")
    print(f"  Volume at burnout: {res['V'][-1]:.3f} m^3")
    print(f"  Average thrust: {I_tot / 110:.1f} N")

    _try_plot(res)


if __name__ == "__main__":
    main()
