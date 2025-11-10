# Erosive burning SRM simulator (0D internal ballistics, single cylindrical port)
# - Base burning law: r0 = a * p^n
# - Erosive increment (Lenoir–Robillard): r = r0 + α G^0.8 D^-0.2 exp(-β ρ_p r / G)
# - Quasi-steady nozzle: m_dot_noz = p * At / Cstar(T0, gamma, R)
# - Chamber pressure ODE: dp/dt = (R T0 / V)(m_dot_gen - m_dot_noz) - (p/V) dV/dt
# # - Geometry: cylindrical port in a grain of length L; S_burn = 2*pi*R*L, V = V_free + pi*R^2*L
# - Geometry: constant burn area and increasing volume (for simplicity/testing): S_burn = R_case*L, V = V0 + k*t
#
# Notes:
# - All units SI. Time in seconds, Pa, m, kg.
# - This script uses only the Python stdlib; plotting is optional if matplotlib is installed.

from __future__ import annotations
import math
from typing import Dict, List, Tuple, Optional


def Cstar(T0: float, gamma: float, Rgas: float) -> float:
    """Characteristic velocity c* [m/s] for ideal gas at chamber T0.
    c* = sqrt(R*T0/gamma) * ((gamma+1)/2)**((gamma+1)/(2*(gamma-1)))
    """
    return math.sqrt(Rgas * T0 / gamma) * ((gamma + 1.0) * 0.5) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))


def nozzle_mdot_from_p(p: float, At: float, cstar: float) -> float:
    """Choked mass flow through nozzle: m_dot = p * At / c*"""
    return p * At / cstar


def area_mach_function(M: float, gamma: float) -> float:
    """A/A* as a function of Mach number (isentropic)."""
    g = gamma
    # term = (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    # return (1.0 / M) * (1.0 + 0.5 * (g - 1.0) * M * M) ** ((g + 1.0) / (2.0 * (g - 1.0))) / term
    return (1.0 / M) * ((1.0 + 0.5 * (g - 1.0) * M * M)/((g+1)/2)) ** ((g + 1.0) / (2.0 * (g - 1.0)))


def solve_exit_mach(area_ratio: float, gamma: float) -> float:
    """Solve for supersonic exit Mach given area ratio Ae/At >= 1."""
    # Simple Newton-Raphson with clamp; start at M=2.5
    M = max(1.5, 0.5 * area_ratio + 1.0)
    for _ in range(50):
        f = area_mach_function(M, gamma) - area_ratio
        # Numerical derivative
        dM = 1e-5 * max(1.0, M)
        df = (area_mach_function(M + dM, gamma) - area_mach_function(M - dM, gamma)) / (2.0 * dM)
        step = -f / df
        M = max(1.01, M + step)
        if abs(step) < 1e-8:
            break
    return M


def nozzle_perf(p_c: float, At: float, Ae: float, gamma: float, Rgas: float, T0: float, p_amb: float) -> Tuple[float, float, float]:
    """Compute thrust and exit conditions assuming isentropic nozzle with given Ae/At.
    Returns (F, mdot_noz, v_e). Pressure thrust uses actual exit pressure from isentropic flow; may over/under expand vs ambient.
    """
    cstar = Cstar(T0, gamma, Rgas)
    mdot = nozzle_mdot_from_p(p_c, At, cstar)
    eps = Ae / At
    Me = solve_exit_mach(max(1.0, eps), gamma)
    # print(f"Mach number {Me}")
    # Exit static conditions relative to chamber total
    Te = T0 / (1.0 + 0.5 * (gamma - 1.0) * Me * Me)
    pe = p_c * (Te / T0) ** (gamma / (gamma - 1.0))
    a_e = math.sqrt(gamma * Rgas * Te)
    v_e = Me * a_e
    F = mdot * v_e + (pe - p_amb) * Ae
    return F, mdot, v_e


def alpha_er_from_properties(
    cp_gas: float,
    mu_gas: float,
    Pr: float,
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
    - Pr: gas Prandtl number
    - cs_solid: heat capacity of the solid propellant
    - rho_p: density of unburned propellant
    - T1: gas reaction/combustion temperature (K)
    - Ts: propellant surface temperature (K)
    - T2: near-wall gas temperature (K)
    - Tp: initial propellant temperature (K)
    """
    if (T2 - Tp) == 0:
        return 0.0
    return 0.0288 * cp_gas * (mu_gas ** 0.2) * (Pr ** (-2.0 / 3.0)) / (rho_p * cs_solid) * ((T1 - Ts) / (T2 - Tp))


def simulate(params: Dict) -> Dict[str, List[float]]:
    # Unpack
    L = params.get("L_grain", 0.20)  # m
    # R0 = params.get("R_port0", 0.010)  # m initial port radius
    R_case = params.get("R_case", 0.040)  # m inner case radius (burnout when R>=R_case)
    R0 = R_case
    V_free = params.get("V_free", 1.0e-4)  # m^3 free volume not tied to port (voids, plenum)

    At = params.get("A_t", 5.0e-5)  # m^2 throat
    Ae = params.get("A_e", 2.5e-4)  # m^2 exit area
    p_amb = params.get("p_amb", 101325.0)  # Pa

    rho_p = params.get("rho_p", 1750.0)  # kg/m^3
    # Typical composite propellant: r ≈ 5 mm/s at ~10 MPa with n≈0.35
    # In SI with p in Pa => a ≈ 2e-5 m/s/Pa^n (order of magnitude)
    a = params.get("a", 2.0e-5)  # m/s/Pa^n (with p in Pa)
    n = params.get("n", 0.35)

    gamma = params.get("gamma", 1.20)
    Rgas = params.get("Rgas", 330.0)  # J/kg-K
    T0 = params.get("T0", 3200.0)  # K

    # Erosive model parameters (Lenoir–Robillard):
    #   r = r0 + re,   re = alpha_er * G^0.8 * D^-0.2 * exp(-beta_er * rho_p * r / G)
    # Default alpha_er chosen to give modest increments for G~200 kg/m^2/s and D~0.02 m
    alpha_er = params.get("alpha_er", 2.0e-4)
    beta_er = params.get("beta_er", 53.0)

    # Integration
    t_end = params.get("t_end", 4.0)  # s safeguard; true stop is web burnout
    dt = params.get("dt", 0.0005)
    p0 = params.get("p0", 1.1e6)  # Pa initial guess

    # Pre-compute cstar
    cstar = Cstar(T0, gamma, Rgas)

    # State
    t = 0.0
    p = max(1.01 * p_amb, p0)
    R = R0

    # Outputs
    T_list: List[float] = []
    P_list: List[float] = []
    R_list: List[float] = []
    R0_list: List[float] = []
    Rtot_list: List[float] = []
    G_list: List[float] = []
    mdot_g_list: List[float] = []
    mdot_n_list: List[float] = []
    F_list: List[float] = []

    # Helper lambdas
    def port_area(r: float) -> float:
        return math.pi * r * r

    def burn_area(r: float) -> float:
        return 2.0 * math.pi * r * L

    def chamber_volume(r: float) -> float:
        return V_free + math.pi * r * r * L

    def hydraulic_diam(r: float) -> float:
        # Circular port => Dh = 2R
        return 2.0 * r

    # Integrate until burnout or t_end
    # while t <= t_end and R < (R_case - 1e-6):
    while t <= t_end:
        A_port = port_area(R)
        S_burn = burn_area(R)
        V = chamber_volume(R)
        Dh = hydraulic_diam(R)

        # Previous step nozzle flow as pass-through estimate for core G
        # First, compute nozzle mass flow and thrust from current p
        F_now, mdot_noz, v_e = nozzle_perf(p, At, Ae, gamma, Rgas, T0, p_amb)

        # Mass flux at port entrance (engineering shortcut)
        G = (mdot_noz / A_port) if A_port > 0.0 else 0.0

        # Burning rate: base + Lenoir–Robillard erosive addition
        r0 = a * (p ** n)
        if G > 1e-9 and Dh > 1e-9:
            # Fixed-point iterate r = r0 + alpha * G^0.8 * D^-0.2 * exp(-beta * rho_p * r / G)
            D_char = Dh  # For circular port, D = 4Ap/S = 2R = Dh
            re_scale = (G ** 0.8) * (D_char ** -0.2)
            r_iter = r0
            for _ in range(12):
                expo = -beta_er * rho_p * r_iter / G
                # Guard extreme negatives to avoid underflow
                if expo < -60.0:
                    re = 0.0
                else:
                    re = alpha_er * re_scale * math.exp(expo)
                r_new = r0 + re
                if abs(r_new - r_iter) <= 1e-6 * max(1e-4, r_new):
                    r_iter = r_new
                    break
                r_iter = r_new
            r = max(0.0, r_iter)
        else:
            r = r0

        # Generation and nozzle outflow
        mdot_gen = rho_p * r * S_burn
        mdot_noz = nozzle_mdot_from_p(p, At, cstar)  # use c* for stability in ODE

        # Pressure ODE (0D)
        dVdt = S_burn * r
        dpdt = (Rgas * T0 / V) * (mdot_gen - mdot_noz) - (p / V) * dVdt

        # Advance (semi-implicit for geometry)
        p_new = p + dpdt * dt
        p = max(p_amb * 1.0001, p_new)
        # R = min(R_case, R + r * dt)
        t += dt

        # Store
        T_list.append(t)
        P_list.append(p)
        R_list.append(R)
        R0_list.append(r0)
        Rtot_list.append(r)
        G_list.append(G)
        mdot_g_list.append(mdot_gen)
        mdot_n_list.append(mdot_noz)
        F_list.append(F_now)

    # Summaries
    I_tot = 0.0
    if T_list:
        # Trapezoidal impulse
        for i in range(1, len(T_list)):
            I_tot += 0.5 * (F_list[i] + F_list[i - 1]) * (T_list[i] - T_list[i - 1])

    return {
        "t": T_list,
        "p": P_list,
        "R": R_list,
        "r0": R0_list,
        "r": Rtot_list,
        "G": G_list,
        "mdot_gen": mdot_g_list,
        "mdot_noz": mdot_n_list,
        "F": F_list,
        "I_tot": [I_tot],
        "burnout": [1.0 if R >= (R_case - 1e-6) else 0.0],
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
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(t, [x * 1e-6 for x in res["p"]])
    axs[0].set_ylabel("p [MPa]")
    axs[0].grid(True)
    axs[1].plot(t, res["G"]) ; axs[1].set_ylabel("G [kg/m^2/s]") ; axs[1].grid(True)
    axs[2].plot(t, res["r0"], label="r0")
    axs[2].plot(t, res["r"], label="r total")
    axs[2].set_ylabel("r [m/s]")
    axs[2].set_xlabel("t [s]")
    axs[2].legend()
    axs[2].grid(True)
    plt.tight_layout()
    plt.show()


def example_params() -> Dict:
    return {
        # Geometry
        "L_grain": 0.20,
        "R_port0": 0.010,
        "R_case": 0.040,
        "V_free": 2.0e-4,
        # Nozzle
        "A_t": 7.5e-5,          # ~9.8 mm throat dia
        "A_e": 3.0e-4,          # eps=4 (example)
        "p_amb": 101325.0,
        # Propellant
        "rho_p": 1750.0,
        "a": 2.0e-5,
        "n": 0.35,
        # Erosive (Lenoir–Robillard)
        "alpha_er": 2.0e-4,     # or compute from properties using alpha_er_from_properties(...)
        "beta_er": 53.0,
        # Integration
        "t_end": 4.0,
        "dt": 5.0e-4,
        "p0": 1.2e6,
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

    _try_plot(res)


if __name__ == "__main__":
    main()
