# Converted from /C:/Users/ionst/Downloads/mrcs1.f90
# Minimal direct translation of the Fortran program into Python

import math
import random


def main():
  # constants / initial values
  tburn = 5.5
  rbref = 30e-3 / tburn  # Lungime baton propelant / timp ardere
  mgref = 17e-3 / tburn  # Masa baton propelant / timp ardere
  Itotref = 20.0
  Fref = Itotref / tburn

  gama = 1.29
  R = 287.0

  T0 = 1100.0
  Cstar = math.sqrt(R * T0 / gama) * ((gama + 1.0) * 0.5) ** ((gama + 1.0) / (2.0 * gama - 2.0))

  pi = math.pi
  n = 0.5  # Constanta exponentiala a legii de ardere (initial, will be randomized)
  a = 4.8e-3  # Constanta legii de ardere (initial, will be randomized)
  Pa = 1e5  # Presiunea ambientala

  Mach = 2.5
  ve = 2.5 * math.sqrt(1.33 * 287.0 * 700.0)  # viteza de evacuare
  Db = 2.5e-3
  At = 0.25 * pi * Db ** 2
  Ab = 0.25 * pi * (17e-3) ** 2

  print("Cstar=", Cstar)
  print("ve=", ve)

  a1 = 2.5
  a2 = 3.1

  n1 = 0.2
  n2 = 0.3

  ro1 = 5.4
  ro2 = 5.8

  critmin = 1e6

  Fmin = None
  amin = None
  nmin = None
  romin = None

  # Monte Carlo loop (1_000_000 iterations as in original)
  for _ in range(1_000_000):
    a = a1 + random.random() * (a2 - a1)
    n = n1 + random.random() * (n2 - n1)
    rho_p = ro1 + random.random() * (ro2 - ro1)

    # Compute chamber pressure from burning law and mass flow balance
    denom = At * Cstar
    base = (rho_p * a * Ab) / denom
    # protect against negative/zero base (shouldn't happen with given ranges)
    if base <= 0.0:
      continue

    Pc = base ** (1.0 / (1.0 - n))

    rb = a * Pc ** n
    mg = rho_p * Ab * rb

    F = mg * ve + (Pc - Pa) * At
    Itot = F * tburn

    crit = abs(F - Fref) / Fref

    if crit <= critmin:
      critmin = crit
      Fmin = F
      amin = a
      nmin = n
      romin = rho_p

  # Summary of best found
  print("Fmin", Fmin)
  print("amin", amin)
  print("n", nmin)
  print("rho_p", romin)

  # Recompute with best parameters
  a = amin
  n = nmin
  rho_p = romin

  base = (rho_p * a * Ab) / (At * Cstar)
  Pc = base ** (1.0 / (1.0 - n))

  rb = a * Pc ** n
  mg = rho_p * Ab * rb

  F = mg * ve + (Pc - Pa) * At
  Itot = F * tburn

  print("Pc", Pc)
  print("rho_p", rho_p)
  print("rb, rbr", rb, rbref)
  print("mg, mgr", mg, mgref)
  print("F, Itot", F, Itot)
  print("mg*ve, (Pc-Pa)*At", mg * ve, (Pc - Pa) * At)
  print()


if __name__ == "__main__":
  main()
