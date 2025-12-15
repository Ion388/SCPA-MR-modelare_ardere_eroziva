import numpy as np
import matplotlib.pyplot as plt
# a = 49.81390
# b = 297.3000	
# c = -186.1620
# d = 128.3380
# e = 0.245965
# t = 300/1000
# molecular_weight = 27/1000 # kg/mol
# # cp = a + b*t + c*(t**2) + d*(t**3) + e/(t**2) # J/mol/K
# cp = 900
# # cp = 0.239*cp / (molecular_weight*1000) # kJ/kg/K
# cp = cp / molecular_weight # J/kg/K
# print("cp=", 0.239*cp/1000) # kcal/kg/K

# print("cp_total=", 0.69*1.1 + 0.19*0.9 + 0.12*2.5)
# # print(np.exp(-50))

# r_lim = 1.45
# theta_spread = 0.6
# y = np.linspace(r_lim-0.2, r_lim, 21)
# x = (1-np.pi*theta_spread*(r_lim-y)*(r_lim+y))/(4*y)
# print(np.vstack((y, x)).T)
# thickness = y/x
# print("thickness=", thickness)
# plt.plot(y, x)
# plt.grid()
# plt.show()"

# k = 1.14
# pressure_ratio = np.arange(0.00001, 0.02, 0.00001)
# right_side = 2*k/(k-1)*pressure_ratio**(2/k)*(1 - pressure_ratio**((k-1)/k))
# plt.plot(pressure_ratio, right_side)
# # plt.yscale("log")
# # plt.xscale("log")
# plt.axhline(y=0.00158304515625, color='red', linestyle='--', label='y=0.00158304515625')
# plt.legend()
# plt.grid()
# plt.xlabel("p0/pe")
# plt.ylabel("Right side")
# plt.title("Isentropic flow relation")
# plt.show()

# k = 1.14
# R = 280
# T1 = 3550
# cstar = np.sqrt(R * T1) / (np.sqrt(k) * np.sqrt((2/(k+1))**((k+1)/(k-1))))
# print("cstar=", cstar)
# cstar = 1/0.6366*np.sqrt(R * T1)
# print("cstar=", cstar)

# M = 3.4
# k = 1.14
# Ar = (1.0 / M) * (((1.0 + 0.5 * (k - 1.0) * M * M)/(0.5 * (k + 1))) ** ((k + 1.0) / (2.0 * (k - 1.0))))
# print("Ar=", Ar)

gasses_list = ["CO", "CO2", "Cl", "H", "HCl", "H2", "H2O", "N2"]
fractions_list = [0.224, 0.008, 0.006, 0.025, 0.12, 0.33, 0.09, 0.1]
gas_viscosity = []
print(np.exp(-5))