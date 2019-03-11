import numpy as np


x0 = np.array([10., 0., 0., 0., 0., 0., 1.])
kg_to_mass_unit__coeff = 1 / 16290
meter_to_distance_unit_coeff = 1 / 637100
time_coff = 1 / 60
newton_to_force_unit_coeff = kg_to_mass_unit__coeff * meter_to_distance_unit_coeff / (time_coff ** 2)
T = 600 * time_coff
n = 50
dt = T / n
grad_time = 0
J_time = 0
Grav = 6.673 * (10 ** (-11)) * meter_to_distance_unit_coeff ** 3 / kg_to_mass_unit__coeff / (time_coff ** 2)
M = 5.972 * (10 ** 24) * kg_to_mass_unit__coeff
alpha = Grav * M
p0 = 101325 * kg_to_mass_unit__coeff / meter_to_distance_unit_coeff / (time_coff ** 2)
h0 = 8635 * meter_to_distance_unit_coeff
A = (0.25 * 0.75 ** 2) * np.pi * meter_to_distance_unit_coeff ** 2
Cd = 0.2
isp = 400 * time_coff
g0 = 9.81 * meter_to_distance_unit_coeff / (time_coff ** 2)
e0 = 0.25
a0 = 10 + 900000 * meter_to_distance_unit_coeff
v_ideal = 7400 * meter_to_distance_unit_coeff / time_coff
rocket_radius = 0.75 * meter_to_distance_unit_coeff
rocket_height = 16 * meter_to_distance_unit_coeff
rocket_inertia = 1 / 12 * (3 * rocket_radius ** 2 + rocket_height ** 2) + 1 / 4 * rocket_height ** 2
deg = 5
X_i, W_i = np.polynomial.legendre.leggauss(deg)
X_i = (X_i + 1) / 2
W_i /= 2
time_array_u = np.linspace(0, T, n)
inv_isp_g0 = 1 / (isp * g0)
half_A_Cd_p0 = .5 * A * Cd * p0
_3_4ths_meter_to_distance_unit_coeff__over__rocket_inertia = .75 * meter_to_distance_unit_coeff / rocket_inertia
inv_h0 = 1. / h0
ideal_orbit_x = a0 * np.array([np.cos(i * 2 * np.pi / 5000) for i in range(0, 5000)])
ideal_orbit_y = a0 * np.array([np.sin(i * 2 * np.pi / 5000) for i in range(0, 5000)])
drag_constant = 0.5 * p0 * Cd * A

