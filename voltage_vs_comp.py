import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from icet import ClusterExpansion, StructureContainer

# =========================
# SETTINGS
# =========================
ce_file = "/media/sampk/350GB/1_qpivolta/5_OCV curve/1_LFP/2_cluster_expansion/3x3x3_90/cluster_3x3x3_90/LFP_CE_lasso.ce"
sc_file = "/media/sampk/350GB/1_qpivolta/5_OCV curve/1_LFP/2_cluster_expansion/3x3x3_90/cluster_3x3x3_90/LFP_training_data.sc"

TOTAL_LI_SITES = 27*4
E_Li_metal = -1.9010885  # eV per Li atom

# =========================
# LOAD
# =========================
ce = ClusterExpansion.read(ce_file)
sc = StructureContainer.read(sc_file)

# =========================
# Group energies by n_Li
# =========================
energy_dict = defaultdict(list)

for fit_struct in sc:
    atoms = fit_struct.structure

    n_Li = sum(1 for atom in atoms if atom.symbol == "Li")
    E_pred = ce.predict(atoms)  # supercell energy

    energy_dict[n_Li].append(E_pred)

# =========================
# Compute min/avg/max energies
# =========================
n_Li_vals = []
x_vals = []

E_min = []
E_avg = []
E_max = []

print("\nComposition Table (Energy Statistics)")
print("-------------------------------------------------------------------")
print(" n_Li     x        E_min (eV)      E_avg (eV)      E_max (eV)")
print("-------------------------------------------------------------------")

for n_Li in sorted(energy_dict.keys()):

    energies = energy_dict[n_Li]

    e_min = np.min(energies)
    e_avg = np.mean(energies)
    e_max = np.max(energies)

    x = n_Li / TOTAL_LI_SITES

    n_Li_vals.append(n_Li)
    x_vals.append(x)

    E_min.append(e_min)
    E_avg.append(e_avg)
    E_max.append(e_max)

    print(f"{n_Li:5d}   {x:6.3f}   {e_min:12.6f}   {e_avg:12.6f}   {e_max:12.6f}")

# =========================
# Compute incremental voltages
# =========================
print("\nIncremental Voltage (vs Li/Li+)")
print("--------------------------------------------------------------")
print(" x1 -> x2      V_min (V)     V_avg (V)     V_max (V)")

volt_min = []
volt_avg = []
volt_max = []

x_midpoints = []

for i in range(len(n_Li_vals) - 1):

    n1 = n_Li_vals[i]
    n2 = n_Li_vals[i+1]

    delta_n = n2 - n1

    # Energies
    E1_min, E2_min = E_min[i], E_min[i+1]
    E1_avg, E2_avg = E_avg[i], E_avg[i+1]
    E1_max, E2_max = E_max[i], E_max[i+1]

    # Voltage calculation
    Vmin = -(E2_min - E1_min - delta_n * E_Li_metal) / delta_n
    Vavg = -(E2_avg - E1_avg - delta_n * E_Li_metal) / delta_n
    Vmax = -(E2_max - E1_max - delta_n * E_Li_metal) / delta_n

    volt_min.append(Vmin)
    volt_avg.append(Vavg)
    volt_max.append(Vmax)

    x1 = x_vals[i]
    x2 = x_vals[i+1]

    x_midpoints.append((x1 + x2) / 2)

    print(f"{x1:5.3f}->{x2:5.3f}     {Vmin:8.4f}      {Vavg:8.4f}      {Vmax:8.4f}")



from scipy.optimize import curve_fit

# =========================
# OCV fitting function
# =========================

def ocv_fit(x, a, b, c, d, e, f):
    return a + b*x + c*np.exp(d*x) + e*np.exp(f*(1-x))


# Convert lists to numpy
x_data = np.array(x_midpoints)

V_min = np.array(volt_min)
V_avg = np.array(volt_avg)
V_max = np.array(volt_max)

# =========================
# Fit curves
# =========================
initial_guess = [3.4, -0.02, 0.5, -150, -0.9, -30]

params_min, _ = curve_fit(ocv_fit, x_data, V_min, p0=initial_guess, maxfev=100000)
params_avg, _ = curve_fit(ocv_fit, x_data, V_avg, p0=initial_guess, maxfev=100000)
params_max, _ = curve_fit(ocv_fit, x_data, V_max, p0=initial_guess, maxfev=100000)

# =========================
# Print parameters
# =========================

print("\nFitted OCV parameters(Afshar2017 form)")
print("---------------------------")

print("\nMIN voltage fit:")
print(f"a={params_min[0]:.6f}, b={params_min[1]:.6f}, c={params_min[2]:.6f}, d={params_min[3]:.6f}, e={params_min[4]:.6f}, f={params_min[5]:.6f}")

print("\nAVG voltage fit:")
print(f"a={params_avg[0]:.6f}, b={params_avg[1]:.6f}, c={params_avg[2]:.6f}, d={params_avg[3]:.6f}, e={params_avg[4]:.6f}, f={params_avg[5]:.6f}")

print("\nMAX voltage fit:")
print(f"a={params_max[0]:.6f}, b={params_max[1]:.6f}, c={params_max[2]:.6f}, d={params_max[3]:.6f}, e={params_max[4]:.6f}, f={params_max[5]:.6f}")

# =========================
# Plot OCV Curves WITH FITS
# =========================
plt.figure(figsize=(7,5))

# Stepwise CE voltage data
plt.step(x_midpoints, volt_min, where='mid', linewidth=2, label="Min Energy Voltage (CE)")
plt.step(x_midpoints, volt_avg, where='mid', linewidth=2, label="Avg Energy Voltage (CE)")
plt.step(x_midpoints, volt_max, where='mid', linewidth=2, label="Max Energy Voltage (CE)")

# Smooth fitted curves
x_smooth = np.linspace(0,1,400)
plt.plot(x_smooth, ocv_fit(x_smooth, *params_min), linestyle="--", linewidth=2, label="Min Fit")
plt.plot(x_smooth, ocv_fit(x_smooth, *params_avg), linestyle="-", linewidth=2, label="Avg Fit")
plt.plot(x_smooth, ocv_fit(x_smooth, *params_max), linestyle="--", linewidth=2, label="Max Fit")

plt.xlabel("Li fraction (x)")
plt.ylabel("Voltage vs Li/Li⁺ (V)")
plt.title("OCV Curve from CE Energies (0 K)")

plt.xlim(0,1)
plt.ylim(0,5)

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



# =========================
# Plot OCV Curves
# =========================
plt.figure(figsize=(7,5))

plt.step(x_midpoints, volt_min, where='mid', linewidth=2, label="Min Energy Voltage")
plt.step(x_midpoints, volt_avg, where='mid', linewidth=2, label="Avg Energy Voltage")
plt.step(x_midpoints, volt_max, where='mid', linewidth=2, label="Max Energy Voltage")

plt.xlabel("Li fraction (x)")
plt.ylabel("Voltage vs Li/Li⁺ (V)")
plt.title("OCV Curve from CE Energies (0 K)")

plt.xlim(0,1)
plt.ylim(0,5)

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()