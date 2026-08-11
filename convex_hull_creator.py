import numpy as np
import matplotlib.pyplot as plt
from icet import ClusterExpansion, StructureContainer
from scipy.spatial import ConvexHull

# =========================
# SETTINGS
# =========================
ce_files = ["/media/sampk/350GB/1_qpivolta/5_OCV curve/1_LFP/2_cluster_expansion/3x3x3/cluster_3x3x3/LFP_CE_lasso.ce", "/media/sampk/350GB/1_qpivolta/5_OCV curve/1_LFP/2_cluster_expansion/3x3x3/cluster_3x3x3/LFP_CE_ridge.ce", "/media/sampk/350GB/1_qpivolta/5_OCV curve/1_LFP/2_cluster_expansion/3x3x3/cluster_3x3x3/LFP_CE_least-squares.ce"]  # list of CE files
sc_file = "/media/sampk/350GB/1_qpivolta/5_OCV curve/1_LFP/2_cluster_expansion/3x3x3/cluster_3x3x3/LFP_training_data.sc"  # training structures

# Reference energies (supercell)
E_LFP_supercell = -1526.177002 *27/8     #change for different supercell
E_delith_supercell = -1355.017212 *27/8   #change for different supercell
N_fu = 3*3*3     #change for different supercell
# Labels and colors for plotting
labels = ["Lasso CE", "Ridge CE", "Least Squares CE"]
colors = ["blue", "green", "orange"]

# =========================
# LOAD TRAINING DATA
# =========================
sc = StructureContainer.read(sc_file)

# Li site indices from first structure
li_indices = [i for i, a in enumerate(sc[0].structure) if a.symbol == "Li"]
total_li_sites = len(li_indices)
# =========================
# FUNCTION TO COMPUTE FORMATION ENERGIES
# =========================
def compute_formation_energy(ce):
    fractions = []
    formation_energies = []

    for fit_struct in sc:
        atoms = fit_struct.structure
        n_Li = sum(1 for i in li_indices if atoms[i].symbol == "Li")
        x = n_Li / total_li_sites
        fractions.append(x)

        E_pred = ce.predict(atoms)
        #print(E_pred)
        # Formation energy per formula unit (divide by 8 supercell size)
        E_form = (E_pred - x * E_LFP_supercell - (1 - x) * E_delith_supercell)/N_fu
        formation_energies.append(E_form)

    return np.array(fractions), np.array(formation_energies)

# =========================
# PLOT MULTIPLE CEs WITH CONVEX HULL
# =========================
plt.figure(figsize=(7,5))

for ce_file, label, color in zip(ce_files, labels, colors):
    ce = ClusterExpansion.read(ce_file)
    fractions, formation_energies = compute_formation_energy(ce)

    # Scatter points
    plt.scatter(fractions, formation_energies, color=color, alpha=0.6, label=f"{label} energies")

    # Convex hull
    points = np.column_stack([fractions, formation_energies])
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_points = hull_points[np.argsort(hull_points[:,0])]

    # Keep only points on lower envelope
    lower_hull = [hull_points[0]]
    for p in hull_points[1:]:
        while len(lower_hull) >= 2:
            x1, y1 = lower_hull[-2]
            x2, y2 = lower_hull[-1]
            x3, y3 = p
            if (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1) < 0:
                lower_hull.pop()
            else:
                break
        lower_hull.append(p)
    lower_hull = np.array(lower_hull)

    # Plot convex hull
    plt.plot(lower_hull[:,0], lower_hull[:,1], color=color, lw=2, label=f"{label} hull")

plt.xlabel("Li fraction (x)")
plt.ylabel("Formation energy per formula unit (eV)")
plt.title("Li‑vacancy Convex Hull Comparison")
plt.legend()
plt.tight_layout()
plt.show()