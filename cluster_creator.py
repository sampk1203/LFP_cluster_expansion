import os
import numpy as np
import csv
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from icet import ClusterSpace, StructureContainer
from icet.tools import enumerate_structures
from icet import ClusterExpansion
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
import random
random.seed(42)
np.random.seed(42)
# =============================
# SETTINGS
# =============================

cif_file = "LFP.cif"
supercell = [3, 3, 3]
cutoffs = [10.0, 10.0]
n_training = 5000
samples_per_composition = 90
fmax = 0.05
max_steps=500
output_folder = "1_cutoff_10_10_set_5000"

os.makedirs(output_folder, exist_ok=True)

# =============================
# 1. Build 2x2x2 supercell
# =============================

prim = Structure.from_file(cif_file)
prim.make_supercell(supercell)
ase_full = AseAtomsAdaptor.get_atoms(prim)

write(os.path.join(output_folder, "supercell_full.cif"), ase_full)

li_indices = [i for i, a in enumerate(ase_full) if a.symbol == "Li"]
n_li = len(li_indices)

print("Total Li sites:", n_li)

full_frac = ase_full.get_scaled_positions()
full_cart = ase_full.get_positions()

# =============================
# 2. Cluster space
# =============================

chemical_symbols = []
for i in range(len(ase_full)):
    if i in li_indices:
        chemical_symbols.append(['Li', 'X'])
    else:
        chemical_symbols.append([ase_full[i].symbol])

cs = ClusterSpace(
    ase_full,
    cutoffs=cutoffs,
    chemical_symbols=chemical_symbols
)
cs.write("LFP_cluster_space.cs")
sc = StructureContainer(cs)
# Track unique occupation patterns to avoid duplicates
seen_configurations = set()
# =============================
# 3. ORB model (default 1)
# =============================

orbff = pretrained.orb_v3_conservative_inf_omat(
    device="cuda", precision="float32-high"
)
calc = ORBCalculator(orbff, device="cuda")
#print("MLIP loaded")
# =============================
# 4. Generate training structures
# =============================

structures = []



for n_remove in range(n_li + 1):

    for _ in range(samples_per_composition):

        s = ase_full.copy()

        remove_indices = random.sample(li_indices, n_remove)

        for idx in remove_indices:
            s[idx].symbol = "X"

        structures.append(s)

random.shuffle(structures)
structures = structures[:n_training]

print("Generated random structures:", len(structures))
# =============================
# 5. Process each structure
# =============================

for idx, s in enumerate(structures):
    # ---- DUPLICATE CHECK BEFORE ANYTHING ----
    occupation = tuple(cs.get_cluster_vector(s))

    if occupation in seen_configurations:
        print("Duplicate configuration detected — skipping")
        continue
    else:
        seen_configurations.add(occupation)

    print(f"\nProcessing structure {idx}")

    atoms_with_X = s.copy()

    # -------------------------
    # Identify removed Li sites
    # -------------------------

    removed_data = []

    for site_index in li_indices:
        if atoms_with_X[site_index].symbol == "X":

            frac = full_frac[site_index]
            cart = full_cart[site_index]

            removed_data.append([
                f"removed_{idx}.cif",  # filename
                None,                  # placeholder for energy
                site_index,
                frac[0], frac[1], frac[2],
                cart[0], cart[1], cart[2]
            ])

    # -------------------------
    # Write generated structure (with X)
    # -------------------------

    write(
        os.path.join(output_folder, f"generated_{idx}.cif"),
        atoms_with_X
    )

    # -------------------------
    # Physically remove X atoms
    # -------------------------

    keep_mask = [a.symbol != "X" for a in atoms_with_X]
    removed_atoms = atoms_with_X[keep_mask]

    # Write structure AFTER removal, BEFORE optimisation
    removed_path = os.path.join(output_folder, f"removed_{idx}.cif")
    write(removed_path, removed_atoms)

    # -------------------------
    # Relax with ORB
    # -------------------------

    removed_atoms.calc = calc
    dyn = BFGS(removed_atoms, logfile="-")
    converged = dyn.run(fmax=fmax, steps=max_steps)

    if not converged:
        print("WARNING: Structure did not converge — skipping")
        continue

    energy = removed_atoms.get_potential_energy()

    sc.add_structure(
    s,
    properties={"energy": energy}
    )
    print("Relaxation done")
    # -------------------------
    # Write relaxed structure
    # -------------------------

    relaxed_path = os.path.join(output_folder, f"relaxed_{idx}.cif")
    write(relaxed_path, removed_atoms)

    # -------------------------
    # Write CSV (with energy)
    # -------------------------

    csv_path = os.path.join(output_folder, f"removed_Li_{idx}.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "structure_filename",
            "energy_eV",
            "supercell_index",
            "frac_x", "frac_y", "frac_z",
            "cart_x", "cart_y", "cart_z"
        ])

        for row in removed_data:
            row[1] = energy  # insert energy
            writer.writerow(row)

    # -------------------------
    # Add to CE container
    # -------------------------

    print(f"Energy: {energy:.6f} eV")

# filter out any structures with NaN energies
clean_sc = StructureContainer(cs)

for idx, fit_struct in enumerate(sc):
    # fit_struct is a FitStructure object
    energy = fit_struct.properties["energy"]
    atoms = fit_struct.structure  # <-- get ASE Atoms object

    if not np.isnan(energy):
        clean_sc.add_structure(atoms, properties={"energy": energy})
    else:
        print(f"Skipping NaN energy for structure {idx}")

# overwrite container
sc = clean_sc


from icet import ClusterExpansion
import json

# =============================
# 6. Fit Cluster Expansion (Multiple Methods)
# =============================

from trainstation import CrossValidationEstimator

fit_data = sc.get_fit_data(key="energy")

methods = [
    "least-squares",
    "ridge",
    "lasso",
    "elasticnet",
    "ardr",
    "bayesian-ridge",
    "omp",
    "rfe",
    "split-bregman"
]

for method in methods:
    print(f"\nFitting using {method}...")

    try:
        opt = CrossValidationEstimator(fit_data, fit_method=method)
        opt.train()  # fit on all data
    except Exception as e:
        print(f"Skipping {method}, error: {e}")
        continue

    ce = ClusterExpansion(cluster_space=cs, parameters=opt.parameters)

    ce_filename = f"LFP_CE_{method}.ce"
    ce.write(ce_filename)

    print(f"{method} fit saved to {ce_filename}")

# Save training container
sc.write("LFP_training_data.sc")

print("\nAll models fitted and saved.")
