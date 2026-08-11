# LFP Open-Circuit Voltage (OCV) Curve from Cluster Expansion

Predicts the 0 K open-circuit voltage curve of LiFePO₄ (LFP) as a function of lithiation
fraction, using a **cluster expansion (CE)** trained on machine-learning-relaxed
Li/vacancy configurations. The pipeline goes: generate delithiated structures → relax
with an MLIP → fit a cluster expansion to the relaxed energies → derive formation
energies, a convex hull, and a smooth OCV curve suitable for use as a physics-based
voltage function (e.g. in PyBaMM).

## Pipeline overview

```
cluster_creator.py        →  convex_hull_creator.py   →  voltage_vs_comp.py
(generate + relax +          (formation energy +          (voltage vs. composition +
 fit CE)                      convex hull plot)             OCV curve fit)
```

### 1. `cluster_creator.py`
- Builds a supercell of LFP from a CIF file and enumerates random Li/vacancy
  configurations across the full composition range (0 → 100% delithiated).
- Relaxes each configuration with the **ORB** MLIP (`orb_v3_conservative_inf_omat`,
  via `orb-models` + ASE `BFGS`).
- Skips duplicate configurations (by cluster vector) and unconverged relaxations.
- Fits a **cluster expansion** to the relaxed energies using [icet](https://icet.materialsmodeling.org/),
  trying multiple regression methods (least-squares, ridge, lasso, elastic net, ARDR,
  Bayesian ridge, OMP, RFE, split-Bregman) via `trainstation`.
- Outputs: per-structure CIFs (generated / delithiated / relaxed), a CSV of removed-Li
  site data per structure, one `.ce` file per fitting method, and the training
  `StructureContainer` (`.sc`).

### 2. `convex_hull_creator.py`
- Loads one or more fitted `.ce` files and the training structure container.
- Computes **formation energy per formula unit** at each Li fraction relative to fully
  lithiated (LFP) and fully delithiated (FePO₄) reference energies.
- Constructs the **lower convex hull** (stable phase boundary) for each CE fit and
  plots them together for comparison.

### 3. `voltage_vs_comp.py`
- Loads a single fitted CE and its training data.
- Groups predicted energies by Li count, computing min/avg/max energy at each
  composition.
- Computes the **incremental voltage** between adjacent compositions via
  ΔG = -(E₂ - E₁ - Δn·E_Li_metal) / Δn (vs. Li/Li⁺).
- Fits the resulting stepwise voltage profile to a smooth empirical OCV function
  (Afshar 2017 form: `a + b·x + c·exp(d·x) + e·exp(f·(1-x))`) via `scipy.optimize.curve_fit`.
- Plots the raw stepwise voltage curve alongside the smooth fit, for min/avg/max
  energy cases.

## Dependencies

- [pymatgen](https://pymatgen.org/)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [icet](https://icet.materialsmodeling.org/) (`ClusterSpace`, `ClusterExpansion`, `StructureContainer`)
- [trainstation](https://trainstation.materialsmodeling.org/) (`CrossValidationEstimator`)
- [orb-models](https://github.com/orbital-materials/orb-models) (MLIP; requires a CUDA-capable GPU as configured)
- numpy, scipy, matplotlib

```bash
pip install pymatgen ase icet trainstation orb-models numpy scipy matplotlib
```

## Usage

1. Place your structure file (e.g. `LFP.cif`) in the working directory.
2. Edit the `SETTINGS` block at the top of `cluster_creator.py` (supercell size, CE
   cutoffs, number of training structures, output folder) and run it:
   ```bash
   python cluster_creator.py
   ```
3. Update the file paths at the top of `convex_hull_creator.py` to point at the `.ce`
   and `.sc` files produced above, and set the reference energies
   (`E_LFP_supercell`, `E_delith_supercell`) for your supercell size. Run:
   ```bash
   python convex_hull_creator.py
   ```
4. Update the file paths and `TOTAL_LI_SITES` / `E_Li_metal` in `voltage_vs_comp.py`,
   then run:
   ```bash
   python voltage_vs_comp.py
   ```

**Note:** file paths in `convex_hull_creator.py` and `voltage_vs_comp.py` are
currently hardcoded to a local directory structure and will need to be updated for
your own setup.

## Method notes

- Random seeds are fixed (`random.seed(42)`, `np.random.seed(42)`) for reproducible
  structure generation.
- Duplicate configurations are detected via their cluster vector before any relaxation
  is attempted, to avoid wasted compute.
- The OCV fit form and voltage sign convention follow standard practice for
  first-principles-derived voltage curves relative to Li metal.

## Citation / acknowledgments

This workflow builds on:
- `icet` for cluster expansion construction and fitting
- `orb-models` (Orbital Materials) for MLIP-based structure relaxation
- `pymatgen` / `ASE` for structure handling

## License

Add a license (e.g. MIT) here before wider distribution.
