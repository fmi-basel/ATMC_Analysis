# ATMC_Analysis

A notebook-driven pipeline for extracting morphology and intensity features from segmented objects stored as **OME-Zarr plates**, and exporting results as `.csv` and `.h5ad` (AnnData) for downstream QC, filtering, and analysis.

---

## Workflow Overview

The pipeline is organized into a series of Jupyter notebooks, each handling a specific stage of the analysis:

| Notebook | Description |
|---|---|
| `1_FeatureExtraction.ipynb` | Loads plate layout Excel files and OME-Zarr data, estimates thresholds, and extracts features from segmented objects. **Main entry point.** |
| `1_FeatureLoading.ipynb` | Use *instead of* notebook 1 when features were already computed elsewhere (e.g. by a Fractal task) and are stored as AnnData tables inside the OME-Zarr. Loads and merges them rather than measuring anything. |
| `1b_Match1to1.ipynb` | Matches and merges objects between two runs by centroid proximity (e.g. pairing cells with nuclei). |
| `2_Filtering.ipynb` | Applies QC gates and outlier removal to the extracted feature tables. |
| `3_QualityControl.ipynb` | Computes outgrowth, normalizes features per group, and generates plate-bias heatmaps. |
| `4_DM.ipynb` | Downstream analysis: dimensionality reduction (PCA, diffusion map, UMAP, t-SNE), clustering, and trajectory inference. |
| `5_Plotting.ipynb` | Advanced result visualization and figure generation utilities. |

Notebooks 2 → 5 each take the `.h5ad` written by the previous one, so run them in order.

---

## Function Modules (`Functions/`)

| Module | Description |
|---|---|
| `Fcts_FE.py` | Feature extraction primitives and experiment setup parsing. |
| `Fcts_Base.py` | I/O helpers, Excel layout parsing, barcode↔folder resolution, and saving utilities. |
| `Fcts_DimRed.py` | Dimensionality reduction and clustering methods. |
| `Fcts_Filtering.py` | Filtering functions and the galleries that show what a filter removed. |
| `Fcts_Matching.py` | Matching and merging of datasets. |
| `Fcts_Plotting.py` | Plotting and visualization utilities. |
| `Fcts_QC.py` | Quality control, normalization, and plate bias analysis. |

---

## Plate Layout Files (`Layout_96.xlsx`, `Layout_384.xlsx`)

Excel-based plate layout files define experimental metadata and well conditions. Exactly one
`Layout*.xlsx` must sit in the experiment folder; it is required for correct feature extraction
and annotation. Templates for both plate formats are in `PlateLayout_Templates/`.

Each file contains four plate-grid sheets plus one immunostaining table.

### The four plate-grid sheets

| Sheet | Becomes | Typical use |
|---|---|---|
| `MediumLayout` | `Medium` | Treatment / medium condition |
| `StainingLayout` | `ABs` | Which antibody mix the well was stained with — must match an **Antibody Mix** name in the immunostaining table |
| `LineLayout` | `Cell_line` | Cell line or donor |
| `OtherLayout` | `Other` | Any remaining condition. Threshold estimation and the QC plots group by this column, so this is where a **timepoint / day** belongs |

Each sheet holds one grid per plate, introduced by a `Barcode:` label cell. A well is included
in the analysis only if **all four sheets** give it a value; a well left empty everywhere is
skipped silently, and a well filled in some sheets but not others raises an error naming the
well. Wells present in the OME-Zarr but absent from the layout are never processed.

### Linking a barcode to its OME-Zarr folder

Each plate block starts with a `Barcode:` label cell followed by the barcode. Optionally add a
`Folder:` label and value on the same row to declare which OME-Zarr folder that plate lives in:

```
| Barcode: | 260429LG001ACAajACAakD10 | Folder: | acaak_d10 |
```

The hint only has to be a substring that **uniquely** identifies one OME-Zarr folder; it is
matched case-, `_`- and `-`-insensitively against the folder name first and then against the
full path (so e.g. `output/2` can disambiguate identical folder names in different parent
directories). It needs to appear in only one sheet, but repeating it is harmless as long as the
hints agree.

If no `Folder:` cell is given, the barcode itself is matched against the folder name, which is
enough whenever your OME-Zarr folders are named after their barcode. Barcodes are **never**
paired with folders by sort order: anything that cannot be resolved to exactly one folder raises
an error listing the barcodes and the available folders. `make_experiment` prints the resolved
pairing — check it before continuing.

### The immunostaining table (in `StainingLayout`)

To the right of the plate grid, a small table maps each antibody mix to the stains it carries
and the channel each stain was imaged on:

| Target | Imaging Channel | Round | Antibody Mix |
|---|---|---|---|
| DAPI | 1 | 0 | AB1 |
| KRT7 | 2 | 0 | AB1 |
| AQP5 | 3 | 0 | AB1 |

- **`Imaging Channel` is 1-based and is the channel's real position in the image**, not a rank.
  Channel 1 is the first channel of the OME-Zarr channel axis. If a channel carries no stain of
  its own — brightfield and secondary-antibody rows are filtered out, and a layout may simply
  skip a number — leave the gap: the remaining stains keep their true channel. `get_stainings`
  prints which channels were left empty.
- **The number is relative to that round's own channel axis.** In a multiplexed (4i)
  acquisition, rounds routinely differ in how many channels they carry and in which order, so
  the same fluorophore sits at a different index in different rounds. For example, a round
  imaged as `DAPI, FITC, Texas Red, Cy5` puts Cy5 on channel 4, while a round imaged as
  `DAPI, Cy5` puts Cy5 on channel **2**. Number each row for the round it belongs to, not by a
  fixed fluorophore slot. If a number exceeds the channels the round actually has, extraction
  stops and names the stain and round rather than reading the wrong channel.
- **A stain must sit on the same channel in every mix of a round.** Thresholds are stored per
  stain, so a conflict raises rather than silently using one mix's channel for the other's images.
- **`Antibody Mix`** may list several mixes separated by commas to share one row.
- **`Round`** is optional. Omit the column (or leave it at `0`) for a single-round experiment;
  give a numeric round per row for multiplexed experiments, and the stains are stored per round.
- Rows whose `Target` contains *brightfield* are ignored, as are rows whose `Type` column says
  *secondary* when that legacy column is present.

---

## Data Requirements

### OME-Zarr Plates

Your experiment folder should contain one or more OME-Zarr plate folders. They may sit in
subdirectories; `find_zarr_dirs` searches breadth-first and takes every match at the shallowest
depth where it finds any.

**Assumptions:**
- The ROI table and the label image exist in the segmentation round (commonly `0`).
- Additional rounds may be intensity-only — no labels or tables of their own. Segmentation is
  assumed identical across rounds, and all rounds must be registered onto the same pixel grid;
  a mismatch between a round's image and the segmentation mask raises an error naming the object.
- A **round id is the `image_name` group inside each well**, i.e. the pipeline reads round `1`
  from `<plate>.zarr/<row>/<col>/1/`. Rounds are loaded with `ez_zarr.import_plate(..., image_name="1")`.

**Round ids need not be contiguous.** A plate whose wells contain image groups
`0, 1, 2, 3, 4, 8, 14, ... 28` is handled as-is; list exactly the ones you want in
`rounds_to_extract`.

**Scale warning for many rounds.** `compute_cross_round_pearson=True` correlates every pair of
(round, stain) intensity vectors, which grows quadratically: an 18-round panel with 57 stains
produces 1,653 correlation columns, about 40% of a ~4,000-column table. Set it to `False`
unless you need the cross-round correlations, and expect the merged `.csv` to reach tens of GB
at that width — the `.h5ad` is the practical output.

**`file_ending`** selects which folders count as plates. It defaults to `.zarr`, which picks up
every plate. Narrow it when a folder holds several variants of the same plates and you want only
one — for example `_mip.zarr` to analyse maximum-intensity projections only.

### Excel Layout (`Layout*.xlsx`)

Place exactly one `Layout*.xlsx` in the experiment folder alongside the plates. If several are
found the pipeline warns and uses the first in natural-sort order, which is rarely what you want.

---

## Getting Started

### 1. Install uv

Choose the instructions for your operating system.

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

After installation, restart your terminal and verify it worked:
```bash
uv --version
```

### 2. Clone the Repository

```bash
git clone <your-repo-url>
cd <project-folder>
```

Make sure `pyproject.toml`, `uv.lock`, and `.python-version` are present in this folder before continuing.

### 3. Create the Environment and Install Dependencies

Run this single command on **any** OS — uv reads `.python-version`, downloads that Python version if it's not already installed, creates a `.venv` folder, and installs the exact package versions from `uv.lock`.

```bash
uv sync
```

> **On a shared Linux server (e.g. via SSH)**, where installing into the base Python is not
> permitted, use a personal conda environment instead of `.venv` — this replaces steps 3–5:
>
> ```bash
> conda create -n atmc_hcs_env python=3.13 -c conda-forge --override-channels
> conda activate atmc_hcs_env
> pip install uv
> export UV_PROJECT_ENVIRONMENT=$CONDA_PREFIX
> uv sync
> python -m ipykernel install --user --name=ATMC_HCS_Analysis --display-name="ATMC HCS Analysis"
> ```
>
> If `conda create` fails with a `403 FORBIDDEN` error on the `pkgs/main` channel, this is due to
> Anaconda's default channel access restrictions — using `-c conda-forge --override-channels` as
> shown above resolves this.

### 4. Activate the Environment

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\activate
```

**Windows (Command Prompt)**
```cmd
.venv\Scripts\activate.bat
```

### 5. Register the Jupyter Kernel

With the environment activated, run:

```bash
python -m ipykernel install --user --name=ATMC_HCS_Analysis --display-name="ATMC HCS Analysis"
```

### 6. Run `1_FeatureExtraction.ipynb`

Set the parameters in the user-input cells, then execute top to bottom.

**Paths and data**

```python
source              # experiment folder containing the .zarr plate(s) + Layout*.xlsx
analysis_dir        # where results go; None = same as source
file_ending         # ".zarr" — narrow it to select a subset of plates
experiment_ID       # free-text label stored on every object
table_name          # name of the ROI table inside the OME-Zarr (e.g. "organoids_ROI_table")
label_name          # name of the label image (e.g. "organoids")
```

`table_name` and `label_name` must match what is actually in your plates. The names are
printed by `ez_zarr` when a plate is loaded, and a typo surfaces as an empty or missing table.

**Measurement settings**

```python
pyramid_level       # resolution level used for measuring; 0 = full resolution
quantiles_to_calc   # intensity quantiles computed per stain
segmentation_round  # round holding the ROI table + labels (typically 0)
rounds_to_extract   # rounds to measure intensities in, e.g. [0] or [0, 1, 2]
do_moments_features / do_skeleton_features /
do_thresholded_features / do_substructure_features   # feature groups on/off
```

**Thresholds.** The threshold cells estimate one value per stain per round and show the
distributions plus example crops. Override any of them in the *Manual change* cell, keyed as
`"R{round}__{stain}"`:

```python
thresholds_to_change = {"R0__DAPI": 1000, "R1__EPCAM": 750}
```

Every stain in `rounds_to_extract` needs a finite threshold before extraction starts. A missing
or `NaN` threshold would make each thresholded feature silently zero, so extraction refuses and
names the offenders instead. Either fix the value, or switch off `do_thresholded_features` and
`do_substructure_features`.

**Crypt (skeleton) settings.** Tune these in the *Test of crypt features extraction* cell and
the same values are used for the real extraction:

```python
sigma_skeleton          # smoothing of the mask before skeletonization
n_angle_determination   # pixels used to estimate each branch direction
radius_multiplier       # fraction of the max-inscribed-circle radius treated as crypt-free
```

---

## Outputs

`analysis_dir` gets a four-folder structure, and results land in `2_Tables/`:

```
<analysis_dir>/
├── 1_Scripts/
├── 2_Tables/     <- .csv and .h5ad outputs
├── 3_Plots/
└── 4_Results/
```

- **Per-plate `.csv`**, one per barcode, plus a merged `.csv` over all plates. Filenames carry a
  timestamp; nothing is ever overwritten.
- **AnnData (`.h5ad`)**, the input to notebook 2:
  - `X` — the numerical feature matrix
  - `obs` — per-object metadata: `Organoid_ID`, `Barcode`, `Well`, `Object`, `Medium`, `ABs`,
    `Cell_line`, `Other`, `PATH`, `Experiment_ID`, the ROI bounding box and centroid in µm, and
    the `Staining_R{r}_Ch{n}` columns recording which stain sat on which channel
  - `uns` — the configuration needed to re-open the images later: `stainings`,
    `thresholds_by_round`, `experiment_setup`, `folders` (the resolved `{barcode: folder}` map),
    `source_dir`, `table_name`, `label_name`, `pixel_spacing`, `pyramid_level`,
    `rounds_to_extract`, `segmentation_round`, `table_dir`, `plot_dir`
  - `layers` — added by notebook 3: `log1p`, `z_scaled`, `robust_scaled`, `minmax_scaled`

Each plate's extraction prints how many objects it produced and, if any were dropped, why —
for example an ROI whose mask is not a single connected component.

---

## Feature Reference

**Units.** Every length is in µm and every area in µm², derived from the pixel size at the
`pyramid_level` you measured at. The only exception is `StraightEdge_longest`, which is a pixel
count and therefore depends on `pyramid_level`.

**Naming.** Intensity features are prefixed with the round and stain: `R0__DAPI_mean`.
Morphology features have no prefix, because they are measured once from the segmentation round.

| Group | Features |
|---|---|
| Size / shape | `area`, `area_bbox`, `area_convex`, `axis_major_length`, `axis_minor_length`, `eccentricity`, `equivalent_diameter_area`, `extent`, `feret_diameter_max`, `perimeter`, `solidity`, `circularity`, `AxisRatio`, `aspectRatio_equivalentDiameter` |
| Convexity | `concavity`, `concavity_count`, `asymmetry` |
| Edge artefacts | `StraightEdge_longest`, `StraightEdge_longest_fraction` — high values flag objects clipped by a field border |
| Crypts | `crypt_count`, `crypt_length_total`, `crypt_length_max` |
| Intensity (per stain) | `_min`, `_mean`, `_max`, `_std`, `_Q{q}` over all pixels in the mask |
| Thresholded (per stain) | `_T_min/_T_mean/_T_max/_T_std/_T_Q{q}` over pixels above the stain threshold, plus `_area_T` (µm²), `_area_T_ratio`, `_asymmetry`, `_potency` (mean × thresholded area) |
| Substructure (per stain) | `_substructures_count`, `_substructures_size` (mean µm²), `_substructures_intensity` — connected components of the thresholded signal **inside** the object |
| Moments | `moments-*` and `{stain}_moments_weighted-*` from `regionprops` |
| Co-localization | `R0__DAPI--R1__EPCAM_PearsonR` — Pearson correlation between every pair of (round, stain) intensity vectors within the mask |

`centroid-0` / `centroid-1` are the object's position inside its own crop. They carry no
phenotype, so they are kept in `obs` as metadata rather than fed to PCA/UMAP/clustering.

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `Could not resolve barcode -> OME-Zarr folder unambiguously` | Add a `Folder:` cell next to the listed barcode, or make an existing hint more specific. The error lists every available folder. |
| `No directories ending in '.zarr' found under ...` | Check `source` and `file_ending`. The search stops at the shallowest depth that contains any match. |
| `... have no usable threshold` | A stain has a missing or `NaN` threshold. Re-run the threshold estimation cell, set the value in `thresholds_to_change`, or disable `do_thresholded_features` and `do_substructure_features`. |
| `Stain 'X' is on imaging channel A in mix ... but on channel B in mix ...` | The same stain sits on different channels in two antibody mixes of one round. Thresholds are keyed by stain, so align the channels in the layout. |
| `declares stain '...' on imaging channel N, but the image has only M channel(s)` | An `Imaging Channel` number in the layout exceeds the channels present. Remember the numbering is 1-based. |
| `round R image has shape ... but the segmentation-round mask has shape ...` | The rounds are at different pixel sizes or are not registered onto a common grid. |
| Some wells are skipped | Only wells annotated in **all four** layout sheets are processed. The per-plate summary printed during extraction says how many wells and objects were dropped and why. |
| Fewer objects than expected | Objects whose mask is not a single connected component are skipped by design. The count appears in the per-plate summary. |
| A stain-specific filter removes everything | Objects stained with a different antibody mix have `NaN` for that feature. `filter_organoids_by` keeps them by default (`keep_na=True`); pass `keep_na=False` to drop them deliberately. |
| `subsample_anndata_geosketch needs the optional 'geosketch' package` | Geometric sketching is not part of the pinned dependency set. Add it with `uv add geosketch`, or use `random_subset_anndata_frac` instead. |
| Saving `.h5ad` fails with `"dict key is int"` | Something you added to `adata.uns` has non-string dict keys. Convert them to strings before writing; the pipeline's own round-indexed dicts are already converted. |

---

## License

The ATMC_Analysis workflow was created for the **Applied Tissue Models Center (ATMC)** at the **Friedrich Miescher Institute for Biomedical Research (FMI)** and is released under the [MIT License](LICENSE). All copyright belongs to the FMI.
