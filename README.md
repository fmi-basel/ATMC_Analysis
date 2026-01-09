# ATMC_Analysis

Notebook-driven pipeline for extracting morphology + intensity features from segmented organoids stored as OME-Zarr plates, and exporting results as `.csv` and `.h5ad` (AnnData) for downstream QC, filtering, and analysis.

## Workflow

Run notebooks in this order:

1. `1_FeatureExtraction.ipynb` – load layout + OME-Zarr, estimate thresholds, extract features.
2. `2_Filtering.ipynb` – feature/table filtering (QC gates, outlier removal, etc.).
3. `3_QualityControl.ipynb` – QC plots and sanity checks.
4. `4_DM.ipynb` – downstream analysis / dimensionality reduction.
5. `5_Plotting.ipynb` – plotting utilities and result visualization.

Core functionality lives in:
- `Fcts_FE_MultiRound.py` (multi-round feature extraction)
- `Fcts_FE.py` (feature primitives)
- `Fcts_Base.py` (I/O helpers, Excel layout parsing, saving)

## Data requirements

### OME-Zarr plates
Your experiment folder should contain one or more OME-Zarr plate folders (`*.zarr`).

Assumptions:
- ROI table + labels exist in `segmentation_round` (commonly `0`).
- Additional rounds may exist as intensity-only rounds.
- “Round id” is assumed to match ez_zarr `image_name` groups (e.g. round `1` lives under image name `"1"`).

### Excel layout (`Layout*.xlsx`)
Place a `Layout*.xlsx` in the same experiment folder.

The pipeline reads stain/channel metadata from a `StainingLayout` sheet (antibody mix → stains → imaging channels).
- Single-round layouts: no `Round` column → treated as round `0`.
- Multi-round layouts: include numeric `Round` column → stains are stored per round.

## Getting started (conda)

### 1) Create environment
From a terminal:

```bash
conda create -n atmc_analysis python=3.12
conda activate atmc_analysis
```

### 2) Install dependencies
Install core packages from conda-forge (recommended):

```bash
conda install -c conda-forge ez-zarr anndata zarr
```

Then install a typical scientific Python stack as needed:

```bash
conda install -c conda-forge numpy pandas scipy scikit-image scikit-learn matplotlib seaborn tqdm h5py openpyxl
```

If something is missing, fall back to pip inside the environment:

```bash
pip install <missing-package>
```

### 3) Run notebooks
From the repo root:

```bash
jupyter lab
```

Open `1_FeatureExtraction.ipynb` and configure:
- `source` (experiment folder containing the `.zarr` plate(s) + `Layout*.xlsx`)
- `table_name` / `label_name`
- `segmentation_round`
- `rounds_to_extract`

Then execute top-to-bottom.

## Outputs

Typical outputs:
- Tables (`.csv`) saved under an analysis directory (e.g. `.../2_Tables/`).
- AnnData (`.h5ad`) containing:
  - numerical features in `X`
  - metadata in `obs`
  - configuration in `uns`

## Troubleshooting

- Saving `.h5ad` fails with “dict key is int”: convert nested dict keys in `adata.uns` to strings before writing when using round-indexed dicts.
- Segmentation overlay looks “rainbow / smeared”: use nearest-neighbor interpolation for label overlays to avoid color blending during resampling.
- Some wells are skipped: only wells included in the Excel-derived experiment setup are processed; missing annotations in the layout mean the well is not part of the run.

## License / citation

Add lab/license/citation information here.
