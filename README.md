# ATMC_Analysis

A notebook-driven pipeline for extracting morphology and intensity features from segmented objects stored as **OME-Zarr plates**, and exporting results as `.csv` and `.h5ad` (AnnData) for downstream QC, filtering, and analysis.

---

## Workflow Overview

The pipeline is organized into a series of Jupyter notebooks, each handling a specific stage of the analysis:

| Notebook | Description |
|---|---|
| `1_FeatureExtraction.ipynb` | Loads plate layout Excel files and OME-Zarr data, estimates thresholds, and extracts features from segmented objects. **Main entry point.** |
| `1_FeatureLoading.ipynb` | Alternatively loads and merges pre-computed feature tables from multiple rounds or plates for downstream analysis. |
| `1b_Match1to1.ipynb` | Matches and merges objects between datasets (e.g., for paired or longitudinal analysis). |
| `2_Filtering.ipynb` | Applies QC gates, outlier removal, and other filtering steps to the extracted feature tables. |
| `3_QualityControl.ipynb` | Generates QC plots and checks for plate or batch effects to identify technical artifacts. |
| `4_DM.ipynb` | Downstream analysis: dimensionality reduction (PCA, UMAP, t-SNE), clustering, and trajectory inference. |
| `5_Plotting.ipynb` | Advanced result visualization and figure generation utilities. |

---

## Function Modules (`Functions/`)

| Module | Description |
|---|---|
| `Fcts_FE.py` | Feature extraction primitives and experiment setup parsing. |
| `Fcts_Base.py` | I/O helpers, Excel layout parsing, and saving utilities. |
| `Fcts_DimRed.py` | Dimensionality reduction and clustering methods. |
| `Fcts_Filtering.py` | Filtering and QC functions. |
| `Fcts_Matching.py` | Matching and merging of datasets. |
| `Fcts_Plotting.py` | Plotting and visualization utilities. |
| `Fcts_QC.py` | Quality control and plate bias analysis. |

---

## Plate Layout Files (`Layout_96.xlsx`, `Layout_384.xlsx`)

Excel-based plate layout files define experimental metadata and well conditions. These files must be placed in the experiment folder and are required for correct feature extraction and annotation.

- **Purpose:** Map wells to experimental conditions (medium, antibody mix, cell line) and imaging channels/stains.
- **Structure:** Each file contains sheets such as `StainingLayout`, `MediumLayout`, `LineLayout`, and `OtherLayout`. The `StainingLayout` sheet maps stains to imaging channels and, optionally, to experimental rounds.
- **Supported formats:** Both 96-well and 384-well plate templates are supported.

**Usage:**
- **Single-round layouts:** 0 or no `Round` column → treated as round 0.
- **Multi-round layouts:** Include a numeric `Round` column → stains are stored per round.

---

## Data Requirements

### OME-Zarr Plates

Your experiment folder should contain one or more OME-Zarr plate folders (`*.zarr`). If you have multiple `*.zarr` in your experimental folder but only want to analyze a subset, such as maximum intensity projections, adjust the file_ending parameter.

**Assumptions:**
- ROI table and labels exist in the segmentation round (commonly `0`).
- Additional rounds may exist as intensity-only rounds.
- `Round id` is assumed to match `ez_zarr` `image_name` groups (e.g., round `1` lives under image name `"1"`).

### Excel Layout (`Layout*.xlsx`)

Place a `Layout*.xlsx` in the same experiment folder. The pipeline reads stain/channel metadata from a `StainingLayout` sheet (antibody mix → stains → imaging channels).

---

## Getting Started

### 1. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate atmc_analysis
```

If a dependency is missing, install it via pip inside the environment:

```bash
pip install <package>
```

### 2. Run the notebooks

From the repo root:

```bash
jupyter notebook
```

Open `1_FeatureExtraction.ipynb` and configure the following parameters:

```python
source              # experiment folder containing .zarr plate(s) + Layout*.xlsx
table_name          # name of the ROI table
label_name          # name of the label image
segmentation_round  # round used for segmentation (typically 0)
rounds_to_extract   # list of rounds to extract intensity features from
```

Then execute the notebook top-to-bottom.

---

## Outputs

Typical outputs produced by the pipeline:

- **Tables (`.csv`)** saved under an `analysis` directory (e.g., `.../2_Tables/`).
- **AnnData (`.h5ad`)** containing:
  - Numerical features in `X`
  - Metadata in `obs`
  - Configuration in `uns`

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Saving `.h5ad` fails with `"dict key is int"` | Convert nested dict keys in `adata.uns` to strings before writing when using round-indexed dicts. |
| Some wells are skipped | Only wells included in the Excel-derived experiment setup are processed. Missing annotations in the layout file mean the well is excluded from the run. |

---

## License

The ATMC_Analysis workflow was created for the **Applied Tissue Models Center (ATMC)** at the **Friedrich Miescher Institute for Biomedical Research (FMI)** and is released under the [MIT License](LICENSE). All copyright belongs to the FMI.