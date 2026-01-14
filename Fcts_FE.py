import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from skimage import filters, measure, morphology
from natsort import natsorted, natsort_keygen
from skan import Skeleton, summarize
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.ndimage import label
from tqdm.notebook import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import itertools
import datetime
import anndata
import random
import glob
import copy
import math
import cv2
import os

from Fcts_Base import find_staining_in_ABs, load_img_mask_by_UID, find_barcodes_with_day, save_adata

# Disable pandas performance warnings
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


"""
***
FEATURE EXTRACTION (MULTICYCLE COMPATIBLE)
***

Implements multi-round feature extraction for datasets where the OME-Zarr structure is:
  <plate>.zarr/<row>/<col>/<round>/...

Important note for ez_zarr:
- ez_zarr.import_plate(..., image_name="0") selects the image group within each well (often "0").
- For multiplexing cycles stored as "0", "1", ... image groups, each round must be imported
  separately with image_name=str(round) to access that data.

Assumptions:
- ROI table exists in segmentation_round (default 0)
- labels exist in segmentation_round (default 0)
- additional rounds can be missing labels and tables
- segmentation is identical across rounds

Strategy:
- morphology features: computed once from segmentation_round labels
- intensity features: computed per round and prefixed with R{round}__
- Pearson correlations: computed across all (round, stain) vectors (within + between rounds)
"""


def make_experiment(source, layout_sheets = ["MediumLayout", "StainingLayout", "LineLayout", "OtherLayout"]):
    """Parses an experiment's plate layout Excel file and returns a nested dictionary of well metadata."""

    filename = glob.glob(source+'/Layout*.xlsx')[0]

    def extract_sheet_plates(df):
        plates = {}
        nrows, ncols = df.shape
        i = 0
        while i < nrows:
            barcode = None
            for j in range(i, nrows):
                row = df.iloc[j, :]
                for k, v in enumerate(row):
                    if str(v).strip() == "Barcode:" and (k+1)<ncols:
                        barcode = str(row[k+1]).strip()
                        i = j+1
                        break
                if barcode: break
            else:
                break

            grid_start_col, head_row, n_cols_detected = None, None, None
            for j in range(i, min(i+10, nrows)):
                row = df.iloc[j, :]
                max_search = ncols-6
                for window in range(max_search):
                    nums = []
                    for q in range(0,24):
                        cell = row[window+q] if window+q < ncols else None
                        try:
                            n = int(float(str(cell)))
                            nums.append(n)
                        except:
                            break
                    if nums == list(range(1,25)):
                        grid_start_col, head_row, n_cols_detected = window, j, 24
                        break
                    elif nums == list(range(1,13)):
                        grid_start_col, head_row, n_cols_detected = window, j, 12
                        break
                if head_row is not None:
                    break
            if head_row is None:
                i += 1
                continue

            if n_cols_detected == 24:
                expected_rows = 16
                allowed_labels = list("ABCDEFGHIJKLMNOP")
            elif n_cols_detected == 12:
                expected_rows = 8
                allowed_labels = list("ABCDEFGH")
            else:
                raise ValueError(f"Unrecognized plate format after barcode {barcode}")

            wells = []
            well_labels = []
            for j in range(head_row+1, head_row+1+expected_rows):
                if j >= nrows: break
                row = df.iloc[j, :]
                rlab = None
                for offset in [grid_start_col-1, grid_start_col-2, 2, 1]:
                    if 0 <= offset < ncols:
                        cell = row[offset]
                        if isinstance(cell, str) and len(cell.strip())==1 and cell.strip().isalpha():
                            rlab = cell.strip()
                            break
                if rlab and rlab in allowed_labels:
                    arr = list(row[grid_start_col:grid_start_col+n_cols_detected])
                    if len(arr) < n_cols_detected:
                        arr = arr + [np.nan]*(n_cols_detected-len(arr))
                    wells.append(arr)
                    well_labels.append(rlab)
            if wells:
                plates[barcode] = pd.DataFrame(
                    wells,
                    index=well_labels,
                    columns=[f"{i:02d}" for i in range(1, n_cols_detected+1)]
                )
            i = head_row + expected_rows + 2
        return plates

    all_plates = {}
    for sheetname in layout_sheets:
        df = pd.read_excel(filename, sheet_name=sheetname, header=None)
        all_plates[sheetname] = extract_sheet_plates(df)

    barcodes = set(all_plates[layout_sheets[0]].keys())
    for others in layout_sheets[1:]:
        barcodes &= set(all_plates[others].keys())
    barcodes = sorted(list(barcodes))

    result = {}
    for barcode in barcodes:
        dfs = [all_plates[s][barcode] for s in layout_sheets]
        result[barcode] = {}
        for row in dfs[0].index:
            for col in dfs[0].columns:
                well = f"{row}{col}"
                values = [df.at[row, col] if (row in df.index and col in df.columns) else np.nan for df in dfs]
                arr = np.array(values, dtype=object)
                if np.all(pd.isna(arr)):
                    continue
                if np.any(pd.isna(arr)) and not np.all(pd.isna(arr)):
                    raise ValueError(
                        f"Mixture of nan/values in {barcode} {well}: {values}. "
                        "Make sure all used wells have information across sheets."
                    )
                result[barcode][well] = values

    display_experiment_setup(result)
    barcodes = list(result.keys())

    print(f"Found {len(barcodes)} barcodes in experiment setup:")
    for bc in barcodes:
        print(bc)
    return result, barcodes


def display_experiment_setup(experiment):
    """Display the experiment setup for each barcode."""
    for barcode in experiment.keys():
        current_plate = experiment[barcode]
        rows = list(set([x[0] for x in current_plate.keys()]))
        cols = list(set([x[1:] for x in current_plate.keys()]))
        rows.sort(), cols.sort()
        df_hm = pd.DataFrame(index = rows, columns = cols)
        for well in current_plate.keys():
            df_hm.loc[well[0], well[1:]] = current_plate[well]
        print("\n\nPlate-Barcode: %s"%barcode)
        display(df_hm)
    return None


# ... existing functions unchanged ...


def _stringify_dict_keys(obj):
    """Convert nested dict keys to str so AnnData can be written to .h5ad safely."""
    if isinstance(obj, dict):
        return {str(k): _stringify_dict_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_dict_keys(x) for x in obj]
    return obj


def _prefix_vars_with_round(ad_t, round_id: int):
    """Prefix AnnData var_names with R{round}__ to match 1_FeatureExtraction naming."""
    ad_cp = ad_t.copy()
    ad_cp.var_names = [f"R{int(round_id)}__{v}" for v in ad_cp.var_names]
    return ad_cp


def merge_feature_tables_from_zarr(
    source: str,
    folder: list[str],
    experiment_setup: dict,
    stainings: dict | None,
    experiment_ID: str,
    result_file_name: str = "1_FeatureLoading",
    analysis_dir: str | None = None,
    feature_table_names: str | list[str] = "features",
    roi_table_name: str = "nuclei_ROI_table",
    label_name: str = "nuclei",
    multiplexing_round: int = 0,
    file_ending: str = ".zarr",
):
    """Load and merge precomputed AnnData feature tables stored in OME-Zarr under <round>/tables/<name>."""

    import os
    import anndata as ad
    import pandas as pd
    from anndata.io import read_zarr
    from ez_zarr import ome_zarr

    if analysis_dir is None:
        analysis_dir = source

    multiplexing_round = int(multiplexing_round)

    if isinstance(feature_table_names, str):
        feature_table_names = [feature_table_names]
    if len(feature_table_names) == 0:
        raise ValueError("feature_table_names is empty. Provide at least one table name.")

    def _load_plate_for_round(plate_path: str, round_id: int):
        return ome_zarr.import_plate(plate_path, image_name=str(int(round_id)))

    def _read_table_anndata(table_zarr_path: str):
        return read_zarr(table_zarr_path)

    tables_all = []

    for plate_folder in folder:
        plate_path = os.path.join(source, plate_folder)
        plate = _load_plate_for_round(plate_path, multiplexing_round)

        wells = plate.get_names()
        well_paths = plate.paths

        barcode_guess = None
        for bc in experiment_setup.keys():
            if bc in plate_folder:
                barcode_guess = bc
                break
        if barcode_guess is None:
            barcode_guess = list(experiment_setup.keys())[folder.index(plate_folder)]

        meta = experiment_setup[barcode_guess]

        for well, path_in_plate in zip(wells, well_paths):
            ad_list = []
            for tname in feature_table_names:
                table_path = os.path.join(plate_path, path_in_plate, "tables", tname)
                if not os.path.exists(table_path):
                    raise FileNotFoundError(f"Cannot find table: {table_path}")
                ad_list.append(_read_table_anndata(table_path))

            obs0 = ad_list[0].obs_names
            for j, ad_t in enumerate(ad_list[1:], start=1):
                if ad_t.n_obs != ad_list[0].n_obs:
                    raise ValueError(
                        f"n_obs mismatch in well {well} between table {feature_table_names[0]} and {feature_table_names[j]}"
                    )
                if not obs0.equals(ad_t.obs_names):
                    raise ValueError(
                        f"obs_names order mismatch in well {well} between table {feature_table_names[0]} and {feature_table_names[j]}"
                    )

            ad_list_pref = []
            for tname, ad_t in zip(feature_table_names, ad_list):
                ad_cp = ad_t.copy()
                ad_cp.var_names = [f"{tname}__{v}" for v in ad_cp.var_names]
                ad_list_pref.append(ad_cp)

            ad_well = ad.concat(ad_list_pref, axis=1, merge="same", join="outer")
            ad_well = _prefix_vars_with_round(ad_well, multiplexing_round)

            ad_well.obs = ad_well.obs.copy()
            ad_well.obs["Barcode"] = barcode_guess
            ad_well.obs["Well"] = well
            ad_well.obs["PATH"] = path_in_plate
            ad_well.obs["Multiplexing_Round"] = multiplexing_round

            idx_in_well = pd.Series(range(ad_well.n_obs), index=ad_well.obs_names)
            ad_well.obs["Organoid_ID"] = barcode_guess + "-" + well + "-" + idx_in_well.astype(str).values

            exp_info = meta.get(well, [None, None, None, None])
            ad_well.obs["Medium"] = exp_info[0]
            ad_well.obs["ABs"] = exp_info[1]
            ad_well.obs["Cell_line"] = exp_info[2]
            ad_well.obs["Other"] = exp_info[3] if len(exp_info) > 3 else None
            ad_well.obs["Experiment_ID"] = experiment_ID

            tables_all.append(ad_well)

    if len(tables_all) == 0:
        raise RuntimeError("No feature tables loaded. Check table names and OME-Zarr structure.")

    ad_all = ad.concat(tables_all, axis=0, merge="same", join="outer", index_unique=None)
    ad_all.obs = ad_all.obs.copy()
    ad_all.obs_names = ad_all.obs["Organoid_ID"].astype(str)

    ad_all.uns["stainings"] = _stringify_dict_keys(stainings) if stainings is not None else {}
    ad_all.uns["experiment_setup"] = _stringify_dict_keys(experiment_setup)
    ad_all.uns["folders"] = folder
    ad_all.uns["source_dir"] = source
    ad_all.uns["table_name"] = roi_table_name
    ad_all.uns["label_name"] = label_name
    ad_all.uns["feature_table_names"] = feature_table_names
    ad_all.uns["multiplexing_round"] = multiplexing_round
    ad_all.uns["experiment_ID"] = experiment_ID
    ad_all.uns["table_dir"] = os.path.join(analysis_dir, "2_Tables")

    save_adata(ad_all, f"{result_file_name}_R{multiplexing_round}")

    return ad_all


def merge_feature_tables_from_zarr_rounds(
    source: str,
    folder: list[str],
    experiment_setup: dict,
    stainings: dict | None,
    experiment_ID: str,
    multiplexing_rounds: int | list[int] = 0,
    result_file_name: str = "1_FeatureLoading",
    analysis_dir: str | None = None,
    feature_table_names: str | list[str] = "features",
    roi_table_name: str = "nuclei_ROI_table",
    label_name: str = "nuclei",
    file_ending: str = ".zarr",
    validate_obs_names: bool = True,
    save_merged: bool = True,
):
    """Load one merged AnnData per round, then concatenate rounds along vars.

    This is the one-call wrapper intended for 1_FeatureLoading.ipynb.

    Notes
    -----
    - Each round's features are already prefixed with R{round}__ by merge_feature_tables_from_zarr.
    - Each round is saved as <result_file_name>_R{round} (inside merge_feature_tables_from_zarr).
    - Optionally also saves the merged multi-round AnnData as <result_file_name>_R<r0>-<rN>.
    """

    import anndata as ad

    if isinstance(multiplexing_rounds, int):
        rounds = [multiplexing_rounds]
    else:
        rounds = list(multiplexing_rounds)

    if len(rounds) == 0:
        raise ValueError("multiplexing_rounds is empty. Provide at least one round id.")

    ad_list = []
    for r in rounds:
        ad_r = merge_feature_tables_from_zarr(
            source=source,
            folder=folder,
            experiment_setup=experiment_setup,
            stainings=stainings,
            experiment_ID=experiment_ID,
            result_file_name=result_file_name,
            analysis_dir=analysis_dir,
            feature_table_names=feature_table_names,
            roi_table_name=roi_table_name,
            label_name=label_name,
            multiplexing_round=int(r),
            file_ending=file_ending,
        )
        ad_list.append(ad_r)

    if len(ad_list) == 1:
        ad_all = ad_list[0]
    else:
        if validate_obs_names:
            obs0 = ad_list[0].obs_names
            for j, ad_r in enumerate(ad_list[1:], start=1):
                if not obs0.equals(ad_r.obs_names):
                    raise ValueError(f"obs_names mismatch between rounds {rounds[0]} and {rounds[j]}")
        ad_all = ad.concat(ad_list, axis=1, merge="same", join="outer")

    # Carry round list in .uns for downstream
    ad_all.uns["multiplexing_rounds"] = [int(r) for r in rounds]

    if save_merged:
        rtag = "-".join([str(int(r)) for r in rounds])
        save_adata(ad_all, f"{result_file_name}_R{rtag}")

    return ad_all
