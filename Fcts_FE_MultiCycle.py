import os
import datetime
import itertools
import numpy as np
import pandas as pd
import anndata

from skimage import filters
from ez_zarr import ome_zarr

from Fcts_Base import (
    find_staining_in_ABs,
    find_barcodes_with_day,
    save_adata,
)

# Re-use the already implemented feature functions
from Fcts_FE import (
    shape_calc_mask,
    convex_hull_features,
    get_border_fraction,
    skeleton_feats,
    channel_mask_feat_calc,
    intensity_feat_calc,
    moments_channel_mask,
)


"""***
FEATURE EXTRACTION (MULTICYCLE)
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


def _as_path_str(p) -> str:
    if isinstance(p, (list, tuple)) and len(p) > 0:
        p = p[0]
    return str(p)


def _parse_well_round_from_path(path: str) -> tuple[str, int] | tuple[None, None]:
    # Expect .../<row>/<col>/<round>
    parts = _as_path_str(path).replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 3:
        return None, None
    row = parts[-3]
    col = parts[-2]
    rnd = parts[-1]
    try:
        rnd_int = int(rnd)
    except Exception:
        return None, None
    return f"{row}{col}", rnd_int


def build_well_round_map(plate) -> dict[tuple[str, int], object]:
    """Map (well, round) -> ez_zarr Image object (for the images contained in this plate object)."""
    m = {}
    for img in plate.images:
        well, rnd = _parse_well_round_from_path(img.get_path())
        if well is None:
            continue
        m[(well, rnd)] = img
    return m


# --- Round-aware plate loading (ez_zarr image_name) ---
_PLATE_CACHE: dict[tuple[str, int], object] = {}


def _infer_plate_root_from_image_path(img_path: str) -> str:
    """Infer the plate root (.zarr) path from an image path ending with /<row>/<col>/<image_name>."""
    parts = _as_path_str(img_path).replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 4:
        return _as_path_str(img_path)
    # remove last 3: row/col/image_name
    return "/".join(parts[:-3])


def _default_round_from_plate(plate) -> int:
    """Attempt to infer which image_name was used to import this plate (usually 0)."""
    try:
        if hasattr(plate, "images") and len(plate.images) > 0:
            _, rnd = _parse_well_round_from_path(plate.images[0].get_path())
            if rnd is not None:
                return int(rnd)
    except Exception:
        pass
    return 0


def _plate_root_from_plate(plate) -> str:
    """Infer plate root path from the first available image path."""
    if hasattr(plate, "images") and len(plate.images) > 0:
        return _infer_plate_root_from_image_path(plate.images[0].get_path())
    # Fallback: try plate.paths if available
    if hasattr(plate, "paths") and plate.paths:
        return _infer_plate_root_from_image_path(plate.paths[0])
    raise ValueError("Cannot infer plate root path from plate object")


def get_plate_for_round(plate, round_id: int):
    """Return a plate object for a given round (image_name), caching imports."""
    round_id = int(round_id)
    default_r = _default_round_from_plate(plate)
    if round_id == default_r:
        return plate

    root = _plate_root_from_plate(plate)
    key = (root, round_id)
    if key not in _PLATE_CACHE:
        _PLATE_CACHE[key] = ome_zarr.import_plate(root, image_name=str(round_id))
    return _PLATE_CACHE[key]


def build_well_round_map_multi(plate, rounds: set[int]) -> dict[tuple[str, int], object]:
    """Build a (well, round)->image map across multiple rounds by importing each round plate."""
    m_all: dict[tuple[str, int], object] = {}
    for r in sorted({int(x) for x in rounds}):
        pr = get_plate_for_round(plate, r)
        m_all.update(build_well_round_map(pr))
    return m_all


def get_stains_for_round(stainings: dict, ab_key: str, round_id: int) -> list[str]:
    v = stainings.get(ab_key, [])
    if isinstance(v, dict):
        return v.get(int(round_id), [])
    return v


def normalize_thresholds(thresholds):
    """Normalize thresholds into dict[int, dict[str, float]]."""
    if thresholds is None:
        return {}

    # thresholds provided as dict[stain] -> [ch, ..., value] (legacy)
    try:
        any_v = next(iter(thresholds.values()))
    except StopIteration:
        return {}

    if isinstance(any_v, (list, tuple)) and len(any_v) >= 3:
        return {0: {k: float(v[2]) for k, v in thresholds.items()}}

    # thresholds provided as dict[round] -> dict[stain] -> value
    if isinstance(any_v, dict):
        out = {}
        for r, d in thresholds.items():
            out[int(r)] = {k: float(v) for k, v in d.items()}
        return out

    # thresholds provided as dict[stain] -> value
    if isinstance(any_v, (int, float, np.number)):
        return {0: {k: float(v) for k, v in thresholds.items()}}

    raise ValueError("Unrecognized thresholds format")


def morphology_features(mask, row_data, OID, spacing, sigma_skeleton, radius_multiplier):
    row_data = shape_calc_mask(mask, row_data, OID, spacing)
    row_data = convex_hull_features(mask, row_data, OID, spacing, min_area_fraction=0.005)
    row_data = get_border_fraction(mask, row_data, OID)
    row_data = skeleton_feats({"Mask": mask}, row_data, OID, spacing, sigma_skeleton, radius_multiplier)
    return row_data


def intensity_features_for_round(
    img_stack,
    mask,
    stain_names,
    thresholds_round: dict[str, float],
    quantiles_to_calc,
    round_id: int,
    spacing: float,
    OID: str,
    sigma: float = 3,
):
    """Compute intensity features for a single round.

    Returns
    -------
    row_data_round : dict
        Feature names are prefixed via staining name: R{round}__{stain}
    vectors : dict[str, np.ndarray]
        Masked intensity vectors per (round, stain) for Pearson correlations.
    """

    row_data_round = {}
    vectors = {}

    mask_bool = mask.astype(bool)

    for ch_idx, stain in enumerate(stain_names):
        if ch_idx >= img_stack.shape[0]:
            # channel missing in this round
            continue

        staining_key = f"R{int(round_id)}__{stain}"

        img = img_stack[ch_idx, 0]

        # Smooth and mask
        img_proc = filters.gaussian(img, sigma=sigma, preserve_range=True)
        img_proc[~mask_bool] = 0

        # Threshold mask for this stain/round
        thr = thresholds_round.get(stain, 0)
        mask_channel = (img_proc > thr) & mask_bool

        # Provide images in the same structure expected by existing funcs
        images = {
            "Mask": mask,
            "C01": img,
            "C01_Mask": mask_channel.astype(bool),
        }

        # Compute features (use real pixel spacing)
        row_data_round = channel_mask_feat_calc(mask, images["C01_Mask"], staining_key, row_data_round, OID=OID, spacing=spacing)
        row_data_round = intensity_feat_calc(img, mask, images["C01_Mask"], row_data_round, staining_key, OID=OID, quantiles_to_calc=quantiles_to_calc)
        row_data_round = moments_channel_mask(mask, img, row_data_round, OID=OID, staining=staining_key, spacing=spacing)

        # Pearson vectors (raw intensities within mask)
        vectors[staining_key] = img[mask_bool]

    return row_data_round, vectors


def pearson_features_from_vectors(vectors: dict[str, np.ndarray]) -> dict:
    out = {}
    keys = sorted(vectors.keys())

    for k1, k2 in itertools.combinations(keys, 2):
        v1 = vectors[k1]
        v2 = vectors[k2]

        if v1.size == 0 or v2.size == 0 or np.std(v1) == 0 or np.std(v2) == 0:
            r = np.nan
        else:
            r = float(np.corrcoef(v1, v2)[0, 1])

        out[f"{k1}--{k2}_PearsonR"] = r

    return out


def estimate_staining_thresholds_multicycle(
    ome_zarr_df,
    ome_zarr_dict,
    stainings,
    experiment_setup,
    table_name,
    label_name,
    pyramid_level=0,
    control_condition=None,
    n=20,
    seed=0,
    sigma=3,
    q=0.5,
    round_id: int = 0,
    segmentation_round: int = 0,
):
    """Estimate thresholds for a specific round using segmentation_round ROI table coordinates."""

    import random
    random.seed(seed)

    round_id = int(round_id)
    segmentation_round = int(segmentation_round)
    rounds_needed = {round_id, segmentation_round}

    # Build stain list for this round per AB mix and map to channel index
    thresholds = {}
    for ab_mix in stainings.keys():
        stains_r = get_stains_for_round(stainings, ab_mix, round_id)
        for i, stain in enumerate(stains_r):
            thresholds[stain] = [i + 1, [], 0]

    # Extract timepoints from experiment setup (Other)
    other = []
    for plate in experiment_setup.keys():
        for well in experiment_setup[plate].keys():
            if experiment_setup[plate][well][3] not in other:
                other.append(experiment_setup[plate][well][3])

    dict_org = {ab: {cond: [] for cond in other} for ab in thresholds.keys()}
    dict_org_lst = {ab: {} for ab in thresholds.keys()}
    timepoints_lst = []

    for stain in thresholds:
        ABs_lst = find_staining_in_ABs(stainings, stain)

        for day in other:
            barcodes_day = find_barcodes_with_day(experiment_setup, day)

            if control_condition is not None:
                filtered_df = ome_zarr_df[
                    (ome_zarr_df['Barcode'].isin(barcodes_day))
                    & (ome_zarr_df['AB'].isin(ABs_lst))
                    & (ome_zarr_df["Medium"] == control_condition)
                ]
            else:
                filtered_df = ome_zarr_df[
                    (ome_zarr_df['Barcode'].isin(barcodes_day))
                    & (ome_zarr_df['AB'].isin(ABs_lst))
                ]

            dict_org[stain][day] = list(filtered_df.UID)
            timepoints_lst = timepoints_lst + [*filtered_df.Day]

    # Unique timepoints
    from natsort import natsorted
    timepoints_lst = natsorted(list(set(timepoints_lst)))

    for stain in dict_org.keys():
        lst = []
        for cond in dict_org[stain].keys():
            if n > len(dict_org[stain][cond]):
                lst = dict_org[stain][cond] + lst
            else:
                lst = random.sample(dict_org[stain][cond], n) + lst
        dict_org_lst[stain] = lst

    # Cache well-round maps per barcode (avoids repeated imports)
    maps_by_barcode: dict[str, dict[tuple[str, int], object]] = {}

    # Compute thresholds
    for stain, uids in dict_org_lst.items():
        for UID in uids:
            # Parse UID
            idx = int(UID.split("-")[-1])
            bc = "-".join(UID.split("-")[:-2])
            well = UID.split("-")[-2]

            plate = ome_zarr_dict[bc]
            if bc not in maps_by_barcode:
                maps_by_barcode[bc] = build_well_round_map_multi(plate, rounds=rounds_needed)
            m = maps_by_barcode[bc]

            img_seg = m.get((well, segmentation_round))
            img_r = m.get((well, round_id))
            if img_seg is None or img_r is None:
                thresholds[stain][1].append(np.nan)
                continue

            table = img_seg.get_table(table_name)
            if table is None or table.empty or idx >= len(table):
                thresholds[stain][1].append(np.nan)
                continue

            entry = table.iloc[idx]
            ul_y, ul_x = entry["y_micrometer"], entry["x_micrometer"]
            lr_y = ul_y + entry["len_y_micrometer"]
            lr_x = ul_x + entry["len_x_micrometer"]

            ch = thresholds[stain][0] - 1
            try:
                img = img_r.get_array_by_coordinate(
                    pyramid_level=pyramid_level,
                    upper_left_yx=(ul_y, ul_x),
                    lower_right_yx=(lr_y, lr_x),
                )
                img = img[ch, 0]
            except Exception:
                thresholds[stain][1].append(np.nan)
                continue

            img = filters.gaussian(img, sigma=sigma, preserve_range=True)

            if np.max(img) != 0:
                thresholds[stain][1].append(filters.threshold_triangle(img))
            else:
                thresholds[stain][1].append(np.nan)

        if len(thresholds[stain][1]) > 0:
            thresholds[stain][2] = float(np.nanquantile(thresholds[stain][1], q=q))
        else:
            thresholds[stain][2] = np.nan

    return thresholds, dict_org, timepoints_lst


def plot_thresholds_multicycle(
    thresholds,
    dict_org,
    timepoints_lst,
    ome_zarr_dict,
    table_name,
    label_name,
    seed=0,
    pyramid_level=0,
    round_id: int = 0,
    segmentation_round: int = 0,
):
    """A minimal round-aware replacement for plot_thresholds (supports Round>0 without labels)."""

    import random
    import matplotlib.pyplot as plt
    import seaborn as sns

    random.seed(seed)

    round_id = int(round_id)
    segmentation_round = int(segmentation_round)
    rounds_needed = {round_id, segmentation_round}

    rows = len(dict_org[list(dict_org.items())[0][0]].keys()) + 1
    cols = len(dict_org.keys())

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 3 * rows))

    for i, stain in enumerate(thresholds.keys()):
        if len(thresholds[stain][1]) > 1:
            sns.kdeplot(
                ax=ax[0, i],
                x=thresholds[stain][1],
                color="green",
                cut=0,
                fill=True,
                linewidth=1,
            )
            ax[0, i].axes.get_yaxis().set_visible(False)
            ax[0, i].set_title(stain, fontsize=12)
            ax[0, i].axvline(thresholds[stain][2], color="red")
            ax[0, i].text(x=0.9, y=0.9, s="T: " + str(int(thresholds[stain][2])), transform=ax[0, i].transAxes, ha="right")
        else:
            ax[0, i].imshow(np.zeros((200, 200)), aspect="auto", cmap="binary")
            ax[0, i].set_title(stain, fontsize=12)
            ax[0, i].set_axis_off()
            ax[0, i].text(x=0.5, y=0.5, s="No threshold found", transform=ax[0, i].transAxes, ha="center")

    # Cache per-barcode maps
    maps_by_barcode: dict[str, dict[tuple[str, int], object]] = {}

    # Show one random ROI per timepoint
    for col, stain in enumerate(thresholds.keys()):
        for row in range(1, len(timepoints_lst) + 1):
            uids = dict_org[stain][timepoints_lst[row - 1]]
            if len(uids) == 0:
                ax[row, col].imshow(np.zeros((200, 200)), aspect="auto", cmap="binary")
                ax[row, col].set_axis_off()
                ax[row, col].text(x=0.5, y=0.5, s="No image found", transform=ax[row, col].transAxes, ha="center")
                continue

            UID = random.sample(uids, 1)[0]
            idx = int(UID.split("-")[-1])
            bc = "-".join(UID.split("-")[:-2])
            well = UID.split("-")[-2]

            plate = ome_zarr_dict[bc]
            if bc not in maps_by_barcode:
                maps_by_barcode[bc] = build_well_round_map_multi(plate, rounds=rounds_needed)
            m = maps_by_barcode[bc]

            img_seg = m.get((well, segmentation_round))
            img_r = m.get((well, round_id))

            if img_seg is None or img_r is None:
                ax[row, col].imshow(np.zeros((200, 200)), aspect="auto", cmap="binary")
                ax[row, col].set_axis_off()
                continue

            table = img_seg.get_table(table_name)
            entry = table.iloc[idx]
            ul_y, ul_x = entry["y_micrometer"], entry["x_micrometer"]
            lr_y = ul_y + entry["len_y_micrometer"]
            lr_x = ul_x + entry["len_x_micrometer"]

            ch = thresholds[stain][0] - 1
            try:
                img = img_r.get_array_by_coordinate(
                    pyramid_level=pyramid_level,
                    upper_left_yx=(ul_y, ul_x),
                    lower_right_yx=(lr_y, lr_x),
                )
                img = img[ch, 0]
            except Exception:
                img = np.zeros((200, 200))

            vmin = min(thresholds[stain][2], np.max(img)) if np.max(img) > 0 else 0
            ax[row, col].imshow(img, vmin=vmin, aspect="auto", cmap="inferno")
            ax[row, col].set_axis_off()

    plt.tight_layout()
    return fig


def extract_features_multicycle(
    ome_zarrs_dict,
    table_name,
    label_name,
    pyramid_level,
    source,
    folder,
    analysis_dir,
    barcodes,
    experiment_setup,
    thresholds,
    stainings,
    result_file_name,
    radius_multiplier,
    sigma_skeleton,
    quantiles_to_calc,
    experiment_ID,
    rounds_to_extract: list[int] | None = None,
    segmentation_round: int = 0,
    sigma_intensity: float = 3,
    compute_cross_round_pearson: bool = True,
):
    """Extract features with multi-round intensity support."""

    thresholds_by_round = normalize_thresholds(thresholds)

    if rounds_to_extract is None:
        rounds_to_extract = [0]
    rounds_to_extract = [int(r) for r in rounds_to_extract]
    segmentation_round = int(segmentation_round)

    rounds_needed = set(rounds_to_extract + [segmentation_round])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")

    for subfolder in ["1_Scripts", "2_Tables", "3_Plots", "4_Results"]:
        os.makedirs(os.path.join(analysis_dir, subfolder), exist_ok=True)

    df_total_list = []

    for i, bc in enumerate(ome_zarrs_dict.keys()):
        plate = ome_zarrs_dict[bc]
        m = build_well_round_map_multi(plate, rounds=rounds_needed)

        wells_in_exp = experiment_setup[bc]
        rows = []

        for well in wells_in_exp.keys():
            img_seg = m.get((well, segmentation_round))
            if img_seg is None:
                continue

            exp_info = wells_in_exp[well]
            ab_key = exp_info[1]

            table = img_seg.get_table(table_name)
            if table is None or table.empty:
                continue

            pixel_spacing = img_seg.get_scale(pyramid_level=pyramid_level)[-1]

            for row_idx, row in enumerate(table.itertuples()):
                OID = f"{bc}-{well}-{row_idx}"

                ul_y, ul_x = row.y_micrometer, row.x_micrometer
                lr_y = row.y_micrometer + row.len_y_micrometer
                lr_x = row.x_micrometer + row.len_x_micrometer

                # Load segmentation round pair for mask
                try:
                    _, mask0 = img_seg.get_array_pair_by_coordinate(
                        label_name=label_name,
                        pyramid_level=pyramid_level,
                        upper_left_yx=(ul_y, ul_x),
                        lower_right_yx=(lr_y, lr_x),
                    )
                except Exception:
                    continue

                mask_arr = mask0[label_name][0]
                mask_for_oid = (mask_arr == int(row_idx) + 1).astype(mask_arr.dtype)

                # Skip multi-label ROIs
                from skimage import measure
                if np.max(measure.label(mask_for_oid.astype(bool))) != 1:
                    continue

                row_data = {
                    "Organoid_ID": OID,
                    "Barcode": bc,
                    "Well": well,
                    "Object": str(row_idx),
                    "Medium": exp_info[0],
                    "ABs": exp_info[1],
                    "Cell_line": exp_info[2],
                    "Other": exp_info[3],
                    "PATH": img_seg.get_path(),
                    "y_micrometer": row.y_micrometer,
                    "x_micrometer": row.x_micrometer,
                    "len_y_micrometer": row.len_y_micrometer,
                    "len_x_micrometer": row.len_x_micrometer,
                    "centroid_y_micrometer": row.y_micrometer + row.len_y_micrometer / 2.0,
                    "centroid_x_micrometer": row.x_micrometer + row.len_x_micrometer / 2.0,
                    "Experiment_ID": experiment_ID,
                }

                # Add staining columns for each round
                for r in sorted(rounds_needed):
                    stains_r = get_stains_for_round(stainings, ab_key, r)
                    for ch_i, stain in enumerate(stains_r):
                        if r == 0:
                            row_data[f"Staining_Ch{ch_i+1}"] = stain
                        row_data[f"Staining_R{r}_Ch{ch_i+1}"] = stain

                # Morphology once (segmentation round)
                row_data = morphology_features(
                    mask_for_oid,
                    row_data,
                    OID,
                    spacing=pixel_spacing,
                    sigma_skeleton=sigma_skeleton,
                    radius_multiplier=radius_multiplier,
                )

                # Intensity per round
                pearson_vectors = {}
                for r in rounds_to_extract:
                    img_r_obj = m.get((well, int(r)))
                    if img_r_obj is None:
                        continue

                    stains_r = get_stains_for_round(stainings, ab_key, r)
                    if len(stains_r) == 0:
                        continue

                    try:
                        img_r = img_r_obj.get_array_by_coordinate(
                            pyramid_level=pyramid_level,
                            upper_left_yx=(ul_y, ul_x),
                            lower_right_yx=(lr_y, lr_x),
                        )
                    except Exception:
                        continue

                    thr_r = thresholds_by_round.get(int(r), {})

                    feats_r, vecs_r = intensity_features_for_round(
                        img_stack=img_r,
                        mask=mask_for_oid,
                        stain_names=stains_r,
                        thresholds_round=thr_r,
                        quantiles_to_calc=quantiles_to_calc,
                        round_id=int(r),
                        spacing=pixel_spacing,
                        OID=OID,
                        sigma=sigma_intensity,
                    )

                    row_data.update(feats_r)
                    pearson_vectors.update(vecs_r)

                # Pearson across all rounds/channels
                if compute_cross_round_pearson and len(pearson_vectors) >= 2:
                    row_data.update(pearson_features_from_vectors(pearson_vectors))

                rows.append(row_data)

        if not rows:
            continue

        df = pd.DataFrame(rows)
        df = df.drop(columns=[col for col in df.columns if "moments-0-0" in col or "moments_weighted-0-0" in col], errors="ignore")

        # Save per-plate
        save_path = os.path.join(source, folder[i], f"{result_file_name}_{barcodes[i]}_{timestamp}.csv")
        df.to_csv(save_path)
        print(f"Saved {bc} results as {save_path}.")

        df_total_list.append(df)

    if not df_total_list:
        raise ValueError("No features extracted for any barcode; check your input data.")

    df_total = pd.concat(df_total_list, axis=0)
    df_total = df_total.set_index("Organoid_ID", drop=False)

    total_save_path = os.path.join(analysis_dir, "2_Tables", f"{result_file_name}_{timestamp}.csv")
    df_total.to_csv(total_save_path)
    print(f"Saved merged csv as {total_save_path}.")

    feats_categorical = [
        "Organoid_ID", "Barcode", "Well", "Object", "Medium",
        "ABs", "Cell_line", "Other", "PATH", "Experiment_ID",
    ]
    feats_categorical += [col for col in df_total.columns if col.startswith("Staining_")]
    feats_numerical = [col for col in df_total.columns if col not in feats_categorical]

    ad = anndata.AnnData(
        X=df_total[feats_numerical].values,
        obs=df_total.loc[:, feats_categorical],
        var=pd.DataFrame(index=feats_numerical),
    )

    ad.uns["stainings"] = stainings
    ad.uns["experiment_setup"] = experiment_setup
    ad.uns["pixel_spacing"] = pixel_spacing
    ad.uns["pyramid_level"] = pyramid_level
    ad.uns["folders"] = folder
    ad.uns["source_dir"] = source
    ad.uns["table_dir"] = os.path.join(analysis_dir, "2_Tables")
    ad.uns["plot_dir"] = os.path.join(analysis_dir, "3_Plots")
    ad.uns["table_name"] = table_name
    ad.uns["label_name"] = label_name
    ad.uns["thresholds_by_round"] = thresholds_by_round
    ad.uns["rounds_to_extract"] = rounds_to_extract
    ad.uns["segmentation_round"] = int(segmentation_round)

    save_adata(ad, f"{result_file_name}_{timestamp}")
    return ad
