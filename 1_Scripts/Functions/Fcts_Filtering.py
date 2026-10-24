from Functions.Fcts_Base import add_zarr_uns, remove_organoids
from Functions.Fcts_Plotting import load_img_mask_by_UID
from Functions.Fcts_QC import (_BLUE, _DEFAULT_CONDITIONS, _INK_SECONDARY, _SEQUENTIAL, _border_contacts,
                               _condition_labels, _feature_frame, _image_size, _map_wells, _normal_scores,
                               _pca_scores, _plot_plates)
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

"""
***
FILTERING FUNCTIONS
***
"""


def filter_organoids_by(ad, feature, values, channel, pyramid_level=1, add_boundary=False, keep_na=True):
    """
    Remove organoids outside a range of a feature (e.g. debris or failed segmentations) and
    visualize the removed organoids.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object; left unchanged.
    - feature (str): Feature in ad.var_names or column in ad.obs.
    - values (tuple): (lower_bound, upper_bound); None for no bound on that side.
    - channel (str): Channel for image visualization.
    - pyramid_level (int): Pyramid level of images for visualization.
    - add_boundary (bool): If True, draw the segmentation boundary on the images.
    - keep_na (bool): If True, objects whose value is NaN are kept. For a stain-specific feature,
      NaN means "this object's antibody mix does not include that stain", so removing them would
      remove every object from the other panels.

    Returns:
    - ad (anndata.AnnData): A copy without the removed organoids, recorded in
      ad.uns["removed_organoids"].
    """
    lower_bound, upper_bound = values
    vals = np.asarray(ad.obs_vector(feature), dtype=float)
    is_na = np.isnan(vals)
    none = np.zeros(len(vals), dtype=bool)
    too_low = vals < lower_bound if lower_bound is not None else none
    too_high = vals > upper_bound if upper_bound is not None else none
    remove = too_low | too_high | (is_na & (not keep_na))

    for side, bound, mask in [("lower", lower_bound, too_low), ("upper", upper_bound, too_high)]:
        if bound is not None:
            print(f"{mask.sum()} objects removed due to {side} boundary ({bound}) of {feature}.")
    if is_na.any():
        print(f"{is_na.sum()} objects have no value for {feature} and were "
              f"{'kept' if keep_na else 'removed'} (keep_na={keep_na}).")
    print(f"{(~remove).sum()} objects remain after filtering.")

    for mask, sign, bound in [(too_low, "<", lower_bound), (too_high, ">", upper_bound)]:
        _plot_organoid_images(ad, ad.obs_names[mask], [f"{feature}: {v:.2f}" for v in vals[mask]],
                              f"Objects with {feature} {sign} {bound}", channel, pyramid_level, add_boundary)

    reason = np.where(too_low, f"{feature} < {lower_bound}",
                      np.where(too_high, f"{feature} > {upper_bound}", f"no value for {feature}"))
    return remove_organoids(ad, remove, "filter_organoids_by", reason)


def calculate_outgrowth(ad, n_seeded):
    """
    Calculate outgrowth efficiency for each well.

    Organoids removed as cut off at the image border (remove_cut_organoids, recorded in
    ad.uns["removed_organoids"]) still count: they are organoids, only not completely imaged.
    Debris must not count, so run this after the debris filters.

    Parameters:
    - ad (anndata.AnnData): AnnData object containing organoid data.
    - n_seeded (int): Number of cells seeded per well (assumed constant).
    """
    experiment_setup = ad.uns["experiment_setup"]
    group_cols = ["Other", "Cell_line", "Medium", "Well", "Barcode"]
    df = ad.obs[group_cols].astype(str)
    removed = ad.uns.get("removed_organoids")
    if removed is not None:
        df = pd.concat([df, removed.loc[removed["step"] == "remove_cut_organoids", group_cols].astype(str)])

    # Count number of objects (e.g., organoids) per well/barcode/condition
    outgrowth = df.groupby(group_cols).size().reset_index(name="Organoid_No")

    # Ensure all wells from experiment_setup are represented
    missing_rows = []
    for bc in experiment_setup:
        for well in experiment_setup[bc]:
            is_present = (
                (outgrowth["Barcode"] == bc) & (outgrowth["Well"] == well)
            )
            if not is_present.any():
                setup = experiment_setup[bc][well]
                # df comes from ad.obs.astype(str), so stringify here too - otherwise these
                # rows never match the `df_out.Other == str(day)` comparisons downstream.
                new_row = {
                    "Other": str(setup[3]),
                    "Cell_line": str(setup[2]),
                    "Medium": str(setup[0]),
                    "Well": well,
                    "Barcode": bc,
                    "Organoid_No": 0
                }
                missing_rows.append(new_row)
    if missing_rows:
        outgrowth = pd.concat([outgrowth, pd.DataFrame(missing_rows)], ignore_index=True)

    # Assign the same seeded cell count for all wells
    outgrowth["Cells_Seeded_Day0"] = n_seeded

    # Calculate %Outgrowth
    outgrowth["%Outgrowth"] = 100 * outgrowth["Organoid_No"] / outgrowth["Cells_Seeded_Day0"]

    ad.uns["Outgrowth_DF"] = outgrowth
    return ad


def remove_cut_organoids(ad, box_cols=("x_micrometer", "y_micrometer", "len_x_micrometer", "len_y_micrometer"),
                         plate_col="Barcode", border_tolerance=1.0, channel="R0__DAPI", pyramid_level=1,
                         add_boundary=False, plot=True):
    """
    Remove organoids cut off at the border of their well image and visualize a sample of them.

    An organoid whose bounding box touches the image border is only partly imaged, so its size,
    shape and intensities are wrong. The image size per plate is the largest extent of all bounding
    boxes on the plate, since cut-off organoids reach the border. It is stored in
    ad.uns["image_size"], so that later steps (check_well_positions, or running this again) use the
    same size instead of estimating it from the remaining organoids, which would cut off the next
    outermost ones.

    Cut-off organoids still count in calculate_outgrowth. Without bounding boxes, filter on
    StraightEdge_longest with filter_organoids_by instead.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object; left unchanged.
    - box_cols (tuple of str): Columns in ad.obs with the bounding box origin x, y and size x, y (µm).
    - plate_col (str): Column in ad.obs with the plate ID.
    - border_tolerance (float): Distance to the image border (µm) that still counts as touching.
    - channel (str): Channel for image visualization.
    - pyramid_level (int): Pyramid level of images for visualization.
    - add_boundary (bool): If True, draw the segmentation boundary on the images.
    - plot (bool): Visualize a sample of the removed organoids.

    Returns:
    - ad (anndata.AnnData): A copy without the cut-off organoids (recorded in
      ad.uns["removed_organoids"]), with ad.uns["image_size"] (width and height per plate, µm).
    """
    missing = [c for c in box_cols if c not in ad.obs.columns]
    if missing:
        raise KeyError(f"Bounding box column(s) {missing} not found in ad.obs; filter on StraightEdge_longest "
                       f"with filter_organoids_by instead.")
    plate = ad.obs[plate_col].astype(str).to_numpy()
    bx, by, bw, bh = (pd.to_numeric(ad.obs[c], errors="coerce").to_numpy(dtype=float) for c in box_cols)
    right, bottom = bx + bw, by + bh
    width, height = _image_size(ad, plate, right, bottom)
    contacts = _border_contacts(bx, by, right, bottom, width, height, border_tolerance)
    cut = contacts.any(axis=1).to_numpy()
    sides = np.array([", ".join(contacts.columns[row]) for row in contacts.to_numpy()], dtype=object)

    size = pd.DataFrame({"plate": plate, "width": width, "height": height}).groupby("plate", sort=False).first()
    dims = [f"{w:.0f} x {h:.0f}" for w, h in size.itertuples(index=False)]
    per_plate = pd.Series(cut).groupby(plate, sort=False).mean()
    print(f"Cut-off organoids: {cut.sum()} of {len(cut)} ({cut.mean():.1%}) touch the border of their well image "
          f"and were removed (per plate: " + ", ".join(f"{p} {f:.1%}" for p, f in per_plate.items()) + "). "
          f"Image size (µm): " + (dims[0] if len(set(dims)) == 1 else
                                  ", ".join(f"{p} {d}" for p, d in zip(size.index, dims))) + ".")

    if plot:
        _plot_organoid_images(ad, ad.obs_names[cut], [f"{s} border" for s in sides[cut]],
                              "Organoids cut off at the image border", channel, pyramid_level, add_boundary)
    filtered = remove_organoids(ad, cut, "remove_cut_organoids", "cut off at the " + sides + " image border")
    filtered.uns["image_size"] = size.reset_index()
    return filtered


def find_outlier_organoids(adata, condition_cols=_DEFAULT_CONDITIONS, plate_col="Barcode", well_col="Well",
                           features=None, n_neighbors=20, threshold=2.0, variance=0.9, min_group_size=10,
                           remove_outliers=False, plate_format=None, channel="R0__DAPI", pyramid_level=1,
                           add_boundary=False, plot=True):
    """
    Find organoids that look unlike any other organoid of their condition (likely artifacts such as
    merged organoids, debris, segmentation errors or out-of-focus objects).

    Within each plate x condition, every feature is converted to rank-based normal scores, so skew,
    units and condition differences do not matter. The scores are reduced to the principal
    components that cover `variance` of the variance, and the local outlier factor (LOF) is computed
    within each plate x condition: how much sparser an organoid's neighborhood is than that of its
    n_neighbors nearest neighbors. LOF is density-based, so heterogeneous but common phenotypes
    (e.g. a subpopulation of budding organoids) are not flagged; isolated organoids are. An
    organoid with LOF >= threshold is an outlier; the features in which it is most extreme are
    listed (score in SD units).

    Run it after remove_cut_organoids: cut-off organoids are common, so they would form their own
    dense group and blur the neighborhoods.

    Parameters:
    adata (anndata.AnnData): The AnnData object; adata.X holds the features.
    condition_cols (list of str): Columns in adata.obs that together define a condition.
    plate_col, well_col (str): Columns in adata.obs with the plate ID and the well name.
    features (list of str, optional): Features to use. None uses all features without missing values.
    n_neighbors (int): Neighbors for the LOF.
    threshold (float): LOF from which an organoid is an outlier (1 = as dense as its neighbors,
        2 = half as dense).
    variance (float): Share of the variance kept by the principal components.
    min_group_size (int): Plate x condition groups with fewer organoids are skipped.
    remove_outliers (bool): Remove the outliers (recorded in adata.uns["removed_organoids"]).
    plate_format (int, optional): Wells per plate for the plate map; None infers it.
    channel (str): Channel for the images of the outliers.
    pyramid_level (int): Pyramid level of the images.
    add_boundary (bool): If True, draw the segmentation boundary on the images.
    plot (bool): Plot the LOF distribution, the share of outliers per well and images of outliers.

    Returns:
    adata (anndata.AnnData): With adata.obs["outlier_score"] (LOF), ["organoid_outlier"] (bool),
        ["outlier_class"] ("outlier" / "typical", e.g. for plot_random_organoids_per_cluster) and
        ["outlier_features"] (most extreme features of outliers).
    """
    X = _feature_frame(adata, features, complete=True)
    plate = adata.obs[plate_col].astype(str).to_numpy()
    condition = _condition_labels(adata.obs, [c for c in condition_cols if c in adata.obs.columns])
    scores = _normal_scores(X, [plate, condition]).to_numpy()
    components = _pca_scores(scores, variance)

    lof = np.full(len(X), np.nan)
    for idx in pd.Series(np.arange(len(X))).groupby([plate, condition]).indices.values():
        if len(idx) < min_group_size:
            continue
        model = LocalOutlierFactor(n_neighbors=min(n_neighbors, len(idx) - 1)).fit(components[idx])
        lof[idx] = -model.negative_outlier_factor_
    outlier = lof >= threshold

    reasons = np.full(len(X), "", dtype=object)
    columns = X.columns.to_numpy()
    for i in np.flatnonzero(outlier):
        top = np.argsort(-np.abs(scores[i]), kind="stable")[:3]  # ties in feature order, as np.argmax below
        reasons[i] = ", ".join(f"{columns[j]} ({scores[i, j]:+.1f})" for j in top)
    adata.obs["outlier_score"] = lof
    adata.obs["organoid_outlier"] = outlier
    adata.obs["outlier_class"] = pd.Categorical(np.where(outlier, "outlier", "typical"),
                                                categories=["typical", "outlier"])
    adata.obs["outlier_features"] = reasons

    print(f"Outlier organoids: {outlier.sum()} of {len(X)} ({outlier.mean():.1%}) with LOF >= {threshold:g}, "
          f"based on {X.shape[1]} features ({components.shape[1]} principal components). "
          f"{np.isnan(lof).sum()} organoids in groups smaller than {min_group_size} were not assessed.")
    extreme = [columns[np.argmax(np.abs(scores[i]))] for i in np.flatnonzero(outlier)]
    if outlier.any():
        drivers = pd.Series(extreme).value_counts()
        print("Most frequent most-extreme feature of the outliers: "
              + ", ".join(f"{f} ({n})" for f, n in drivers.head(5).items()) + ".")

    if plot:
        wells, shape = _map_wells(adata, plate_col, well_col, condition_cols, plate_format=plate_format)
        share = pd.Series(outlier).groupby([plate, adata.obs[well_col].astype(str).to_numpy()]).mean()
        share.index.names = ["plate", "well"]
        _plot_outlier_scores(lof, threshold)
        _plot_plates(wells, share.reindex(wells.index), shape, _SEQUENTIAL, 0, max(0.2, float(share.max())),
                     "share of outlier organoids", pd.Series(False, index=wells.index), None,
                     "Outlier organoids per well")
        _plot_organoid_images(adata, adata.obs_names[outlier],
                              [f"LOF {s:.1f}, {f}" for s, f in zip(lof[outlier], extreme)],
                              f"Outlier organoids (LOF ≥ {threshold:g})", channel, pyramid_level, add_boundary)

    if remove_outliers:
        print(f"Removed {outlier.sum()} outlier organoids.")
        adata = remove_organoids(adata, outlier, "find_outlier_organoids",
                                 [f"LOF {s:.1f}; most extreme: {r}" for s, r in zip(lof, reasons)])
    return adata


def plot_random_organoids(ad, feature, rows=10, cols=10, channel="R0__DAPI", seed=0, pyramid_level=1,
                          add_boundary=False):
    """
    Plot random organoids remaining after filtering, and summarize what was removed.

    Parameters:
    - ad (anndata.AnnData): AnnData object after filtering.
    - feature (str): Feature to display in subplot titles.
    - rows (int): Number of rows in subplot grid.
    - cols (int): Number of columns in subplot grid.
    - channel (str): Channel for image visualization.
    - seed (int): Random seed for reproducibility.
    - pyramid_level (int): Pyramid level for image loading.
    - add_boundary (bool): If True, draw the segmentation boundary on the images.
    """
    removed = ad.uns.get("removed_organoids")
    n_removed = 0 if removed is None else len(removed)
    by_step = "" if not n_removed else " (" + ", ".join(
        f"{step}: {n}" for step, n in removed["step"].value_counts(sort=False).items()) + ")"
    print(f"A total of {n_removed} objects have been removed during the filtering process{by_step}. "
          f"{ad.n_obs} objects remain for further analysis.\n")
    if ad.n_obs == 0:
        print("No objects remain after filtering; nothing to plot.")
        return None

    vals = np.asarray(ad.obs_vector(feature), dtype=float)
    return _plot_organoid_images(ad, ad.obs_names, [f"{feature}: {v:.2f}" for v in vals], "Remaining Objects",
                                 channel, pyramid_level, add_boundary, max_rows=rows, cols=cols, seed=seed)


def _plot_organoid_images(ad, ids, labels, title, channel, pyramid_level, add_boundary=False, max_rows=4, cols=9,
                          seed=0):
    """
    Plot a random sample of up to max_rows x cols organoids, masked to the organoid and titled with
    their ID and label. Opens the OME-Zarr plates if they are not loaded yet (kept in ad.uns).
    """
    ids, labels = list(ids), list(labels)
    if not ids:
        return None
    if "ome_zarr_dict" not in ad.uns:
        try:
            add_zarr_uns(ad)
        except Exception as e:
            print(f"Images not shown: could not open the OME-Zarr plates ({e}).")
            return None

    rows = min(max_rows, math.ceil(len(ids) / cols))
    cols = min(cols, len(ids))
    shown = np.sort(np.random.default_rng(seed).choice(len(ids), size=min(rows * cols, len(ids)), replace=False))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2), squeeze=False)
    fig.suptitle(title, fontsize=18, y=1.01)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, shown):
        try:
            img, _ = load_img_mask_by_UID(ids[i], ad.uns["stainings"], ad.uns["experiment_setup"],
                                          ad.uns["ome_zarr_dict"], ad.uns["table_name"], ad.uns["label_name"],
                                          pyramid_level, str(channel), add_boundary=add_boundary,
                                          segmentation_round=ad.uns.get("segmentation_round", 0))
            ax.imshow(img, interpolation="nearest", cmap="magma")
            ax.set_title(f"{ids[i]}\n{labels[i]}", fontsize=8)
        except Exception as e:
            ax.text(0.5, 0.5, f"Failed to load\n{ids[i]}", ha="center", va="center")
            print(f"Warning: failed to load image for {ids[i]}: {e}")

    fig.tight_layout()
    return fig


def _plot_outlier_scores(lof, threshold):
    values = lof[np.isfinite(lof)]
    fig, ax = plt.subplots(figsize=(5, 2.4), layout="constrained")
    upper = max(threshold * 1.5, np.quantile(values, 0.999))
    ax.hist(np.clip(values, None, upper), bins=60, color=_BLUE)
    ax.axvline(threshold, color=_INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)))
    ax.set_yscale("log")
    clipped = f" (values above {upper:.1f} shown at {upper:.1f})" if (values > upper).any() else ""
    ax.set_xlabel(f"Local outlier factor{clipped}")
    ax.set_ylabel("organoids")
    ax.text(threshold, ax.get_ylim()[1], f" threshold {threshold:g}", va="top", color=_INK_SECONDARY)
    fig.suptitle("Outlier scores of all organoids")
    plt.show()
