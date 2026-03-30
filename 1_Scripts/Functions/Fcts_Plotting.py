from __future__ import annotations

from Functions.Fcts_Base import save_fig, remove_uns
from skimage import segmentation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
import random

def load_img_mask_by_UID(UID, stainings, experiment_setup, ome_zarr_dict, table_name, label_name, pyramid_level, channel_str, add_boundary=False):
    """
    Retrieve an image and corresponding mask for a specific entry in an OME-Zarr plate dataset,
    based on a unique identifier (UID) and additional parameters.

    Parameters:
        UID (str): Unique identifier of the data instance, formatted as 'barcode-well-index'.
        ome_zarr_dict (dict): Dictionary mapping barcodes to Plate objects, assumed to adhere to an OME-Zarr interface.
        table_name (str): Name of the table to extract entries from for the well (e.g., 'organoids').
        label_name (str): Name of the label within the mask to extract (e.g., 'organoid').
        pyramid_level (int): Pyramid level to extract the image/mask from.
        channel_str (str): Channel identifier string, e.g., 'R2__DAPI'.
        add_boundary (bool): If True, add the segmentation boundary to the image using a high value.
    """

    from Functions.Fcts_FE import get_plate_for_round
    
    # UID format: 'barcode-well-index'
    try:
        bc, well, index_str = UID.rsplit("-", 2)
        index = int(index_str)
    except Exception as e:
        raise ValueError(f"UID format error: {UID} should be 'barcode-well-index'") from e

    if bc not in ome_zarr_dict:
        raise KeyError(f"Barcode {bc} not found in ome_zarr_dict.")
    plate = ome_zarr_dict[bc]

    well_names = plate.get_names()
    if well not in well_names:
        raise KeyError(f"Well {well} not found in barcode {bc}.")
    well_idx = well_names.index(well)
    well_ov = plate.images[well_idx]

    table = well_ov.get_table(table_name)
    if table.empty:
        raise ValueError(f"Table {table_name} is empty for well {well} in barcode {bc}.")
    if index < 0 or index >= len(table):
        raise IndexError(f"Index {index} out of bounds for table length {len(table)}")

    entry = table.loc[str(index), :]
    ul_y, ul_x = entry["y_micrometer"], entry["x_micrometer"]
    lr_y = ul_y + entry["len_y_micrometer"]
    lr_x = ul_x + entry["len_x_micrometer"]

    _, mask = well_ov.get_array_pair_by_coordinate(
        label_name = label_name,
        pyramid_level = pyramid_level,
        upper_left_yx = (ul_y, ul_x),
        lower_right_yx = (lr_y, lr_x)
    )

    if label_name not in mask:
        raise KeyError(f"Label {label_name} not found in mask for well {well} in barcode {bc}.")


    # Parse round and marker from channel_str
    if "__" in channel_str:
        round, marker = channel_str.split("__", 1)
        round = round[1:]  # Remove leading 'R'
    else:
        raise ValueError(f"Channel string '{channel_str}' is not in expected format 'R#__MARKER'")

    # Find the channel index for the requested channel_str
    channel_idx = list(stainings[experiment_setup[bc][well][1]][round]).index(marker)

    # load plate ov for the requested round
    plate_channel = get_plate_for_round(plate, round)
    well_names = plate_channel.get_names()
    if well not in well_names:
        raise KeyError(f"Well {well} not found in barcode {bc}.")
    well_idx = well_names.index(well)
    well_ov_img = plate_channel.images[well_idx]

    img = well_ov_img.get_array_by_coordinate(
        label_name = None,
        pyramid_level = pyramid_level,
        upper_left_yx = (ul_y, ul_x),
        lower_right_yx = (lr_y, lr_x)
    )


    img = img[channel_idx, 0]
    img = np.pad(img, pad_width=20, mode="constant", constant_values=0)
    mask = mask[label_name][0]
    mask = np.pad(mask, pad_width=20, mode="constant", constant_values=0)
    mask[mask != int(index)] = 0
    img[~mask.astype(bool)] = 0

    if add_boundary:
        boundary = segmentation.find_boundaries(mask.astype(bool), mode="inner")
        img[boundary] = np.max(img)

    return img, mask



def segmentation_fidelity_check(ome_zarrs_dict, channel, channel_color, channel_range, n, label_name, alpha, pyramid_lvl_plot=4):
    """
    Plot randomly selected images and their corresponding segmentation masks from OME-Zarr plates.

    Parameters
    ----------
    ome_zarrs_dict : dict
    channel : int
        Index of the image channel to display from the returned `img` array.
    channel_color : str or matplotlib colormap
        Colormap used to render the intensity image channel.
    channel_range : list[int, int]
    n : int
        Number of wells to randomly sample and plot per barcode.
    label_name : str
        Name/key of the label to retrieve and overlay.
    alpha : float
        Opacity (0-1) used for non-zero label regions in the overlay.
    pyramid_lvl_plot : int, default=4
        Pyramid level to load for both image and label.
    """
    
    # Loop over barcodes
    for barcode in ome_zarrs_dict:
        
        wells_in_bc = ome_zarrs_dict[barcode].names
        # Get random wells
        wells = random.sample(ome_zarrs_dict[barcode].names, n)

        # Loop over wells
        for well in wells:
            _, ax = plt.subplots(figsize=(15, 15))
            ax.set_title(f"{barcode} - {well}", pad=10)

            img, label = ome_zarrs_dict[barcode][wells_in_bc.index(well)].get_array_pair_by_coordinate(
                    label_name=label_name,
                    pyramid_level=pyramid_lvl_plot,
                )
            img = img[channel][0]
            label = label[label_name][0]

            ax.imshow(img, vmin = channel_range[0], vmax = channel_range[1], cmap = channel_color, interpolation="nearest")

            n_max = int(np.max(label))
            if n_max == 0:
                ax.set_axis_off()
                return

            rng = np.random.default_rng()
            colors = rng.random((n_max + 1, 4))
            colors[0] = (0, 0, 0, 0)
            colors[1:, 3] = alpha
            cmap = mpl.colors.ListedColormap(colors)

            lab = np.ma.masked_where(label == 0, label)
            ax.imshow(lab, cmap=cmap, interpolation="nearest")
            ax.set_axis_off()
            plt.show()



def plot_well(ome_zarrs_dict, barcode, well, channels, channel_colors, channel_ranges, label_name, pyramid_lvl_plot=4):
    """
    Plot a specific well from the OME-ZARR files.

    Parameters:
    - ome_zarrs_dict (dict): Dictionary containing OME-ZARR files.
    - barcode (str): Barcode of the well to plot.
    - well (str): Well identifier to plot.
    - channels (list): List of channels to plot.
    - channel_colors (list): List of colors for each channel.
    - channel_ranges (list): List of ranges for each channel.
    - label_name (str): Name of the label to plot.
    - pyramid_lvl_plot (int): Pyramid level to plot.
    """
    ome_zarrs_dict[barcode][ome_zarrs_dict[barcode].names.index(well)].plot(
        label_name=label_name, 
        pyramid_level=pyramid_lvl_plot, 
        channels=channels, 
        channel_colors=channel_colors, 
        channel_ranges=channel_ranges,
        fig_width_inch=15, 
        fig_height_inch=15,
        scalebar_micrometer = 100,
        show_scalebar_label=True,
        title=f"{barcode} - {well}"
    )


def plot_random_organoids_per_cluster(
    ad,
    cluster_key="kmeans_labels",      # e.g., "kmeans_labels", "phenograph_labels", or with suffixes like "_12"
    n_per_cluster=10,
    stainings="R0__DAPI",                 # str or list[str], e.g., "R0__DAPI" or ["R0__DAPI","R1__KRT7"]
    pyramid_level=1,
    seed=0,
    cmap="magma",
    title_prefix=None,
    save_plot=False,                   # If True, save the figure to ad.uns["plot_dir"] with a default name
    figure_width_per_col=2.0,
    figure_height_per_row=2.0,
    # scalebar (bar only, no label) and style:
    bar_length_um=None,               # desired scalebar length in microns; if None, auto-pick a “nice” value
    bar_height_frac=0.02,             # scalebar height as fraction of image height
    bar_pad_frac=0.03,                # padding from image edges as fraction of width/height
    bar_color="white",
    bar_alpha=1.0,
    # intensity scaling per staining:
    thresholds=None,                  # None or list/tuple of (vmin, vmax) for each staining in order
    # normalization option:
    normalize_sizes=False,            # If True, pad all images to a common size; only the top-left image gets a scalebar
    add_boundary=False,               # If True, overlays the segmentation boundary on the image as np.max(img)
):
    """
    Plot n random organoids per cluster using only information in `ad`, with:
      - Black background, white titles.
      - Scalebar (bar only) whose physical size (µm) is displayed in the figure title.
      - Support for one or multiple stainings; when multiple, render multiple rows per cluster (one row per staining).
      - Optional per-staining intensity thresholds (vmin, vmax) for display.
      - Optional size normalization: pad all images to the maximum (H,W) within the grid so they display uniformly.
        Only the image in the first row and first column (top-left) will display the scalebar.

    Parameters:
        stainings: str or list[str]. If list, a separate row per staining will be plotted within each cluster.
        thresholds: None or list/tuple of (vmin, vmax) per staining, in the same order as 'stainings'.
        normalize_sizes: If True, pad all images to the maximum (H,W) within the grid so they display uniformly.
                         Only the image in the first row and first column (top-left) will display the scalebar.
        add_boundary: If True, overlays the segmentation boundary on the image as np.max(img).
    """
    from Functions.Fcts_Base import extract_ome_zarr_tables
    ad.uns["ome_zarr_dict"], ad.uns["ome_zarr_df"] = extract_ome_zarr_tables(ad.uns["experiment_setup"], ad.uns["source_dir"], ad.uns["folders"], ad.uns["table_name"])

    # ---------- validations ----------
    required_uns = ("ome_zarr_dict", "table_name", "label_name", "pixel_spacing", "stainings")
    for k in required_uns:
        if (k not in ad.uns):
            raise ValueError(f"ad.uns must contain '{k}' (missing: {k}).")
    if cluster_key not in ad.obs.columns:
        raise ValueError(f"cluster_key '{cluster_key}' not found in ad.obs.")
    if "ABs" not in ad.obs.columns:
        raise ValueError("ad.obs must contain an 'ABs' column to map organoids to staining panels.")

    # Normalize stainings to list
    if isinstance(stainings, str):
        stainings = [stainings]
    if not isinstance(stainings, (list, tuple)) or len(stainings) == 0:
        raise ValueError("stainings must be a non-empty string or list of strings.")

    # Normalize thresholds
    if thresholds is not None:
        if not (isinstance(thresholds, (list, tuple)) and len(thresholds) == len(stainings)):
            raise ValueError("thresholds must be a list/tuple with same length as 'stainings', or None.")
        for t in thresholds:
            if t is not None and not (isinstance(t, (list, tuple)) and len(t) == 2):
                raise ValueError("Each thresholds entry must be a (vmin, vmax) tuple or None.")
    else:
        thresholds = [None] * len(stainings)

    # Pixel size at requested pyramid level
    base_px_um = float(ad.uns["pixel_spacing"])
    if base_px_um <= 0:
        raise ValueError("ad.uns['pixel_spacing'] must be a positive float.")
    pixel_size_um = base_px_um * (2 ** int(pyramid_level))

    # All organoids and clusters
    all_ids = ad.obs.index
    if len(all_ids) == 0:
        raise ValueError("ad.obs.index is empty; no organoids available to plot.")
    obs_clusters = ad.obs.loc[all_ids, cluster_key]
    if not pd.api.types.is_categorical_dtype(obs_clusters):
        obs_clusters = obs_clusters.astype(str).astype("category")
    clusters = list(obs_clusters.cat.categories)
    if not clusters:
        raise ValueError(f"No clusters found under '{cluster_key}'.")

    # Filter to organoids whose AB supports ALL requested stainings (multiplexed rounds aware)
    def ab_supports_all_stainings(ab_name):
        ab_dict = ad.uns["stainings"].get(ab_name, {})
        for st in stainings:
            if "__" not in st:
                return False
            round_str, marker = st.split("__", 1)
            round_num = round_str[1:] if round_str.startswith("R") else round_str
            # Try both string and int keys for round
            round_keys = list(ab_dict.keys())
            found_round = None
            for rk in round_keys:
                if str(rk) == str(round_num):
                    found_round = rk
                    break
            if found_round is None:
                return False
            markers = ab_dict[found_round]
            if marker not in list(markers):
                return False
        return True

    rng = np.random.default_rng(seed)
    per_cluster_ids = []
    max_cols = 0
    total_to_plot = 0
    for cname in clusters:
        ids_c = [oid for oid in all_ids[obs_clusters == cname] if ab_supports_all_stainings(ad.obs.loc[oid, "ABs"])]
        if len(ids_c) == 0:
            per_cluster_ids.append([])
            continue
        if len(ids_c) > n_per_cluster:
            sampled = list(rng.choice(ids_c, size=n_per_cluster, replace=False))
        else:
            sampled = ids_c
        per_cluster_ids.append(sampled)
        max_cols = max(max_cols, len(sampled))
        total_to_plot += len(sampled)
    if total_to_plot == 0:
        raise ValueError("No organoids available to plot after per-cluster sampling for the requested stainings.")

    example_width = None
    for row_ids in per_cluster_ids:
        if row_ids:
            oid0 = row_ids[0]
            try:
                st0 = stainings[0]
                img0, _ = load_img_mask_by_UID(
                    oid0,
                    ad.uns["stainings"], 
                    ad.uns["experiment_setup"], 
                    ad.uns["ome_zarr_dict"],
                    ad.uns["table_name"],
                    ad.uns["label_name"],
                    pyramid_level,
                    st0,
                )
                example_width = img0.shape[1]
                break
            except Exception:
                continue
    if example_width is None:
        example_width = 512
    final_bar_um = choose_bar_um(example_width, pixel_size_um, bar_length_um)
    # Figure and axes (black background)
    rows = len(clusters) * len(stainings)  # rows per cluster times stainings
    cols = max(1, max_cols)
    fig_w = cols * figure_width_per_col
    fig_h = rows * figure_height_per_row
    fig, ax = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False, facecolor="black")
    for a in ax.flatten():
        a.set_facecolor("black")

    # Title includes scalebar size
    st_list_str = ", ".join(stainings)
    if title_prefix is None:
        title_prefix = f"Random organoids per {cluster_key} — {st_list_str}"
    fig.suptitle(
        f"{title_prefix} — scalebar: {int(final_bar_um) if final_bar_um >= 10 else final_bar_um:g} µm",
        fontsize=16, y=1.02, color="white"
    )

    # Optional: if normalize_sizes=True, pre-load all images to compute max H and W, and store them to avoid reloading
    cache = {}  # (oid, st) -> (img, mask)
    maxH = 0
    maxW = 0
    if normalize_sizes:
        for ci, cname in enumerate(clusters):
            sampled_ids = per_cluster_ids[ci]
            for oid in sampled_ids:
                for st in stainings:
                    try:
                        img, mask = load_img_mask_by_UID(
                            oid,
                            ad.uns["stainings"],
                            ad.uns["experiment_setup"],
                            ad.uns["ome_zarr_dict"],
                            ad.uns["table_name"],
                            ad.uns["label_name"],
                            pyramid_level,
                            st,
                            add_boundary=add_boundary,
                        )
                        # Apply mask if compatible
                        if mask is not None and mask.shape == img.shape:
                            img = img.copy()
                            img[mask == 0] = 0
                        cache[(oid, st)] = (img, mask)
                        H, W = img.shape[:2]
                        maxH = max(maxH, H)
                        maxW = max(maxW, W)
                    except Exception as e:
                        cache[(oid, st)] = (None, None)
                        print(f"Warning: failed to preload image for {oid} ({st}): {e}")
        def pad_to_max(image, Hmax, Wmax):
            if image is None:
                return None
            H, W = image.shape[:2]
            pad_top = (Hmax - H) // 2
            pad_bottom = Hmax - H - pad_top
            pad_left = (Wmax - W) // 2
            pad_right = Wmax - W - pad_left
            return np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode="constant", constant_values=0)

    # Render images
    for ci, cname in enumerate(clusters):
        sampled_ids = per_cluster_ids[ci]
        for si, st in enumerate(stainings):
            row_idx = ci * len(stainings) + si
            for cidx in range(cols):
                ax_curr = ax[row_idx, cidx]
                if cidx >= len(sampled_ids):
                    ax_curr.axis("off")
                    continue

                oid = sampled_ids[cidx]
                ab = ad.obs.loc[oid, "ABs"]

                try:
                    if normalize_sizes:
                        img, mask = cache.get((oid, st), (None, None))
                        if img is None:
                            raise RuntimeError("missing image in cache")
                        img_disp = pad_to_max(img, maxH, maxW)
                        img_for_bar = img_disp
                    else:
                        img, mask = load_img_mask_by_UID(
                            oid,
                            ad.uns["stainings"],
                            ad.uns["experiment_setup"],
                            ad.uns["ome_zarr_dict"],
                            ad.uns["table_name"],
                            ad.uns["label_name"],
                            pyramid_level,
                            st,
                            add_boundary=add_boundary,
                        )
                        img = img.copy()
                        if mask is not None and mask.shape == img.shape:
                            img[mask == 0] = 0
                        img_disp = img
                        img_for_bar = img

                    vmin, vmax = (None, None)
                    if thresholds[si] is not None:
                        vmin, vmax = thresholds[si]

                    ax_curr.imshow(img_disp, interpolation="nearest", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

                    if cidx == 0:
                        ax_curr.set_title(f"Cluster {cname} — {st}\n{oid}", fontsize=9, color="white")
                    else:
                        ax_curr.set_title(f"{oid}", fontsize=8, color="white")

                    # Remove axis ticks and labels
                    ax_curr.set_xticks([])
                    ax_curr.set_yticks([])
                    ax_curr.set_xlabel("")
                    ax_curr.set_ylabel("")
                    ax_curr.tick_params(left=False, bottom=False)
                    ax_curr.axis("off")

                    if not normalize_sizes:
                        draw_scalebar(
                            ax_curr, img_for_bar.shape, pixel_size_um, final_bar_um,
                            bar_color, bar_alpha, bar_pad_frac, bar_height_frac
                        )
                    else:
                        if row_idx == 0 and cidx == 0:
                            draw_scalebar(
                                ax_curr, img_for_bar.shape, pixel_size_um, final_bar_um,
                                bar_color, bar_alpha, bar_pad_frac, bar_height_frac
                            )

                except Exception as e:
                    ax_curr.text(0.5, 0.5, f"Failed\n{oid}", ha="center", va="center",
                                 fontsize=8, color="white")
                    ax_curr.axis("off")

    fig.tight_layout()

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "Staining_Gallery", dpi=300)
    
    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])

    return fig

def choose_bar_um(img_width_pixels, px_um, requested_um):
    if requested_um is not None:
        return float(requested_um)
    target_um = (img_width_pixels * px_um) / 5.0
    nice = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000], dtype=float)
    return float(nice[(np.abs(nice - target_um)).argmin()])

# Helper: draw scalebar rectangle
def draw_scalebar(ax_, img_shape, px_um, bar_um, color, alpha, pad_frac, height_frac):
    H, W = img_shape[0], img_shape[1]
    bar_px = bar_um / px_um
    if bar_px < 1:
        return
    x_pad = pad_frac * W
    y_pad = pad_frac * H
    bar_h = max(1, int(height_frac * H))
    x0 = int(x_pad)
    y0 = int(H - y_pad - bar_h)
    rect = Rectangle((x0, y0), width=bar_px, height=bar_h, color=color, alpha=alpha)
    ax_.add_patch(rect)

def pad_to_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img

    size = max(h, w)
    pad_h = size - h
    pad_w = size - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 2:
        pad_width = ((top, bottom), (left, right))
    else:
        pad_width = ((top, bottom), (left, right), (0, 0))

    return np.pad(img, pad_width, mode="constant", constant_values=0)


def plot_all_stainings_per_UID(
    ad,
    uids,
    pyramid_level = 0,
    scalebar_um = None,
    add_boundary: bool = False,
    figsize_per_panel: tuple[float, float] = (4, 4),
    cmap = "magma",
    save_plot = False,
):
    """
    Plot all stainings for one UID or a list of UIDs.

    Returns
    -------
    dict
        Nested dict of the form:
        {uid: {staining: (img, msk)}}
    """
    from Functions.Fcts_Base import extract_ome_zarr_tables
    ad.uns["ome_zarr_dict"], ad.uns["ome_zarr_df"] = extract_ome_zarr_tables(ad.uns["experiment_setup"], ad.uns["source_dir"], ad.uns["folders"], ad.uns["table_name"])

    if isinstance(uids, str):
        uids = [uids]
    else:
        uids = list(uids)

    pixel_size_um = float(ad.uns["pixel_spacing"]) * (2 ** int(pyramid_level))

    for uid in uids:
        bc, well, _ = uid.rsplit("-", 2)
        uid_stainings = [f"R{k}__{marker}" for k, markers in ad.uns["stainings"][ad.uns["experiment_setup"][bc][well][1]].items() for marker in markers]

        n = len(uid_stainings)
        fig, axes = plt.subplots(1, n, figsize=(figsize_per_panel[0] * n, figsize_per_panel[1]))
        if n == 1:
            axes = [axes]

        for i, (ax, staining) in enumerate(zip(axes, uid_stainings)):
            
            img, _ = load_img_mask_by_UID(
                uid,
                ad.uns["stainings"],
                ad.uns["experiment_setup"],
                ad.uns["ome_zarr_dict"],
                ad.uns["table_name"],
                ad.uns["label_name"],
                pyramid_level = pyramid_level,
                channel_str = staining,
                add_boundary = add_boundary,
            )

            img = pad_to_square(img)
            
            ax.imshow(img, interpolation="nearest", cmap=cmap, vmin = np.percentile(img, 1), vmax = np.percentile(img, 99))
            ax.set_title(f"{uid}\n{staining}")
            ax.axis("off")

            if i == 0:
                final_bar_um = choose_bar_um(img.shape[1], pixel_size_um, scalebar_um)

                print(f"Drawing scalebar of {final_bar_um} µm for UID {uid}")

                draw_scalebar(
                    ax,
                    img.shape,
                    pixel_size_um,
                    final_bar_um,
                    color = "white",
                    alpha = 1,
                    pad_frac = 0.03,
                    height_frac = 0.02,
                )

    
        fig.tight_layout()

        if save_plot:
            save_fig(fig, ad.uns["plot_dir"], f"All_Stainings_{uid}", dpi=300)

    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])


def highlight_UID_in_obsm(
    ad,
    UID,
    highlight_obs = None,
    dims = (0, 1),
    base_color = "lightgray",
    highlight_color = "red",
    alpha = 0.5,
    s = 8,
    highlight_s = 60,
    label = True,
    label_offset = (5, 5),
    figsize = (5, 5),
    title = None,
    save_plot = False,
):
    """
    Plot any 2D embedding stored in ad.obsm and highlight selected obs_names.
    """

    if UID not in ad.obsm:
        raise ValueError(f"'{UID}' not found in ad.obsm.")

    coords = ad.obsm[UID]

    if coords.shape[1] <= max(dims):
        raise ValueError(f"'{UID}' has shape {coords.shape}, so dims={dims} is out of range.")

    if highlight_obs is None:
        highlight_obs = []
    elif isinstance(highlight_obs, str):
        highlight_obs = [highlight_obs]

    missing_obs = [x for x in highlight_obs if x not in ad.obs_names]
    if missing_obs:
        raise ValueError(f"These obs_names were not found in ad.obs_names: {missing_obs}")

    highlight_mask = ad.obs_names.isin(highlight_obs)

    fig, ax = plt.subplots(figsize=figsize)

    xdim, ydim = dims

    ax.scatter(
        coords[:, xdim],
        coords[:, ydim],
        c=base_color,
        s=s,
        alpha=alpha,
        linewidths=0,
    )

    if np.any(highlight_mask):
        ax.scatter(
            coords[highlight_mask, xdim],
            coords[highlight_mask, ydim],
            c=highlight_color,
            s=highlight_s,
            alpha=1.0,
            linewidths=0.5,
            edgecolors="black",
            zorder=3,
        )

        if label:
            dx, dy = label_offset
            for obs_name in ad.obs_names[highlight_mask]:
                idx = ad.obs_names.get_loc(obs_name)
                x = coords[idx, xdim]
                y = coords[idx, ydim]

                ax.annotate(
                    str(obs_name),
                    xy=(x, y),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    color=highlight_color,
                    ha="left",
                    va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
                    arrowprops=dict(arrowstyle="-", color=highlight_color, lw=0.8),
                    zorder=4,
                )

    ax.set_xlabel(f"{UID}_{xdim + 1}")
    ax.set_ylabel(f"{UID}_{ydim + 1}")
    ax.set_title(title if title is not None else UID)
    ax.set_axis_off()

    plt.tight_layout()

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], f"{UID}_highlight", dpi=300)