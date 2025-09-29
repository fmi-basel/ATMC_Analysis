from Fcts_Base import load_img_mask_by_UID, save_fig
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

def segmentation_fidelity_check(ome_zarrs_dict, channels, channel_colors, channel_ranges, n, label_name, scalebar_micrometer = 100, pyramid_lvl_plot=4):
    """
    Plot randomly selected images and their corresponding masks from OME-ZARR files.

    Parameters:
    - ome_zarrs_dict (dict): Dictionary containing OME-ZARR files.
    - channels (list): List of channels to plot.
    - channel_colors (list): List of colors for each channel.
    - channel_ranges (list): List of ranges for each channel.
    - n (int): Number of images to plot per barcode.
    - label_name (str): Name of the label to plot.
    - pyramid_lvl_plot (int): Pyramid level to plot.
    """
    # Loop over barcodes
    for barcode in ome_zarrs_dict:
        
        wells_in_bc = ome_zarrs_dict[barcode].names
        # Get random wells
        wells = random.sample(ome_zarrs_dict[barcode].names, n)

        # Loop over wells
        for well in wells:

            ome_zarrs_dict[barcode][wells_in_bc.index(well)].plot(
                label_name=label_name, 
                pyramid_level=pyramid_lvl_plot, 
                channels=channels, 
                channel_colors=channel_colors, 
                channel_ranges=channel_ranges,
                fig_width_inch=15, 
                fig_height_inch=15,
                scalebar_micrometer = scalebar_micrometer,
                show_scalebar_label=True,
                title=f"{barcode} - {well}"
            )


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
    stainings="DAPI",                 # str or list[str], e.g., "DAPI" or ["DAPI","KRT7"]
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
    normalize_sizes=False,            # If True: pad all images to a common size; only the top-left image gets a scalebar
    
):
    """
    Plot n random organoids per cluster using only information in `ad`, with:
      - Black background, white titles.
      - Scalebar (bar only) whose physical size (µm) is displayed in the figure title.
      - Support for one or multiple stainings; when multiple, render multiple rows per cluster (one row per staining).
      - Optional per-staining intensity thresholds (vmin, vmax) for display.
      - Optional size normalization: pad all images to the same (H,W) using black pixels; only the top-left image shows the scalebar.

    Parameters:
        stainings: str or list[str]. If list, a separate row per staining will be plotted within each cluster.
        thresholds: None or list/tuple of (vmin, vmax) per staining, in the same order as 'stainings'.
        normalize_sizes: If True, pad all images to the maximum (H,W) within the grid so they display uniformly.
                         Only the image in the first row and first column (top-left) will display the scalebar.
    """

    # ---------- validations ----------
    required_uns = ("ome_zarr_dict", "table_name", "label_name", "pixel_spacing", "stainings")
    for k in required_uns:
        if k not in ad.uns:
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

    # Build staining -> {AB -> channel_index}
    stainings_cfg = ad.uns["stainings"]
    staining_to_ab_channel = {}
    for st in stainings:
        ab_to_channel = {}
        for ab, arr in stainings_cfg.items():
            arr_list = list(arr)
            if st in arr_list:
                ab_to_channel[ab] = arr_list.index(st)
        if not ab_to_channel:
            raise ValueError(f"Staining '{st}' not found in any entry of ad.uns['stainings'].")
        staining_to_ab_channel[st] = ab_to_channel

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

    # Filter to organoids whose AB supports ALL requested stainings
    def ab_supports_all_stainings(ab_name):
        return all(ab_name in staining_to_ab_channel[st] for st in stainings)

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

    # Determine bar length for title (auto if None). Estimate width using a probe image.
    def choose_bar_um(img_width_pixels, px_um, requested_um):
        if requested_um is not None:
            return float(requested_um)
        target_um = (img_width_pixels * px_um) / 5.0
        nice = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000], dtype=float)
        return float(nice[(np.abs(nice - target_um)).argmin()])

    example_width = None
    for row_ids in per_cluster_ids:
        if row_ids:
            oid0 = row_ids[0]
            try:
                st0 = stainings[0]
                ab0 = ad.obs.loc[oid0, "ABs"]
                ch0 = staining_to_ab_channel[st0][ab0]
                img0, _ = load_img_mask_by_UID(
                    oid0,
                    ad.uns["ome_zarr_dict"],
                    ad.uns["table_name"],
                    ad.uns["label_name"],
                    pyramid_level,
                    ch0,
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

    # Optional: if normalize_sizes=True, pre-load all images to compute max H and W, and store them to avoid reloading
    cache = {}  # (oid, st) -> (img, mask)
    maxH = 0
    maxW = 0
    if normalize_sizes:
        for ci, cname in enumerate(clusters):
            sampled_ids = per_cluster_ids[ci]
            for oid in sampled_ids:
                ab = ad.obs.loc[oid, "ABs"]
                for st in stainings:
                    ch = staining_to_ab_channel[st][ab]
                    try:
                        img, mask = load_img_mask_by_UID(
                            oid,
                            ad.uns["ome_zarr_dict"],
                            ad.uns["table_name"],
                            ad.uns["label_name"],
                            pyramid_level,
                            ch,
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
                        # mark as missing
                        cache[(oid, st)] = (None, None)
                        print(f"Warning: failed to preload image for {oid} ({st}): {e}")
        # Define a padding function
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
                if ab not in staining_to_ab_channel[st]:
                    ax_curr.axis("off")
                    continue
                ch = staining_to_ab_channel[st][ab]

                try:
                    # Load/pick image
                    if normalize_sizes:
                        img, mask = cache.get((oid, st), (None, None))
                        if img is None:
                            raise RuntimeError("missing image in cache")
                        img_disp = pad_to_max(img, maxH, maxW)
                        img_for_bar = img_disp  # padded size for consistent bar geometry on top-left
                    else:
                        img, mask = load_img_mask_by_UID(
                            oid,
                            ad.uns["ome_zarr_dict"],
                            ad.uns["table_name"],
                            ad.uns["label_name"],
                            pyramid_level,
                            ch,
                        )
                        img = img.copy()
                        if mask is not None and mask.shape == img.shape:
                            img[mask == 0] = 0
                        img_disp = img
                        img_for_bar = img  # individual size

                    # thresholds per staining
                    vmin, vmax = (None, None)
                    if thresholds[si] is not None:
                        vmin, vmax = thresholds[si]

                    ax_curr.imshow(img_disp, interpolation="nearest", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

                    # Row titles on first column
                    if cidx == 0:
                        ax_curr.set_title(f"Cluster {cname} — {st}\n{oid}", fontsize=9, color="white")
                    else:
                        ax_curr.set_title(f"{oid}", fontsize=8, color="white")

                    # Draw scalebar:
                    # - If normalize_sizes is False: draw on every tile (as before)
                    # - If normalize_sizes is True: only draw on the global top-left tile (row 0, col 0)
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
                    print(f"Warning: failed to load image for {oid} (staining={st}): {e}")
                ax_curr.axis("off")

    fig.tight_layout()

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "Staining_Gallery", dpi=300)
    
    return fig


