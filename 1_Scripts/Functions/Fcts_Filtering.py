from Functions.Fcts_Plotting import load_img_mask_by_UID
import matplotlib.pyplot as plt
import random
import math

"""
***
FILTERING FUNCTIONS
***
"""

import pandas as pd

def filter_rows_by_percentile_bounds(
    df: pd.DataFrame,
    features,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    keep_na: bool = True,
    return_details: bool = False,
    flag_prefix: str = "OutlierPct",
):
    """
    Filter out rows where any of the specified features lies outside the
    [lower_q, upper_q] percentile interval (computed per feature).

    Tagging happens first for all features; filtering happens only after all tags exist.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    features : str | list[str]
        Feature name(s) to evaluate.
    lower_q, upper_q : float
        Quantile bounds in [0, 1]. Example: 0.01 and 0.99.
    keep_na : bool
        If True: NaNs do NOT trigger outlier removal (NaNs are kept).
        If False: NaNs are treated as outliers (row gets flagged).
    return_details : bool
        If True: return (df_filtered, df_annotated, bounds_df).
        If False: return df_filtered.
    flag_prefix : str
        Prefix for added flag columns in df_annotated.
    """
    if isinstance(features, str):
        features = [features]
    features = list(features)

    if not (0 <= lower_q <= 1 and 0 <= upper_q <= 1 and lower_q < upper_q):
        raise ValueError("Quantiles must satisfy 0 <= lower_q < upper_q <= 1.")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Feature(s) not found in df: {missing}")

    df_annot = df.copy()

    # Build per-feature flags and record bounds
    flags = pd.DataFrame(index=df_annot.index)
    bounds_rows = []

    for feat in features:
        s = df_annot[feat]
        lo = s.quantile(lower_q)
        hi = s.quantile(upper_q)

        if keep_na:
            flag = s.notna() & (s.lt(lo) | s.gt(hi))
        else:
            flag = s.isna() | s.lt(lo) | s.gt(hi)

        flags[feat] = flag
        df_annot[f"{flag_prefix}__{feat}"] = flag

        bounds_rows.append(
            {"feature": feat, "lower_q": lower_q, "upper_q": upper_q, "lower": lo, "upper": hi}
        )

    # Summary tagging (before filtering)
    df_annot[f"{flag_prefix}__any"] = flags.any(axis=1)
    df_annot[f"{flag_prefix}__features"] = flags.apply(
        lambda r: ",".join(r.index[r.values]),
        axis=1,
    )

    # Now filter (single pass)
    df_filtered = df_annot.loc[~df_annot[f"{flag_prefix}__any"]].copy()

    bounds_df = pd.DataFrame(bounds_rows).set_index("feature")

    print(f"Removed {len(df) - len(df_filtered)} out of {len(df)} objects.")

    if return_details:
        return df_filtered, df_annot, bounds_df
    return df_filtered



def filter_organoids_by(ad, df, feature, values, channel, pyramid_level=1, add_boundary = False,
                        keep_na = True):
    """
    Filter organoids based on a specified numerical feature range and visualize the removed organoids.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object.
    - df (pandas.DataFrame): Input DataFrame containing organoid information. Its index should correspond to organoid IDs.
    - feature (str): Name of the numerical feature to filter.
    - values (tuple): Tuple (lower_bound, upper_bound) for filtering bounds.
    - channel (str): Channel for image visualization.
    - pyramid_level (int): Pyramid level of images for visualization.
    - keep_na (bool): If True (default, matching filter_rows_by_percentile_bounds), objects whose
      value is NaN are kept. A NaN fails both bound tests, so the previous behaviour dropped
      them and counted them twice in the printout - and for a stain-specific feature, NaN means
      "this object's antibody mix does not include that stain", so dropping silently removed
      every object from the other panels.
    """
    lower_bound, upper_bound = values

    vals = df[feature]
    is_na = vals.isna()

    # Apply filter
    mask_lower = vals >= lower_bound
    mask_upper = vals <= upper_bound
    mask = mask_lower & mask_upper
    if keep_na:
        mask = mask | is_na
    df_filtered = df[mask]

    n_removed_lower = int((~is_na & ~mask_lower).sum())
    n_removed_upper = int((~is_na & ~mask_upper).sum())
    n_na = int(is_na.sum())
    print(f"{n_removed_lower} objects removed due to lower boundary ({lower_bound}) of {feature}.")
    print(f"{n_removed_upper} objects removed due to upper boundary ({upper_bound}) of {feature}.")
    if n_na:
        print(f"{n_na} objects have no value for {feature} and were "
              f"{'kept' if keep_na else 'removed'} (keep_na={keep_na}).")
    print(f"{len(df_filtered)} objects remain after filtering.")

    # Visualize removed due to lower boundary
    if n_removed_lower > 0:
        removed_lower = df[~is_na & ~mask_lower].index
        n_lower = len(removed_lower)
        rows_lower = 1 if n_lower < 9 else min(4, math.ceil(n_lower / 9))
        cols_lower = min(9, n_lower) if rows_lower == 1 else 9
        get_deleted_organoids(ad, df.loc[removed_lower], df_filtered, rows_lower, cols_lower,
                              f"Objects with {feature} < {lower_bound}", feature, channel, pyramid_level, add_boundary=add_boundary)

    # Visualize removed due to upper boundary
    if n_removed_upper > 0:
        removed_upper = df[~is_na & ~mask_upper].index
        n_upper = len(removed_upper)
        rows_upper = 1 if n_upper < 9 else min(4, math.ceil(n_upper / 9))
        cols_upper = min(9, n_upper) if rows_upper == 1 else 9
        get_deleted_organoids(ad, df.loc[removed_upper], df_filtered, rows_upper, cols_upper,
                              f"Objects with {feature} > {upper_bound}", feature, channel, pyramid_level, add_boundary=add_boundary)

    return df_filtered

def get_deleted_organoids(ad, df_removed, df_filtered, rows, cols, title, feature, channel, pyramid_level, add_boundary):
    """
    Plot organoids that are unique to one DataFrame; used to visualize organoids removed during filtering.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object.
    - df_removed (pandas.DataFrame): DataFrame of removed organoids to visualize.
    - df_filtered (pandas.DataFrame): DataFrame of kept organoids for comparison (may not be used directly).
    - rows (int): Number of rows in the plot grid.
    - cols (int): Number of columns in the plot grid.
    - title (str): Title of the plot.
    - feature (str): Feature name to display in subplot titles.
    - channel (str): Channel for image visualization.
    - pyramid_level (int): Pyramid level for visualization.
    """
    
    if df_removed.empty:
        return None

    n_display = min(rows * cols, len(df_removed))
    sample_removed = df_removed.sample(n=n_display)
    OIDs = sample_removed.index.tolist()
    feature_values = sample_removed[feature].tolist()

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(title, fontsize=18, y=1.01)

    # Flatten axes array for easy indexing even if rows=1 or cols=1
    import numpy as np
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = np.array(axes).flatten().tolist()

    for i in range(rows * cols):
        if i >= n_display:
            axes_flat[i].axis('off')
            continue
        OID = OIDs[i]
        try:
            img, mask = load_img_mask_by_UID(OID, ad.uns["stainings"], ad.uns["experiment_setup"], ad.uns["ome_zarr_dict"], ad.uns["table_name"],
                                            ad.uns["label_name"], pyramid_level, str(channel), add_boundary=add_boundary)
            img = img.copy()
            img[~mask.astype(bool)] = 0
            axes_flat[i].imshow(img, interpolation="nearest", aspect="auto", cmap="magma")
            axes_flat[i].set_title(f"{OID}\n{feature}: {feature_values[i]:.2f}", fontsize=8)
            axes_flat[i].axis('off')
        except Exception as e:
            axes_flat[i].text(0.5, 0.5, f"Failed to load\n{OID}", ha='center', va='center')
            axes_flat[i].axis('off')
            print(f"Warning: failed to load image for {OID}: {e}")

    plt.tight_layout()
    return fig

def plot_random_organoids(ad, df_raw, df, feature, rows=10, cols=10, channel="R0__DAPI", seed=0, pyramid_level=1, add_boundary=False):
    """
    Plot random organoids surviving after filtering from the DataFrame.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object.
    - df_raw (pandas.DataFrame): DataFrame before filtering.
    - df (pandas.DataFrame): DataFrame after filtering.
    - feature (str): Feature to display in subplot titles.
    - rows (int): Number of rows in subplot grid.
    - cols (int): Number of columns in subplot grid.
    - channel (str: Channel for image visualization.
    - seed (int): Random seed for reproducibility.
    - pyramid_level (int): Pyramid level for image loading.
    """

    n_available = len(df)
    n_requested = rows * cols

    print(f"A total of {len(df_raw) - n_available} objects have been removed during the filtering process. "
          f"{n_available} objects remain for further analysis.\n")

    # Adjust grid size if fewer organoids than requested
    if n_available == 0:
        print("No objects remain after filtering; nothing to plot.")
        return None

    if n_available < n_requested:
        print(f"Only {n_available} objects available but grid requires {n_requested}. Adjusting grid size accordingly.")
        n_to_plot = n_available
        # Compute new rows and cols to have a nearly square layout
        cols = max(1, min(cols, n_to_plot))
        rows = math.ceil(n_to_plot / cols)
    else:
        n_to_plot = n_requested

    random.seed(seed)
    sampled = df.sample(n=n_to_plot, replace=False, random_state=seed)
    removed_OID = list(sampled.index)
    removed_size = list(sampled[feature])

    fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle("Remaining Objects", fontsize=18, y=1.00)

    # Flatten axes array for easy iteration regardless of shape
    import numpy as np
    if rows == 1 and cols == 1:
        axes_flat = [ax]
    else:
        axes_flat = np.array(ax).flatten().tolist()

    for i in range(rows * cols):
        if i >= n_to_plot:
            # Turn off unused axes
            axes_flat[i].axis('off')
            continue
        OID = removed_OID[i]
        try:
            img, mask = load_img_mask_by_UID(OID, ad.uns["stainings"], ad.uns["experiment_setup"], ad.uns["ome_zarr_dict"], ad.uns["table_name"],
                                            ad.uns["label_name"], pyramid_level, str(channel), add_boundary=add_boundary)
            img = img.copy()
            img[mask == 0] = 0
            axes_flat[i].imshow(img, interpolation="nearest", aspect="auto", cmap="magma")
            axes_flat[i].set_title(f"{OID}\n{feature}: {round(removed_size[i], 2)}", fontsize=8)
            axes_flat[i].axis('off')
        except Exception as e:
            axes_flat[i].text(0.5, 0.5, f"Failed to load\n{OID}", ha='center', va='center')
            axes_flat[i].axis('off')
            print(f"Warning: failed to load image for {OID}: {e}")

    fig.tight_layout()
    return fig
