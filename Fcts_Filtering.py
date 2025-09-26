import matplotlib.pyplot as plt
import random
import math
from Fcts_Base import load_img_mask_by_UID

"""
***
FILTERING FUNCTIONS
***
"""

def filter_organoids_by(ad, df, feature, values, channel, pyramid_level=1):
    """
    Filter organoids based on a specified numerical feature range and visualize the removed organoids.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object.
    - df (pandas.DataFrame): Input DataFrame containing organoid information. Its index should correspond to organoid IDs.
    - feature (str): Name of the numerical feature to filter.
    - values (tuple): Tuple (lower_bound, upper_bound) for filtering bounds.
    - channel (int): Channel index for image visualization.
    - pyramid_level (int): Pyramid level of images for visualization.
    """
    lower_bound, upper_bound = values

    # Apply filter
    mask_lower = df[feature] >= lower_bound
    mask_upper = df[feature] <= upper_bound
    mask = mask_lower & mask_upper
    df_filtered = df[mask]

    n_removed_lower = (~mask_lower).sum()
    n_removed_upper = mask_lower.sum() - mask.sum()
    print(f"{n_removed_lower} objects removed due to lower boundary ({lower_bound}) of {feature}.")
    print(f"{n_removed_upper} objects removed due to upper boundary ({upper_bound}) of {feature}.")
    print(f"{len(df_filtered)} objects remain after filtering.")

    # Visualize removed due to lower boundary
    if n_removed_lower > 0:
        removed_lower = df.index.difference(df_filtered.index)
        n_lower = len(removed_lower)
        rows_lower = 1 if n_lower < 9 else min(4, math.ceil(n_lower / 9))
        cols_lower = min(9, n_lower) if rows_lower == 1 else 9
        fig1 = get_deleted_organoids(ad, df.loc[removed_lower], df_filtered, rows_lower, cols_lower,
                                    f"Objects with {feature} < {lower_bound}", feature, channel, pyramid_level)
    else:
        fig1 = None

    # Visualize removed due to upper boundary
    if n_removed_upper > 0:
        removed_upper = df_filtered.index.difference(df.index[mask_lower])
        n_upper = len(removed_upper)
        rows_upper = 1 if n_upper < 9 else min(4, math.ceil(n_upper / 9))
        cols_upper = min(9, n_upper) if rows_upper == 1 else 9
        fig2 = get_deleted_organoids(ad, df.loc[removed_upper], df_filtered, rows_upper, cols_upper,
                                    f"Objects with {feature} > {upper_bound}", feature, channel, pyramid_level)
    else:
        fig2 = None

    return df_filtered

def get_deleted_organoids(ad, df_removed, df_filtered, rows, cols, title, feature, channel, pyramid_level):
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
    - channel (int): Channel index for image visualization.
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
    if rows == 1 and cols == 1:
        axes_flat = [axes]
    elif rows == 1 or cols == 1:
        axes_flat = axes.flatten() if hasattr(axes, 'flatten') else axes
    else:
        axes_flat = axes.flatten()

    for i in range(rows * cols):
        if i >= n_display:
            axes_flat[i].axis('off')
            continue
        OID = OIDs[i]
        try:
            img, mask = load_img_mask_by_UID(OID, ad.uns["ome_zarr_dict"], ad.uns["table_name"],
                                            ad.uns["label_name"], pyramid_level, channel)
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

def plot_random_organoids(ad, df_raw, df, feature, rows=10, cols=10, channel=0, seed=0, pyramid_level=1):
    """
    Plot random organoids surviving after filtering from the DataFrame.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object.
    - df_raw (pandas.DataFrame): DataFrame before filtering.
    - df (pandas.DataFrame): DataFrame after filtering.
    - feature (str): Feature to display in subplot titles.
    - rows (int): Number of rows in subplot grid.
    - cols (int): Number of columns in subplot grid.
    - seed (int): Random seed for reproducibility.
    - pyramid_level (int): Pyramid level for image loading.
    """

    n_available = len(df)
    n_requested = rows * cols

    print(f"A total of {len(df_raw) - n_available} objects have been removed during the filtering process. "
          f"{n_available} objects remain for further analysis.\n")

    # Adjust grid size if fewer organoids than requested
    if n_available < n_requested:
        print(f"Only {n_available} objects available but grid requires {n_requested}. Adjusting grid size accordingly.")
        n_to_plot = n_available
        # Compute new rows and cols to have a nearly square layout
        cols = min(cols, n_to_plot)
        rows = math.ceil(n_to_plot / cols)
    else:
        n_to_plot = n_requested

    random.seed(seed)
    sampled = df.sample(n=n_to_plot, replace=False, random_state=seed)
    removed_OID = list(sampled.index)
    removed_size = list(sampled[feature])

    fig, ax = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    fig.suptitle("Surviving Organoids", fontsize=18, y=1.00)

    # Flatten axes array for easy iteration regardless of shape
    if rows == 1 and cols == 1:
        axes_flat = [ax]
    elif rows == 1 or cols == 1:
        axes_flat = ax.flatten() if hasattr(ax, 'flatten') else ax
    else:
        axes_flat = ax.flatten()

    for i in range(rows * cols):
        if i >= n_to_plot:
            # Turn off unused axes
            axes_flat[i].axis('off')
            continue
        OID = removed_OID[i]
        try:
            img, mask = load_img_mask_by_UID(OID, ad.uns["ome_zarr_dict"], ad.uns["table_name"],
                                            ad.uns["label_name"], pyramid_level, channel)
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
