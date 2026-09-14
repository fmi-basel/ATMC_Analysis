import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse
import pandas as pd
import numpy as np

"""
***
QUALITY CONTROL FUNCTIONS
***
"""


def calculate_outgrowth(ad, n_seeded):
    """
    Calculate outgrowth efficiency for each well.

    Parameters:
    - ad (anndata.AnnData): AnnData object containing organoid data.
    - n_seeded (int): Number of cells seeded per well (assumed constant).
    """
    experiment_setup = ad.uns["experiment_setup"]
    df = pd.concat([ad.to_df(), ad.obs.astype(str)], axis=1)
    group_cols = ["Other", "Cell_line", "Medium", "Well", "Barcode"]

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


def normalize_groups(adata, group_by, control=None):
    """
    Normalize data in an AnnData object by groups, applying log1p transformation
    to area-associated features, and then creating separate layers for robust scaling,
    z-normalization and min-max scaling.

    Parameters:
    adata (anndata.AnnData): The AnnData object containing the data to be normalized.
    group_by (str or list of str): The column name(s) in adata.obs to group the data by.
    control (dict, optional): A dictionary specifying the control group for normalization.
        The key is the column name, and the value is the control group value.
        If None, each group is normalized independently.

    Notes:
    - Cells with NaN in a feature are ignored when fitting scalers for that feature,
      but are NOT dropped. Each feature is scaled independently using its own
      non-NaN control (or group) cells.
    - NaN values in the output layers remain NaN.
    - A ValueError is raised if no control cells are found in a group, or if any
      feature has zero non-NaN control cells.
    """

    if isinstance(group_by, str):
        group_by = [group_by]

    # Fix pandas grouping observed warning by specifying observed=True
    groups = adata.obs.groupby(group_by, observed=True).groups

    # Handle sparse matrix: convert to dense if needed
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    lower_var_names = [v.lower() for v in adata.var_names]
    to_transform = [
        adata.var_names[i] for i, v in enumerate(lower_var_names)
        if any(key in v for key in ["area", "moments", "potency", "substructures_size"])
        and "ratio" not in v
    ]

    # Log1p transform (clip non-negative values)
    df_log = df.copy()
    df_log[to_transform] = np.log1p(df_log[to_transform].clip(lower=0))

    adata.layers["log1p"] = df_log.values

    robust_normalized_data = np.full_like(df_log.values, np.nan, dtype=np.float64)
    z_normalized_data      = np.full_like(df_log.values, np.nan, dtype=np.float64)
    minmax_normalized_data = np.full_like(df_log.values, np.nan, dtype=np.float64)

    if control is not None:
        if len(control) != 1:
            raise ValueError("Control dictionary must have exactly one key-value pair")
        control_key, control_value = list(control.items())[0]

    for group, indices in groups.items():
        int_indices = adata.obs_names.get_indexer(indices)
        group_data = adata.layers["log1p"][int_indices]
        n_features = group_data.shape[1]

        # Determine which cells to fit scalers on (NaNs handled per-feature below)
        if control is None:
            fit_data = group_data
        else:
            subset_obs = adata.obs.loc[indices]
            control_mask = (subset_obs[control_key] == control_value).values
            fit_data = group_data[control_mask]

            if fit_data.shape[0] == 0:
                raise ValueError(
                    f"No control cells ('{control_value}') found in group '{group}'. "
                    f"Cannot fit scalers."
                )

            # Raise if any feature has zero non-NaN control cells
            n_valid_per_feature = (~np.isnan(fit_data)).sum(axis=0)
            missing_features = [
                adata.var_names[i] for i, n in enumerate(n_valid_per_feature) if n == 0
            ]
            if missing_features:
                raise ValueError(
                    f"Group '{group}': the following features have NO non-NaN control "
                    f"cells and cannot be normalized:\n{missing_features}"
                )

        # --- Per-feature scaler fitting (NaN-aware, no row dropping) ---
        robust_params = []   # (center, scale)  per feature
        z_params      = []   # (mean,   std)     per feature
        minmax_params = []   # (min,    range)   per feature

        for j in range(n_features):
            col = fit_data[:, j]
            valid = col[~np.isnan(col)]

            if valid.shape[0] == 0:
                # No non-NaN values in this feature for this group — keep NaN in output
                robust_params.append((np.nan, 1.0))
                z_params.append((np.nan, 1.0))
                minmax_params.append((np.nan, 1.0))
                continue

            # RobustScaler: center = median, scale = IQR
            center = np.median(valid)
            q75, q25 = np.percentile(valid, [75, 25])
            iqr = q75 - q25
            robust_params.append((center, iqr if iqr != 0 else 1.0))

            # StandardScaler: mean, std
            mu    = valid.mean()
            sigma = valid.std()
            z_params.append((mu, sigma if sigma != 0 else 1.0))

            # MinMaxScaler: min, range
            vmin, vmax = valid.min(), valid.max()
            vrange = vmax - vmin
            minmax_params.append((vmin, vrange if vrange != 0 else 1.0))

        # --- Transform all group cells per-feature, preserving NaNs ---
        robust_scaled = np.full_like(group_data, np.nan, dtype=np.float64)
        z_scaled      = np.full_like(group_data, np.nan, dtype=np.float64)
        minmax_scaled = np.full_like(group_data, np.nan, dtype=np.float64)

        for j in range(n_features):
            col        = group_data[:, j]
            valid_mask = ~np.isnan(col)

            r_center, r_scale  = robust_params[j]
            z_mu,     z_sigma  = z_params[j]
            mm_min,   mm_range = minmax_params[j]

            if np.isnan(r_center):
                continue  # feature had no valid fit data — leave as NaN

            robust_scaled[valid_mask, j] = (col[valid_mask] - r_center) / r_scale
            z_scaled[valid_mask, j]      = (col[valid_mask] - z_mu)     / z_sigma
            minmax_scaled[valid_mask, j] = (col[valid_mask] - mm_min)   / mm_range

        robust_normalized_data[int_indices] = robust_scaled
        z_normalized_data[int_indices]      = z_scaled
        minmax_normalized_data[int_indices] = minmax_scaled

    adata.layers["robust_scaled"] = robust_normalized_data
    adata.layers["z_scaled"]      = z_normalized_data
    adata.layers["minmax_scaled"] = minmax_normalized_data

    if control is not None:
        check_control_normalization(adata, control=control, group_by=group_by)
    return adata


def check_control_normalization(adata, control, group_by, tol=0.5):
    """
    Sanity check: for the control group, 
    - z_scaled features should have mean ≈ 0
    - robust_scaled features should have median ≈ 0
    
    Parameters:
        adata: AnnData object after normalization
        control: dict, e.g. {"Cell_line": "WT_11"}
        group_by: str or list of str — same as used in normalize_groups
        tol: tolerance threshold to flag deviations (default 0.5)
    """
    if isinstance(group_by, str):
        group_by = [group_by]

    control_key, control_value = list(control.items())[0]

    results = {}
    for layer, metric_name, metric_fn in [
        ("z_scaled",      "mean",   lambda x: np.nanmean(x, axis=0)),
        ("robust_scaled", "median", lambda x: np.nanmedian(x, axis=0)),
    ]:
        # Check per group_by group
        for group, indices in adata.obs.groupby(group_by, observed=True).groups.items():
            int_idx = adata.obs_names.get_indexer(indices)
            group_obs = adata.obs.loc[indices]
            ctrl_within_group = group_obs[control_key] == control_value
            ctrl_int_idx = int_idx[ctrl_within_group.values]

            if len(ctrl_int_idx) == 0:
                print(f"[{layer}] Group '{group}': no control samples found, skipping.")
                continue

            data = adata.layers[layer][ctrl_int_idx]
            stat = metric_fn(data)
            max_dev = np.nanmax(np.abs(stat))
            status = "✅ OK" if max_dev < tol else "⚠️  WARNING"

            print(f"[{layer}] Group '{group}' | control '{control_value}' | "
                  f"max |{metric_name}| = {max_dev:.4f}  {status}")

            results[(layer, str(group))] = {"metric": metric_name, "max_deviation": max_dev}

    return results


def build_heatmap_df(plate_size):
    """
    Build an empty dataframe representing the plate layout for heatmap visualization.

    Parameters:
    - plate_size (int): Plate size, either 96 or 384.
    """
    if plate_size == 384:
        rows = list("ABCDEFGHIJKLMNOP")
        cols = [f"{i:02d}" for i in range(1, 25)]
    elif plate_size == 96:
        rows = list("ABCDEFGH")
        cols = [f"{i:02d}" for i in range(1, 13)]
    else:
        raise ValueError("Unsupported plate size. Choose 96 or 384.")
    return pd.DataFrame(np.nan, index=rows, columns=cols)


def plot_heatmap_with_means(data_df, title, ax=None):
    """
    Plot plate layout heatmap with extra row and column showing means per row and per column.
    Adds a closed rectangular border around the entire heatmap.

    Parameters:
    - data_df: pd.DataFrame formatted with rows and columns matching plate layout
    - title: str, title for the heatmap
    - ax: matplotlib Axes or None
    """
    # Calculate means ignoring NaNs
    row_means = data_df.mean(axis=1)
    col_means = data_df.mean(axis=0)

    # Append 'RowMean' column
    data_df = data_df.copy()
    data_df["Row Avg"] = row_means

    # Append 'ColMean' row (a DataFrame with one row)
    col_means_df = pd.DataFrame([list(col_means) + [np.nan]], index=["Column Avg"], columns=data_df.columns)
    data_df_aug = pd.concat([data_df, col_means_df])

    # Plot heatmap
    if ax is None:
        plt.figure(figsize=(12, 6))
        ax = plt.gca()

    sns.heatmap(
        data_df_aug,
        cmap="viridis",
        square=True,
        linewidth=0.5,
        linecolor="gray",
        cbar_kws={"label": title},
        ax=ax,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 5},
    )

    ax.xaxis.set_ticks_position("top")
    ax.tick_params(left=False, top=False)
    ax.set_xticklabels([str(lbl) for lbl in data_df_aug.columns], rotation=45, ha="left")
    ax.set_yticklabels([str(lbl) for lbl in data_df_aug.index], rotation=0)

    # Draw closed border lines to “close” wells on right and bottom sides
    n_rows, n_cols = data_df_aug.shape
    ax.axhline(0, color='black', linewidth=1.5)           # top border
    ax.axhline(n_rows, color='black', linewidth=1.5)     # bottom border
    ax.axvline(0, color='black', linewidth=1.5)           # left border
    ax.axvline(n_cols, color='black', linewidth=1.5)      # right border

    ax.set_title(title, fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()


def plate_bias_overview(
    plt_features,
    ad,
    plate_size,
    control_condition=None,
    control_only=False,
):
    """
    Generate plate-layout heatmaps per feature and day to visualize spatial bias.
    Optionally filter to only plot wells matching a specified control condition.

    Parameters:
    - plt_features: list of feature column names to visualize (mean values)
    - ad: AnnData object with organoid data
    - plate_size: int, 96 or 384
    - control_condition: dict, optional, e.g. {"Medium": "Ctrl"}
        Specifies condition(s) to filter wells for plotting.
    - control_only: bool, default False
        If True, only wells matching control_condition are plotted.
    """
    def passes_control_condition(obs_row, condition):
        # Check if the observation row satisfies all key-value pairs in condition dict
        return all(obs_row.get(k, None) == v for k, v in condition.items())

    # Fail on a misspelled feature instead of silently drawing an all-NaN plate.
    unknown = [f for f in plt_features if f not in ad.var_names]
    if unknown:
        raise ValueError(f"Feature(s) not found in ad.var_names: {unknown}")

    # ad.to_df() materialises the whole feature matrix; build it once rather than once per
    # (feature, day) pair.
    df_all = ad.to_df()
    obs_str = ad.obs.astype(str)

    for feat in plt_features:
        for day in ad.obs["Other"].unique():

            # Filter observations by day first
            obs_filtered = ad.obs[ad.obs["Other"] == day]

            # If filtering only control wells, keep only those rows satisfying control_condition
            if control_only:
                if not control_condition:
                    raise ValueError("control_condition must be specified if control_only=True")
                mask = obs_filtered.apply(lambda row: passes_control_condition(row, control_condition), axis=1)
                obs_filtered = obs_filtered[mask]

            # Combine the one needed feature column with the metadata
            df = pd.concat([df_all.loc[obs_filtered.index, [feat]],
                            obs_str.loc[obs_filtered.index]], axis=1)

            # One heatmap per physical plate (Barcode): a Well label alone does not
            # uniquely identify a well across plates, so pooling wells from different
            # barcodes into one heatmap would silently mix or overwrite unrelated data.
            for barcode in df.Barcode.unique():
                df_HM = build_heatmap_df(plate_size)
                if df_HM is None:
                    continue

                df_bc = df[df.Barcode == barcode]

                for cellline in df_bc.Cell_line.unique():
                    for medium in df_bc.Medium.unique():
                        df_plt = df_bc[(df_bc.Cell_line == cellline) & (df_bc.Medium == medium)].copy()
                        if df_plt.empty or feat not in df_plt.columns:
                            continue

                        # Compute mean of feature per well (no z-scoring)
                        grouped = df_plt.groupby("Well")[feat].mean()

                        # Fill heatmap DataFrame
                        for well, val in grouped.items():
                            row = well[0]
                            col = well[1:].lstrip("0").zfill(2)
                            if row in df_HM.index and col in df_HM.columns:
                                df_HM.loc[row, col] = val

                plot_heatmap_with_means(df_HM, title=f"{barcode} {day} {feat}")

    # Similarly for Outgrowth plotting, apply the same filtering
    df_out = ad.uns.get("Outgrowth_DF")
    if df_out is None:
        print("Outgrowth_DF not found in ad.uns")
        return

    for day in ad.obs["Other"].unique():

        df_out_day = df_out[df_out.Other == str(day)]

        if control_only:
            if not control_condition:
                raise ValueError("control_condition must be specified if control_only=True")
            # Filter outgrowth DF by control_condition keys and values
            for k, v in control_condition.items():
                df_out_day = df_out_day[df_out_day[k] == v]

        # One heatmap per physical plate (Barcode) - see comment above for why.
        for barcode in df_out_day.Barcode.unique():
            df_HM = build_heatmap_df(plate_size)
            df_out_bc = df_out_day[df_out_day.Barcode == barcode]

            for cellline in df_out_bc.Cell_line.unique():

                for medium in df_out_bc.Medium.unique():
                    df_plt = df_out_bc[(df_out_bc.Cell_line == cellline) & (df_out_bc.Medium == medium)].copy()

                    if df_plt.empty:
                        continue

                    grouped = df_plt.groupby("Well")["Organoid_No"].mean()

                    for well, val in grouped.items():
                        row = well[0]
                        col = well[1:].lstrip("0").zfill(2)
                        if row in df_HM.index and col in df_HM.columns:
                            df_HM.loc[row, col] = val

            plot_heatmap_with_means(df_HM, title=f"{barcode} {day} Outgrowth")
