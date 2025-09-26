import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing
import scipy.sparse
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

from Fcts_Base import save_fig

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
    import pandas as pd

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
                new_row = {
                    "Other": setup[3],
                    "Cell_line": setup[2],
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

def build_heatmap_df(plate_size):
    """
    Build an empty DataFrame for heatmap plotting based on the plate size.

    Parameters:
    - plate_size (int): Size of the plate (96 or 384).
    """ 

    if plate_size == 384:
        index_lst = ["A", "B" , "C", "D", "E", "F", "G", "H", "I", "J", "K" , "L" , "M" , "N" , "O", "P"]
        col_lst = [str(x) for x in range(1,25)]
        for i,el in enumerate(col_lst):
            if len(el) == 1:
                col_lst[i] = str(0)+el
                
    if plate_size == 96:
        index_lst = ["A", "B" , "C", "D", "E", "F", "G", "H"]
        col_lst = [str(x) for x in range(1,13)]
        for i,el in enumerate(col_lst):
            if len(el) == 1:
                col_lst[i] = str(0)+el
        
    df_plot_HM = pd.DataFrame(np.nan, index = index_lst, columns = col_lst)

    if (plate_size != 96) and (plate_size != 384):
        print("plate size not configured.")

    return df_plot_HM

def plate_bias_overview(plt_features, ad, plate_size):
    """
    Generate heatmaps illustrating plate bias based on specified features and experimental conditions.

    Parameters:
    - plt_features (list): List of features for heatmap plotting.
    - ad (anndata.AnnData): AnnData object containing organoid data.
    - plate_size (int): Size of the plate (96 or 384).
    """   
    
    for feat in plt_features:   
        
        for day in ad.obs["Other"].unique():

            fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize =(10,5))
            
            # Built empty DF
            df_HM = build_heatmap_df(plate_size)
        
            # Go through conds and use compute minmax scale
            df = pd.concat([ad[ad.obs.Other == day].to_df(), ad[ad.obs.Other == day].obs.astype(str)], axis = 1)
            
            for cellline in df.Cell_line.unique():
                
                for medium in df.Medium.unique():   

                    # Take values and filter
                    df_plt = df[(df.Cell_line == cellline) & (df.Medium == medium)].copy(deep = True)

                    # z-scoring  
                    mean = np.mean(df_plt[feat])
                    sd = np.std(df_plt[feat], axis = 0)
                    df_plt.loc[:,"plt"] = abs(df_plt[feat].transform(lambda x : (x - mean)/sd))
                    
                    # GroupBy
                    grouped = df_plt.groupby(["Well"])["plt"].mean().to_frame() 

                    # Put into heatmap based on well
                    for well in grouped.index:
                        df_HM.loc[well[0], well[1:]] = grouped.loc[well]["plt"]
            
            
            # Plot
            f1 = sns.heatmap(data= df_HM,
                    linewidth = 1,
                    square = True,
                    cmap='RdBu_r',
                    robust = False)

            f1.xaxis.set_ticks_position("top")
            f1.tick_params(left=False, top=False)

            ax.set_title(day+" "+feat, fontsize = 18, y = 1.05)

            fig.tight_layout()
    
        
    for day in ad.obs["Other"].unique():

        fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize =(10,5))

        # Built empty DF
        df_HM = build_heatmap_df(plate_size)

        # Go through conds and use compute minmax scale
        df = ad.uns["Outgrowth_DF"]
        df = df[df.Other == day]

        for cellline in df.Cell_line.unique():

            for medium in df.Medium.unique():   

                # Take values and filter
                df_plt = df[(df.Cell_line == cellline) & (df.Medium == medium)].copy(deep = True)

                # z-scoring  
                df_plt.loc[:,"plt"] = abs(df_plt["Organoid_No"].transform(lambda x : (x - np.mean(df_plt[["Organoid_No"]]))/np.std(df_plt[["Organoid_No"]], axis = 0)))

                # Group and compute medians
                grouped = df_plt.groupby(["Well"])["plt"].mean().to_frame() 


                # Put into heatmap based on well
                for well in grouped.index:
                    df_HM.loc[well[0], well[1:]] = grouped.loc[well]["plt"]
                
        # Plot outgrowth
        f1 = sns.heatmap(data= df_HM,
                linewidth = 1,
                square = True,
                cmap='RdBu_r',
                robust = False)

        f1.xaxis.set_ticks_position("top")
        f1.tick_params(left=False, top=False)

        ax.set_title(day+" Outgrowth", fontsize = 18, y = 1.05)

        fig.tight_layout()

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
        if any(key in v for key in ["area", "moments", "potency"]) and "ratio" not in v
    ]

    # Log1p transform (clip non-negative values)
    df_log = df.copy()
    df_log[to_transform] = np.log1p(df_log[to_transform].clip(lower=0))

    adata.layers["log1p"] = df_log.values

    robust_normalized_data = np.full_like(df_log.values, np.nan, dtype=np.float64)
    z_normalized_data = np.full_like(df_log.values, np.nan, dtype=np.float64)
    minmax_normalized_data = np.full_like(df_log.values, np.nan, dtype=np.float64)

    if control is not None:
        if len(control) != 1:
            raise ValueError("Control dictionary must have exactly one key-value pair")
        control_key, control_value = list(control.items())[0]

    for group, indices in groups.items():
        int_indices = adata.obs_names.get_indexer(indices)
        group_data = adata.layers["log1p"][int_indices]

        # Create mask of NaNs to preserve after scaling
        nan_mask = np.isnan(group_data)

        # Determine data to fit scalers on
        if control is None:
            fit_data = group_data[~nan_mask.any(axis=1)]  # rows without NaNs
            if fit_data.shape[0] == 0:
                # fallback: fill NaNs temporarily with col mean for fitting
                fill_data = np.nan_to_num(group_data, nan=np.nanmean(np.nan_to_num(group_data, nan=0)))
                fit_data = fill_data
        else:
            subset_obs = adata.obs.loc[indices]
            control_mask = subset_obs[control_key] == control_value
            control_indices_within_group = np.where(control_mask)[0]
            control_data = group_data[control_indices_within_group]
            control_data_no_nan = control_data[~np.isnan(control_data).any(axis=1)]
            if control_data_no_nan.shape[0] == 0:
                fit_data = group_data[~nan_mask.any(axis=1)]
                if fit_data.shape[0] == 0:
                    fill_data = np.nan_to_num(group_data, nan=np.nanmean(np.nan_to_num(group_data, nan=0)))
                    fit_data = fill_data
            else:
                fit_data = control_data_no_nan

        # Fit scalers
        robust_scaler = RobustScaler().fit(fit_data)
        z_scaler = StandardScaler().fit(fit_data)
        minmax_scaler = MinMaxScaler().fit(fit_data)

        # Temporarily impute NaNs for transformation (column medians)
        group_data_filled = group_data.copy()
        col_medians = np.nanmedian(group_data_filled, axis=0)
        inds_nan = np.where(np.isnan(group_data_filled))
        group_data_filled[inds_nan] = np.take(col_medians, inds_nan[1])

        robust_scaled = robust_scaler.transform(group_data_filled)
        z_scaled = z_scaler.transform(group_data_filled)
        minmax_scaled = minmax_scaler.transform(group_data_filled)

        # Reapply NaNs to original positions
        robust_scaled[nan_mask] = np.nan
        z_scaled[nan_mask] = np.nan
        minmax_scaled[nan_mask] = np.nan

        robust_normalized_data[int_indices] = robust_scaled
        z_normalized_data[int_indices] = z_scaled
        minmax_normalized_data[int_indices] = minmax_scaled

    adata.layers["robust_scaled"] = robust_normalized_data
    adata.layers["z_scaled"] = z_normalized_data
    adata.layers["minmax_scaled"] = minmax_normalized_data

    return adata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

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

    for feat in plt_features:
        for day in ad.obs["Other"].unique():

            df_HM = build_heatmap_df(plate_size)
            if df_HM is None:
                continue

            # Filter observations by day first
            obs_filtered = ad.obs[ad.obs["Other"] == day]

            # If filtering only control wells, keep only those rows satisfying control_condition
            if control_only:
                if not control_condition:
                    raise ValueError("control_condition must be specified if control_only=True")
                mask = obs_filtered.apply(lambda row: passes_control_condition(row, control_condition), axis=1)
                obs_filtered = obs_filtered[mask]

            # Get the expression/dataframe for filtered observations
            df_data = ad.to_df().loc[obs_filtered.index]

            # Combine data and obs as before
            df = pd.concat([df_data, obs_filtered.astype(str)], axis=1)

            for cellline in df.Cell_line.unique():
                for medium in df.Medium.unique():
                    df_plt = df[(df.Cell_line == cellline) & (df.Medium == medium)].copy()
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

            plot_heatmap_with_means(df_HM, title=f"{day} {feat}")

    # Similarly for Outgrowth plotting, apply the same filtering
    for day in ad.obs["Other"].unique():
        df_HM = build_heatmap_df(plate_size)
        df_out = ad.uns.get("Outgrowth_DF")
        if df_out is None:
            print("Outgrowth_DF not found in ad.uns")
            return
        df_out_day = df_out[df_out.Other == day]

        if control_only:
            if not control_condition:
                raise ValueError("control_condition must be specified if control_only=True")
            # Filter outgrowth DF by control_condition keys and values
            for k, v in control_condition.items():
                df_out_day = df_out_day[df_out_day[k] == v]

        for cellline in df_out_day.Cell_line.unique():
            for medium in df_out_day.Medium.unique():
                df_plt = df_out_day[(df_out_day.Cell_line == cellline) & (df_out_day.Medium == medium)].copy()
                if df_plt.empty:
                    continue

                grouped = df_plt.groupby("Well")["Organoid_No"].mean()

                for well, val in grouped.items():
                    row = well[0]
                    col = well[1:].lstrip("0").zfill(2)
                    if row in df_HM.index and col in df_HM.columns:
                        df_HM.loc[row, col] = val

        plot_heatmap_with_means(df_HM, title=f"{day} Outgrowth")