from __future__ import annotations

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from typing import Optional, Union
from pyslingshot import Slingshot
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import pandas as pd
import scanpy as sc
import numpy as np
import phenograph
import anndata

from Functions.Fcts_Base import save_fig, load_img_mask_by_UID, remove_uns

"""
***
TRAJECTORY FUNCTIONS
***
"""


def select_dimred_features(ad, remove_features):
    df = ad.to_df()
    common_feats = df.columns[~df.isna().any()]
    used_features = [x for x in common_feats if not any(rf in x for rf in remove_features)]
    return used_features


def compute_PCA(ad, plot_feat, number_components = 20, dimred_feats = None, representation = None, palette = "magma", save_plot = False):
    """
    Compute PCA on selected features of an AnnData object and visualize principal component scatter plots colored by specified features.

      Parameters:
        ad (anndata.AnnData): The AnnData object containing expression data and annotations.
        plot_feat (list of str): Features to use for coloring the scatter plots.
        number_components (int, optional): Number of PCA components to compute. Default is 20.
        dimred_feats (list of str or None, optional): Subset of features (from `ad.var_names`) to use as input for PCA. If `None`, all features from the selected data matrix are used.
        representation (str or None, optional):  Specifies which data matrix to use as PCA input:
            - If a valid key in `ad.layers`, that layer matrix will be used.
            - If a valid key in `ad.obsm`, that representation will be used.
            - If `None`, defaults to `ad.X`.
        palette (str, optional): Colormap to use for continuous features in scatter plots. Default is `"magma"`.
        save_plot (bool, optional): Whether to save the generated plots to the directory specified in `ad.uns['plot_dir']`. Default is `False`.
    """
    if isinstance(plot_feat, str):
        plot_feat = [plot_feat]

    # Select data matrix from layers, obsm, or X
    if representation is not None:
        if representation in ad.layers:
            data_all = ad.layers[representation]
            layer_for_pca = representation
            obsm_for_pca = None
        elif representation in ad.obsm:
            data_all = ad.obsm[representation]
            layer_for_pca = None
            obsm_for_pca = representation
        else:
            raise ValueError(f"Layer '{representation}' not found in ad.layers or ad.obsm.")
    else:
        data_all = ad.X
        layer_for_pca = None
        obsm_for_pca = None

    # Subset features used for PCA input if specified (only possible if using var_names-aligned matrix)
    if dimred_feats is not None:
        missing_feats = [f for f in dimred_feats if f not in ad.var_names]
        if missing_feats:
            raise ValueError(f"Some features in dimred_feats not found: {missing_feats}")
        feat_indices = [ad.var_names.get_loc(f) for f in dimred_feats]
        data = data_all[:, feat_indices]
        # Make a temporary view with the selected features to let sc.pp.pca handle bookkeeping
        ad_pca = ad[:, dimred_feats].copy()
    else:
        data = data_all
        ad_pca = ad  # run PCA on the whole object

    # Guard against NaNs in the matrix Scanpy will read
    if hasattr(data, "toarray"):
        # sparse or array-like
        arr_chk = data.toarray() if hasattr(data, "toarray") else np.asarray(data)
    else:
        arr_chk = np.asarray(data)
    if np.isnan(arr_chk).any():
        raise ValueError("Input data contains NaNs. Please preprocess or filter before PCA.")

    # Run Scanpy PCA. Prefer passing layer if representation selects a layer; otherwise PCA uses X or an obsm is unsupported by pp.pca.
    # Note: scanpy.pp.pca does not accept arbitrary obsm input; if representation refers to obsm, rely on ad.obsm being precomputed finite.
    sc.pp.pca(
        ad_pca,
        n_comps=number_components,
        layer=layer_for_pca,          # only used when representation is a layer or None
        random_state=0,
        copy=False,
    )

    # If PCA ran on a sliced copy, copy results back to the original AnnData
    # Embed PCs for all cells; when dimred_feats was used, PCs are based on those genes
    ad.obsm["X_pca"] = ad_pca.obsm["X_pca"]
    # Store Scanpy-standard metadata for explained variance
    ad.uns["pca"] = ad_pca.uns.get("pca", {})
    # Store loadings aligned to ad.var_names; if subset was used, place into full varm with NaN for non-selected genes
    if dimred_feats is None:
        ad.varm["PCs"] = ad_pca.varm["PCs"]
    else:
        # Initialize full matrix with NaNs then fill selected rows
        comps = ad_pca.varm["PCs"].shape[1]
        PCs_full = np.full((ad.n_vars, comps), np.nan, dtype=np.float32)
        sel_idx = [ad.var_names.get_loc(f) for f in dimred_feats]
        PCs_full[sel_idx, :] = ad_pca.varm["PCs"]
        ad.varm["PCs"] = PCs_full

    # Pull arrays for plotting
    pcs = ad.obsm["X_pca"]
    var_ratio = np.asarray(ad.uns["pca"].get("variance_ratio", []))
    total_var = float(np.nansum(var_ratio) * 100.0) if var_ratio.size else 0.0

    n_pc_pairs = number_components // 2
    n_feats = len(plot_feat)

    fig, ax = plt.subplots(
        nrows=n_pc_pairs, ncols=n_feats,
        figsize=(4 * n_feats, 3.5 * n_pc_pairs),
        gridspec_kw={"hspace": 0.4, "wspace": 0.5},
        squeeze=False
    )
    fig.subplots_adjust(top=0.90)

    for col, feat in enumerate(plot_feat):
        iterator = 1
        for row in range(n_pc_pairs):
            axis = ax[row, col]

            if feat in ad.obs.columns:
                hue_data = ad.obs[feat]
                n_cat = hue_data.nunique() if hasattr(hue_data, "nunique") else None
                palette_used = "tab10" if n_cat is not None and n_cat <= 10 else "Set2"
                sns.scatterplot(
                    ax=axis,
                    x=pcs[:, iterator - 1],
                    y=pcs[:, iterator],
                    hue=hue_data,
                    s=20,
                    alpha=0.7,
                    palette=palette_used,
                    legend='brief' if (row == 0) else False,
                    edgecolor='none'
                )
            elif feat in ad.var_names:
                gene_exp = ad[:, feat].X
                gene_exp = gene_exp.toarray().flatten() if hasattr(gene_exp, "toarray") else np.array(gene_exp).flatten()
                scatter = axis.scatter(
                    pcs[:, iterator - 1], pcs[:, iterator],
                    c=gene_exp, cmap=palette, s=20, alpha=0.7, edgecolor='none'
                )
                if row == 0:
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(scatter, cax=cax, orientation="vertical")
                    cbar.ax.tick_params(labelsize=8)
            else:
                axis.text(0.5, 0.5, f"Feature '{feat}' not found", ha="center", va="center", fontsize=10)
                axis.set_axis_off()
                iterator += 2
                continue

            # Axis labeling with scanpy's explained variance
            if var_ratio.size >= iterator + 0:
                x_var = float(var_ratio[iterator - 1] * 100.0)
            else:
                x_var = np.nan
            if var_ratio.size >= iterator + 1:
                y_var = float(var_ratio[iterator] * 100.0)
            else:
                y_var = np.nan

            axis.set_xlabel(f"PC{iterator}\nExplained var: {x_var:.2f}%", fontsize=9)
            axis.set_ylabel(f"PC{iterator+1}\nExplained var: {y_var:.2f}%", fontsize=9)
            axis.tick_params(left=True, bottom=True, labelsize=8)
            axis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

            if row == 0:
                axis.set_title(feat, fontsize=12, y=1.03)
            else:
                axis.set_title("")

            axis.set_xticklabels([])
            axis.set_yticklabels([])

            iterator += 2

    for col in range(n_feats):
        for row in range(n_pc_pairs):
            if (row + 1) * 2 > number_components:
                ax[row, col].set_visible(False)

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "PCA", dpi=300)

    # PCA loadings heatmap from scanpy varm['PCs']
    comps = [f"PC{i + 1}" for i in range(number_components)]
    if "PCs" in ad.varm and ad.varm["PCs"] is not None:
        pca_loadings = pd.DataFrame(ad.varm["PCs"][:, :number_components], index=ad.var_names, columns=comps)
        fig, ax = plt.subplots(1, 1, figsize=(6, 15))
        sns.heatmap(pca_loadings, ax=ax, cmap="RdBu_r", center=0, robust=True, cbar=True, cbar_kws={'shrink': 0.5})
        ax.set_title("PCA Loadings", fontsize=16, y=1.02)
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis='x', labelrotation=45, labelsize=10)
        if save_plot:
            save_fig(fig, ad.uns["plot_dir"], "PCA_Heatmap", dpi=300)

    return ad

# (rest of file unchanged)
