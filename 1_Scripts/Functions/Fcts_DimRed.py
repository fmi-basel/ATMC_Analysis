from __future__ import annotations

from itertools import product
from typing import Optional, Union

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from pyslingshot import Slingshot

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scanpy as sc
import numpy as np
import phenograph
import anndata

from Functions.Fcts_Base import remove_uns
from Functions.Fcts_Plotting import save_fig


"""
***
TRAJECTORY / DIMRED / CLUSTERING FUNCTIONS
***
"""


def _as_dense_array(x):
    return x.toarray() if hasattr(x, "toarray") else np.asarray(x)


def _flatten_vector(x):
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    return np.asarray(x).reshape(-1)


def _validate_dimred_feats(ad, dimred_feats, allow_none=False):
    if dimred_feats is None:
        if allow_none:
            return
        raise ValueError("dimred_feats is empty.")
    if not len(dimred_feats):
        raise ValueError("dimred_feats is empty.")
    missing = [f for f in dimred_feats if f not in ad.var_names]
    if missing:
        raise ValueError(f"Some features in dimred_feats not found: {missing}")


def _check_finite_matrix(matrix, err_msg):
    arr = _as_dense_array(matrix)
    if not np.isfinite(arr).all():
        raise ValueError(err_msg)


def _validate_color_percentiles(color_percentiles):
    if color_percentiles is None:
        return None
    if not isinstance(color_percentiles, (tuple, list)) or len(color_percentiles) != 2:
        raise ValueError("color_percentiles must be a tuple/list of length 2, e.g. (1, 99).")
    lo, hi = color_percentiles
    lo = float(lo)
    hi = float(hi)
    if not (0 <= lo < hi <= 100):
        raise ValueError("color_percentiles must satisfy 0 <= low < high <= 100.")
    return (lo, hi)


def _compute_color_limits(values, color_percentiles=(1, 99)):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None

    cp = _validate_color_percentiles(color_percentiles)
    if cp is None:
        return float(np.min(finite)), float(np.max(finite))

    vmin, vmax = np.percentile(finite, cp)
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return float(np.min(finite)), float(np.max(finite))

    if vmin == vmax:
        return float(np.min(finite)), float(np.max(finite))

    return float(vmin), float(vmax)


def _get_feature_matrix(ad, dimred_feats=None, representation=None, allow_obsm=False, copy=True):
    """
    Returns
    -------
    ad_sub : AnnData
    matrix : matrix-like
    use_rep : str
        "X" or an obsm key
    rep_kind : str
        one of {"X", "layer", "obsm"}
    """
    if representation is None:
        ad_sub = ad[:, dimred_feats].copy() if dimred_feats is not None else (ad.copy() if copy else ad)
        return ad_sub, ad_sub.X, "X", "X"

    if representation in ad.layers:
        ad_sub = ad[:, dimred_feats].copy() if dimred_feats is not None else (ad.copy() if copy else ad)
        ad_sub.X = ad_sub.layers[representation].copy() if copy else ad_sub.layers[representation]
        return ad_sub, ad_sub.X, "X", "layer"

    if representation in ad.obsm:
        if not allow_obsm:
            raise ValueError(
                f"Representation '{representation}' is in ad.obsm, but this function expects X/layer-style feature matrices."
            )
        ad_sub = ad.copy() if copy else ad
        return ad_sub, ad_sub.obsm[representation], representation, "obsm"

    raise ValueError(f"Representation '{representation}' not found in ad.layers or ad.obsm.")


def _get_plot_vector(ad, feat, representation=None):
    if feat in ad.obs.columns:
        return ad.obs[feat], "obs"

    if feat in ad.var_names:
        if representation is not None and representation in ad.layers:
            vec = ad[:, feat].layers[representation]
        else:
            vec = ad[:, feat].X
        return _flatten_vector(vec), "var"

    return None, "missing"


def _get_2d_coords_for_plotting(ad, visualize_on=None, fallback_matrix=None, fallback_to_pca=True, random_state=0):
    def _get_coords_from_obsm(key):
        if key in ad.obsm:
            X = ad.obsm[key]
            if X.ndim == 2 and X.shape[1] >= 2:
                return X[:, :2], key
        return None, None

    if visualize_on is not None:
        coords, name = _get_coords_from_obsm(visualize_on)
        if coords is not None:
            return coords, name
        if not fallback_to_pca:
            raise ValueError(f"Requested visualize_on='{visualize_on}' not available or not 2D.")

    for key in ["X_umap", "X_tsne", "X_pca", "X_diffmap"]:
        coords, name = _get_coords_from_obsm(key)
        if coords is not None:
            return coords, name

    if not fallback_to_pca or fallback_matrix is None:
        raise ValueError("No 2D embedding available for plotting and fallback_to_pca is False.")

    M = _as_dense_array(fallback_matrix)
    if not np.isfinite(M).all():
        raise ValueError("Fallback PCA input contains non-finite values.")

    tmp = anndata.AnnData(X=M)
    sc.pp.pca(tmp, n_comps=2, random_state=random_state, copy=False)
    return tmp.obsm["X_pca"][:, :2], "PCA(2) fallback"


def select_dimred_features(ad, remove_features):
    """Select the features usable for dimensionality reduction.

    Keeps only features present for *every* object, because PCA/UMAP/clustering cannot handle
    NaN. A feature is NaN for an object whenever its antibody mix does not include that stain,
    so this drops the panel-specific columns and keeps the common core.

    Parameters:
    - ad (anndata.AnnData): Input object; ad.X is inspected.
    - remove_features (list of str): Substrings; any feature whose name contains one is dropped.

    Returns:
    - used_features (list of str): Feature names safe to pass as `dimred_feats`.
    """
    df = ad.to_df()
    common_feats = df.columns[~df.isna().any()]
    used_features = [x for x in common_feats if not any(rf in x for rf in remove_features)]
    if not used_features:
        import warnings
        warnings.warn("select_dimred_features: no features remain after filtering. Check remove_features or NaN coverage.")
    return used_features


def compute_PCA(
    ad,
    plot_feat,
    number_components=20,
    dimred_feats=None,
    representation=None,
    palette="magma",
    save_plot=False,
    dot_size=5,
    alpha=0.7,
    color_percentiles=(1, 99),
):
    """Run PCA and plot consecutive principal-component pairs coloured by chosen features.

    Parameters:
    - ad (anndata.AnnData): Input object. Not modified; a copy is returned.
    - plot_feat (str or list of str): Features to colour by. Each may be an obs column
      (categorical hue) or a var name (continuous colour map).
    - number_components (int): Number of principal components to compute.
    - dimred_feats (list of str or None): Features to run PCA on; None uses all of them.
    - representation (str or None): Key in ad.layers to use instead of ad.X (e.g. "z_scaled").
    - palette (str): Colour map for continuous features.
    - save_plot (bool): Save the figures to ad.uns["plot_dir"].
    - dot_size (float), alpha (float): Scatter point size and opacity.
    - color_percentiles (tuple or None): Percentile range for continuous colour limits.

    Returns:
    - ad (anndata.AnnData): Copy with X_pca in .obsm, PCs in .varm and pca info in .uns.
    """
    if isinstance(plot_feat, str):
        plot_feat = [plot_feat]

    if number_components < 2:
        raise ValueError("number_components must be at least 2.")

    color_percentiles = _validate_color_percentiles(color_percentiles)

    if dimred_feats is not None:
        _validate_dimred_feats(ad, dimred_feats, allow_none=True)

    ad = ad.copy()

    ad_pca, data, _, _ = _get_feature_matrix(
        ad,
        dimred_feats=dimred_feats,
        representation=representation,
        allow_obsm=False,
        copy=True,
    )

    _check_finite_matrix(data, "Input data contains non-finite values. Please preprocess or filter before PCA.")

    ad_pca.obsm.pop("X_pca", None)
    ad_pca.uns.pop("pca", None)
    ad_pca.varm.pop("PCs", None)

    n_comps_max = min(ad_pca.n_obs, ad_pca.n_vars)
    if number_components > n_comps_max:
        raise ValueError(f"number_components={number_components} exceeds min(n_obs, n_vars)={n_comps_max}.")

    sc.pp.pca(
        ad_pca,
        n_comps=number_components,
        layer=None,
        random_state=0,
        copy=False,
    )

    ad.obsm["X_pca"] = ad_pca.obsm["X_pca"].copy()
    ad.uns["pca"] = ad_pca.uns.get("pca", {}).copy()

    if dimred_feats is None:
        ad.varm["PCs"] = ad_pca.varm["PCs"].copy()
    else:
        comps = ad_pca.varm["PCs"].shape[1]
        pcs_full = np.full((ad.n_vars, comps), np.nan, dtype=np.float32)
        sel_idx = [ad.var_names.get_loc(f) for f in dimred_feats]
        pcs_full[sel_idx, :] = ad_pca.varm["PCs"]
        ad.varm["PCs"] = pcs_full

    pcs = ad.obsm["X_pca"]
    var_ratio = np.asarray(ad.uns["pca"].get("variance_ratio", []))

    n_pc_pairs = pcs.shape[1] // 2
    n_feats = len(plot_feat)

    fig, ax = plt.subplots(
        nrows=n_pc_pairs,
        ncols=n_feats,
        figsize=(4 * n_feats, 3.5 * n_pc_pairs),
        gridspec_kw={"hspace": 0.4, "wspace": 0.5},
        squeeze=False,
    )
    fig.subplots_adjust(top=0.90, wspace=0.5, hspace=0.4)

    for col, feat in enumerate(plot_feat):
        iterator = 1
        for row in range(n_pc_pairs):
            axis = ax[row, col]
            values, kind = _get_plot_vector(ad, feat, representation=representation)

            if kind == "obs":
                hue_data = values
                n_cat = hue_data.nunique() if hasattr(hue_data, "nunique") else None
                palette_used = "tab10" if n_cat is not None and n_cat <= 10 else "Set2"
                sns.scatterplot(
                    ax=axis,
                    x=pcs[:, iterator - 1],
                    y=pcs[:, iterator],
                    hue=hue_data,
                    s=dot_size,
                    alpha=alpha,
                    palette=palette_used,
                    legend="brief" if row == 0 else False,
                    edgecolor="none",
                )
            elif kind == "var":
                vmin, vmax = _compute_color_limits(values, color_percentiles=color_percentiles)
                scatter = axis.scatter(
                    pcs[:, iterator - 1],
                    pcs[:, iterator],
                    c=values,
                    cmap=palette,
                    s=dot_size,
                    alpha=alpha,
                    edgecolor="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                if row == 0:
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(scatter, cax=cax, orientation="vertical")
                    cbar.ax.tick_params(labelsize=8)
            else:
                axis.text(0.5, 0.5, f"Feature '{feat}' not found", ha="center", va="center", fontsize=10)
                axis.set_axis_off()
                iterator += 2   # keep the PC pairing in step with the row index
                continue

            x_var = float(var_ratio[iterator - 1] * 100.0) if var_ratio.size >= iterator else np.nan
            y_var = float(var_ratio[iterator] * 100.0) if var_ratio.size >= iterator + 1 else np.nan

            axis.set_xlabel(f"PC{iterator}\nExplained var: {x_var:.2f}%", fontsize=9)
            axis.set_ylabel(f"PC{iterator + 1}\nExplained var: {y_var:.2f}%", fontsize=9)
            axis.tick_params(left=True, bottom=True, labelsize=8)
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            axis.set_title(feat if row == 0 else "", fontsize=12 if row == 0 else None, y=1.03 if row == 0 else None)
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            iterator += 2

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "PCA", dpi=300)

    if "PCs" in ad.varm and ad.varm["PCs"] is not None:
        n_loadings = ad.varm["PCs"].shape[1]
        comps = [f"PC{i + 1}" for i in range(n_loadings)]
        mask = ~np.isnan(ad.varm["PCs"]).any(axis=1)

        pca_loadings = pd.DataFrame(
            ad.varm["PCs"][mask, :n_loadings],
            index=ad.var_names[mask],
            columns=comps,
        )

        fig, ax = plt.subplots(1, 1, figsize=(6, 15))
        sns.heatmap(
            pca_loadings,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            robust=True,
            cbar=True,
            cbar_kws={"shrink": 0.5},
        )
        ax.set_title("PCA Loadings", fontsize=16, y=1.02)
        ax.xaxis.set_ticks_position("bottom")
        ax.tick_params(axis="x", labelrotation=45, labelsize=10)
        fig.subplots_adjust(top=0.97, bottom=0.08, left=0.18, right=0.95)

        if save_plot:
            save_fig(fig, ad.uns["plot_dir"], "PCA_Heatmap", dpi=300)

    return ad


def compute_DC(
    ad,
    plot_feat,
    number_components=20,
    n_neighbors=15,
    dimred_feats=None,
    representation=None,
    palette="magma",
    save_plot=False,
    dot_size=5,
    alpha=0.7,
    color_percentiles=(1, 99),
):
    """Run a diffusion map and plot consecutive diffusion-component pairs.

    The first diffusion component is constant and is skipped, so plotting starts at DC2.

    Parameters:
    - ad (anndata.AnnData): Input object. Not modified; a copy is returned.
    - plot_feat (str or list of str): Features to colour by (obs column or var name).
    - number_components (int): Number of diffusion components to compute.
    - n_neighbors (int): Neighbourhood size for the underlying kNN graph.
    - dimred_feats (list of str or None): Features to use; None uses all of them.
    - representation (str or None): Key in ad.layers, or in ad.obsm (e.g. "X_pca").
    - palette, save_plot, dot_size, alpha, color_percentiles: As in compute_PCA.

    Returns:
    - ad (anndata.AnnData): Copy with X_diffmap in .obsm and diffmap_evals in .uns.
    """
    if isinstance(plot_feat, str):
        plot_feat = [plot_feat]

    color_percentiles = _validate_color_percentiles(color_percentiles)
    _validate_dimred_feats(ad, dimred_feats, allow_none=True)

    ad = ad.copy()

    ad_sub, data, use_rep, _ = _get_feature_matrix(
        ad,
        dimred_feats=dimred_feats,
        representation=representation,
        allow_obsm=True,
        copy=True,
    )

    _check_finite_matrix(data, "Input data contains non-finite values. Please preprocess or filter before diffusion mapping.")

    ad_sub.uns.pop("neighbors", None)
    ad_sub.obsm.pop("X_diffmap", None)
    ad_sub.uns.pop("diffmap_evals", None)

    sc.pp.neighbors(ad_sub, n_neighbors=n_neighbors, use_rep=use_rep)
    sc.tl.diffmap(ad_sub, n_comps=number_components)

    diffmap_all = ad_sub.obsm["X_diffmap"].copy()
    ad.obsm["X_diffmap"] = diffmap_all
    ad.uns["diffmap_evals"] = ad_sub.uns["diffmap_evals"].copy()

    start_idx = 1
    end_idx = min(number_components + 1, diffmap_all.shape[1])
    pcs = diffmap_all[:, start_idx:end_idx]

    n_dc_pairs = pcs.shape[1] // 2
    n_feats = len(plot_feat)

    fig, ax = plt.subplots(
        nrows=n_dc_pairs,
        ncols=n_feats,
        figsize=(4 * n_feats, 3.5 * n_dc_pairs),
        gridspec_kw={"hspace": 0.4, "wspace": 0.3},
        squeeze=False,
    )
    fig.subplots_adjust(top=0.95, wspace=0.3, hspace=0.4)

    for col, feat in enumerate(plot_feat):
        idx = 0
        for row in range(n_dc_pairs):
            axis = ax[row, col]
            values, kind = _get_plot_vector(ad, feat, representation=representation)

            if kind == "obs":
                hue_data = values
                n_cat = hue_data.nunique() if hasattr(hue_data, "nunique") else None
                palette_used = "tab10" if n_cat is not None and n_cat <= 10 else "Set2"
                sns.scatterplot(
                    ax=axis,
                    x=pcs[:, idx],
                    y=pcs[:, idx + 1],
                    hue=hue_data,
                    s=dot_size,
                    alpha=alpha,
                    palette=palette_used,
                    legend="brief" if row == 0 else False,
                    edgecolor="none",
                )
            elif kind == "var":
                vmin, vmax = _compute_color_limits(values, color_percentiles=color_percentiles)
                scatter = axis.scatter(
                    pcs[:, idx],
                    pcs[:, idx + 1],
                    c=values,
                    cmap=palette,
                    s=dot_size,
                    alpha=alpha,
                    edgecolor="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                if row == 0:
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(scatter, cax=cax, orientation="vertical")
                    cbar.ax.tick_params(labelsize=8)
            else:
                axis.text(0.5, 0.5, f"Feature '{feat}' not found", ha="center", va="center", fontsize=10)
                axis.set_axis_off()
                idx += 2
                continue

            axis.set_xlabel(f"DC{idx + 1}", fontsize=9)
            axis.set_ylabel(f"DC{idx + 2}", fontsize=9)
            axis.tick_params(left=True, bottom=True, labelsize=8)
            axis.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            axis.set_title(feat if row == 0 else "", fontsize=12 if row == 0 else None)
            axis.set_xticklabels([])
            axis.set_yticklabels([])
            idx += 2

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "DiffusionComponents", dpi=300)

    return ad


def compute_UMAP(
    ad,
    plot_feat,
    n_neighbors,
    min_dist,
    dimred_feats,
    save_plot=False,
    representation=None,
    palette="magma",
    dot_size=5,
    alpha=0.7,
    color_percentiles=(1, 99),
):
    """Run UMAP, optionally sweeping several parameter combinations.

    Passing lists for n_neighbors and/or min_dist plots every combination as a grid. Only one
    embedding can be stored, so .obsm["X_umap"] holds the last combination computed; the
    function prints which one that was.

    Parameters:
    - ad (anndata.AnnData): Input object. Not modified; a copy is returned.
    - plot_feat (str or list of str): Features to colour by. In a sweep only the first is used.
    - n_neighbors (int or list of int): UMAP neighbourhood size(s).
    - min_dist (float or list of float): UMAP minimum distance(s).
    - dimred_feats (list of str): Features to use. Ignored when `representation` is an obsm key.
    - save_plot (bool): Save the figure to ad.uns["plot_dir"].
    - representation (str or None): Key in ad.layers, or in ad.obsm (e.g. "X_pca").
    - palette, dot_size, alpha, color_percentiles: As in compute_PCA.

    Returns:
    - ad (anndata.AnnData): Copy with X_umap in .obsm.
    """
    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])
    _validate_dimred_feats(ad, dimred_feats, allow_none=False)
    color_percentiles = _validate_color_percentiles(color_percentiles)

    ad = ad.copy()

    if isinstance(plot_feat, str):
        plot_feat = [plot_feat]
    if isinstance(n_neighbors, int):
        n_neighbors = [n_neighbors]
    if isinstance(min_dist, (int, float)):
        min_dist = [min_dist]

    combos = list(product(n_neighbors, min_dist))
    nrows = len(min_dist)
    ncols = len(n_neighbors)

    if len(combos) == 1:
        fig, ax_grid = plt.subplots(1, len(plot_feat), figsize=(4 * len(plot_feat), 4))
        if len(plot_feat) == 1:
            ax_grid = [ax_grid]
        fig.subplots_adjust(top=0.82, wspace=0.25)
    else:
        fig, ax_grid = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
        fig.subplots_adjust(top=0.92, wspace=0.25, hspace=0.25)

    coords = None
    last_ad_sub = None

    for n, d in combos:
        ad_sub, data, use_rep, _ = _get_feature_matrix(
            ad,
            dimred_feats=dimred_feats,
            representation=representation,
            allow_obsm=True,
            copy=True,
        )
        _check_finite_matrix(data, "Input data contains non-finite values. Please preprocess or filter before UMAP.")

        ad_sub.uns.pop("neighbors", None)
        ad_sub.obsm.pop("X_umap", None)

        sc.pp.neighbors(ad_sub, n_neighbors=n, use_rep=use_rep)
        sc.tl.umap(ad_sub, min_dist=d)

        coords = ad_sub.obsm["X_umap"].copy()
        last_ad_sub = ad_sub

        if len(combos) == 1:
            for col, feat in enumerate(plot_feat):
                ax = ax_grid[col]
                values, kind = _get_plot_vector(ad, feat, representation=representation)

                if kind == "obs":
                    sns.scatterplot(
                        ax=ax,
                        x=coords[:, 0],
                        y=coords[:, 1],
                        hue=values,
                        s=dot_size,
                        alpha=alpha,
                        palette="pastel",
                        legend=True,
                        edgecolor="none",
                    )
                elif kind == "var":
                    vmin, vmax = _compute_color_limits(values, color_percentiles=color_percentiles)
                    scatter = ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=values,
                        cmap=palette,
                        s=dot_size,
                        alpha=alpha,
                        edgecolor="none",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    cb = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.02)
                    cb.set_label(feat)
                else:
                    ax.set_title(f"{feat} (not found)", fontsize=14)

                if kind != "missing":
                    ax.set_title(str(feat), fontsize=12)
                ax.set_axis_off()

            fig.suptitle(
                f"UMAP (Neighbors={n}, MinDist={d}, rep={representation if representation is not None else 'X'})",
                fontsize=18,
                y=0.98,
            )
        else:
            row = min_dist.index(d)
            col_index = n_neighbors.index(n)
            ax = ax_grid[row][col_index]
            feat = plot_feat[0]
            values, kind = _get_plot_vector(ad, feat, representation=representation)

            if kind == "obs":
                sns.scatterplot(
                    ax=ax,
                    x=coords[:, 0],
                    y=coords[:, 1],
                    hue=values,
                    s=dot_size,
                    alpha=alpha,
                    palette="pastel",
                    legend=False,
                    edgecolor="none",
                )
                ax.set_title(f"n={n}, min_dist={d}, {feat}", fontsize=10)
            elif kind == "var":
                vmin, vmax = _compute_color_limits(values, color_percentiles=color_percentiles)
                scatter = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=values,
                    cmap=palette,
                    s=dot_size,
                    alpha=alpha,
                    edgecolor="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"n={n}, min_dist={d}, {feat}", fontsize=10)
                fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.02)
            else:
                ax.set_title(f"{feat} (not found)", fontsize=10)

            ax.set_axis_off()

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "UMAP" if len(combos) == 1 else "UMAP_grid", dpi=300)

    # Only one embedding can live in .obsm; when sweeping, that is the last combination.
    if coords is not None:
        ad.obsm["X_umap"] = coords
        if len(combos) > 1:
            n, d = combos[-1]
            print(f"ad.obsm['X_umap'] holds the last combination plotted: n_neighbors={n}, min_dist={d}.")
    if last_ad_sub is not None and "umap" in last_ad_sub.uns:
        ad.uns["umap"] = last_ad_sub.uns["umap"].copy()

    plt.show()
    return ad


def compute_tSNE(
    ad,
    plot_feat,
    dimred_feats,
    perplexity=30.0,
    learning_rate=200.0,
    save_plot=False,
    representation=None,
    early_exaggeration=12.0,
    random_state=0,
    metric="euclidean",
    palette="magma",
    dot_size=5,
    alpha=0.7,
    color_percentiles=(1, 99),
):
    """Run t-SNE, optionally sweeping perplexity and learning rate.

    Passing lists for perplexity and/or learning_rate plots every combination as a grid. Only
    one embedding can be stored, so .obsm["X_tsne"] holds the last combination computed; the
    function prints which one that was.

    Parameters:
    - ad (anndata.AnnData): Input object. Not modified; a copy is returned.
    - plot_feat (str or list of str): Features to colour by. In a sweep only the first is used.
    - dimred_feats (list of str): Features to use. Ignored when `representation` is an obsm key.
    - perplexity (float or list of float): t-SNE perplexity; must be < number of objects.
    - learning_rate (float or list of float): t-SNE learning rate.
    - save_plot (bool): Save the figure to ad.uns["plot_dir"].
    - representation (str or None): Key in ad.layers, or in ad.obsm (e.g. "X_pca").
    - early_exaggeration (float), random_state (int), metric (str): Passed to scanpy's t-SNE.
    - palette, dot_size, alpha, color_percentiles: As in compute_PCA.

    Returns:
    - ad (anndata.AnnData): Copy with X_tsne in .obsm.
    """
    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])
    _validate_dimred_feats(ad, dimred_feats, allow_none=False)
    color_percentiles = _validate_color_percentiles(color_percentiles)

    ad = ad.copy()

    if isinstance(plot_feat, str):
        plot_feat = [plot_feat]
    if isinstance(perplexity, (int, float)):
        perplexity = [float(perplexity)]
    if isinstance(learning_rate, (int, float)):
        learning_rate = [float(learning_rate)]

    combos = list(product(perplexity, learning_rate))
    nrows = len(perplexity)
    ncols = len(learning_rate)

    if len(combos) == 1:
        fig, ax_grid = plt.subplots(1, len(plot_feat), figsize=(4 * len(plot_feat), 4))
        if len(plot_feat) == 1:
            ax_grid = [ax_grid]
        fig.subplots_adjust(top=0.82, wspace=0.25)
    else:
        fig, ax_grid = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
        fig.subplots_adjust(top=0.92, wspace=0.25, hspace=0.25)

    coords = None

    for ppx, lr in combos:
        ad_sub, data, use_rep, _ = _get_feature_matrix(
            ad,
            dimred_feats=dimred_feats,
            representation=representation,
            allow_obsm=True,
            copy=True,
        )
        _check_finite_matrix(data, "Input data contains non-finite values. Please preprocess or filter before t-SNE.")

        ad_sub.obsm.pop("X_tsne", None)

        if ppx >= ad_sub.n_obs:
            raise ValueError(f"perplexity ({ppx}) must be less than the number of cells ({ad_sub.n_obs}).")

        sc.tl.tsne(
            ad_sub,
            use_rep=None if use_rep == "X" else use_rep,
            n_pcs=None,
            perplexity=ppx,
            learning_rate=lr,
            early_exaggeration=early_exaggeration,
            random_state=random_state,
            metric=metric,
            use_fast_tsne=False,
        )

        coords = ad_sub.obsm["X_tsne"].copy()

        if len(combos) == 1:
            for col, feat in enumerate(plot_feat):
                ax = ax_grid[col]
                values, kind = _get_plot_vector(ad, feat, representation=representation)

                if kind == "obs":
                    sns.scatterplot(
                        ax=ax,
                        x=coords[:, 0],
                        y=coords[:, 1],
                        hue=values,
                        s=dot_size,
                        alpha=alpha,
                        palette="pastel",
                        legend=True,
                        edgecolor="none",
                    )
                elif kind == "var":
                    vmin, vmax = _compute_color_limits(values, color_percentiles=color_percentiles)
                    scatter = ax.scatter(
                        coords[:, 0],
                        coords[:, 1],
                        c=values,
                        cmap=palette,
                        s=dot_size,
                        alpha=alpha,
                        edgecolor="none",
                        vmin=vmin,
                        vmax=vmax,
                    )
                    cb = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.02)
                    cb.set_label(feat)
                else:
                    ax.set_title(f"{feat} (not found)", fontsize=14)
                ax.set_axis_off()

            fig.suptitle(f"t-SNE (Perplexity={ppx}, LR={lr})", fontsize=18, y=0.98)
        else:
            row = perplexity.index(ppx)
            col_index = learning_rate.index(lr)
            ax = ax_grid[row][col_index]
            feat = plot_feat[0]
            values, kind = _get_plot_vector(ad, feat, representation=representation)

            if kind == "obs":
                sns.scatterplot(
                    ax=ax,
                    x=coords[:, 0],
                    y=coords[:, 1],
                    hue=values,
                    s=dot_size,
                    alpha=alpha,
                    palette="pastel",
                    legend=False,
                    edgecolor="none",
                )
                ax.set_title(f"perp={ppx}, lr={lr}, {feat}", fontsize=10)
            elif kind == "var":
                vmin, vmax = _compute_color_limits(values, color_percentiles=color_percentiles)
                scatter = ax.scatter(
                    coords[:, 0],
                    coords[:, 1],
                    c=values,
                    cmap=palette,
                    s=dot_size,
                    alpha=alpha,
                    edgecolor="none",
                    vmin=vmin,
                    vmax=vmax,
                )
                ax.set_title(f"perp={ppx}, lr={lr}, {feat}", fontsize=10)
                fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.02)
            else:
                ax.set_title(f"{feat} (not found)", fontsize=10)

            ax.set_axis_off()

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "tSNE" if len(combos) == 1 else "tSNE_grid", dpi=300)

    # Only one embedding can live in .obsm; when sweeping, that is the last combination.
    if coords is not None:
        ad.obsm["X_tsne"] = coords
        if len(combos) > 1:
            ppx, lr = combos[-1]
            print(f"ad.obsm['X_tsne'] holds the last combination plotted: perplexity={ppx}, learning_rate={lr}.")

    return ad


def compute_kmeans(
    ad,
    n_clusters,
    dimred_feats,
    save_plot=False,
    representation=None,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=0,
    visualize_on=None,
    fallback_to_pca=True,
    point_size=6,
):
    """Cluster objects with k-means, optionally for several values of k.

    Parameters:
    - ad (anndata.AnnData): Input object. Not modified; a copy is returned.
    - n_clusters (int or list of int): Number of clusters; a list computes each in turn.
    - dimred_feats (list of str): Features to cluster on. Ignored when `representation` is obsm.
    - save_plot (bool): Save the figure to ad.uns["plot_dir"].
    - representation (str or None): Key in ad.layers, or in ad.obsm (e.g. "X_pca").
    - init, n_init, max_iter, random_state: Passed through to sklearn's KMeans.
    - visualize_on (str or None): obsm key used for the scatter plot; None auto-picks X_umap,
      X_tsne, X_pca or X_diffmap, in that order.
    - fallback_to_pca (bool): If no embedding exists, compute a 2-component PCA for plotting.
    - point_size (float): Scatter point size.

    Returns:
    - ad (anndata.AnnData): Copy with "kmeans_labels_{k}" per k in .obs, plus "kmeans_labels"
      holding the last k computed.
    """
    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])
    _validate_dimred_feats(ad, dimred_feats, allow_none=False)

    ad = ad.copy()

    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    _, M, _, _ = _get_feature_matrix(
        ad,
        dimred_feats=dimred_feats,
        representation=representation,
        allow_obsm=True,
        copy=True,
    )

    M = _as_dense_array(M)
    _check_finite_matrix(M, "KMeans input contains non-finite values.")

    coords, coords_name = _get_2d_coords_for_plotting(
        ad,
        visualize_on=visualize_on,
        fallback_matrix=M,
        fallback_to_pca=fallback_to_pca,
        random_state=random_state,
    )

    multi_k = len(n_clusters) > 1
    if multi_k:
        fig, ax_grid = plt.subplots(len(n_clusters), 1, figsize=(5, 3.5 * len(n_clusters)), squeeze=False)
        fig.subplots_adjust(top=0.95, hspace=0.35)
    else:
        fig, ax_grid = plt.subplots(1, 1, figsize=(4, 4))
        fig.subplots_adjust(top=0.92)

    last_labels = None

    for i, k in enumerate(n_clusters):
        km = KMeans(
            n_clusters=k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
        )
        labels = km.fit_predict(M)
        ad.obs[f"kmeans_labels_{k}"] = pd.Categorical(labels.astype(str))
        last_labels = labels

        ax = ax_grid[i][0] if multi_k else ax_grid
        sns.scatterplot(
            ax=ax,
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels.astype(str),
            s=point_size,
            palette="tab20",
            legend=not multi_k,
            edgecolor="none",
        )
        ax.set_title(f"k={k} on {coords_name}", fontsize=10)
        ax.set_axis_off()

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "KMeans_grid" if multi_k else "KMeans", dpi=300)

    if last_labels is not None:
        ad.obs["kmeans_labels"] = pd.Categorical(last_labels.astype(str))

    return ad


def compute_phenograph(
    ad,
    k,
    dimred_feats,
    save_plot=False,
    representation=None,
    clustering_metric="euclidean",
    prune=False,
    jaccard=True,
    min_cluster_size=10,
    visualize_on=None,
    fallback_to_pca=True,
    random_state=0,
    point_size=6,
):
    """Cluster objects with PhenoGraph (Louvain on a kNN graph), optionally for several k.

    Parameters:
    - ad (anndata.AnnData): Input object. Not modified; a copy is returned.
    - k (int or list of int): Neighbourhood size(s); must be smaller than the object count.
    - dimred_feats (list of str): Features to cluster on. Ignored when `representation` is obsm.
    - save_plot (bool): Save the figure to ad.uns["plot_dir"].
    - representation (str or None): Key in ad.layers, or in ad.obsm (e.g. "X_pca").
    - clustering_metric (str): Distance metric for the kNN graph.
    - prune, jaccard, min_cluster_size, random_state: Passed through to phenograph.cluster.
    - visualize_on, fallback_to_pca, point_size: As in compute_kmeans.

    Returns:
    - ad (anndata.AnnData): Copy with "phenograph_labels_{k}" per k in .obs, plus
      "phenograph_labels" holding the last k computed.
    """
    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])
    _validate_dimred_feats(ad, dimred_feats, allow_none=False)

    ad = ad.copy()

    if isinstance(k, int):
        k = [k]

    _, M, _, _ = _get_feature_matrix(
        ad,
        dimred_feats=dimred_feats,
        representation=representation,
        allow_obsm=True,
        copy=True,
    )

    M = _as_dense_array(M)
    _check_finite_matrix(M, "PhenoGraph input contains non-finite values.")

    n_cells = M.shape[0]
    for ki in k:
        if ki >= n_cells:
            raise ValueError(f"k ({ki}) must be less than the number of cells ({n_cells}).")

    coords, coords_name = _get_2d_coords_for_plotting(
        ad,
        visualize_on=visualize_on,
        fallback_matrix=M,
        fallback_to_pca=fallback_to_pca,
        random_state=random_state,
    )

    multi_k = len(k) > 1
    if multi_k:
        fig, ax_grid = plt.subplots(len(k), 1, figsize=(5, 3.5 * len(k)), squeeze=False)
        fig.subplots_adjust(top=0.95, hspace=0.35)
    else:
        fig, ax_grid = plt.subplots(1, 1, figsize=(4, 4))
        fig.subplots_adjust(top=0.92)

    last_labels = None

    for i, ki in enumerate(k):
        communities, graph, Q = phenograph.cluster(
            M,
            k=ki,
            directed=False,
            prune=prune,
            jaccard=jaccard,
            primary_metric=clustering_metric,
            min_cluster_size=min_cluster_size,
            cluster_algorithm="louvain",
            n_jobs=-1,
            seed=random_state,
        )

        labels = np.asarray(communities, dtype=int)
        ad.obs[f"phenograph_labels_{ki}"] = pd.Categorical(labels.astype(str))
        last_labels = labels

        ax = ax_grid[i][0] if multi_k else ax_grid
        sns.scatterplot(
            ax=ax,
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels.astype(str),
            s=point_size,
            palette="tab20",
            legend=not multi_k,
            edgecolor="none",
        )
        ax.set_title(f"PhenoGraph k={ki} on {coords_name}", fontsize=10)
        ax.set_axis_off()

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "PhenoGraph_grid" if multi_k else "PhenoGraph", dpi=300)

    if last_labels is not None:
        ad.obs["phenograph_labels"] = pd.Categorical(last_labels.astype(str))

    return ad


def run_phenograph(
    ad,
    ad_dimred,
    k,
    plot_save_dir=None,
    save_plot=False,
    layer="minmax_transformed",
    cluster_key="phenograph_labels",
    embedding_key="X_umap",
    point_size=20,
    alpha=0.5,
):
    """Cluster with PhenoGraph using one AnnData for the features and another for the embedding.

    `ad` supplies the embedding and receives the labels; `ad_dimred` supplies the feature matrix.
    Labels are assigned by position, so both must describe the same objects in the same order -
    this is checked and raises if not.

    Parameters:
    - ad (anndata.AnnData): Object holding the embedding; a labelled copy is returned.
    - ad_dimred (anndata.AnnData): Object holding the clustering features.
    - k (int): PhenoGraph neighbourhood size.
    - plot_save_dir (str or None): Output directory; None uses ad.uns["plot_dir"].
    - save_plot (bool): Save the overview and per-cluster figures.
    - layer (str): Layer of `ad_dimred` to cluster on.
    - cluster_key (str): obs column the labels are written to.
    - embedding_key (str): obsm key used for plotting.
    - point_size (float), alpha (float): Scatter point size and opacity.

    Returns:
    - ad (anndata.AnnData): Copy with the cluster labels in .obs[cluster_key].
    """
    if layer not in ad_dimred.layers:
        raise ValueError(f"Layer '{layer}' not found in ad_dimred.layers.")

    if embedding_key not in ad.obsm:
        raise ValueError(f"Embedding '{embedding_key}' not found in ad.obsm.")

    coords = ad.obsm[embedding_key]
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Embedding '{embedding_key}' must have at least 2 columns.")

    # Labels are written back onto `ad` by position, so the two objects must describe the
    # same observations in the same order.
    if ad_dimred.n_obs != ad.n_obs or not ad_dimred.obs_names.equals(ad.obs_names):
        raise ValueError(
            f"ad_dimred and ad must contain the same observations in the same order "
            f"(got {ad_dimred.n_obs} vs {ad.n_obs}); cluster labels would be misassigned."
        )

    X = ad_dimred.to_df(layer=layer).to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("Clustering input contains non-finite values.")

    ad = ad.copy()

    labels, _, _ = phenograph.cluster(X, k=k, seed=0)
    labels = np.asarray(labels)
    ad.obs[cluster_key] = pd.Categorical(labels.astype(str))

    unique_labels = list(pd.Categorical(ad.obs[cluster_key]).categories)
    cols = 3
    rows = int(np.ceil(len(unique_labels) / cols))

    fig1, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
    sns.scatterplot(
        ax=ax,
        x=coords[:, 0],
        y=coords[:, 1],
        s=point_size,
        hue=ad.obs[cluster_key],
        palette="tab20",
        alpha=alpha,
        legend=True,
        edgecolor="none",
    )
    ax.set_axis_off()
    ax.set_title(f"PhenoGraph (k={k})", fontsize=12)
    fig1.subplots_adjust(top=0.92)

    fig2, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 2.5, rows * 2.5), squeeze=False)
    fig2.subplots_adjust(top=0.92, wspace=0.15, hspace=0.25)

    for ax_i in axs.flat:
        ax_i.axis("off")

    for idx, lab in enumerate(unique_labels):
        r, c = divmod(idx, cols)
        ax_i = axs[r, c]
        mask = ad.obs[cluster_key].astype(str) == str(lab)

        sns.scatterplot(
            ax=ax_i,
            x=coords[:, 0],
            y=coords[:, 1],
            s=max(1, point_size * 0.25),
            color="lightgray",
            alpha=0.15,
            legend=False,
            edgecolor="none",
        )
        sns.scatterplot(
            ax=ax_i,
            x=coords[mask, 0],
            y=coords[mask, 1],
            s=max(2, point_size * 0.5),
            color="red",
            alpha=0.7,
            legend=False,
            edgecolor="none",
        )
        ax_i.set_title(str(lab), fontsize=10)
        ax_i.set_axis_off()

    if save_plot:
        out_dir = plot_save_dir if plot_save_dir is not None else ad.uns["plot_dir"]
        save_fig(fig1, out_dir, "4Trajectory_PhenoGraph")
        save_fig(fig2, out_dir, "4Trajectory_PhenoGraph_Subplots")

    return ad



def run_slingshot(
    ad,
    start_cluster,
    num_epochs=3,
    save_plot=False,
    cluster_key="phenograph_labels",
    visualize_on="X_umap",
    point_size=15,
    cmap_pseudotime="magma",
):
    """Infer lineages and pseudotime over an existing embedding and clustering with Slingshot.

    Parameters:
    - ad (anndata.AnnData): Input object. Not modified; a copy is returned.
    - start_cluster (str or int): Cluster the trajectory starts from - either a label from
      `cluster_key` or its integer index among the categories.
    - num_epochs (int): Slingshot fitting iterations.
    - save_plot (bool): Save the fitting and pseudotime figures to ad.uns["plot_dir"].
    - cluster_key (str): obs column holding the cluster labels. Every object must be assigned.
    - visualize_on (str): obsm key of the 2D embedding to fit on.
    - point_size (float): Scatter point size.
    - cmap_pseudotime (str): Colour map for the pseudotime plot.

    Returns:
    - ad (anndata.AnnData): Copy with "slingshot_pseudotime" (min-max scaled) and one
      "slingshot_weight_lineage_{i}" column per lineage in .obs, and settings in .uns.
    """
    if visualize_on not in ad.obsm:
        raise ValueError(f"Embedding '{visualize_on}' not found in ad.obsm.")
    coords = ad.obsm[visualize_on]
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"ad.obsm['{visualize_on}'] must be 2D with at least 2 columns.")
    coords_2d = coords[:, :2]

    if cluster_key not in ad.obs.columns:
        raise ValueError(f"cluster_key '{cluster_key}' not found in ad.obs.")

    ad = ad.copy()

    clusters_raw = ad.obs[cluster_key]
    if isinstance(clusters_raw.dtype, pd.CategoricalDtype):
        clusters = clusters_raw.copy()
    else:
        str_labels = clusters_raw.astype(str)
        seen_order = list(dict.fromkeys(str_labels))
        clusters = pd.Series(pd.Categorical(str_labels, categories=seen_order), index=clusters_raw.index)

    categories = list(clusters.cat.categories)
    cluster_int = clusters.cat.codes.to_numpy()
    if np.any(cluster_int < 0):
        raise ValueError("Found missing cluster labels; all cells must have valid cluster assignments.")

    label_to_int = {lab: i for i, lab in enumerate(categories)}

    if isinstance(start_cluster, (int, np.integer)):
        if 0 <= int(start_cluster) < len(categories):
            start_node = int(start_cluster)
        elif str(start_cluster) in label_to_int:
            start_node = label_to_int[str(start_cluster)]
        else:
            raise ValueError(f"start_cluster={start_cluster} not found among cluster labels or valid indices.")
    else:
        start_cluster_str = str(start_cluster)
        if start_cluster_str not in label_to_int:
            raise ValueError(f"start_cluster='{start_cluster}' not found among cluster labels {categories}.")
        start_node = label_to_int[start_cluster_str]

    n = ad.n_obs
    K = len(categories)
    part = np.zeros((n, K), dtype=float)
    part[np.arange(n), cluster_int] = 1.0

    sl = Slingshot(coords_2d, part, start_node=start_node)

    fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    fig1.subplots_adjust(top=0.95, wspace=0.15, hspace=0.15)
    for axis in axes.flat:
        axis.axis("off")

    sl.fit(num_epochs=num_epochs, debug_axes=axes)

    if save_plot:
        save_fig(fig1, ad.uns["plot_dir"], "Trajectory_SlingshotComputation")

    pt = sl.unified_pseudotime
    if pt is None:
        raise RuntimeError("Slingshot did not produce unified pseudotime.")

    pt_scaled = MinMaxScaler().fit_transform(np.asarray(pt).reshape(-1, 1)).flatten()
    ad.obs["slingshot_pseudotime"] = pt_scaled

    if getattr(sl, "lineages", None) is None or getattr(sl, "cell_weights", None) is None:
        raise RuntimeError("Slingshot did not return lineages/cell_weights.")

    n_lineages = len(sl.lineages)
    weights = np.asarray(sl.cell_weights)
    if weights.ndim == 1:
        weights = weights.reshape(-1, 1)
    if weights.ndim != 2 or weights.shape[1] != n_lineages:
        raise RuntimeError(
            f"cell_weights shape {weights.shape} is incompatible with n_lineages={n_lineages}."
        )
    for i in range(n_lineages):
        ad.obs[f"slingshot_weight_lineage_{i}"] = weights[:, i]

    fig2, ax2 = plt.subplots(ncols=1, figsize=(10, 10))
    fig2.subplots_adjust(top=0.95)
    ax2.set_title("Pseudotime")

    try:
        sl.plotter.clusters(ax2, color_mode="pseudotime", s=point_size, cmap=cmap_pseudotime)
    except Exception:
        sca = ax2.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=pt_scaled,
            s=point_size,
            cmap=cmap_pseudotime,
            edgecolor="none",
        )
        fig2.colorbar(sca, ax=ax2, orientation="vertical", pad=0.02)

    ax2.set_axis_off()

    if save_plot:
        save_fig(fig2, ad.uns["plot_dir"], "Trajectory_SlingshotPseudotime")

    ad.uns["slingshot"] = {
        "start_cluster": start_cluster,
        "cluster_key": cluster_key,
        "visualize_on": visualize_on,
        "num_epochs": num_epochs,
        "categories": categories,
    }

    return ad


def subsample_anndata_geosketch(ad, fraction=0.1, use_rep="X_pca", random_state=0):
    """Geometric-sketch subsampling. Requires the optional `geosketch` package."""
    try:
        import geosketch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "subsample_anndata_geosketch needs the optional 'geosketch' package, which is not "
            "part of this project's dependencies. Install it into the project environment with "
            "`uv add geosketch` (or `pip install geosketch`), or use random_subset_anndata_frac "
            "instead."
        ) from exc

    X = ad.obsm[use_rep]
    n_sketch = int(ad.n_obs * fraction)
    sketch_index = geosketch.gs(X, n_sketch, replace=False, seed=random_state)
    ad_subsampled = ad[sketch_index].copy()

    return ad_subsampled


def random_subset_anndata(adata, n, random_state):
    """Take a random subset of a fixed number of objects, keeping their original order.

    Parameters:
    - adata (anndata.AnnData): Input object.
    - n (int): Number of objects to keep; must be between 0 and adata.n_obs.
    - random_state (int): Random seed.

    Returns:
    - adata_sub (anndata.AnnData): Copy containing the sampled objects.
    """
    if n < 0 or n > adata.n_obs:
        raise ValueError(f"n must be between 0 and {adata.n_obs}, got {n}")

    rng = np.random.default_rng(random_state)
    idx = rng.choice(adata.n_obs, size=n, replace=False)
    idx.sort()

    adata_sub = adata[idx, :].copy()
    return adata_sub


def random_subset_anndata_frac(adata, fraction, random_state):
    """Take a random subset of a fraction of objects, keeping their original order.

    Parameters:
    - adata (anndata.AnnData): Input object.
    - fraction (float): Fraction to keep, in (0, 1]. At least one object is always kept.
    - random_state (int): Random seed.

    Returns:
    - adata_sub (anndata.AnnData): Copy containing the sampled objects.
    """
    if fraction <= 0 or fraction > 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    if adata.n_obs == 0:
        raise ValueError("Cannot sample from an AnnData with zero observations")

    n = max(1, int(round(fraction * adata.n_obs)))
    rng = np.random.default_rng(random_state)
    idx = rng.choice(adata.n_obs, size=n, replace=False)
    idx.sort()

    adata_sub = adata[idx, :].copy()
    return adata_sub


def knn_label_transfer(
    adata_ref: anndata.AnnData,
    adata_target: anndata.AnnData,
    label_key: str,
    n_neighbors: int = 15,
    layer: Optional[str] = None,
    obsm_key: Optional[str] = None,
    features: Optional[Union[list[str], np.ndarray, pd.Index]] = None,
    target_obs_key: Optional[str] = None,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """Transfer a label from a reference AnnData onto a target AnnData with a kNN classifier.

    Parameters:
    - adata_ref (anndata.AnnData): Labelled reference objects.
    - adata_target (anndata.AnnData): Objects to predict labels for.
    - label_key (str): obs column in the reference holding the labels.
    - n_neighbors (int): Number of neighbours used for the vote.
    - layer (str or None): Layer to use as the feature matrix; mutually exclusive with obsm_key.
    - obsm_key (str or None): obsm key to use instead (e.g. "X_pca").
    - features (list of str or None): Restrict to these var names. Use it whenever the two objects
      may not share the same variable order, since otherwise columns are matched by position.
    - target_obs_key (str or None): obs column to write into; defaults to `label_key`.
    - copy (bool): Return a labelled copy instead of writing into adata_target in place.

    Returns:
    - anndata.AnnData if copy=True, otherwise None (adata_target is modified in place).
    """
    if (layer is not None) and (obsm_key is not None):
        raise ValueError("Provide at most one of 'layer' or 'obsm_key', not both.")

    if label_key not in adata_ref.obs.columns:
        raise ValueError(f"Label column '{label_key}' not found in adata_ref.obs.")

    target_obs_key = target_obs_key or label_key
    y_train = adata_ref.obs[label_key].to_numpy()

    def _get_matrix(a: anndata.AnnData) -> np.ndarray:
        if obsm_key is not None:
            if obsm_key not in a.obsm:
                raise ValueError(f"obsm_key '{obsm_key}' not found in a.obsm.")
            M = a.obsm[obsm_key]
            M = M.A if hasattr(M, "A") else (M.toarray() if hasattr(M, "toarray") else np.asarray(M))
            return M

        if layer is not None:
            if layer not in a.layers:
                raise ValueError(f"layer '{layer}' not found in a.layers.")
            M = a.layers[layer]
        else:
            M = a.X

        if features is not None:
            feats = pd.Index(features)
            missing = feats.difference(a.var_names)
            if len(missing) > 0:
                raise ValueError(f"Some requested features are missing: {list(missing)}")
            idx = a.var_names.get_indexer(feats)
            M = M[:, idx]

        M = M.A if hasattr(M, "A") else (M.toarray() if hasattr(M, "toarray") else np.asarray(M))
        return M

    X_train = _get_matrix(adata_ref)
    X_full = _get_matrix(adata_target)

    if X_train.ndim != 2 or X_full.ndim != 2:
        raise ValueError("Feature matrices must be 2D.")
    if X_train.shape[1] != X_full.shape[1]:
        raise ValueError(f"Feature dimension mismatch: ref={X_train.shape[1]}, target={X_full.shape[1]}.")

    if not np.isfinite(X_train).all():
        raise ValueError("Training matrix contains non-finite values (NaN/Inf). Clean or filter before training.")
    if not np.isfinite(X_full).all():
        raise ValueError("Target matrix contains non-finite values (NaN/Inf). Clean or filter before prediction.")

    knn = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    y_pred = knn.predict(X_full)

    if copy:
        ad_out = adata_target.copy()
        ad_out.obs[target_obs_key] = pd.Categorical(y_pred) if isinstance(adata_ref.obs[label_key].dtype, pd.CategoricalDtype) else y_pred
        return ad_out
    else:
        adata_target.obs[target_obs_key] = pd.Categorical(y_pred) if isinstance(adata_ref.obs[label_key].dtype, pd.CategoricalDtype) else y_pred
        return None