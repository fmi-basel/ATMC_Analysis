import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
from itertools import product
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
import anndata
import phenograph
from pyslingshot import Slingshot
from Fcts_Base import save_fig, load_img_mask_by_UID, remove_uns, add_zarr_uns

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
        elif representation in ad.obsm:
            data_all = ad.obsm[representation]
        else:
            raise ValueError(f"Layer '{representation}' not found in ad.layers or ad.obsm.")
    else:
        data_all = ad.X

    # Subset features used for PCA input if specified
    if dimred_feats is not None:
        missing_feats = [f for f in dimred_feats if f not in ad.var_names]
        if missing_feats:
            raise ValueError(f"Some features in dimred_feats not found: {missing_feats}")
        feat_indices = [ad.var_names.get_loc(f) for f in dimred_feats]
        data = data_all[:, feat_indices]
    else:
        data = data_all

    if np.isnan(data).any():
        raise ValueError("Input data contains NaNs. Please preprocess or filter before PCA.")

    # Perform PCA
    pca = PCA(n_components=number_components, random_state=0)
    pcs = pca.fit_transform(data)
    ad.obsm["X_pca"] = pcs

    total_var = np.sum(pca.explained_variance_ratio_) * 100
    n_pc_pairs = number_components // 2
    n_feats = len(plot_feat)

    # Setup figure and axes grid: rows = PC pairs, cols = features
    fig, ax = plt.subplots(
        nrows=n_pc_pairs, ncols=n_feats,
        figsize=(4 * n_feats, 3.5 * n_pc_pairs),
        gridspec_kw={"hspace": 0.4, "wspace": 0.5},
        squeeze=False
    )

    #fig.suptitle(f"Total Variance Explained: {total_var:.2f}%", fontsize=16)
    fig.subplots_adjust(top=0.90)

    for col, feat in enumerate(plot_feat):
        iterator = 1
        for row in range(n_pc_pairs):
            axis = ax[row, col]

            # Plotting: categorical (obs) or continuous (var_names)
            if feat in ad.obs.columns:  # categorical
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
            elif feat in ad.var_names:  # continuous
                gene_exp = ad[:, feat].X
                if hasattr(gene_exp, "toarray"):
                    gene_exp = gene_exp.toarray().flatten()
                else:
                    gene_exp = np.array(gene_exp).flatten()
                scatter = axis.scatter(
                    pcs[:, iterator - 1], pcs[:, iterator],
                    c=gene_exp, cmap=palette, s=20, alpha=0.7, edgecolor='none'
                )
                if row == 0:  # colorbar once per column
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(scatter, cax=cax, orientation="vertical")
                    # cbar.set_label(feat, fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
            else:
                axis.text(0.5, 0.5, f"Feature '{feat}' not found", ha="center", va="center", fontsize=10)
                axis.set_axis_off()
                iterator += 2
                continue

            # Axis labeling & grid
            x_var = pca.explained_variance_ratio_[iterator - 1] * 100
            y_var = pca.explained_variance_ratio_[iterator] * 100
            axis.set_xlabel(f"PC{iterator}\nExplained var: {x_var:.2f}%", fontsize=9)
            axis.set_ylabel(f"PC{iterator+1}\nExplained var: {y_var:.2f}%", fontsize=9)
            axis.tick_params(left=True, bottom=True, labelsize=8)
            axis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

            # Feature title only in the first row
            if row == 0:
                axis.set_title(feat, fontsize = 12, y = 1.03)
            else:
                axis.set_title("")

            # Remove tick labels to reduce clutter
            axis.set_xticklabels([])
            axis.set_yticklabels([])

            iterator += 2

    # Hide any unused axes if number of PC pairs * features grid is bigger than needed
    for col in range(n_feats):
        for row in range(n_pc_pairs):
            # If the PCs to plot exceed available components, hide the axis
            if (row + 1) * 2 > number_components:
                ax[row, col].set_visible(False)

    #plt.tight_layout(rect=[0, 0, 1, 0.90])

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "PCA", dpi=300)


    # PCA Loadings heatmap
    comps = [f"PC{i + 1}" for i in range(number_components)]
    feature_index = ad.var_names if dimred_feats is None else dimred_feats
    pca_loadings = pd.DataFrame(pca.components_.T, index=feature_index, columns=comps)

    fig, ax = plt.subplots(1, 1, figsize=(6, 15))
    sns.heatmap(pca_loadings, ax=ax, cmap="RdBu_r", center=0, robust=True, cbar=True,
                cbar_kws={'shrink': 0.5})

    ax.set_title("PCA Loadings", fontsize=16, y=1.02)
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis='x', labelrotation=45, labelsize=10)

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "PCA_Heatmap", dpi=300)

    return ad

def compute_DC(ad, plot_feat, number_components=20, n_neighbors=15, dimred_feats=None, representation=None, palette="magma", save_plot=False):
    """
    Compute and visualize diffusion map components for an AnnData object.

    Parameters:
        ad (anndata.AnnData): Input AnnData object.
        plot_feat (list of str): Features for coloring diffusion component scatter plots.
                                 Features can be categorical columns in `ad.obs` or continuous
                                 genes/numeric features in `ad.var_names`.
        number_components (int): Number of diffusion components to compute.
        n_neighbors (int): Number of neighbors for `scanpy.pp.neighbors`.
        dimred_feats (list of str or None): Subset of features (genes) from `ad.var_names` to use.
                                            If None, all features in the selected data are used.
        representation (str or None): Key in `ad.layers` or `ad.obsm` to use as input data.
                                      If None, uses `ad.X`.
        palette (str): Colormap for continuous feature scatter plots (default "magma").
        save_plot (bool): Whether to save the generated plots to directory in `ad.uns['plot_dir']`.
    """

    if isinstance(plot_feat, str):
        plot_feat = [plot_feat]

    # Select data matrix
    if representation is not None:
        if representation in ad.layers:
            data_all = ad.layers[representation]
        elif representation in ad.obsm:
            data_all = ad.obsm[representation]
        else:
            raise ValueError(f"Representation '{representation}' not found in ad.layers or ad.obsm.")
    else:
        data_all = ad.X

    # Subset features if specified
    if dimred_feats is not None:
        missing = [f for f in dimred_feats if f not in ad.var_names]
        if missing:
            raise ValueError(f"Features in dimred_feats not found: {missing}")
        indices = [ad.var_names.get_loc(f) for f in dimred_feats]
        data = data_all[:, indices]
    else:
        data = data_all

    if np.isnan(data).any():
        raise ValueError("Input data contains NaNs. Please preprocess or filter before diffusion mapping.")

    # Prepare AnnData for diffusion map
    ad_sub = anndata.AnnData(data.copy(), obs=ad.obs.copy(),
                           var=ad.var if dimred_feats is None else ad.var.loc[dimred_feats].copy())

    # Compute neighborhood graph (required for diffusion map)
    sc.pp.neighbors(ad_sub, n_neighbors=n_neighbors)

    # Compute diffusion map embedding
    sc.tl.diffmap(ad_sub, n_comps=number_components)

    # Extract diffusion components (skip DC0)
    diffmap_all = ad_sub.obsm["X_diffmap"]
    n_avail = diffmap_all.shape[1]
    start_idx = 1
    end_idx = min(number_components + 1, n_avail)
    pcs = diffmap_all[:, start_idx:end_idx]

    # Store full diffusion components in original AnnData
    ad.obsm["X_diffmap"] = diffmap_all
    ad.uns["diffmap_evals"] = ad_sub.uns["diffmap_evals"]

    n_dc_pairs = (end_idx - start_idx) // 2
    n_feats = len(plot_feat)

    fig, ax = plt.subplots(nrows=n_dc_pairs, ncols=n_feats,
                           figsize=(4 * n_feats, 3.5 * n_dc_pairs),
                           gridspec_kw={"hspace": 0.4, "wspace": 0.3},
                           squeeze=False)

    #fig.suptitle(f"Diffusion Components", fontsize=16)
    #fig.subplots_adjust(top=0.95)

    for col, feat in enumerate(plot_feat):
        idx = 0
        for row in range(n_dc_pairs):
            axis = ax[row, col]

            if feat in ad.obs.columns:  # categorical
                hue_data = ad.obs[feat]
                n_cat = hue_data.nunique() if hasattr(hue_data, "nunique") else None
                palette_used = "tab10" if n_cat is not None and n_cat <= 10 else "Set2"
                sns.scatterplot(ax=axis, x=pcs[:, idx], y=pcs[:, idx + 1],
                                hue=hue_data, s=20, alpha=0.7,
                                palette=palette_used,
                                legend='brief' if (row == 0) else False,
                                edgecolor='none')
            elif feat in ad.var_names:  # continuous
                gene_exp = ad[:, feat].X
                if hasattr(gene_exp, "toarray"):
                    gene_exp = gene_exp.toarray().flatten()
                else:
                    gene_exp = np.array(gene_exp).flatten()
                scatter = axis.scatter(pcs[:, idx], pcs[:, idx + 1],
                                      c=gene_exp, cmap=palette, s=20,
                                      alpha=0.7, edgecolor='none')
                if row == 0:
                    divider = make_axes_locatable(axis)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    cbar = fig.colorbar(scatter, cax=cax, orientation="vertical")
                    # cbar.set_label(feat, fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
            else:
                axis.text(0.5, 0.5, f"Feature '{feat}' not found", ha="center", va="center", fontsize=10)
                axis.set_axis_off()
                idx += 2
                continue

            axis.set_xlabel(f"DC{idx+start_idx}", fontsize=9)
            axis.set_ylabel(f"DC{idx+1+start_idx}", fontsize=9)
            axis.tick_params(left=True, bottom=True, labelsize=8)
            axis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)

            if row == 0:
                axis.text(0.5, 1, feat, fontsize=12, va = "center", ha = "center", transform = axis.transAxes)
            else:
                axis.set_title("")

            axis.set_xticklabels([])
            axis.set_yticklabels([])

            idx += 2

    # Hide unused axes if grid is bigger than necessary
    for col in range(n_feats):
        for row in range(n_dc_pairs):
            if (row + 1) * 2 + start_idx > end_idx:
                ax[row, col].set_visible(False)

    plt.tight_layout()

    if save_plot:
        save_fig(fig, ad.uns["plot_dir"], "DiffusionComponents", dpi=300)

    return ad


def compute_UMAP(ad, plot_feat, n_neighbors, min_dist, dimred_feats, save_plot=False, representation=None):
    """
    Plot UMAP projection(s) for selected features, with grid support for multiple parameter combinations.

    Parameters:
        ad (anndata.AnnData): The input AnnData object.
        plot_feat (str or list of str): Features to visualize on UMAP. obs (categorical) or var_names (continuous/gene).
        n_neighbors (int or list of int): Number of neighbors for UMAP.
        min_dist (float or list of float): Minimum distances for UMAP.
        dimred_feats (list of str): Genes/features for dimensionality reduction (subset columns from ad.var_names).
        save_plot (bool, optional): Whether to save the figure to ad.uns['plot_dir']. Default is False.
        representation (str or None, optional): Key for .layers (expression matrix) or .obsm (representation, e.g. "X_pca"). If None, uses ad.X.

    Returns:
        ad (anndata.AnnData): Returns AnnData with final UMAP embedding in ad.obsm["X_umap"].
    """

    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])

    # Validation of dimred_feats
    if not dimred_feats:
        raise ValueError("dimred_feats is empty.")
    if not all(f in ad.var_names for f in dimred_feats):
        missing = [f for f in dimred_feats if f not in ad.var_names]
        raise ValueError(f"Some features in dimred_feats not found: {missing}")

    if isinstance(plot_feat, str):
        plot_feat = [plot_feat]
    if isinstance(n_neighbors, int):
        n_neighbors = [n_neighbors]
    if isinstance(min_dist, (int, float)):
        min_dist = [min_dist]

    combos = list(product(n_neighbors, min_dist))
    nrows = len(min_dist)
    ncols = len(n_neighbors)

    # Plot setup
    if len(combos) == 1:
        fig, ax_grid = plt.subplots(1, len(plot_feat), figsize=(4 * len(plot_feat), 4))
        if len(plot_feat) == 1:
            ax_grid = [ax_grid]
    else:
        fig, ax_grid = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    coords_dict = {}
    coords = None

    for idx, (n, d) in enumerate(combos):
        ad_sub = ad[:, dimred_feats].copy()
        use_rep = None
        # --- Layer/obsm logic ---
        if representation is not None:
            if representation in ad.layers:   # Use ad.layers
                ad_sub.X = ad_sub.layers[representation]
            elif representation in ad.obsm:   # Use obsm directly as representation
                use_rep = representation
            else:
                raise ValueError(f"Provided layer '{representation}' not in ad.layers or ad.obsm.")
        # --- End logic ---

        sc.pp.neighbors(ad_sub, n_neighbors=n, use_rep=use_rep)
        sc.tl.umap(ad_sub, min_dist=d)
        coords = ad_sub.obsm["X_umap"]
        coords_dict[(n, d)] = coords

        if len(combos) == 1:
            for col, feat in enumerate(plot_feat):
                ax = ax_grid[col]
                if feat in ad.obs.columns:
                    sns.scatterplot(ax=ax, x=coords[:, 0], y=coords[:, 1],
                                    hue=ad.obs[feat], s=5, palette="pastel", legend=True)
                elif feat in ad.var_names:
                    gene_exp = ad[:, feat].X if representation is None or (representation in ad.obsm) else ad[:, feat].layers[representation]
                    if hasattr(gene_exp, 'toarray'):
                        gene_exp = gene_exp.toarray().flatten()
                    else:
                        gene_exp = np.array(gene_exp).flatten()
                    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=gene_exp, cmap="magma", s=5)
                    cb = fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.02)
                    cb.set_label(feat)
                else:
                    ax.set_title(f"{feat} (not found)", fontsize=14)
                ax.set_axis_off()
            fig.suptitle(f"UMAP (Neighbors={n}, MinDist={d})", fontsize=18, y=1.08)
        else:
            row = min_dist.index(d)
            col_index = n_neighbors.index(n)
            ax = ax_grid[row][col_index]
            feat = plot_feat[0]  # Only first feat per panel
            if feat in ad.obs.columns:
                sns.scatterplot(ax=ax, x=coords[:, 0], y=coords[:, 1],
                                hue=ad.obs[feat], s=5, palette="pastel", legend=False)
                ax.set_title(f"n={n}, min_dist={d} {feat}", fontsize=10)
            elif feat in ad.var_names:
                gene_exp = ad[:, feat].X if representation is None or (representation in ad.obsm) else ad[:, feat].layers[representation]
                if hasattr(gene_exp, 'toarray'):
                    gene_exp = gene_exp.toarray().flatten()
                else:
                    gene_exp = np.array(gene_exp).flatten()
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=gene_exp, cmap="magma", s=5)
                ax.set_title(f"n={n}, min_dist={d} {feat}", fontsize=10)
                fig.colorbar(scatter, ax=ax, orientation='horizontal', pad=0.02)
            else:
                ax.set_title(f"{feat} (not found)", fontsize=10)
            ax.set_axis_off()

    plt.tight_layout()
    if save_plot:
        if len(combos) == 1:
            file_name = "UMAP"
        else:
            file_name = "UMAP_grid"
        save_fig(fig, ad.uns["plot_dir"], file_name, dpi=300)

    ad.obsm["X_umap"] = coords
    ad = add_zarr_uns(ad)
    return ad

def compute_tSNE(
    ad,
    plot_feat,
    dimred_feats,
    perplexity = 30.0,
    learning_rate = 200.0,
    save_plot=False,
    representation=None,
    early_exaggeration=12.0,
    random_state=0,
    metric="euclidean",
):
    """
    Plot t-SNE projection(s) for selected features, with grid support for multiple parameter combinations.

    Parameters:
        ad (anndata.AnnData): The input AnnData object.
        plot_feat (str or list of str): Features to visualize on t-SNE. obs (categorical) or var_names (continuous/gene).
        perplexity (float or list of float): t-SNE perplexity values to try.
        learning_rate (float or list of float): t-SNE learning rate values to try.
        dimred_feats (list of str): Genes/features for dimensionality reduction (subset columns from ad.var_names).
        save_plot (bool, optional): Whether to save the figure to ad.uns['plot_dir']. Default is False.
        representation (str or None, optional): Key for .layers (expression matrix) or .obsm (representation, e.g. "X_pca").
                                               If None, uses ad.X.
        early_exaggeration (float): Early exaggeration factor for t-SNE.
        random_state (int): Random seed for reproducibility.
        metric (str): Distance metric for t-SNE when using dense input.
    """
    # --- housekeeping ---
    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])

    # Validation of dimred_feats
    if not dimred_feats:
        raise ValueError("dimred_feats is empty.")
    if not all(f in ad.var_names for f in dimred_feats):
        missing = [f for f in dimred_feats if f not in ad.var_names]
        raise ValueError(f"Some features in dimred_feats not found: {missing}")

    # Normalize input parameter types
    if isinstance(plot_feat, str):
        plot_feat = [plot_feat]
    if isinstance(perplexity, (int, float)):
        perplexity = [float(perplexity)]
    if isinstance(learning_rate, (int, float)):
        learning_rate = [float(learning_rate)]

    combos = list(product(perplexity, learning_rate))
    nrows = len(perplexity)
    ncols = len(learning_rate)

    # Plot setup
    if len(combos) == 1:
        fig, ax_grid = plt.subplots(1, len(plot_feat), figsize=(4 * len(plot_feat), 4))
        if len(plot_feat) == 1:
            ax_grid = [ax_grid]
    else:
        fig, ax_grid = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    coords_dict = {}
    coords = None

    for idx, (ppx, lr) in enumerate(combos):
        # Work on a copy subset with selected features for dimred
        ad_sub = ad[:, dimred_feats].copy()

        # --- Layer/obsm logic ---
        use_rep = None
        if representation is not None:
            if representation in ad.layers:
                ad_sub.X = ad_sub.layers[representation]
            elif representation in ad.obsm:
                use_rep = representation
            else:
                raise ValueError(f"Provided layer '{representation}' not in ad.layers or ad.obsm.")
        # --- End logic ---

        # Clear any stale diffusion-map state which can cause KeyError in some scanpy versions
        if "X_diffmap" in ad_sub.obsm_keys():
            del ad_sub.obsm["X_diffmap"]
        if "diffmap_evals" in ad_sub.uns_keys():
            del ad_sub.uns["diffmap_evals"]

        # Prepare input for t-SNE
        # sc.tl.tsne supports: use_rep to pass an obsm key; otherwise uses ad_sub.X
        # Note: perplexity must be < n_cells; enforce gently
        n_cells = ad_sub.n_obs
        if ppx >= n_cells:
            raise ValueError(f"perplexity ({ppx}) must be less than the number of cells ({n_cells}).")

        # Compute t-SNE
        sc.tl.tsne(
            ad_sub,
            use_rep=use_rep,           # if None, uses ad_sub.X
            n_pcs=None,                # not used when use_rep or X provided already
            perplexity=ppx,
            learning_rate=lr,
            early_exaggeration=early_exaggeration,
            random_state=random_state,
            metric=metric,
            use_fast_tsne=False,       # set True if openTSNE/MulticoreTSNE installed and desired
        )

        coords = ad_sub.obsm["X_tsne"]
        coords_dict[(ppx, lr)] = coords

        # Plotting
        if len(combos) == 1:
            for col, feat in enumerate(plot_feat):
                ax = ax_grid[col]
                if feat in ad.obs.columns:
                    sns.scatterplot(
                        ax=ax,
                        x=coords[:, 0],
                        y=coords[:, 1],
                        hue=ad.obs[feat],
                        s=5,
                        palette="pastel",
                        legend=True,
                    )
                elif feat in ad.var_names:
                    # choose expression from appropriate matrix
                    gene_exp = (
                        ad[:, feat].X
                        if representation is None or (representation in ad.obsm)
                        else ad[:, feat].layers[representation]
                    )
                    if hasattr(gene_exp, "toarray"):
                        gene_exp = gene_exp.toarray().flatten()
                    else:
                        gene_exp = np.array(gene_exp).flatten()
                    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=gene_exp, cmap="magma", s=5)
                    cb = fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.02)
                    cb.set_label(feat)
                else:
                    ax.set_title(f"{feat} (not found)", fontsize=14)
                ax.set_axis_off()
            fig.suptitle(f"t-SNE (Perplexity={ppx}, LR={lr})", fontsize=18, y=1.08)
        else:
            row = perplexity.index(ppx)
            col_index = learning_rate.index(lr)
            ax = ax_grid[row][col_index]
            feat = plot_feat[0]  # Only first feat per panel
            if feat in ad.obs.columns:
                sns.scatterplot(
                    ax=ax,
                    x=coords[:, 0],
                    y=coords[:, 1],
                    hue=ad.obs[feat],
                    s=5,
                    palette="pastel",
                    legend=False,
                )
                ax.set_title(f"perp={ppx}, lr={lr} {feat}", fontsize=10)
            elif feat in ad.var_names:
                gene_exp = (
                    ad[:, feat].X
                    if representation is None or (representation in ad.obsm)
                    else ad[:, feat].layers[representation]
                )
                if hasattr(gene_exp, "toarray"):
                    gene_exp = gene_exp.toarray().flatten()
                else:
                    gene_exp = np.array(gene_exp).flatten()
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=gene_exp, cmap="magma", s=5)
                ax.set_title(f"perp={ppx}, lr={lr} {feat}", fontsize=10)
                fig.colorbar(scatter, ax=ax, orientation="horizontal", pad=0.02)
            else:
                ax.set_title(f"{feat} (not found)", fontsize=10)
            ax.set_axis_off()

    plt.tight_layout()
    if save_plot:
        if len(combos) == 1:
            file_name = "tSNE"
        else:
            file_name = "tSNE_grid"
        save_fig(fig, ad.uns["plot_dir"], file_name, dpi=300)

    # Save final embedding from last combo
    ad.obsm["X_tsne"] = coords
    ad = add_zarr_uns(ad)
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
    visualize_on=None,     # e.g., "X_umap", "X_tsne", "X_pca", or None to auto-pick
    fallback_to_pca=True,  # if visualize_on missing, fall back to auto/PCA
    point_size=6,
):
    """
    Run K-Means clustering on a chosen representation (X, a .layer, or an .obsm matrix),
    optionally across a grid of cluster numbers, and plot results colored by the computed K-Means labels.

    Parameters:
        ad (anndata.AnnData): The input AnnData object.
        n_clusters (int or list of int): Number(s) of clusters for K-Means.
        dimred_feats (list of str): Genes/features used to construct the feature matrix for clustering (subset of ad.var_names).
        save_plot (bool): Whether to save the figure to ad.uns['plot_dir'].
        representation (str or None): If in .layers, uses as X; if in .obsm, uses that matrix; else uses ad.X.
        init (str): K-Means init method.
        n_init (int): Number of K-Means restarts.
        max_iter (int): Max K-Means iterations.
        random_state (int): Random seed for reproducibility.
        add_to_obs_key (str): Base key to store labels in ad.obs. Grid will create suffixes like f"{add_to_obs_key}_{k}".
        visualize_on (str or None): Which 2D embedding to use for plotting (an .obsm key). If None, auto-pick.
        fallback_to_pca (bool): If True and visualize_on is missing or not 2D, compute a quick 2D PCA for plotting only.
        point_size (int): Marker size for scatter plots.
    """

    # --- housekeeping ---
    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])

    # Validation of dimred_feats
    if not dimred_feats:
        raise ValueError("dimred_feats is empty.")
    if not all(f in ad.var_names for f in dimred_feats):
        missing = [f for f in dimred_feats if f not in ad.var_names]
        raise ValueError(f"Some features in dimred_feats not found: {missing}")

    # Normalize parameter types
    if isinstance(n_clusters, int):
        n_clusters = [n_clusters]

    # Determine clustering input matrix M (n_cells x n_features)
    ad_sub = ad[:, dimred_feats].copy()
    if representation is not None:
        if representation in ad.layers:
            M = ad_sub.layers[representation]
        elif representation in ad.obsm:
            M = ad_sub.obsm[representation]
        else:
            raise ValueError(f"Provided representation '{representation}' not in ad.layers or ad.obsm.")
    else:
        M = ad_sub.X

    # Ensure dense for scikit-learn if sparse
    if hasattr(M, "toarray"):
        M = M.toarray()

    # Select 2D coordinates for visualization
    coords = None
    coords_name = None

    def get_coords_from_obsm(key):
        if key in ad.obsm_keys():
            X = ad.obsm[key]
            if X.ndim == 2 and X.shape[1] >= 2:
                return X[:, :2]
        return None

    if visualize_on is not None:
        coords = get_coords_from_obsm(visualize_on)
        coords_name = visualize_on if coords is not None else None

        if coords is None and not fallback_to_pca:
            raise ValueError(
                f"Requested visualize_on='{visualize_on}' not available or not 2D, and fallback_to_pca=False."
            )

    if coords is None:
        # Auto-pick from common 2D embeddings
        for key in ["X_umap", "X_tsne", "X_pca"]:
            coords = get_coords_from_obsm(key)
            if coords is not None:
                coords_name = key
                break

    if coords is None:
        # Compute a quick 2D PCA on M for plotting only
        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(M)
        coords_name = "PCA(2) on clustering matrix"

    # Plot layout:
    multi_k = len(n_clusters) > 1
    if multi_k:
        nrows = len(n_clusters)
        ncols = 1
        fig, ax_grid = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    else:
        fig, ax_grid = plt.subplots(1, 1, figsize=(4, 4))

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

        # store labels
        key_k = f"{"kmeans_labels"}_{k}"
        ad.obs[key_k] = pd.Categorical(labels.astype(str))
        last_labels = labels

        # plotting: always color by computed KMeans labels
        ax = ax_grid[i][0] if multi_k else ax_grid
        sns.scatterplot(
            ax=ax,
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels.astype(str),
            s=point_size,
            palette="tab20",
            legend=not multi_k,  # keep legend for single plot
        )
        ax.set_title(f"k={k} on {coords_name}", fontsize=10)
        ax.set_axis_off()

    plt.tight_layout()
    if save_plot:
        file_name = "KMeans_grid" if multi_k else "KMeans"
        save_fig(fig, ad.uns["plot_dir"], file_name, dpi=300)

    # Save last labels to a canonical key
    ad.obs["kmeans_labels"] = pd.Categorical(last_labels.astype(str))
    ad = add_zarr_uns(ad)
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
    visualize_on=None,     # e.g., "X_umap", "X_tsne", "X_pca", or None to auto-pick
    fallback_to_pca=True,  # if visualize_on missing, fall back to auto/PCA
    random_state=0,
    point_size=6,
):
    """
    Run PhenoGraph clustering (via python-igraph/leidenalg under the hood), optionally across a grid of k,
    and plot results colored by the computed PhenoGraph labels on a chosen 2D embedding.

    Parameters:
        ad (anndata.AnnData): The input AnnData object.
        k (int or list[int]): Neighborhood size(s) for the KNN graph used by PhenoGraph.
        dimred_feats (list[str]): Genes/features used to construct the feature matrix for clustering (subset of ad.var_names).
        save_plot (bool): Whether to save the figure to ad.uns['plot_dir'].
        representation (str or None): If in .layers, uses as X; if in .obsm, uses that matrix; else uses ad.X.
        clustering_metric (str): Distance metric for KNN construction ('euclidean', etc.).
        prune (bool): Whether to prune the graph in PhenoGraph.
        jaccard (bool): Whether to use Jaccard weighting of edges (PhenoGraph default).
        min_cluster_size (int): Clusters smaller than this can be labeled as -1 by PhenoGraph.
        visualize_on (str or None): Which 2D embedding to use for plotting (an .obsm key). If None, auto-pick.
        fallback_to_pca (bool): If True and visualize_on is missing or not 2D, compute a quick 2D PCA for plotting only.
        random_state (int): Random state for reproducibility (passed to PhenoGraph where supported).
        point_size (int): Marker size for scatter plots.
    """

    # --- housekeeping ---
    ad = remove_uns(ad, keys_to_remove=["ome_zarr_dict", "ome_zarr_df"])

    # Normalize parameter types
    if isinstance(k, int):
        k = [k]

    # Determine clustering input matrix M (n_cells x n_features)
    ad_sub = ad[:, dimred_feats].copy()
    if representation is not None:
        if representation in ad.layers:
            M = ad_sub.layers[representation]
        elif representation in ad.obsm:
            M = ad_sub.obsm[representation]
        else:
            raise ValueError(f"Provided representation '{representation}' not in ad.layers or ad.obsm.")
    else:
        M = ad_sub.X

    # Ensure dense for neighbors if sparse
    if hasattr(M, "toarray"):
        M = M.toarray()

    n_cells = M.shape[0]
    for ki in k:
        if ki >= n_cells:
            raise ValueError(f"k ({ki}) must be less than the number of cells ({n_cells}).")

    # Select 2D coordinates for visualization
    coords = None
    coords_name = None

    def get_coords_from_obsm(key):
        if key in ad.obsm_keys():
            X = ad.obsm[key]
            if X.ndim == 2 and X.shape[1] >= 2:
                return X[:, :2]
        return None

    if visualize_on is not None:
        coords = get_coords_from_obsm(visualize_on)
        coords_name = visualize_on if coords is not None else None
        if coords is None and not fallback_to_pca:
            raise ValueError(
                f"Requested visualize_on='{visualize_on}' not available or not 2D, and fallback_to_pca=False."
            )

    if coords is None:
        for key in ["X_umap", "X_tsne", "X_pca"]:
            coords = get_coords_from_obsm(key)
            if coords is not None:
                coords_name = key
                break

    if coords is None:
        pca = PCA(n_components=2, random_state=random_state) # Compute a quick 2D PCA on M for plotting only. PCA in ad obsm["X_pca"] is not changed
        coords = pca.fit_transform(M)
        coords_name = "PCA(2) on clustering matrix"

    # Plot layout
    multi_k = len(k) > 1
    if multi_k:
        nrows = len(k)
        ncols = 1
        fig, ax_grid = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)
    else:
        fig, ax_grid = plt.subplots(1, 1, figsize=(4, 4))

    last_labels = None

    for i, ki in enumerate(k):
        # Build graph with sklearn NN (phenograph can do this internally; we keep explicit control)
        # However, we’ll use phenograph.cluster directly on M so that it handles graph + community detection.
        # cluster returns: communities, graph, Q
        communities, graph, Q = phenograph.cluster(
            M,
            k=ki,
            directed=False,
            prune=prune,
            jaccard=jaccard,
            primary_metric=clustering_metric,
            min_cluster_size = min_cluster_size,
            cluster_algorithm="louvain",  # default in phenograph; could be 'leiden' if supported in installed version
            n_jobs=-1,
            seed=random_state,
        )

        labels = np.array(communities, dtype=int)
        key_k = f"{"phenograph_labels"}_{ki}"
        ad.obs[key_k] = pd.Categorical(labels.astype(str))
        last_labels = labels

        # plotting: always color by computed PhenoGraph labels
        ax = ax_grid[i][0] if multi_k else ax_grid
        sns.scatterplot(
            ax=ax,
            x=coords[:, 0],
            y=coords[:, 1],
            hue=labels.astype(str),
            s=point_size,
            palette="tab20",
            legend=not multi_k,  # keep legend for single plot
        )
        ax.set_title(f"PhenoGraph k={ki} on {coords_name}", fontsize=10)
        ax.set_axis_off()

    plt.tight_layout()
    if save_plot:
        file_name = "PhenoGraph_grid" if multi_k else "PhenoGraph"
        save_fig(fig, ad.uns["plot_dir"], file_name, dpi=300)

    # Save last labels to a canonical key
    ad.obs["phenograph_labels"] = pd.Categorical(last_labels.astype(str))
    ad = add_zarr_uns(ad)
    return ad






def run_phenograph(ad, ad_dimred, k, plot_save_dir, save_plot = False):
    """
    Run PhenoGraph clustering on minmax normalized and log1p-transformed data.

    Parameters:

    ad (anndata.AnnData): The input AnnData object.
    ad_dimred (anndata.AnnData): AnnData object with common features and selected features for dimension reduction.
    k (int): Integer for number of nearest neighbors.
    plot_save_dir (str): Directory to save the generated plots.
    save_plot (bool, optional): Whether to save the plots. Default is False.
    """

    cols = 3

    # Run PhenoGraph
    ad.obs["Phenograph"], _, _ = phenograph.cluster(ad_dimred.to_df(layer = "minmax_transformed"),
                                                    k = k,
                                                    seed = 0)
    # Plot overview
    fig1, ax = plt.subplots(nrows=1, ncols = 1, figsize=(6,6))

    sns.scatterplot(ax = ax,
        x = ad.obsm["UMAP"][:,0],
        y = ad.obsm["UMAP"][:,1],
        s = 20,
        hue = ad.obs["Phenograph"],
        palette = "tab20",
        alpha = 0.5,
        legend = True
        )

    ax.set_axis_off()

    # Plot subplots
    rows = ((np.max(ad.obs.Phenograph)+1)//cols)+1
    fig2, axs = plt.subplots(nrows=rows, ncols = cols, figsize=(cols*2,2*rows))
    
    for ax in axs.flat:
        ax.axis('off')

    for n in range(np.max(ad.obs.Phenograph)+1):
        
        coords = n//cols, n%cols
        
        sns.scatterplot(ax = axs[coords],
            x = ad.obsm["UMAP"][:,0],
            y = ad.obsm["UMAP"][:,1],
            s = 4,
            color = "Gray",
            alpha = 0.1,
            legend = False
                )
        
        sns.scatterplot(ax = axs[coords],
            x = ad[ad.obs.Phenograph == n].obsm["UMAP"][:,0],
            y = ad[ad.obs.Phenograph == n].obsm["UMAP"][:,1],
            s = 10,
            color = "red",
            alpha = 0.5,
            legend = True
                )
        
        axs[coords].set_title(n, fontsize = 10, y = 0.9)

    if save_plot:
        save_fig(fig1, plot_save_dir, f"4Trajectory_PhenoGraph")

    return ad

def plot_random_organoids_cluster(ad, n_organoids, plot_save_dir, channel = 0, seed = 0, save_plot = False, pyramid_level = 1):
    """
    Plot n_organoids random organoids for each PhenoGraph cluster.

    Parameters:

    ad (anndata.AnnData): The input AnnData object.
    n_organoids (int): Number of organoids to plot for each cluster.
    plot_save_dir (str): Directory to save the generated plots.
    seed (int, optional): Seed for reproducibility. Default is 0.
    save_plot (bool, optional): Whether to save the plot. Default is False.
    """

    n = n_organoids

    # Set seeds
    np.random.seed(seed)

    fig, ax = plt.subplots(nrows = np.max(ad.obs.Phenograph)+1, ncols = n, figsize=(n*3,3*np.max(ad.obs.Phenograph)))

    for i in ad.obs["Phenograph"].unique():
        bd_plot = ad[ad.obs["Phenograph"] == i]  # Loop through clusters
        bd_plot = pd.concat([bd_plot.to_df(),bd_plot.obs], axis = 1).sample(n = n).Organoid_ID # Pick n random organoids
        
        for k in range(n):

            img, mask = load_img_mask_by_UID(bd_plot[k], ad.uns["ome_zarr_dict"], ad.uns["table_name"], ad.uns["label_name"], pyramid_level, channel)

            img[mask == 0] = 0
            ax[i,k].imshow(img, interpolation = "nearest", aspect = "auto", cmap = "magma")
            ax[i,k].set_axis_off()  
            ax[i,k].text(x = 0.05, y = 0.93, s =str(i), fontsize = 12, transform=ax[i, k].transAxes, ha = "left", color = "white")
            
    plt.tight_layout()

    if save_plot:
        save_fig(fig, plot_save_dir, f"4Trajectory_OrganoidsPerCluster")

def run_slingshot(
    ad,
    start_cluster,
    num_epochs = 3,
    save_plot = False,
    cluster_key = "phenograph_labels",     # column in ad.obs with cluster labels (categorical or int/str)
    visualize_on = "X_umap",               # 2D embedding in ad.obsm to run/plot on (e.g., "X_umap", "X_tsne", "X_pca")
    point_size = 15,
    cmap_pseudotime = "magma",
):
    """
    Run Slingshot trajectory inference and store results in ad.obs, with clean plotting.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler

    # --------- validations ---------
    if visualize_on not in ad.obsm_keys():
        raise ValueError(f"Embedding '{visualize_on}' not found in ad.obsm.")
    coords = ad.obsm[visualize_on]
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"ad.obsm['{visualize_on}'] must be 2D with at least 2 columns.")
    coords_2d = coords[:, :2]

    if cluster_key not in ad.obs.columns:
        raise ValueError(f"cluster_key '{cluster_key}' not found in ad.obs.")
    clusters_raw = ad.obs[cluster_key]

    # Normalize cluster labels to categorical for stable mapping
    if not pd.api.types.is_categorical_dtype(clusters_raw):
        clusters = clusters_raw.astype(str).astype("category")
    else:
        clusters = clusters_raw.copy()

    # Map cluster labels to integer indices [0..K-1] in the order of categories
    categories = list(clusters.cat.categories)
    label_to_int = {lab: i for i, lab in enumerate(categories)}
    cluster_int = clusters.cat.codes.to_numpy()  # -1 for NaN; disallow
    if np.any(cluster_int < 0):
        raise ValueError("Found NaN cluster labels; please ensure all cells have a valid cluster label.")

    # Determine start node index from provided start_cluster (int or str)
    if isinstance(start_cluster, (int, np.integer)):
        if str(start_cluster) in label_to_int:
            start_node = label_to_int[str(start_cluster)]
        elif 0 <= start_cluster < len(categories):
            start_node = start_cluster
        else:
            raise ValueError(f"start_cluster={start_cluster} not found among cluster labels or valid indices.")
    else:
        if start_cluster in label_to_int:
            start_node = label_to_int[start_cluster]
        else:
            raise ValueError(f"start_cluster='{start_cluster}' not found among cluster labels {categories}.")

    # Build one-hot partition matrix (n_cells x K)
    K = len(categories)
    n = ad.n_obs
    part = np.zeros((n, K), dtype=float)
    part[np.arange(n), cluster_int] = 1.0

    # --------- run Slingshot ---------
    try:
        sl = Slingshot(
            coords_2d,
            part,
            start_node=start_node,
            #debug_level="verbose",
        )
    except NameError as e:
        raise NameError("Slingshot class is not defined/importable in this environment.") from e

    # Diagnostics canvas
    fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    for ax in axes.flat:
        ax.axis("off")

    sl.fit(num_epochs=num_epochs, debug_axes=axes)

    if save_plot:
        save_fig(fig1, ad.uns["plot_dir"], "Trajectory_SlingshotComputation")

    # --------- pseudotime figure ---------
    pt = sl.unified_pseudotime
    if pt is None:
        raise RuntimeError("Slingshot did not produce unified pseudotime.")
    pt_scaled = MinMaxScaler().fit_transform(pt.reshape(-1, 1)).flatten()
    ad.obs["slingshot_pseudotime"] = pt_scaled

    # Per-lineage weights: assign directly as float lists/arrays (no pd.Categorical(None))
    if getattr(sl, "lineages", None) is None or getattr(sl, "cell_weights", None) is None:
        raise RuntimeError("Slingshot did not return lineages/cell_weights.")
    n_lineages = len(sl.lineages)
    # sl.cell_weights is typically a list of arrays per cell; build columns by indexing each lineage
    for i in range(n_lineages):
        ad.obs[f"slingshot_weight_lineage_{i}"] = [w[i] for w in sl.cell_weights]

    # Plot pseudotime on the same 2D embedding
    fig2, ax2 = plt.subplots(ncols=1, figsize=(10, 10))
    ax2.set_title("Pseudotime")
    # Try plotter; fallback to manual scatter
    try:
        sl.plotter.clusters(ax2, color_mode="pseudotime", s=point_size, cmap=cmap_pseudotime)
    except Exception:
        sca = ax2.scatter(
            coords_2d[:, 0],
            coords_2d[:, 1],
            c=pt_scaled,
            s=point_size,
            cmap=cmap_pseudotime,
        )
        fig2.colorbar(sca, ax=ax2, orientation="vertical", pad=0.02)

    ax2.set_axis_off()

    if save_plot:
        save_fig(fig2, ad.uns["plot_dir"], "Trajectory_SlingshotPseudotime")

    # --------- store metadata ---------
    ad.uns["slingshot"] = {
        "start_cluster": start_cluster,
        "cluster_key": cluster_key,
        "visualize_on": visualize_on,
        "num_epochs": num_epochs,
        "categories": categories,
    }
    return ad

