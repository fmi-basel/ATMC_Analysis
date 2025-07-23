import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import umap
import phenograph
from slingshot import Slingshot

from Fcts_Base import save_fig, load_img_mask_by_UID

"""
***
TRAJECTORY FUNCTIONS
***
"""

def ad_dimred_setup(ad ,remove_features):

    """
    Setup AnnData for dimensionality reduction based on common features across all organoids not part of remove_features

    Parameters:
    - ad (anndata.AnnData): The input AnnData object.
    - remove_features (list): A list of feature keywords to be removed from the dimensionality reduction.
    """


    # Get list of common features and get an anndata object with only common features
    df = ad.to_df(layer = "minmax_transformed")
    common_feats = df.columns[~df.isna().any()].tolist()
    ad_dimred = ad[:,common_feats].copy()

    # Retain only features not in remove_features for dimension reduction and trajectory computation
    used_features = [x for x in list(ad_dimred.var_names) if not any(y in x for y in remove_features)]
    ad_dimred = ad_dimred[:,used_features].copy()
    ad.uns["DimRedFeatures"] = used_features 

    return ad, ad_dimred

def test_UMAP_parameters(ad, ad_dimred, feature_to_plot, plot_save_dir, neighbors = [50, 100, 250, 500], distances = [0.05, 0.10, 0.20, 0.30], save_plot = False):
    """
    Test UMAP parameters for various combinations of n_neighbors and min_distances.

    Parameters:
    - ad (anndata.AnnData): The input AnnData object.
    - ad_dimred (anndata.AnnData): AnnData object with common features and selected features for dimension reduction.
    - feature_to_plot (str): The feature to be plotted on UMAP.
    - plot_save_dir (str): Directory to save the generated plots.
    - neighbors (list, optional): List of neighbor values to test. Default is [50, 100, 250, 500].
    - distances (list, optional): List of distance values to test. Default is [0.05, 0.10, 0.20, 0.30].
    - save_plot (bool, optional): Whether to save the plots. Default is False.
    """

    feat = feature_to_plot

    fig, ax = plt.subplots(nrows=len(distances), ncols=len(neighbors), figsize=(3*len(neighbors),3*len(distances)))

    for col, neighbor in enumerate(neighbors):
        
        for row, dist in enumerate(distances):
            
            ad.obsm["UMAP"] = umap.UMAP(random_state = 0,
                                n_neighbors = neighbor,
                                min_dist = dist).fit_transform(ad_dimred.to_df(layer = "minmax_transformed"))
            
            if (row == 0) & (col == 0):
                set_legend = True
            else:
                set_legend = False
                
            sns.scatterplot(ax = ax[row, col],
                            x = ad.obsm["UMAP"][:,0],
                            y = ad.obsm["UMAP"][:,1],
                            hue = ad.obs[feat],
                            s = 10,
                            legend = set_legend)
            
            ax[row, col].text(x = 0.05, y = 0.93, s = "neighbor: %s\ndist: %s "%(str(neighbor), str(dist)), fontsize = 8, transform=ax[row, col].transAxes, ha = "left")
            ax[row, col].set_axis_off()

    if save_plot:
        save_fig(fig, plot_save_dir, f"4Trajectory_UMAP-Parameters_{feat}")

    return ad

def plot_umap(ad, ad_dimred, plot_feat, n_neighbors, min_dist, plot_save_dir, save_plot = False):
    """
    Plot UMAP for selected features.

    Parameters:

    ad (anndata.AnnData): The input AnnData object.
    ad_dimred (anndata.AnnData): AnnData object with common features and selected features for dimension reduction.
    plot_feat (list): List of features to be plotted on UMAP.
    n_neighbors (int): Number of neighbors for UMAP computation.
    min_dist (float): Minimum distance for UMAP computation.
    plot_save_dir (str): Directory to save the generated plots.
    save_plot (bool, optional): Whether to save the plot. Default is False.
    """

    ad.obsm["UMAP"] = umap.UMAP(random_state = 0,
                                n_neighbors = n_neighbors,
                                min_dist = min_dist).fit_transform(ad_dimred.to_df(layer = "minmax_transformed"))

    #plotting
    fig, ax = plt.subplots(nrows=1, ncols=len(plot_feat), figsize=((4*len(plot_feat)),4))
    fig.suptitle("UMAP", fontsize = 22, y = 1.15)

    for col,feat in enumerate(plot_feat):

        #Check if palette is for cont. or discrete values. If cont. then add colorbar, else stay with standard legend
        if feat not in list(ad.obs) :
            
            f = sns.scatterplot(ax = ax[col],
                            x = ad.obsm["UMAP"][:,0],
                            y = ad.obsm["UMAP"][:,1],
                            hue = [x[0] for x in ad[:, feat].X],
                            s = 5,
                            legend = False,
                            palette = "magma")

            ax[col].figure.colorbar(plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(np.min([x[0] for x in ad[:, feat].X]), np.max([x[0] for x in ad[:, feat].X]))),
                                            cax=fig.add_axes([ax[col].get_position().x0, ax[col].get_position().y1+0.005, ax[col].get_position().width, ax[col].get_position().height/20]),
                                            orientation = "horizontal",
                                            ticklocation = "top")               

        else: 
            f = sns.scatterplot(ax = ax[col],
                            x = ad.obsm["UMAP"][:,0],
                            y = ad.obsm["UMAP"][:,1],
                            hue = ad.obs[feat],
                            s = 5,
                            palette = "pastel",
                            legend = True)

        ax[col].set_title(feat, fontsize = 15, y = 1.14)

        ax[col].set_axis_off()

    if save_plot:
        save_fig(fig, plot_save_dir, f"4Trajectory_UMAP")

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

def run_slingshot(ad, start_cluster, num_epochs, plot_save_dir, save_plot = False):

    """
    Run Slingshot analysis for trajectory inference.

    Parameters:

    ad (anndata.AnnData): The input AnnData object.
    start_cluster (int): Cluster index to start Slingshot analysis.
    num_epochs (int): Number of epochs for Slingshot analysis.
    plot_save_dir (str): Directory to save the generated plots.
    save_plot (bool, optional): Whether to save the plots. Default is False.
    """

    # Reshape phenograph clusters
    cluster_labels = np.asarray(ad.obs["Phenograph"])
    cluster_labels_onehot = np.zeros((cluster_labels.shape[0], cluster_labels.max()+1))
    cluster_labels_onehot[np.arange(cluster_labels.shape[0]), cluster_labels] = 1


    # Run Slingshot
    fig1, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))

    for ax in axes.flat:
        ax.axis('off')

    slingshot = Slingshot(ad.obsm["UMAP"],
                        cluster_labels_onehot,
                        start_node = start_cluster,
                        debug_level='verbose')
    
    slingshot.fit(num_epochs = num_epochs, debug_axes = axes)

    if save_plot:
        save_fig(fig1, plot_save_dir, f"4Trajectory_SlingshotComputation")

    fig2, axes = plt.subplots(ncols=1, figsize=(10, 10))
    axes.set_title('Pseudotime')
    slingshot.plotter.clusters(axes, color_mode='pseudotime', s=15)

    axes.set_axis_off()

    if save_plot:
        save_fig(fig2, plot_save_dir, f"4Trajectory_SlingshotPseudotime")


    return slingshot