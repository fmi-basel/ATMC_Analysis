import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn import preprocessing
import scipy

from Fcts_Base import make_experiment, save_fig

"""
***
QUALITY CONTROL FUNCTIONS
***
"""


def CalculateOutgrowth(ad):
    """
    Calculate outgrowth information per well.

    Parameters:
    - ad (anndata.AnnData): AnnData object containing organoid data.
    """
    # Get experimental setup
    experiment_setup = (make_experiment(ad.uns["source_dir"], ad.obs["Barcode"].unique()))

    # get DF from AD for grouping
    df = pd.concat([ad.to_df(), ad.obs.astype(str)], axis = 1)

    # Empty Outgrowth DF
    OutgrowthDF = pd.DataFrame()

    # Group DF by conditions and fill Outgrowth DF
    OutgrowthDF["Other"] = [x[0] for x in df.groupby(["Other", "Cell_line", "Medium", "Well", "Barcode"]).size().to_frame().index]
    OutgrowthDF["Cell_line"] = [x[1] for x in df.groupby(["Other", "Cell_line", "Medium", "Well", "Barcode"]).size().to_frame().index]
    OutgrowthDF["Medium"] = [x[2] for x in df.groupby(["Other", "Cell_line", "Medium", "Well", "Barcode"]).size().to_frame().index]
    OutgrowthDF["Well"] = [x[3] for x in df.groupby(["Other", "Cell_line", "Medium", "Well", "Barcode"]).size().to_frame().index]
    OutgrowthDF["Barcode"] = [x[4] for x in df.groupby(["Other", "Cell_line", "Medium", "Well", "Barcode"]).size().to_frame().index]
    OutgrowthDF["Organoid_No"] = df.groupby(["Other", "Cell_line", "Medium", "Well", "Barcode"]).size().to_list()
    
    # Check which cells from Outgrowth DF are missing in full PlateLayout
    for bc in experiment_setup.keys():
        
        for well in experiment_setup[bc].keys():
            
            if len(OutgrowthDF[(OutgrowthDF.Barcode == bc) & (OutgrowthDF.Well == well)]) == 0:

                # Create a new row to add to the dataframe and set object count to 0
                new_row = {"Other": experiment_setup[bc][well][3], "Cell_line": experiment_setup[bc][well][2], "Medium":str(experiment_setup[bc][well][0]),"Well":well, "Barcode":bc, "Organoid_No": 0}

                # Append the new row to the dataframe
                OutgrowthDF.loc[len(OutgrowthDF)] = new_row        

    ad.uns["Outgrowth_DF"] = OutgrowthDF  
    
    return ad

def transform_features(ad, control_medium, plot_save_dir, save_plot = False):
    """
    Perform log1p transformation on a subset of area-associated features in the AnnData object and visualize the results.

    Parameters:
    - ad (anndata.AnnData): The input AnnData object.
    - control_medium (str): The medium used as a control. Set in the "Medium" tab of the .xls setup file.
    - plot_save_dir (str): The directory to save the generated plots.
    - save_plot (bool, optional): Whether to save the generated plots. Default is False.
    """

    # get list of area-associated features for transformation
    to_transform = [x for x in list(ad.var_names) if any(y in x for y in ["area", "moments", "potency"])]
    to_transform = [x for x in to_transform if not any(y in x for y in ["ratio"])]

    # Extract DF to work with from AD
    df = ad.to_df().copy()

    # log1P transform subset
    df[to_transform] = np.log1p(df[to_transform])

    # Save as layer
    ad.layers["minmax_transformed"] = preprocessing.MinMaxScaler().fit_transform(df)

    #Standard feats
    fig, ax = plt.subplots(nrows=len(to_transform), ncols=3, figsize=(6,1.5*len(to_transform)))

    for n, feat in enumerate(to_transform):  
        
        # Anndata view of control_medium
        bd = ad[ad.obs.Medium == control_medium]

        p1 = sns.kdeplot(data = bd[:,feat].to_df(layer = "minmax_transformed").dropna()[feat],
                    ax = ax[n,0],
                    fill = True,
                    linewidth = 0,
                    color = "Red")
        
        p1 = sns.kdeplot(data = bd[:,feat].to_df(layer = "minmax_transformed").dropna()[feat],
                    ax = ax[n,0],
                    fill = False,
                    linewidth = 2,
                    color = "Red",
                    alpha = 1)
        
        p1 = sns.kdeplot(data = bd[:,feat].to_df(layer = "minmax").dropna()[feat],
                    ax = ax[n,0],
                    fill = True,
                    linewidth = 0,
                    color = "Green")
        
        p1 = sns.kdeplot(data = bd[:,feat].to_df(layer = "minmax").dropna()[feat],
                    ax = ax[n,0],
                    fill = False,
                    linewidth = 2,
                    color = "Green",
                    alpha = 1)
        
        
        #QQPlot
        p2 = sm.qqplot(np.asarray(bd[:,feat].to_df(layer = "minmax").dropna()[feat].tolist()),
                line='45',
                fit = "True",
                ax = ax[n,1],
                marker='.', markerfacecolor='k', alpha=0.6)
        
        #QQPlot
        p3 = sm.qqplot(np.asarray(bd[:,feat].to_df(layer = "minmax_transformed").dropna()[feat].tolist()),
                line='45',
                fit = "True",
                ax = ax[n,2],
                marker='.', markerfacecolor='k', alpha=0.6)

        #Extract R
        _,b = scipy.stats.probplot(np.asarray(bd[:,feat].to_df(layer = "minmax_transformed").dropna()[feat].tolist()))
        _,b_before = scipy.stats.probplot(np.asarray(bd[:,feat].to_df(layer = "minmax").dropna()[feat].tolist()))
        p1.set(yticklabels=[])
        p1.tick_params(left=False)

        ax[n,1].text(x = 0.05, y = 0.93, s = "before\nr: "+str(round(b_before[2],3)), fontsize = 8, transform=ax[n,1].transAxes, ha = "left")
        ax[n,2].text(x = 0.05, y = 0.93, s = "after\nr: "+str(round(b[2],3)), fontsize = 8, transform=ax[n,2].transAxes, ha = "left")
        
        if feat in to_transform:
            ax[n, 0].set_ylabel(feat, fontsize = 6, color = "red")
        else:
            ax[n, 0].set_ylabel(feat, fontsize = 6)
            
        ax[n, 0].set_xlabel("", fontsize = 10) 

        
    plt.tight_layout()

    if save_plot:
        save_fig(fig, plot_save_dir, "3QC_FeatureTransformation")

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