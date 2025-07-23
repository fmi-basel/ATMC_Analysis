import matplotlib.pyplot as plt
import random

from Fcts_Base import load_img_mask_by_UID

"""
***
FILTERING FUNCTIONS
***
"""

def filter_organoids_by(ad, df, feature, values, channel, pyramid_level = 1):
    """
    Filter organoids based on a specified numerical feature range and visualize the removed organoids.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object.
    - df (pd.DataFrame): Input DataFrame containing organoid information.
    - feature (str): Name of the numerical feature to filter.
    - values (tuple): Tuple containing two numerical values representing the lower and upper bounds for filtering.
    """


    df_cut1 = df[df[feature] >= values[0]]
    df_cut2 = df_cut1[df_cut1[feature] <= values[1]]
    print(f"{len(df)-len(df_cut1)} objects removed due to lower boundary ({values[0]}) of {feature}.\n{len(df_cut1)-len(df_cut2)} objects removed due to upper boundary ({values[1]}) of {feature}.\n{len(df_cut2)} objects remain.")

    rows, cols = (4,9)
    if len(df)-len(df_cut1) > 0:
        if len(df)-len(df_cut1) < cols:
            rows = 1
            while len(df)-len(df_cut1) < cols:
                cols -= 1

            fig1 = get_deleted_organoids(ad, df, df_cut1, rows, cols, f"Objects with {feature} <= {values[0]}", feature, channel, pyramid_level)
        else:
            while len(df)-len(df_cut1) < cols*rows:
                rows -= 1

            fig1 = get_deleted_organoids(ad, df, df_cut1, rows, cols, f"Objects with {feature} <= {values[0]}", feature, channel, pyramid_level)

    rows, cols = (4,9)

    if len(df_cut1)-len(df_cut2) > 0:
        if len(df_cut1)-len(df_cut2) < cols:
            rows = 1
            while len(df_cut1)-len(df_cut2) < cols:
                cols -= 1

            fig2 = get_deleted_organoids(ad, df_cut2, df_cut1, rows, cols,f"Objects with {feature} >= {values[1]}", feature, channel, pyramid_level)
        else:
            while len(df_cut1)-len(df_cut2) < cols*rows:
                rows -= 1

            fig2 = get_deleted_organoids(ad, df_cut2, df_cut1, rows, cols,f"Objects with {feature} >= {values[1]}", feature, channel, pyramid_level)
    
    return df_cut2

def get_deleted_organoids(ad, df1, df2, rows, cols, title, feature, channel, pyramid_level):
    """
    Plot organoids that are unique to one Dataframe. Used to get organoids which have been deleted during the filtering step.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object.
    - df1 (pandas.DataFrame): DataFrame2.
    - df2 (pandas.DataFrame): DataFrame1.
    - rows (int): Number of rows in the plot.
    - cols (int): Number of columns in the plot.
    - title (str): Title for the plot.
    - feature (str): Feature to be displayed in the title of each sub-plot.
    - pyramid_level (int): Pyramid level of the image.
    """

    # Check which DataFrame is longer
    if len(df1) >= len(df2):
        df_long = df1
        df_short = df2
    else:
        df_long = df2
        df_short = df1
    
    # Get list of missing organoids in shorter compared to longer DataFrame
    dropped = []
    for x in df_long.index:
        if x not in df_short.index:
            dropped.append(x)
    df = df_long[df_long.index.isin(dropped)]
    if len(df) == 0:
        return
    n = rows*cols
    
    if rows*cols > len(df):
        n = len(df)
    
    # Sample n (row*cols) random organoids of the deleted ones and plot
    removed = df.sample(n=n)
    removed_OID = list(removed.Organoid_ID)
    
    removed.Object =  removed.Object.astype(str)
    removed_filt = list(removed[feature])
    
    fig, ax = plt.subplots(rows, cols, figsize = (cols*2,rows*2))
    fig.suptitle(title, fontsize = 18, y = 1.01)
    i = 0
    for row in range(rows):
        for col in range(cols):
            if  rows == 1:
                coordinates = col
            else:
                coordinates = row,col

            img, mask = load_img_mask_by_UID(removed_OID[i], ad.uns["ome_zarr_dict"], ad.uns["table_name"], ad.uns["label_name"], pyramid_level, channel)

            img[~mask.astype(bool)] = 0
            ax[coordinates].imshow(img, interpolation = "nearest", aspect = "auto", cmap = "magma")
            ax[coordinates].set_title(removed_OID[i]+"\n"+str(removed_filt[i]), fontsize = 8)
            ax[coordinates].set_axis_off()
            if i < n-1:
                i += 1
        
    plt.tight_layout() 

    return fig

def plot_random_organoids(ad, df_raw, df, feature, rows = 10, cols = 10, channel = 0, seed = 0, pyramid_level = 1):
    """
    Plot random organoids from the DataFrame.

    Parameters:
    - ad (anndata.AnnData): Input AnnData object.
    - df_raw (pandas.DataFrame): DataFrame containing organoid information before filtering.
    - df (pandas.DataFrame): DataFrame containing organoid information after filtering.
    - feature (str): Feature to be displayed in the title of each sub-plot.
    - rows (int): Number of rows in the plot.
    - cols (int): Number of columns in the plot.
    - seed (int): Seed for random sampling.
    - pyramid_level (int): Pyramid level of the image.
    """

    print("A total of %d objects have been removed during the filtering process. %d objects remain for further analysis.\n\n" %((len(df_raw)-len(df)), len(df)))

    # Set seed
    random.seed(seed)

    # Get n (row*cols) random organoids from DataFrame
    removed = df.sample(n=rows*cols)
    removed_path = list(removed.PATH)
    removed_OID = list(removed.index)
    removed_size = list(removed[feature])

    fig, ax = plt.subplots(rows, cols, figsize = (rows*2,cols*2))
    fig.suptitle("Surviving Organoids", fontsize = 18, y = 1.00)

    i = 0

    # Load images and plot
    for row in range(rows):
        for col in range(cols):

            img, mask = load_img_mask_by_UID(removed_OID[i], ad.uns["ome_zarr_dict"], ad.uns["table_name"], ad.uns["label_name"], pyramid_level, channel)

            img[mask == 0] = 0
            ax[row,col].imshow(img, interpolation = "nearest", aspect = "auto", cmap = "magma")
            ax[row,col].set_title(removed_OID[i]+"\n"+str(round(removed_size[i],2)), fontsize = 8)
            ax[row,col].set_axis_off()
            i += 1

    fig.tight_layout()

    return fig