from scipy.ndimage import binary_fill_holes, zoom, label, mean, find_objects
from skimage import io, filters, measure, morphology, draw
from ez_zarr import ome_zarr, plotting, utils
from natsort import natsorted, natsort_keygen
from typing import Dict, List, Tuple, Any
from skan import Skeleton, summarize
from skimage.transform import resize
from IPython.display import display
from sklearn import preprocessing
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import pandas as pd
import numpy as np
import scipy.stats
import phenograph
import datetime
import anndata
import random
import glob
import copy
import math
import umap
import sys
import cv2
import os

import warnings
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")


# Disable pandas performance warnings
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


"""
***
SAVING FUNCTIONS
***
"""

def save_df(path, filename, df):
    """
    Save a pandas DataFrame to a CSV file with a unique filename.
    Parameters:
    - path (str): Directory path for saving the file.
    - filename (str): Base filename (without extension).
    - df (pandas.DataFrame): DataFrame to be saved.

    Derived from Suppinger et al., 2023 (https://doi.org/10.1016/j.stem.2023.04.018)
    """
    if not os.path.exists(path):
        os.makedirs(path)

    savepath = os.path.join(path, filename + ".csv")
    i = 0

    # Find unique save path
    while os.path.exists(savepath):
        i += 1
        savepath = os.path.join(path, filename + str(i) + ".csv")

    df.to_csv(savepath, index=False)
    print(f"Saved as:\n{savepath}")

def save_adata(path, filename, adata):
    """
    Save an AnnData object to an h5ad file with a unique filename.

    Parameters:
    - path (str): Directory path for saving the file.
    - filename (str): Base filename (without extension).
    - adata (anndata.AnnData): AnnData object to be saved.

    Derived from Suppinger et al., 2023 (https://doi.org/10.1016/j.stem.2023.04.018)
    """

    if not os.path.exists(path):
        os.makedirs(path)

    savepath = os.path.join(path, filename + ".h5ad")
    i = 0

    # Find unique save path
    while os.path.exists(savepath):
        i += 1
        savepath = os.path.join(path, filename + str(i) + ".h5ad")

    adata.write(savepath, compression="gzip")
    print(f"Saved as:\n{savepath}")

def save_fig(fig, path, filename, dpi=300):
    """
    Save a Matplotlib figure to a PDF and PNG file with a unique filename.

    Parameters:
    - fig (matplotlib.figure.Figure): Matplotlib figure to be saved.
    - path (str): Directory path for saving the files.
    - filename (str): Base filename (without extension).
    - dpi (int): Dots per inch for the figure resolution (default is 300).

    Derived from Suppinger et al., 2023 (https://doi.org/10.1016/j.stem.2023.04.018)
    """

    if not os.path.exists(path):
        os.makedirs(path)

    savepath_pdf = os.path.join(path, filename + ".pdf")
    savepath_png = savepath_pdf.replace(".pdf", ".png")
    i = 0

    # Find unique save path
    while os.path.exists(savepath_pdf) or os.path.exists(savepath_png):
        i += 1
        savepath_pdf = os.path.join(path, filename + str(i) + ".pdf")
        savepath_png = savepath_pdf.replace(".pdf", ".png")

    fig.savefig(savepath_pdf, transparent=True, dpi=dpi, bbox_inches="tight")
    fig.savefig(savepath_png, transparent=True, dpi=dpi, bbox_inches="tight")
    print(f"Saved as:\n{savepath_pdf}\nand\n{savepath_png}")

def save_after_filtering(save_dir, df, df_raw, ad_raw):
    """
    Save the filtered DataFrame, updated AnnData object, and a list of deleted organoid IDs.

    Parameters:
    - save_dir (str): Directory path to save the results.
    - df (pd.DataFrame): Filtered DataFrame containing organoid information.
    - ad_raw (anndata.AnnData): Original AnnData object.
    """

    # Save DF
    save_df(save_dir, "2_FeaturesFiltered"+"_{date:%Y-%m-%d_%Hh%Mmin%Ss}".format(date=datetime.datetime.now()), df)

    # Filter anndata
    ad = ad_raw[df.index,:].copy()

    # Save list of deleted organoid IDs
    ad.uns["deleted_IDs"] = list(set(df_raw.index.to_list()) - set(df.index.to_list()))

    # Save AnnData object
    save_adata(save_dir, "2_FeaturesFiltered", ad)

"""
***
SEGMENTATION FUNCTIONS
***
"""

def segmentation_fidelity_check(ome_zarrs_dict, channels, channel_colors, channel_ranges, n, label_name, pyramid_lvl_plot=4):
    """
    Plot randomly selected images and their corresponding masks from OME-ZARR files.

    Parameters:
    - ome_zarrs_dict (dict): Dictionary containing OME-ZARR files.
    - channels (list): List of channels to plot.
    - channel_colors (list): List of colors for each channel.
    - channel_ranges (list): List of ranges for each channel.
    - n (int): Number of images to plot per barcode.
    - label_name (str): Name of the label to plot.
    - pyramid_lvl_plot (int): Pyramid level to plot.
    """
    # Loop over barcodes
    for barcode in ome_zarrs_dict:

        # Get random wells
        wells = random.sample(ome_zarrs_dict[barcode].names, n)

        # Loop over wells
        for well in wells:

            # Load image and corresponding mask
            img, msk = ome_zarrs_dict[barcode][well].get_array_pair_by_coordinate(label_name, pyramid_level=pyramid_lvl_plot)
            msk = msk[label_name]
            
            # Plot image and mask
            plotting.plot_image(im=img, msk=msk[0], msk_alpha=0.3,
                                channels=channels, channel_colors=channel_colors, channel_ranges=channel_ranges,
                                fig_width_inch=15, fig_height_inch=15,
                                title=f"{barcode} - {well}")

            
            
def bin_img(img, n):
    """
    Bins the input image by a given factor.

    Parameters:
    - img (numpy.ndarray): Input image.
    - n (int): Binning factor.
    """
    # Check if rows are dividable by binning factor. If not, remove one.
    if img.shape[0] % n != 0:
        r_del = img.shape[0] % n
        # Remove a row
        img = img[:-r_del,:]

    # Check if columns are dividable by binning factor. If not, remove one.
    if img.shape[1] % n != 0:
        c_del = img.shape[1] % n
        # remove a column
        img = img[:,:-c_del]

    # Binning
    ydim = int(img.shape[0]/n)
    xdim = int(img.shape[1]/n)
    img = img.reshape(ydim,n,xdim,n).mean(axis=(1,3))
    return img

def load_FilesModel(file_path, segmentation_channel, model_path):
    """
    Loads files and CellPose model

    Parameters:
    - file_path (str): Folder in which files are stored in.
    - segmentation_channel (str): Unique string in file name for the segmentation target channel
    - model_path (str): Path to CellPose model.

    """
        
    # Look for files with segmentation_channel naming
    files = glob.glob(os.path.join(file_path,"*"+segmentation_channel+"*.tif"))
    files.sort()
    print("Found %d files for segmentation."%len(files))

    # Load CellPose model
    model = models.CellposeModel(gpu=True, 
                            pretrained_model=model_path)


    print("Succesfully loaded model %s"%model_path.split("/")[-1])

    return files, model

def test_segmentation(files, model, n, n_bin, flow_threshold, cellprob_threshold, diameter):
    """
    Test the segmentation on a subset of images and visualize the results using Napari.

    Parameters:
    - files (list): List of image files for segmentation.
    - model: Loaded CellPose model.
    - n (int): Number of random sample images for testing.
    - n_bin (int): Binning factor for image processing.
    - flow_threshold (float): Flow threshold for segmentation.
    - cellprob_threshold (float): Cell probability threshold for segmentation.
    - diameter (int): Diameter parameter for segmentation.
    """

    # Get n random sample images from files
    test_subset = random.sample(files, n)

    # Go through subset set run segmentation with specified parameters
    for i in tqdm(range(len(test_subset))):

        filename = test_subset[i]
        print("Currently segmenting %s"%os.path.basename(filename))
        
        # Load Image
        img = io.imread(filename)
        
        # Remove FOV channel if present
        if len(img.shape) != 2:
            img = img[0,:,:] 
    
        # Bin image for faster processing
        if n_bin != 0:
            img = bin_img(img, n_bin)
            
        # Segmentation
        if n_bin != 0:        
            mask = model.eval(img,
                            diameter = diameter/n_bin,
                            channels = [0,0],
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold)[0]
            
        if n_bin == 0:        
            mask = model.eval(img,
                            diameter = diameter,
                            channels = [0,0],
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold)[0]
        
        # Show in napari
        viewer = napari.Viewer(title=os.path.basename(filename))
        
        img = viewer.add_image(img, 
                            colormap = "gray",
                            name = "Image",
                            contrast_limits = [0,5000])
        
        mask = viewer.add_labels(mask,
                                name = "Seg")
        
def segmentation(files, model, n_bin, flow_threshold, cellprob_threshold, diameter, output_folder):
    """
    Perform segmentation and save the segmented masks.

    Parameters:
    - files (list): List of image files for segmentation.
    - model: Loaded CellPose model.
    - n_bin (int): Binning factor for image processing.
    - flow_threshold (float): Flow threshold for segmentation. See CellPose documentatin for further information.
    - cellprob_threshold (float): Cell probability threshold for segmentation. See CellPose documentatin for further information.
    - diameter (float): Diameter parameter for segmentation. Set to 0 for automatic identification. See CellPose documentatin for further information.
    - output_folder (str): Directory to save the segmented masks.
    """

    for i in tqdm(range(len(files))):
        base_name = os.path.basename(files[i])

        print("Currently segmenting %s"%base_name)
        
        # Load image
        img = io.imread(files[i])
        
        # Remove FOV channel if present
        if len(img.shape) != 2:
            img = img[0,:,:] 
        
        # Save original sizes for later upscaling
        y_org_size, x_org_size = img.shape
        
        # Bin image for faster processing
        if n_bin != 0:
            img = bin_img(img, n_bin)
        
        # Run segmentation
        if n_bin != 0:        
            mask = model.eval(img,
                            diameter = diameter/n_bin,
                            channels = [0,0],
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold)[0]
            
        if n_bin == 0:        
            mask = model.eval(img,
                            diameter = diameter,
                            channels = [0,0],
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold)[0]
        
        # Upscale mask again by binning factor
        if n_bin != 0:
            mask = resize(mask, 
                        (y_org_size, x_org_size),
                        anti_aliasing = False,
                        order = 0,
                        preserve_range = True)
        
        
        if os.path.exists(output_folder) == False:
            os.makedirs(output_folder)
            
        # To Image format and apply median filter to smoothen edges
        mask = filters.median(mask)
        mask = Image.fromarray(mask)

        
        # Save segmentation into output folder
          
        mask.save(os.path.join(output_folder,base_name), 
                compression="tiff_lzw")


"""
***
FEATURE EXTRACTION FUNCTIONS
***
"""

def find_staining_in_ABs(stainings, staining_to_find):
    """
    Finds the AB mixes that contain the given staining.
    
    Parameters:
    - stainings (dict): A dictionary where the keys are AB mix names and the values are lists of stainings.
    - staining_to_find (str): The staining to search for.
    """
    found_in = []
    for mix, stainings_list in stainings.items():
        if staining_to_find in stainings_list:
            found_in.append(mix)
    return found_in

def find_barcodes_with_day(experiment_setup, d_string):
    """
    Finds the barcodes where the given "D" string is present.

    Parameters:
    - experiment_setup (dict): The dictionary containing the experiment setup.
    - d_string (str): The "D" string to search for.
    """
    found_barcodes = []
    for barcode, wells in experiment_setup.items():
        for _, treatments in wells.items():
            if d_string in treatments:
                found_barcodes.append(barcode)
                break
    return found_barcodes   

def filter_rows(row):
    """
    Filter rows based on a predefined list of allowed values.

    Parameters:
    - row (str): The value to be checked.
    """

    keep_row = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    return row in keep_row




def extract_ome_zarr_tables(experiment_setup, source, folder, table_name):

    """
    Loads OME-Zarr plates for each barcode, extracts tables, and combines them into a single DataFrame.

    Parameters:
    - experiment_setup: dict
        Nested dictionary with barcode as keys and well-specific metadata as values.
        Example: {barcode: {well: [Medium, AB, Day], ...}, ...}
    - source: str
        Root directory path containing the OME-Zarr folders.
    - folder: list of str
        List of folder names corresponding to each barcode.
    - ome_zarr: module/object
        Module/object providing import_plate and related methods.
    - table_name: str
        Name of the table to extract from each plate.

    Returns:
    - ome_zarr_dict: dict
        Dictionary mapping barcodes to loaded OME-Zarr plate objects.
    - ome_zarr_df: pd.DataFrame
        Combined DataFrame containing all extracted tables, annotated with metadata.
    """

    if len(folder) != len(experiment_setup):
        raise ValueError("Length of folder list must match number of barcodes in experiment_setup.")

    ome_zarr_dict = {}
    all_plate_dfs = []

    for i, barcode in enumerate(experiment_setup):
        plate = ome_zarr.import_plate(os.path.join(source, folder[i]))
        ome_zarr_dict[barcode] = plate
        df_lst = plate.get_table(table_name)

        # Prepare well and path lists
        wells = plate.get_names()
        paths = plate.paths

        # Annotate each DataFrame with well and path
        for df, well, path in zip(df_lst, wells, paths):
            df['well'] = well
            df['path'] = path

        plate_df = pd.concat(df_lst, ignore_index=False)
        plate_df["Barcode"] = barcode
        plate_df["UID"] = barcode + "-" + plate_df["well"].astype(str) + "-" + plate_df.index.astype(str)

        # Extract metadata with safe access
        meta = experiment_setup[barcode]
        plate_df["Medium"] = plate_df["well"].map(lambda w: meta.get(w, [None, None, None])[0])
        plate_df["AB"] = plate_df["well"].map(lambda w: meta.get(w, [None, None, None])[1])
        plate_df["Day"] = plate_df["well"].map(lambda w: meta.get(w, [None, None, None])[-1])

        all_plate_dfs.append(plate_df)

    ome_zarr_df = pd.concat(all_plate_dfs, ignore_index=True)

    return ome_zarr_dict, ome_zarr_df


def make_experiment(folder_path, barcodes):
    """
    Create an experiment setup dictionary from PlateSetup.xls files.

    Parameters:
    - folder_path (str): The directory path containing PlateSetup.xls files.
    - barcodes (list): List of barcode identifiers.

    Returns:
    dict: Experiment setup dictionary.
    """
    experiment = {}
    xl_counter = 0
    ## First make plate layout
    for barcode in barcodes:
                    experiment[barcode] = {}
                    for row in ["A","B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]:
                        for col in range(1,25):
                            if col < 10:
                                experiment[barcode][row+str(0)+str(col)] = ["MEDIUM_PLACEHOLDER", "ANTIBODY_PLACEHOLDER", "CELLLINE_PLACEHOLDER", "OTHER_PLACEHOLDER"]
                            if col > 9:
                                 experiment[barcode][row+str(col)] = ["MEDIUM_PLACEHOLDER", "ANTIBODY_PLACEHOLDER", "CELLLINE_PLACEHOLDER", "OTHER_PLACEHOLDER"] 
    
    ## Load .xls files    
    for fyle in glob.glob(os.path.join(folder_path, '*')):
        
        if (".xl" in fyle) & ("PlateSetup_" in fyle):
            if "_all.x" in fyle:
                xl_counter += 1
                print("Reading: "+fyle)
                plate_excel = pd.read_excel(fyle, sheet_name=["Medium", "Antibodies", "CellLine", "Other"], index_col=(0), engine = "openpyxl")
                # Clean Up        
                for sheet_no in plate_excel:
                    
                    sheet = plate_excel[sheet_no]
                                      
                    rows = sheet.index.tolist()
                    rows_filtered = list(filter(filter_rows,rows))
                    
                    sheet = sheet.loc[rows_filtered]
                    sheet = sheet.loc[:, sheet.columns.isin(range(1,25))]
                    plate_excel[sheet_no] = sheet
                
                    # Fill up Dict          
                    for barcode in barcodes:
                        for row in sheet.index:
                            for col in sheet.columns:
                                if col < 10:
                                    experiment[barcode][row+str(0)+str(col)][0] = plate_excel["Medium"].at[row,col]
                                    experiment[barcode][row+str(0)+str(col)][1] = plate_excel["Antibodies"].at[row,col]
                                    experiment[barcode][row+str(0)+str(col)][2] = plate_excel["CellLine"].at[row,col]
                                    experiment[barcode][row+str(0)+str(col)][3] = plate_excel["Other"].at[row,col]
                                if col > 9:
                                    experiment[barcode][row+str(col)][0] = plate_excel["Medium"].at[row,col]
                                    experiment[barcode][row+str(col)][1] = plate_excel["Antibodies"].at[row,col]
                                    experiment[barcode][row+str(col)][2] = plate_excel["CellLine"].at[row,col]
                                    experiment[barcode][row+str(col)][3] = plate_excel["Other"].at[row,col]
                                    
    for fyle in glob.glob(os.path.join(folder_path, '*')):
        
        if (".xl" in fyle) & ("PlateSetup_" in fyle):
            if "_all.x" not in fyle:
                print("Reading: "+fyle)
                xl_counter += 1
                plate_excel = pd.read_excel(fyle, sheet_name=["Medium", "Antibodies", "CellLine", "Other"], index_col=(0), engine = "openpyxl")
                # Clean Up        
                for sheet_no in plate_excel:
                    
                    sheet = plate_excel[sheet_no]
                                      
                    rows = sheet.index.tolist()
                    rows_filtered = list(filter(filter_rows,rows))
                    
                    sheet = sheet.loc[rows_filtered]
                    sheet = sheet.loc[:, sheet.columns.isin(range(1,25))]
                    plate_excel[sheet_no] = sheet
                
                
                # Fill up Dict                                        
                barcode = os.path.basename(fyle).split("PlateSetup_")[-1].split(".")[0]
                
                if barcode not in barcodes:
                    sys.exit("Error! Barcode in excel-filename not in list of barcodes.")
                
                for row in sheet.index:
                    for col in sheet.columns:
                        if col < 10:
                            experiment[barcode][row+str(0)+str(col)][0] = plate_excel["Medium"].at[row,col]
                            experiment[barcode][row+str(0)+str(col)][1] = plate_excel["Antibodies"].at[row,col]
                            experiment[barcode][row+str(0)+str(col)][2] = plate_excel["CellLine"].at[row,col]
                            experiment[barcode][row+str(0)+str(col)][3] = plate_excel["Other"].at[row,col]
                        if col > 9:
                            experiment[barcode][row+str(col)][0] = plate_excel["Medium"].at[row,col]
                            experiment[barcode][row+str(col)][1] = plate_excel["Antibodies"].at[row,col]
                            experiment[barcode][row+str(col)][2] = plate_excel["CellLine"].at[row,col]
                            experiment[barcode][row+str(col)][3] = plate_excel["Other"].at[row,col]
 
    
    bcs = experiment.keys()
    wells = experiment[barcodes[0]].keys()
    experiment_int = copy.deepcopy(experiment)
    for bc in bcs:
        for well in wells:
            if "PLACEHOLDER" in str(experiment[bc][well][0]):
                experiment_int[bc].pop(well, None)
            elif str(experiment[bc][well][0]) == "nan":
                experiment_int[bc].pop(well, None)
    experiment = experiment_int
    if xl_counter == 0:
        print("No PlateSetup file found at source location. Check naming and source folder.")

    display_experiment_setup(experiment)
    return experiment 

def display_experiment_setup(experiment):
    """
    Display the experiment setup for each barcode.

    Parameters:
    - experiment (dict): Experiment setup dictionary.
    """    
    # Extract row and col names                         
    for barcode in experiment.keys():
        current_plate = experiment[barcode]
        rows = list(set([x[0] for x in current_plate.keys()]))

        cols = list(set([x[1:] for x in current_plate.keys()]))

        rows.sort(), cols.sort()
        # Use them to make DataFrame of the same size
        df_hm = pd.DataFrame(index = rows, columns = cols)

        # Fill up DataFrame with data
        for well in current_plate.keys():
            df_hm.loc[well[0], well[1:]] = current_plate[well]
        print("\n\nPlate-Barcode: %s"%barcode)
        display(df_hm)
    
    return None

def resize_image(img, target_size):
    """
    Resizes a 2D (grayscale) image to the target size while maintaining the aspect ratio.
    
    Parameters:
    - img (numpy.ndarray): The input 2D (grayscale) image.
    - target_size (tuple): A tuple containing the target height and width (height, width).
    """
    # Get the size of the input image
    h, w = img.shape[:2]
    
    # Calculate the scaling factors
    scale_x = target_size[1] / w
    scale_y = target_size[0] / h
    
    # Resize the image while maintaining the aspect ratio
    resized_img = zoom(img, (scale_y, scale_x), order=0, mode = 'nearest')
    
    resized_img[resized_img != 0] = 255

    return resized_img

def load_image_by_UID(ome_zarr_df, UID, channel, label_name, pyramid_level = 0):
        """
        Returns a 2D np.array from a ROI based on the UID (barcode-well-id)
        
        Parameters:
        - ome_zarr_df (dict): Table containing all ome_zarrs tables of the experiment.
        - UID (string): Unique ID for organoids (barcode-well-id).
        - channel (int): Channel to extract.
        - label_name (string): Name of table which contains organoid segmentations.
        - pyramid_level (int): The OME-Zarr pyramid level to extract from the ROI.
        """

        UID = UID.split("-")

        df = ome_zarr_df[(ome_zarr_df.Barcode == UID[0]) & (ome_zarr_df.well == UID[1])].iloc[int(UID[2])]
        well_ov = ome_zarr.Image(df.path)

        img,_ = well_ov.get_array_pair_by_coordinate(label_name, 
            upper_left_yx = (df.y_micrometer, df.x_micrometer), 
            lower_right_yx = (df.y_micrometer+df.len_y_micrometer, df.x_micrometer+df.len_x_micrometer), 
            coordinate_unit = 'micrometer', 
            pyramid_level = pyramid_level, 
            pyramid_level_coord = pyramid_level, 
            verbose = False)

        return img[channel]

def load_mask_by_UID(ome_zarr_df, UID, label_name, pyramid_level = 0):
    
        """
        Returns a 2D np.array from a ROI based on the UID (barcode-well-id)
        
        Parameters:
        - ome_zarr_df (dict): Table containing all ome_zarrs tables of the experiment.
        - UID (string): Unique ID for organoids (barcode-well-id).
        - label_name (string): Name of table which contains organoid segmentations.
        - pyramid_level (int): The OME-Zarr pyramid level to extract from the ROI.
        """
        UID = UID.split("-")
    
        df = ome_zarr_df[(ome_zarr_df.Barcode == UID[0]) & (ome_zarr_df.well == UID[1])].iloc[int(UID[2])]
        well_ov = ome_zarr.Image(df.path)

        _,img = well_ov.get_array_pair_by_coordinate(label_name, 
                             upper_left_yx = (df.y_micrometer, df.x_micrometer), 
                             lower_right_yx = (df.y_micrometer+df.len_y_micrometer, df.x_micrometer+df.len_x_micrometer), 
                             coordinate_unit = 'micrometer', 
                             pyramid_level = pyramid_level, 
                             pyramid_level_coord = pyramid_level, 
                             verbose = False)
    
        img = img[label_name][0].astype("uint8")
        
        img[img!= int(UID[2])+1] = 0 # Remove other labels
        img[img != 0] = 255 # Set label to 1

        return img

def estimate_staining_thresholds(ome_zarr_df, stainings, experiment_setup, label_name, pyramid_level = 0, control_condition = None, n = 20, seed = 0, sigma = 3, q = 0.5):
    """
    Estimate staining thresholds for a set of stainings based on organoid images.

    Parameters:
    - ome_zar_df (dict): Dictionary containing OME-Zarr wrappers for whole experiment with barcode as keys.
    - stainings (dict): Dictionary mapping antibody mix names to staining names.
    - experiment_setup (dict): Experiment setup dictionary.
    - label_name (str): Name of the OME-Zarr organoid segmentations.
    - pyramid_level (int): Pyramid level for image processing (default is 0).
    - control_condition (str): Control condition for threshold estimation (default is "None").
    - n (int): Number of organoids to sample for each staining and timepoint (default is 20).
    - seed (int): Random seed for reproducibility (default is 0).
    - sigma (int): Standard deviation for Gaussian smoothing (default is 3).
    - q (float): Quantile value for threshold calculation (default is 0.5).
    """
    
    # Set random seed
    random.seed(seed)

    # Extract thresholds and couple to stainings
    thresholds = {}
    for ab_mix in stainings.keys():
        for i, stain in enumerate(stainings[ab_mix]):
            thresholds[stain] = [i+1, [], 0]

    # Extract timepoints
    other = []
    for plate in experiment_setup.keys():
        for well in experiment_setup[plate].keys():
            if experiment_setup[plate][well][3] not in other:
                other.append(experiment_setup[plate][well][3])

    dict_org = {}
    for ab in thresholds.keys():
        dict_org[ab] = {}
        for cond in other:
            dict_org[ab][cond] = []

    dict_org_lst = {}
    for ab in thresholds.keys():
        dict_org_lst[ab] = {}

    timepoints_lst = []
    
    # Loop through stainings
    for stain in thresholds:
        # Get list of where staining is found
        ABs_lst = find_staining_in_ABs(stainings, stain)
        
        # Loop through days:
        for day in other:
            # Get list of barcodes which have this timepoint
            barcodes_day = find_barcodes_with_day(experiment_setup, day)
            
            if control_condition != None:
                filtered_df = ome_zarr_df[(ome_zarr_df['Barcode'].isin(barcodes_day)) & (ome_zarr_df['AB'].isin(ABs_lst)) & (ome_zarr_df["Medium"] == control_condition)]
            else:
                filtered_df = ome_zarr_df[(ome_zarr_df['Barcode'].isin(barcodes_day)) & (ome_zarr_df['AB'].isin(ABs_lst))]
            
            dict_org[stain][day] = list(filtered_df.UID)
            timepoints_lst = timepoints_lst+[*filtered_df.Day]

    # Sort timepoints
    timepoints_lst = natsorted(list(set(timepoints_lst)))

    # Loop through stainings and calculate the threshold of n random organoids and save the quantile of it               
    for ab in dict_org:
        lst = []
        for cond in dict_org[ab]:
            if n > len(dict_org[ab][cond]):
                print("Number of organoids in conditon %s is lower than n. Taking all organoids instead (%d)" %(ab+" "+cond, len(dict_org[ab][cond])))
                lst = dict_org[ab][cond] + lst
            else:
                lst = random.sample(dict_org[ab][cond], n) + lst

        dict_org_lst[ab] = lst

    for ab in dict_org.keys():
        for fyle in dict_org_lst[ab]:

            # Load img from OME-ZARR
            img = load_image_by_UID(ome_zarr_df, fyle, thresholds[ab][0]-1, label_name, pyramid_level)

            # Smooth
            img = filters.gaussian(img, sigma = sigma, preserve_range=True)

            # Threshold and add to list
            if np.max(img) != 0:
                thresholds[ab][1].append(filters.threshold_triangle(img))
            else:
                thresholds[ab][1].append(np.nan)
        if len(thresholds[ab][1])>0:
            thresholds[ab][2] = np.nanquantile(thresholds[ab][1], q = q)
        else:
            thresholds[ab][2] = np.nan

    return thresholds, dict_org, timepoints_lst

def threshold_change(threshold, staining, new_threshold):
    """
    Update the threshold value for a specific staining.

    Parameters:
    - threshold (dict): Dictionary containing threshold information.
    - staining (str): Staining name to update the threshold.
    - new_threshold (int): New threshold value.
    """
    threshold[staining][2] = new_threshold

    return threshold

def plot_thresholds(thresholds, dict_org, timepoints_lst, ome_zarr_df, label_name):
    """
    Plot threshold distributions and organoid stainings based on set threshold.

    Parameters:
    - thresholds (dict): Dictionary containing threshold information.
    - dict_org (dict): Dictionary containing original organoid image paths.
    - timepoints_lst (list): List of timepoints.
    - ome_zar_df (dict): Dictionary containing OME-Zarr wrappers for whole experiment with barcode as keys.
    - label_name (string): Name of table which contains organoid segmentations.
    """

    rows = len(dict_org[list(dict_org.items())[0][0]].keys())+1 # Gets timepoints if timepoints saved in "Other" tab of .xls setup file
    cols = len(dict_org.keys()) # Gets number of stainings

    # Set up plot
    fig, ax = plt.subplots(nrows = rows, ncols = len(dict_org.keys()), figsize=(3*cols,3*rows))

    # Plot kdeplot of threshold distribution with vertical line showing currently selected threshold
    for i, ab in enumerate(thresholds.keys()):

        if len(thresholds[ab][1])>1:
            sns.kdeplot(ax = ax[0,i],
                        x = thresholds[ab][1],
                        color = "green",
                        cut = 0,
                        fill = True,
                        linewidth = 1
                        )

            ax[0,i].axes.get_yaxis().set_visible(False)    
            ax[0,i].set_title(ab, fontsize = 12)
            ax[0,i].axvline(thresholds[ab][2], color = "red")
            ax[0,i].text(x = 0.9, y = 0.9, s="T: "+str(int(thresholds[ab][2])), transform=ax[0,i].transAxes, ha = "right")

        else:
            ax[0,i].imshow(np.zeros((500, 500)), aspect = "auto", cmap = "binary")
            ax[0,i].set_title(ab, fontsize = 12)
            ax[0,i].set_axis_off()
            ax[0,i].text(x = 0.5, y = 0.5, s="No threshold found", transform=ax[0,i].transAxes, ha = "center")

    # Plots one random organoid per timepoint with used threshold as minimum value
    for col, ab in enumerate(thresholds.keys()):
        for row in range(1,len(dict_org[list(dict_org.items())[0][0]].keys())+1):

            if len(dict_org[ab][timepoints_lst[row-1]]) > 0:
                fyle = random.sample(dict_org[ab][timepoints_lst[row-1]], 1)[0]

                img = load_image_by_UID(ome_zarr_df, fyle, thresholds[ab][0]-1, label_name)
                
                if thresholds[ab][2] >= np.max(img):
                    vmin = np.max(img)
                else:
                    vmin = thresholds[ab][2]
                ax[row,col].imshow(img, vmin = vmin, aspect = "auto", cmap = "inferno")

                ax[row,col].set_axis_off()

            else:
                ax[row,col].imshow(np.zeros((500, 500)), aspect = "auto", cmap = "binary")
                ax[row,col].set_axis_off()
                ax[row,col].text(x = 0.5, y = 0.5, s="No image found", transform=ax[row,col].transAxes, ha = "center")

    plt.tight_layout()
    
    return fig

def dict_add_feat(d, feat_name, value):
    """
    Add a feature and its value to a dictionary if its already in there. If not, add the value as a list with the feat_name as a new key.

    Parameters:
    - d (dict): Dictionary to which the feature and value will be added.
    - feat_name (str): Name of the feature.
    - value: Value of the feature.
    """
    if feat_name in d.keys():
        d[feat_name].append(value)
    else:
        d[feat_name] = [value]

    return d

def get_max_inscribed_circle(image, radius_multiplier, value):
    """
    Get the largest inscribed circle in a binary image.

    Parameters:
    - image (numpy.ndarray): Binary image.
    - radius_multiplier (float): Multiplier for the circle radius.
    - value (int): Value assigned to the circle pixels.
    Derived from Lei Yang, https://gist.github.com/DIYer22/f82dc329b27c2766b21bec4a563703cc
    """
    
    dist_map = cv2.distanceTransform(image, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    _, radius, _, center = cv2.minMaxLoc(dist_map)

    # Create a mask with the same shape as the image
    mask_circle = np.zeros_like(image[:, :])

    # Generate a circular mask using the center and radius*radius_multiplier
    y_range, x_range = np.ogrid[:image.shape[0], :image.shape[1]]
    mask_circle[(x_range - center[0])**2 + (y_range - center[1])**2 <= (radius*radius_multiplier)**2] = value
    
    return mask_circle, radius, center
 
def calculate_eucl_distance(x1, y1, x2, y2, spacing):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - x1, y1, x2, y2 (integers): Coordinates of the two points.
    - spacing (float): Spacing factor, usually pixel size.
    """
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) * spacing

def define_center_touching_branches(df, center_mask):
    """
    Define branches touching the center circle.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing branch information.
    - center_mask (numpy.ndarray): Binary mask of the center circle.
    """
        
    # Get dilated circle mask to see which branches are touching it
    center_mask_dilated = cv2.dilate(center_mask, np.ones((5, 5), np.uint8), iterations = 1)

    # Get src/dist coordinates of branches and check which ones are touching the circle
    touching = []

    for row in df.index:
        y_src = int(df.loc[row,"image-coord-src-0"])
        x_src = int(df.loc[row,"image-coord-src-1"])
        y_dist = int(df.loc[row,"image-coord-dst-0"])
        x_dist = int(df.loc[row,"image-coord-dst-1"])

        if (center_mask_dilated[y_src,x_src] == 255) | (center_mask_dilated[y_dist,x_dist] == 255):
            touching.append(1)

        else:
            touching.append(0)

    df["Touching"] = touching
    
    return df

def determine_skeleton_endpoints(skeleton_data, center, spacing):
    """
    Determine endpoints of skeleton branches based on their distance to the center circle.

    Parameters:
    - skeleton_data (pandas.DataFrame): DataFrame containing skeleton branch information.
    - center (tuple): Center coordinates.
    - spacing (float): Spacing factor. Usually the pixel size.
    """
    for row in skeleton_data.index:

        y_endpoint_src = skeleton_data.loc[row,"image-coord-src-0"]
        x_endpoint_src = skeleton_data.loc[row,"image-coord-src-1"]
        y_endpoint_dst = skeleton_data.loc[row,"image-coord-dst-0"]
        x_endpoint_dst = skeleton_data.loc[row,"image-coord-dst-1"]        

        # Calculate the Euclidean distance
        distance_src = calculate_eucl_distance(center[0], center[1], x_endpoint_src, y_endpoint_src, spacing)
        distance_dst = calculate_eucl_distance(center[0], center[1], x_endpoint_dst, y_endpoint_dst, spacing)

        skeleton_data.loc[row,"distance_src"] = distance_src
        skeleton_data.loc[row,"distance_dst"] = distance_dst

        # Check if src or dst distance is bigger, and set bigger distance as endpoint
        if distance_src < distance_dst:
            skeleton_data.loc[row,"endpoint-y"] = y_endpoint_dst
            skeleton_data.loc[row,"endpoint-x"] = x_endpoint_dst
        else:
            skeleton_data.loc[row,"endpoint-y"] = y_endpoint_src
            skeleton_data.loc[row,"endpoint-x"] = x_endpoint_src
            
    return skeleton_data

def find_adjacent_nonzero_pixel(pixel_coords, image):
    """
    Find adjacent non-zero pixel coordinates to a skeleton pixel.

    Parameters:
    - pixel_coords (tuple): Coordinates of the initial pixel.
    - image (numpy.ndarray): Binary image.
    """

    # Get the coordinates of the initial pixel
    y, x = pixel_coords

    # Check all adjacent pixels (vertical, horizontal, and diagonal)
    adjacent_coords = [
        (y-1, x), (y+1, x), (y, x-1), (y, x+1),
        (y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)
    ]

    # Iterate through the adjacent pixels
    for adj_y, adj_x in adjacent_coords:
        # Check if the adjacent pixel is within the image bounds and non-zero
        if (
            adj_y >= 0 and adj_y < image.shape[0] and
            adj_x >= 0 and adj_x < image.shape[1] and
            image[adj_y, adj_x] != 0
        ):
            endpoint = 0
            return adj_y, adj_x, endpoint

    # Return same start coordinates if no adjacent pixel found
    endpoint = 1
    return y, x , endpoint

def calculate_angle(pixel1_coords, pixel2_coords):
    """
    Calculate the angle between two pixels.

    Parameters:
    - pixel1_coords, pixel2_coords (tuple of integers): Coordinates of the two pixels (x/y).
    """

    # Get the x and y coordinates of the two pixels
    x1, y1 = pixel1_coords
    x2, y2 = pixel2_coords

    # Calculate the angle using arctan2
    angle = np.arctan2(y2 - y1, x2 - x1)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle)

    # Adjust the angle to be between 0 and 360 degrees
    angle_degrees = (angle_degrees + 360) % 360

    return angle_degrees
 
def get_folder_names(file_path):
    """
    Extracts the names of folders from an absolute path.

    Parameters:
    - file_path (str): The absolute path.
    """

    folders = []
    while file_path and file_path != os.path.dirname(file_path):
        file_path, folder = os.path.split(file_path)
        folders.insert(0, folder)
    return folders

def calculate_endpoint(start_coords, angle_degrees, raw):
    """
    Calculate the point at which the branch will hit the segmentation mask given an angle as the input.
    
    Parameters:
    - start_coords (tuple): The x and y coordinates of the starting pixel.
    - angle_degrees (float): The angle in degrees that defines the direction of the line.
    - raw (numpy.ndarray): The raw image data.
    """

    # Get the x and y coordinates of the start pixel
    x_start, y_start = start_coords

    # Convert the angle from degrees to radians
    angle_radians = np.radians(angle_degrees)

    # Initialize the distance
    distance = 5

    # Iterate along the line defined by the angle until a non-zero pixel is encountered
    while True:
        # Calculate the x and y coordinates at the current distance
        x_current = x_start + distance * np.cos(angle_radians)
        y_current = y_start + distance * np.sin(angle_radians)

        # Round the coordinates to the nearest integers
        x_current = int(round(x_current))
        y_current = int(round(y_current))

        # Check if the current pixel is non-zero or out of bounds. If so, stop.
        if y_current <= 0 or x_current <= 0 or y_current >= raw.shape[0] or x_current >= raw.shape[1]:
            distance -= 5
            break

        if raw[y_current, x_current] == 0:
            distance -= 5
            break

        # Increment the distance
        distance += 5

    # Calculate the x and y coordinates of the endpoint
    x_endpoint = x_start + distance * np.cos(angle_radians)
    y_endpoint = y_start + distance * np.sin(angle_radians)

    # Round the coordinates to the nearest integers
    x_endpoint = int(round(x_endpoint))
    y_endpoint = int(round(y_endpoint))

    return x_endpoint, y_endpoint

def draw_line(image, start_coords, end_coords):
    """
    Draw a line on an image given the starting and ending coordinates.

    Parameters:
    - image (numpy.ndarray): The image to draw the line on.
    - start_coords (tuple): The x and y coordinates of the starting pixel.
    - end_coords (tuple): The x and y coordinates of the ending pixel.
    """

    # Create a copy of the image to draw the line on
    image_with_line = image.copy()

    # Extract the x and y coordinates of the start and end pixels
    x_start, y_start = start_coords
    x_end, y_end = end_coords

    # Compute the difference between the start and end coordinates
    dx = abs(x_end - x_start)
    dy = abs(y_end - y_start)

    # Determine the sign of the x and y directions
    sx = 1 if x_start < x_end else -1
    sy = 1 if y_start < y_end else -1

    # Initialize the error term and current coordinates
    error = dx - dy
    x = x_start
    y = y_start

    # Draw the line by setting the corresponding pixels
    while x != x_end or y != y_end:
        # Set the pixel at the current coordinates
        image_with_line[y, x] = 255

        # Compute the error term and update the coordinates
        error_2 = 2 * error
        if error_2 > -dy:
            error -= dy
            x += sx
        if error_2 < dx:
            error += dx
            y += sy

    # Set the endpoint pixel
    image_with_line[y_end, x_end] = 255

    return image_with_line

def find_indices_except_two_highest(lst):
    """
    Find indices of elements in a list, excluding the indices of the two highest values.

    Parameters:
    - lst (list): The input list.
    """
       
    sorted_indices = sorted(range(len(lst)), key=lambda i: lst[i])  # Sort the indices based on the values
    result_indices = sorted_indices[:-2]  # Retrieve all indices except the last two (highest values)
    return result_indices

def test_skeletonization(barcodes, ome_zarrs_dict, ome_zarr_df, spacing, table_name, label_name, pyramid_level = 0, n = 5, seed = 0, sigma_skeleton = 3, n_angle_determination = 50, radius_multiplier = 0.5):
    """
    Test the skeletonization process on a set of randomly seletected organoid masks and generate visualizations.

    Parameters:
    - barcodes (list): List of barcode information. Needs to have same length and order as folder list.
    - ome_zarrs_dict (dict): Dictionary of OME-Zarr wrappers with barcode as keys.
    - ome_zarrs_df (pandas.DataFrame): Grouped DataFrame of organoid information.
    - spacing (float): Spacing parameter. Usually the pixel size.
    - table_name (str): Table name which contains organoid ROIs.
    - label_name (str): Label name which contains the organoid masks.
    - n (int): Number of masks to sample per plate.
    - seed (int): Seed for randomization.
    - sigma_skeleton (int): Sigma parameter for gaussian blurring before skeletonization.
    - n_angle_determination (int): Number of pixels used to determine angles of skeleton endpoints.
    - radius_multiplier (float): Multiplier for the radius in max inscribed circle calculation. The lower the multiplier, the more sensitive the algorithm for smaller crypts.
    """

    # Set seed
    random.seed(seed)

    # Get list of files for every folder and sample n masks
    for y, folders in enumerate(barcodes): # Loop through imaging plates

        masks_lst = ome_zarr_df[ome_zarr_df["Barcode"] == folders]["UID"].tolist()
        masks = random.sample(masks_lst, n)  # Take random sample of n masks

        # Set up plot
        fig, ax = plt.subplots(nrows=1, ncols=n, figsize=(5*n,8))
        plt.suptitle(barcodes[y], fontsize = 20)

        for col, mask in enumerate(masks):

            obj_id = mask.split("-")[-1]
            well_id = mask.split("-")[-2]
    
            img_shape = load_image_by_UID(ome_zarr_df, mask, 0, label_name, pyramid_level = pyramid_level).shape # Load shape of image of corresponding organoid
            image = load_mask_by_UID(ome_zarr_df, mask, label_name, pyramid_level = pyramid_level) # Load mask

            if image.shape != img_shape:
                image = resize_image(image, img_shape) # Resize image to original size if needed

            
            while np.max(measure.label(image.astype(bool))) != 1:

                mask = random.sample(masks_lst, 1)[0] # Gets new random organoid ID from list
                img_shape = load_image_by_UID(ome_zarr_df, mask, 0, label_name, pyramid_level = pyramid_level).shape # Load shape of image of corresponding organoid
                image = load_mask_by_UID(ome_zarr_df, mask, label_name, pyramid_level = pyramid_level) # Load mask

                image = resize_image(image, img_shape) # Resize image to original size if needed

                obj_id = mask.split("-")[-1]
                well_id = mask.split("-")[-2]

            skeleton = morphology.skeletonize(filters.gaussian(image, sigma = sigma_skeleton), method = "lee")
            mask_circle, radius, center = get_max_inscribed_circle(image, radius_multiplier = radius_multiplier, value = 255)

            # Update skeleton so that circle overlay is deleted
            skeleton[mask_circle.astype(bool)] = 0
            
            # Extract data
            if len(skeleton[skeleton.astype(bool)]) > 1:
                
                skeleton_data = summarize(Skeleton(skeleton))
                skeleton_data = skeleton_data[skeleton_data["branch-type"] != 2]
                elongated_skeleton = np.copy(skeleton)
                skeleton_int = np.copy(skeleton)
                
                skeleton_data = define_center_touching_branches(skeleton_data, mask_circle)

                #Remove branches which are branchtype 1 (junction-endpoint) and touch the circle, as well as branchtype 2 (junction-junction)
                skeleton_data = skeleton_data[~((skeleton_data["branch-type"] == 1) & (skeleton_data["Touching"] == 1))]
                
                # Elongate Skeleton in same angle as last n_angle_determination pixel of branches
                skeleton_data = determine_skeleton_endpoints(skeleton_data, center, spacing)

                for row in skeleton_data.index:

                    start_y = int(skeleton_data.loc[row,"endpoint-y"])
                    start_x = int(skeleton_data.loc[row,"endpoint-x"])
                    adj_y, adj_x = (start_y,start_x)

                    for i in range(n_angle_determination):
                        skeleton_int[adj_y, adj_x] = 0
                        adj_y, adj_x, _ = find_adjacent_nonzero_pixel((adj_y, adj_x), skeleton_int)

                    deg = calculate_angle((adj_x,adj_y),(start_x, start_y))
                    x_end, y_end = calculate_endpoint((start_x,start_y), deg, image)
                    skeleton_data.loc[row, "x_end"] = x_end
                    skeleton_data.loc[row, "y_end"] = y_end

                    elongated_skeleton = draw_line(elongated_skeleton, (start_x,start_y), (x_end,y_end))
                
                skeleton_data = summarize(Skeleton(elongated_skeleton))

                skeleton_data = skeleton_data[skeleton_data["branch-type"] != 2]

                skeleton_data = define_center_touching_branches(skeleton_data, mask_circle)

                # Remove branches which are branchtype 1 (junction-endpoint) and touch the circle, as well as branchtype 2 (junction-junction)
                skeleton_data = skeleton_data[~((skeleton_data["branch-type"] == 1) & (skeleton_data["Touching"] == 1))]

                # Count first endpoint-to-endpoint branches
                crypt_number = len(skeleton_data)               
                crypt_length_total = np.sum(skeleton_data["branch-distance"])*spacing
                longest_crypt = np.max(skeleton_data["branch-distance"])*spacing
                
                skeleton = elongated_skeleton
            else:
                crypt_number = 0
                crypt_length_total = 0
                longest_crypt = 0

            # Plot all into one
            image_plot = np.copy(image)

            image_plot[skeleton.astype(bool)] = 0

            img_plot = cv2.circle(image_plot, tuple(center), int(radius*radius_multiplier), 1)




            ax[col].imshow(img_plot, cmap=plt.cm.gray, aspect = "auto")
            ax[col].set_axis_off()

                
            ax[col].set_title("%s - %s"%(well_id,obj_id), fontsize = 12, loc='right')
            ax[col].set_title("\nNumber of crypts: %d\nCrypt lenght: %d µm\nLongest crypt: %d µm"%(crypt_number, crypt_length_total, longest_crypt), fontsize = 12, loc='left')
        
        plt.tight_layout()

def image_preprocessing(stainings, experiment_setup, images, OID, thresholds, sigma = 3):
    """
    Preprocess images by blurring them with a gaussian kernel, setting values outside the mask to 0, and thresholding image based on stain-specific mask.

    Parameters:
    - stainings (dict): Dictionary mapping staining indices to stain names.
    - experiment_setup (dict): Dictionary containing experimental setup information.
    - images (dict): Dictionary of image data.
    - OID (str): Unique organoid ID.
    - thresholds (dict): Dictionary containing staining-specific thresholds.
    - sigma (int): Sigma parameter for Gaussian blur.
    """
        
    # Go through channels
    channels = list(images.keys())

    for channel in channels:
        
        if "Mask" not in channel:

            # Load image as deepcopy
            image = copy.deepcopy(images[channel])

            # Gaussian blur
            image = filters.gaussian(image, sigma = sigma, preserve_range=True)

            # Set values outside mask to 0
            image[~images["Mask"].astype(bool)] = 0

            # Get threshold for staining
            staining = stainings[experiment_setup[OID.split("-")[0]][OID.split("-")[1]][1]][int(channel[-1:])-1]
            threshold = thresholds[staining][2]

            # Set everything under stain-specific threshold to 0
            image[image<=threshold] = 0

            # Save as mask
            images[channel+"_Mask"] = image.astype(bool)
        
    return images

def get_border_fraction(mask, df, OID):

    """
    Calculate the largest fraction of an object that has a straight edge.

    Parameters:
    - mask (ndarray): Binary mask representing the segmented object.
    - df (DataFrame): DataFrame to store the calculated features.
    - OID (str): Organoid ID for DataFrame indexing.
    """

    # Ensure the mask is binary
    mask = mask > 0

    # Find the indices where the mask is True
    indices = np.argwhere(mask)

    # Get the minimum and maximum indices for both dimensions
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)

    
    """ 
    # Get straight edges along contour
    v_left = np.sum(mask[:,x_min])/mask.shape[0]
    v_right = np.sum(mask[:,x_max])/mask.shape[0]
    h_top = np.sum(mask[y_min,:])/mask.shape[1]
    h_bottom = np.sum(mask[y_max,:])/mask.shape[1]
    """

   
    """
    # Get straight edges along image border
    v_border_left = np.sum(mask[:,0])/mask.shape[0]
    v_border_right = np.sum(mask[:,mask.shape[1]-1])/mask.shape[0]
    h_border_top = np.sum(mask[0,:])/mask.shape[1]
    h_border_bottom = np.sum(mask[mask.shape[0]-1,:])/mask.shape[1] 
    """


    df.loc[OID, "StraightEdge_longest"] = max(np.sum(mask[:,x_min]),  np.sum(mask[:,x_max]), np.sum(mask[y_min,:]), np.sum(mask[y_max,:]))
    df.loc[OID, "StraightEdge_longest_fraction"] = max(np.sum(mask[:,x_min])/mask.shape[0],  np.sum(mask[:,x_max])/mask.shape[0], np.sum(mask[y_min,:])/mask.shape[1], np.sum(mask[y_max,:])/mask.shape[0-1])

    return df

def shape_calc_mask(mask, df, OID, spacing):
    """
    Calculate various shape features for a segmented object.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - df (pandas.DataFrame): DataFrame to store the calculated features.
    - OID (str): Object ID for DataFrame indexing.
    - spacing (float): Pixel spacing.
    """
        
    # Set which regionprops features to calculate
    features = (
        'area',
        'area_bbox',
        'area_convex',
        'axis_major_length',
        'axis_minor_length',
        'centroid',
        'eccentricity',
        'equivalent_diameter_area',
        'extent',
        'feret_diameter_max',
        'moments',
        'perimeter',
        'solidity'
        )

    # Compute region properties
    props = measure.regionprops_table(mask, properties=features, cache = True, spacing = (spacing, spacing))

    # Convert to a Pandas DataFrame
    df_regionprops = pd.DataFrame(props, index = [OID])
    
    # Add to DF
    df.loc[OID,list(df_regionprops)] = df_regionprops.loc[OID,:]
    
    # Calc additional features
    df.loc[OID, "circularity"] = 4*math.pi*(df.loc[OID, "area"]/(df.loc[OID, "perimeter"]**2))
    df.loc[OID, "AxisRatio"] = df.loc[OID, "axis_minor_length"]/df.loc[OID, "axis_major_length"]
    
    #Calc aspectRatio_equivalentDiameter from scMultipleX by Nicole Repina, Liberali Lab
    df.loc[OID, "aspectRatio_equivalentDiameter"] = df.loc[OID, "axis_major_length"]/df.loc[OID, "equivalent_diameter_area"]
    
    return df

def channel_mask_feat_calc(mask, mask_channel, staining, df, OID, spacing):
    
    """
    Calculate staining features related to the whole segmentation mask of an object.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - mask_channel (numpy.ndarray): Binary mask representing a specific staining channel.
    - staining (str): Name of the staining channel.
    - df (pandas.DataFrame): DataFrame to store the calculated features.
    - OID (str): Object ID for DataFrame indexing.
    - spacing (float): Pixel spacing.
    """
            
    if np.max(mask_channel.astype(np.uint8)) > 0:
        prop_mask_channel = measure.regionprops(mask_channel.astype(np.uint8), spacing = (spacing, spacing))[0]
        prop_mask = measure.regionprops(mask.astype(np.uint8), spacing = (spacing, spacing))[0]
        
        df.loc[OID, staining+"_area_T"] = prop_mask_channel.area
        df.loc[OID, staining+"_area_T_ratio"] = prop_mask_channel.area/prop_mask.area
        df.loc[OID, staining+"_asymmetry"] = calculate_eucl_distance(prop_mask_channel.centroid[1], prop_mask_channel.centroid[0], prop_mask.centroid[1], prop_mask.centroid[0], spacing) / np.sqrt(prop_mask.area)
    else:
        df.loc[OID, staining+"_area_T"] = 0
        df.loc[OID, staining+"_area_T_ratio"] = 0
        df.loc[OID, staining+"_asymmetry"] = 1
    return df
                  
def convex_hull_features(mask, df, OID, spacing, min_area_fraction = 0.005):
    """
    Calculate the number of concavities in an object above a minimum size.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - df (pandas.DataFrame): DataFrame to store the calculated features.
    - OID (str): Object ID for DataFrame indexing.
    - spacing (float): Pixel spacing.
    - min_area_fraction (float): Minimum concavity area fraction.

    Functions derived from the scMultipleX package by Nicole Repina, Liberali Lab: https://github.com/fmi-basel/gliberal-scMultipleX
    """
    
    # Regionprops of mask to get needed features
    prop_2D = measure.regionprops(mask, spacing = (spacing,spacing))[0]
    
    
    object_image = prop_2D.image
    convex_image = prop_2D.convex_image
    object_area = prop_2D.area

    diff_img = convex_image ^ object_image

    # Concavity counting above minimum size
    if np.sum(diff_img) > 0:
        labeled_diff_img = measure.label(diff_img, connectivity=1)
        concavity_feat = measure.regionprops(labeled_diff_img, spacing = (spacing,spacing))
        concavity_cnt = 0
        for concavity_2D in concavity_feat:
            if (concavity_2D.area / object_area) > min_area_fraction:
                concavity_cnt += 1
    else:
        concavity_cnt = 0
    
    df.loc[OID, "concavity_count"] = concavity_cnt
    
    """Return the euclidian distance between the centroid of the object label
    and the centroid of the convex hull
    Normalize to the object area; becomes fraction of object composed of divots & indentations
    """

    # Use image that has same size as bounding box (not original seg)
    object_image = prop_2D.image
    object_moments = measure.moments(object_image)
    object_centroid = np.array([object_moments[1, 0] / object_moments[0, 0], object_moments[0, 1] / object_moments[0, 0]])

    # Convex hull image has same size as bounding box
    convex_image = prop_2D.convex_image
    convex_moments = measure.moments(convex_image)
    convex_centroid = np.array([convex_moments[1, 0] / convex_moments[0, 0], convex_moments[0, 1] / convex_moments[0, 0]])

    # calculate 2-norm (Euclidean distance) and normalize
    centroid_dist = np.linalg.norm(object_centroid - convex_centroid) / np.sqrt(prop_2D.area)

    df.loc[OID, "asymmetry"] = centroid_dist
    
    """Return the normalized difference in area between the convex hull and area of the object
    Normalize to the area of the convex hull
    """
    df.loc[OID, "concavity"] = (prop_2D.convex_area - prop_2D.area) / prop_2D.convex_area   
    
    return df

def moments_channel_mask(mask, int_image, df, OID, staining, spacing):
    """
    Calculate moments-based features for a staining channel within the segmented object.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - int_image (numpy.ndarray): Intensity image corresponding to the staining channel.
    - df (pandas.DataFrame): DataFrame to store the calculated features.
    - OID (str): Object ID for DataFrame indexing.
    - staining (str): Name of the staining channel.
    - spacing (float): Pixel spacing.
    """
        
    # Set which regionprops features to calculate
    features = (
        'moments_weighted',
    )

    # Compute region properties
    props = measure.regionprops_table(mask, intensity_image = int_image, properties=features, cache =True, spacing = (spacing, spacing))

    # Convert to a Pandas DataFrame
    df_regionprops = pd.DataFrame(props, index = [OID])
    
    # Rename to add staining name
    df_regionprops.columns = [staining+"_"+x for x in list(df_regionprops)]
    
    # Add to DF
    df.loc[OID,list(df_regionprops)] = df_regionprops.loc[OID,:]
    
    return df

def intensity_feat_calc(img, mask, mask_channel, df, staining, OID, quantiles_to_calc):
    """
    Calculate intensity-based features for a staining channel within a segmented object.

    Parameters:
    - img (numpy.ndarray): Original intensity image.
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - mask_channel (numpy.ndarray): Binary mask representing a specific staining channel.
    - df (pandas.DataFrame): DataFrame to store the calculated features.
    - staining (str): Name of the staining channel.
    - OID (str): Object ID for DataFrame indexing.
    - quantiles_to_calc (list): List of quantiles to calculate.
    """

    #min
    df.loc[OID, staining+"_min"] = np.min(img[mask.astype(bool)])
    
    if len(img[mask_channel.astype(bool)]) != 0:
        df.loc[OID, staining+"_T_min"] = np.min(img[mask_channel.astype(bool)])
    else: 
        df.loc[OID, staining+"_T_min"] = 0
        
    #mean
    df.loc[OID, staining+"_mean"] = np.mean(img[mask.astype(bool)])
    
    if len(img[mask_channel.astype(bool)]) != 0:
        df.loc[OID, staining+"_T_mean"] = np.mean(img[mask_channel.astype(bool)])
    else: 
        df.loc[OID, staining+"_T_mean"] = 0
        
    #max
    df.loc[OID, staining+"_max"] = np.max(img[mask.astype(bool)])
    
    if len(img[mask_channel.astype(bool)]) != 0:
        df.loc[OID, staining+"_T_max"] = np.max(img[mask_channel.astype(bool)])
    else: 
        df.loc[OID, staining+"_T_max"] = 0        

    #std
    df.loc[OID, staining+"_std"] = np.std(img[mask.astype(bool)])
    
    if len(img[mask_channel.astype(bool)]) != 0:
        df.loc[OID, staining+"_T_std"] = np.std(img[mask_channel.astype(bool)])
    else: 
        df.loc[OID, staining+"_T_std"] = 0     
        
    #quantiles
    for q in quantiles_to_calc:
        
        q_name = str(int(q * 100))
        df.loc[OID, staining+"_Q"+q_name] = np.quantile(img[mask.astype(bool)], q = q)

        if len(img[mask_channel.astype(bool)]) != 0:
            df.loc[OID, staining+"_T_Q"+q_name] = np.quantile(img[mask_channel.astype(bool)], q = q)
        else: 
            df.loc[OID, staining+"_T_Q"+q_name] = 0
            
    #potency
    df.loc[OID, staining+"_potency"] = df.loc[OID, staining+"_mean"]*df.loc[OID, staining+"_area_T"]
        
    #substructures

    ## Threshold substructures
    binary_image = img > filters.threshold_otsu(img)
    
    ## Label connected components
    labeled_image, num_speckles = label(binary_image)

    ## Focus on current object
    binary_image[~mask.astype(bool)] == 0

    ## Filter  small speckles
    min_size = 15
    filtered_speckles = np.zeros_like(labeled_image)
    current_label = 1

    for i in range(1, num_speckles + 1):
        speckle_mask = labeled_image == i
        speckle_size = np.sum(speckle_mask)
        
        if speckle_size >= min_size:
            filtered_speckles[speckle_mask] = current_label
            current_label += 1

    ## Quantify remaining speckles
    remaining_labels = np.unique(filtered_speckles)
    remaining_labels = remaining_labels[remaining_labels != 0]  # Exclude background

    speckle_sizes = []
    speckle_intensities = []

    for label_id in remaining_labels:
        speckle_mask = filtered_speckles == label_id
        speckle_size = np.sum(speckle_mask)
        speckle_intensity = np.mean(img[speckle_mask])
        
        speckle_sizes.append(speckle_size)
        speckle_intensities.append(speckle_intensity)

    ## Save
    df.loc[OID, staining+"_substructures_size"] = np.mean(speckle_sizes) if speckle_sizes else 0
    df.loc[OID, staining+"_substructures_intensity"] = np.mean(speckle_intensities) if speckle_intensities else 0
    df.loc[OID, staining+"_substructures_count"] = len(speckle_sizes)

    return df

def intensity_pearsonR(stainings, experiment_setup, images, barcode, well, OID, df):
    """
    Calculate Pearson correlation coefficients between staining channels within a segmented object.

    Parameters:
    - stainings (dict): Dictionary mapping staining IDs to staining names.
    - experiment_setup (dict): Dictionary containing experimental setup information.
    - images (dict): Dictionary of staining channel images.
    - barcode (str): Barcode identifier.
    - well (str): Well identifier.
    - OID (str): Object ID for DataFrame indexing.
    - df (pandas.DataFrame): DataFrame to store the calculated features.
    """
    # Set to keep track of seen pairs
    seen_pairs = set()

    # Loop through channels
    for channel1 in images.keys():

        if "Mask" not in channel1:
            
            # Get first staining
            staining1 = stainings[experiment_setup[barcode][well][1]][int(channel1[-1:])-1]
            
            # Loop through channels again to get other staining
            for channel2 in images.keys():

                if ("Mask" not in channel2) & (channel1 != channel2):
                    
                    #G et second staining
                    staining2 = stainings[experiment_setup[barcode][well][1]][int(channel2[-1:])-1]
                    
                    # Make a pair and check if it was already calculated
                    pair = frozenset([staining1, staining2]) #Use a frozenset for unordered pairs
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)

                        # Return PearsonR
                        r = np.corrcoef(images[channel1][images["Mask"].astype(bool)], images[channel2][images["Mask"].astype(bool)])[0, 1]

                        # Add to DF
                        if staining1 > staining2:
                            df.loc[OID, "%s-%s_PearsonR"%(staining1, staining2)] = r
                        else:
                            df.loc[OID, "%s-%s_PearsonR"%(staining2, staining1)] = r
    return df
    
def skeleton_feats(images, df, OID, spacing, sigma_skeleton, radius_multiplier = 0.5):   
    """
    Extract skeleton features from the provided binary mask image.

    Parameters:
    - images (dict): Dictionary of images, including the binary mask.
    - df (pandas.DataFrame): DataFrame to store the calculated features.
    - d_feats (dict): Dictionary to store additional features.
    - OID (str): Object ID for DataFrame indexing.
    - step_distance (int): Step distance for skeleton elongation.
    - spacing (float): Pixel spacing.
    - stainings (dict): Dictionary mapping staining IDs to staining names.
    - experiment_setup (dict): Dictionary containing experimental setup information.
    - sigma_skeleton (int): Sigma parameter for Gaussian filtering before skeletonization.
    - radius_multiplier (float, optional): Multiplier for the maximum inscribed circle radius. Default is 0.5.
    """
    n_angle_determination = 50
    
    image = images["Mask"]

    skeleton = morphology.skeletonize(filters.gaussian(image, sigma = sigma_skeleton), method = "lee")

    mask_circle, radius, center = get_max_inscribed_circle(image, radius_multiplier = radius_multiplier, value = 255)

    # Update skeleton so that circle overlay is deleted
    skeleton[mask_circle.astype(bool)] = 0


    # Extract data
    if len(skeleton[skeleton.astype(bool)]) > 2:

        skeleton_data = summarize(Skeleton(skeleton))
        skeleton_data = skeleton_data[skeleton_data["branch-type"] != 2]

        elongated_skeleton = np.copy(skeleton)

        skeleton_int = np.copy(skeleton)

        skeleton_data = define_center_touching_branches(skeleton_data, mask_circle)


        #Remove branches which are branchtype 1 (junction-endpoint) and touch the circle
        skeleton_data = skeleton_data[~((skeleton_data["branch-type"] == 1) & (skeleton_data["Touching"] == 1))]

        # Elongate Skeleton in same angle as last n_angle_determination pixel of branches
        skeleton_data = determine_skeleton_endpoints(skeleton_data, center, spacing)

        for row in skeleton_data.index:

            start_y = int(skeleton_data.loc[row,"endpoint-y"])
            start_x = int(skeleton_data.loc[row,"endpoint-x"])
            adj_y, adj_x = (start_y,start_x)

            for i in range(n_angle_determination):

                skeleton_int[adj_y, adj_x] = 0
                adj_y, adj_x, _ = find_adjacent_nonzero_pixel((adj_y, adj_x), skeleton_int)

            deg = calculate_angle((adj_x,adj_y),(start_x, start_y))

            x_end, y_end = calculate_endpoint((start_x,start_y), deg, image)

            skeleton_data.loc[row, "x_end"] = x_end
            skeleton_data.loc[row, "y_end"] = y_end

            elongated_skeleton = draw_line(elongated_skeleton, (start_x,start_y), (x_end,y_end))

        skeleton_data = summarize(Skeleton(elongated_skeleton))

        skeleton_data = skeleton_data[skeleton_data["branch-type"] != 2]

        skeleton_data = define_center_touching_branches(skeleton_data, mask_circle)

        # Remove branches which are branchtype 1 (junction-endpoint) and touch the circle, as well as branchtype 2 (junction-junction)
        skeleton_data = skeleton_data[~((skeleton_data["branch-type"] == 1) & (skeleton_data["Touching"] == 1))]

        # Calculate features
        crypt_number = len(skeleton_data)               
        crypt_length_total = np.sum(skeleton_data["branch-distance"])*spacing
        longest_crypt = np.max(skeleton_data["branch-distance"])*spacing

    else:
        crypt_number = 0
        crypt_length_total = 0
        longest_crypt = 0 

    df.loc[OID, "crypt_count"] = crypt_number
    df.loc[OID, "crypt_length_total"] = crypt_length_total
    df.loc[OID, "crypt_length_max"] = longest_crypt

    return df

def image_analysis(images, OID, df, quantiles_to_calc, sigma_skeleton, spacing, stainings, experiment_setup, radius_multiplier):
    """
    Perform image analysis tasks.

    Parameters:
    - images (dict): Dictionary of images.
    - OID (str): Object ID for DataFrame indexing.
    - df (pandas.DataFrame): DataFrame to store the calculated features.
    - d_feats_slicing (dict): Dictionary to store additional features related to skeleton slicing.
    - quantiles_to_calc (list): List of quantiles to calculate for intensity features.
    - sigma_skeleton (int): Sigma parameter for Gaussian blurring before skeletonization.
    - step_distance (int): Step distance for skeleton slicing of crypts.
    - spacing (float): Pixel spacing.
    - stainings (dict): Dictionary mapping staining IDs to staining names.
    - experiment_setup (dict): Dictionary containing experimental setup information.
    - radius_multiplier (float): Multiplier for the maximum inscribed circle radius.
    """

    # Get plate and well from OID
    barcode = OID.split("-")[0]
    well = OID.split("-")[1]
    
    # Shape features based on mask
    df = shape_calc_mask(images["Mask"], df, OID, spacing)
    
    # Calculate convex-hull associated features similar to scMultipleX by Nicole Repina, Liberali Lab
    df = convex_hull_features(images["Mask"], df, OID, spacing, min_area_fraction = 0.005)
    
    # Calculate border fraction for filtering
    df = get_border_fraction(images["Mask"], df, OID)
    
    # Skeleton features
    df = skeleton_feats(images, df, OID,  spacing, sigma_skeleton, radius_multiplier)
    
    #Loop through channels
    for channel in images.keys():
        
        if "Mask" not in channel:
            
            # Copy images for easier manipulation and set current staining based on well and barcode
            image = copy.deepcopy(images[channel])
            mask = copy.deepcopy(images["Mask"])
            mask_channel = copy.deepcopy(images[channel+"_Mask"])
            staining = stainings[experiment_setup[barcode][well][1]][int(channel[-1:])-1]
            
            # Calculate channel_mask features
            df = channel_mask_feat_calc(mask, mask_channel, staining, df, OID, spacing)
            
            # Calculate intensity features
            df = intensity_feat_calc(image, mask, mask_channel, df, staining, OID, quantiles_to_calc)
            
            # Calculate weighted moments based on intensity images for each channel
            df = moments_channel_mask(mask, image, df, OID, staining, spacing)
     
    # Calculate PearsonR for all combinations
    df = intensity_pearsonR(stainings, experiment_setup, images, barcode, well, OID, df)
    
    return df

def extract_features(ome_zarrs_dict, ome_zarrs_df, table_name, label_name, pyramid_level, source, folder, analysis_dir, barcodes, experiment_setup, thresholds, stainings, result_file_name, radius_multiplier, pixel_spacing, sigma_skeleton, quantiles_to_calc):
    """
    Extract features from organoid images and save results in CSV and AnnData format.

    Parameters:
    - ome_zarrs_dict (dict): Dictionary of OME Zarrs.
    - ome_zarrs_df (pandas.DataFrame): DataFrame of grouped organoid table information.
    - table_name (str): Table name which contains organoid ROIs.
    - label_name (str): Label name which contains organoid IDs.
    - pyramid_level (int): Pyramid level for image processing.
    - source (str): Root directory containing subdirectories for different plates.
    - folder (list): List of subdirectory names within the source directory.
    - analysis_dir (str): Directory to store the analysis results.
    - barcodes (list): List of barcodes corresponding to each subdirectory.
    - experiment_setup (dict): Dictionary containing experimental setup information.
    - thresholds (dict): Dictionary containing threshold values for image processing of each staining channel.
    - stainings (list): List of stainings used in the experiment.
    - result_file_name (str): Base name for the result files.
    - radius_multiplier (float): Multiplier for the radius in skeleton feature extraction.
    - pixel_spacing (float): Pixel spacing for image processing.
    - step_distance (int): Step distance for image analysis.
    - sigma_skeleton (int): Sigma value gaussian bluring before skeletonization.
    - quantiles_to_calc (list): List of quantiles to calculate in feature extraction.
    """
      
    # Set up complete DF
    df_total = pd.DataFrame()

    pbar = tqdm(total = len(ome_zarrs_df), desc = f'Processing {len(ome_zarrs_df)} objects.')
    for i, barcode in enumerate(barcodes):

        # Set up results DF
        df = pd.DataFrame()

        # Loop through folders and set well, barcode, organoid ID
        df_int = ome_zarrs_df[ome_zarrs_df.Barcode == barcode]
        for _, row in df_int.iterrows():
            
            well_id = row["well"]

            if well_id in experiment_setup[barcode].keys():                   

                images = {}

                # Loop through numbers of channels add as image to images
                for channel in range(len(stainings[row["AB"]])):
                    images["C0"+str(channel+1)] = load_image_by_UID(ome_zarrs_df, row["UID"], channel, label_name, pyramid_level = pyramid_level)
                
                img_shape = images["C0"+str(channel+1)].shape
                
                # Load mask into MASK image
                images["Mask"] = load_mask_by_UID(ome_zarrs_df, row["UID"], label_name, pyramid_level = pyramid_level) # Load mask

                if images["Mask"].shape != img_shape:
                    images["Mask"]  = resize_image(images["Mask"], img_shape) # Resize image to original size if needed
                    
                PATH = ome_zarrs_dict[barcode].get_path()[0].split(".zarr")[0]+".zarr"

                # Check for multiple DAPI labels

                                # Check for multiple DAPI labels
                if np.max(measure.label(images["Mask"].astype(bool))) != 1:
                    print(f"Skipping {row['UID']} because it contains multiple labels. Expected exactly one label per organoid.")

                if np.max(measure.label(images["Mask"].astype(bool))) == 1:

                    # Set unique organoid ID
                    OID = row["UID"]
    
                    # Set general information                  
                    df.loc[OID, "Organoid_ID"] = OID
                    df.loc[OID, "Barcode"] = OID.split("-")[0]
                    df.loc[OID, "Well"] = OID.split("-")[1]
                    df.loc[OID, "Object"] = OID.split("-")[2]
                    df.loc[OID, "Medium"] = experiment_setup[OID.split("-")[0]][OID.split("-")[1]][0]
                    df.loc[OID, "ABs"] = experiment_setup[OID.split("-")[0]][OID.split("-")[1]][1]
                    df.loc[OID, "PATH"] = PATH
                    for channel in range(len(images)-1):   
                        df.loc[OID, "Staining_Ch"+str(channel+1)] = stainings[experiment_setup[OID.split("-")[-3]][OID.split("-")[-2]][1]][channel]
    
                    df.loc[OID, "Cell_line"] = experiment_setup[OID.split("-")[0]][OID.split("-")[1]][2]
                    df.loc[OID, "Other"] = experiment_setup[OID.split("-")[0]][OID.split("-")[1]][3]
    
                    # Preprocess images and build channel masks
                    images = image_preprocessing(stainings, experiment_setup, images, OID, thresholds, sigma = 3)
    
                    # Calculate and return features
                    df = image_analysis(images = images,
                                                        OID = OID,
                                                        df = df,
                                                        quantiles_to_calc = quantiles_to_calc,
                                                        sigma_skeleton = sigma_skeleton,
                                                        spacing = pixel_spacing,
                                                        stainings = stainings,
                                                        experiment_setup = experiment_setup,
                                                        radius_multiplier = radius_multiplier)
                    
                    pbar.update(1)

        # Drop moments-0-0 as it is the same as area
        feats_moments_0_0 = [x for x in list(df) if any(y in x for y in ["moments-0-0", "moments_weighted-0-0"])]
        df = df.drop(columns = feats_moments_0_0)
        
        # Save DF for current folder after natsorting for organoid ID
        df = df.sort_values(by="Organoid_ID", key=natsort_keygen())
        save_path = os.path.join(source, folder[i], f"{result_file_name}_{barcodes[i]}"+"_{date:%Y-%m-%d_%Hh%Mmin%Ss}".format(date=datetime.datetime.now())+".csv")
        df.to_csv(save_path)
        print("Saved %s results as %s."%(barcode, save_path))
        
        # Merge into complete DF
        df_total = pd.concat([df_total, df])
            
    # Create Analysis directory if not present
    for folder in ["1_Scripts", "2_Tables", "3_Plots", "4_Results"]:
        path = os.path.join(analysis_dir, folder)
        
        if os.path.exists(path)==False:
            os.makedirs(path)

    # Save complete DF and AnnData
    total_save_path = os.path.join(analysis_dir, "2_Tables", result_file_name+"_{date:%Y-%m-%d_%Hh%Mmin%Ss}".format(date=datetime.datetime.now())+".csv")
    df_total.to_csv(total_save_path)
    print("Saved merged csv as %s."%total_save_path)

    # Get categorical features
    feats_categorical = ["Organoid_ID",
                        "Barcode",
                        "Well",
                        "Object",
                        "Medium",
                        "ABs",
                        "Cell_line",
                        "Other",
                        "PATH"]

    feats_categorical = feats_categorical + [x for x in list(df_total) if any(y in x for y in ["Staining_"])]
    feats_numerical = [x for x in list(df_total) if not any(y in x for y in feats_categorical)]

    # Compose AD
    ad = anndata.AnnData(X=df_total[feats_numerical], obs = df_total[feats_categorical])

    # Add misc. slicing df and misc information as unstructured object
    ad.uns["stainings"] = stainings
    ad.uns["experiment_setup"] = experiment_setup
    ad.uns["pixel_spacing"] = pixel_spacing
    ad.uns["pyramid_level"] = pyramid_level
    ad.uns["source_dir"] = source
    ad.uns["table_name"] = table_name
    ad.uns["label_name"] = label_name
    ad.uns["pyramid_level"] = pyramid_level
    ad.uns["ome_zarr_df"] = ome_zarrs_df
    ad.uns["ome_zarrs_dict"] = ome_zarrs_dict

    thresholds_to_save = {}
    for staining in thresholds.keys():
        thresholds_to_save[staining] = thresholds[staining][2]
            
    ad.uns["thresholds"] = thresholds_to_save

    # Save AD
    save_adata(os.path.join(analysis_dir, "2_Tables"), result_file_name+"_{date:%Y-%m-%d_%Hh%Mmin%Ss}".format(date=datetime.datetime.now()), ad)
   



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

            img = load_image_by_UID(ome_zarr_df=ad.uns["ome_zarr_df"], UID = removed_OID[i], channel=channel, label_name=ad.uns["label_name"], pyramid_level=pyramid_level)
            mask = load_mask_by_UID(ome_zarr_df=ad.uns["ome_zarr_df"], UID = removed_OID[i], label_name = ad.uns["label_name"], pyramid_level = pyramid_level)

            if img.shape != mask.shape:
                mask = resize_image(mask, img.shape)

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

            ome_zarr_dict = {removed_OID[i].split("-")[0]: hcs_wrappers.FractalZarr(ad.obs.loc[removed_OID[i], "PATH"], name = removed_OID[i].split("-")[0])}

            img = load_image_by_UID(ome_zarr_df=ad.uns["ome_zarr_df"], UID = removed_OID[i], channel = channel, label_name = ad.uns["label_name"], pyramid_level = pyramid_level)
            mask = load_mask_by_UID(ome_zarr_df=ad.uns["ome_zarr_df"], UID = removed_OID[i], label_name = ad.uns["label_name"], pyramid_level = pyramid_level)

            if img.shape != mask.shape:
                mask = resize_image(mask, img.shape)

            # img = io.imread(removed_path[i].replace("_MASK.","_TIF-OVR-MIP."))
            # mask = io.imread(removed_path[i])
            img[mask == 0] = 0
            ax[row,col].imshow(img, interpolation = "nearest", aspect = "auto", cmap = "magma")
            ax[row,col].set_title(removed_OID[i]+"\n"+str(round(removed_size[i],2)), fontsize = 8)
            ax[row,col].set_axis_off()
            i += 1

    fig.tight_layout()

    return fig

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

            img = load_image_by_UID(ome_zarr_df = ad.uns["ome_zarr_df"], UID = bd_plot[k], channel = channel, label_name = ad.uns["label_name"], pyramid_level = pyramid_level)
            mask = load_mask_by_UID(ome_zarr_df = ad.uns["ome_zarr_df"], UID = bd_plot[k], label_name = ad.uns["label_name"], pyramid_level = pyramid_level)

            if img.shape != mask.shape:
                mask = resize_image(mask, img.shape)

            #img = io.imread(bd_plot[k].replace("_MASK", "_TIF-OVR-MIP"))
            #mask = io.imread(bd_plot[k])
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