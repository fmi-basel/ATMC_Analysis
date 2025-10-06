import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from skimage import filters, measure, morphology
from natsort import natsorted, natsort_keygen
from skan import Skeleton, summarize
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.ndimage import label
from tqdm.notebook import tqdm
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import anndata
import random
import glob
import copy
import math
import cv2
import os

from Fcts_Base import find_staining_in_ABs, load_img_mask_by_UID, find_barcodes_with_day, save_adata

# Disable pandas performance warnings
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



"""
***
FEATURE EXTRACTION FUNCTIONS
***
"""       

def make_experiment(source, layout_sheets = ["MediumLayout", "StainingLayout", "LineLayout", "OtherLayout"]):
    """
    Parses an experiment's plate layout Excel file and returns a nested dictionary of well metadata.

    For each plate (barcode), this function extracts all well information from several defined sheet types
    (e.g. MediumLayout, StainingLayout, etc.) within the first Excel file matching "Layout*.xlsx" under the source directory.
    The result is a dictionary: 
        {barcode: {well: [medium, staining, line, other]}}
    Only wells with information in *all* sheets are included (all-NaN wells are omitted, error if mixed NaN/values).

    Parameters
    ----------
    source : str
        Path to a directory containing the Excel file that describes the experiment layout.
    layout_sheets : list of str, optional
        The names of the worksheet tabs to extract (default: ["MediumLayout", "StainingLayout", "LineLayout", "OtherLayout"]).
    """


    filename = glob.glob(source+'/Layout*.xlsx')[0]

    def extract_sheet_plates(df):
        plates = {}
        nrows, ncols = df.shape
        i = 0
        while i < nrows:
            # 1. Find next barcode
            barcode = None
            for j in range(i, nrows):
                row = df.iloc[j, :]
                for k, v in enumerate(row):
                    if str(v).strip() == "Barcode:" and (k+1)<ncols:
                        barcode = str(row[k+1]).strip()
                        i = j+1
                        break
                if barcode: break
            else:
                break

            # 2. Find header row with consecutive numbers (for columns)
            grid_start_col, head_row, n_cols_detected = None, None, None
            for j in range(i, min(i+10, nrows)):
                row = df.iloc[j, :]
                max_search = ncols-6  # don't overshoot end
                for window in range(max_search):
                    nums = []
                    for q in range(0,24):  # Can be less
                        cell = row[window+q] if window+q < ncols else None
                        try:
                            n = int(float(str(cell)))
                            nums.append(n)
                        except:
                            break
                    # Is this 1-24 (384well) or 1-12 (96well)?
                    if nums == list(range(1,25)):
                        grid_start_col, head_row, n_cols_detected = window, j, 24    # 384
                        break
                    elif nums == list(range(1,13)):
                        grid_start_col, head_row, n_cols_detected = window, j, 12    # 96
                        break
                if head_row is not None:
                    break
            if head_row is None:
                i += 1
                continue

            # 3. Find well rows (A-P or A-H accordingly)
            if n_cols_detected == 24:
                expected_rows = 16
                allowed_labels = list("ABCDEFGHIJKLMNOP") # 384 wells
            elif n_cols_detected == 12:
                expected_rows = 8
                allowed_labels = list("ABCDEFGH") # 96 wells
            else:
                raise ValueError(f"Unrecognized plate format after barcode {barcode}")

            wells = []
            well_labels = []
            for j in range(head_row+1, head_row+1+expected_rows):
                if j >= nrows: break
                row = df.iloc[j, :]
                rlab = None
                for offset in [grid_start_col-1, grid_start_col-2, 2, 1]:
                    if 0 <= offset < ncols:
                        cell = row[offset]
                        if isinstance(cell, str) and len(cell.strip())==1 and cell.strip().isalpha():
                            rlab = cell.strip()
                            break
                if rlab and rlab in allowed_labels:
                    arr = list(row[grid_start_col:grid_start_col+n_cols_detected])
                    if len(arr) < n_cols_detected:
                        arr = arr + [np.nan]*(n_cols_detected-len(arr))
                    wells.append(arr)
                    well_labels.append(rlab)
            if wells:
                plates[barcode] = pd.DataFrame(
                    wells,
                    index=well_labels,
                    columns=[f"{i:02d}" for i in range(1, n_cols_detected+1)]
                )
            i = head_row + expected_rows + 2  # advance past block and notes/comments
        return plates

    # 1. Parse plates for each sheet
    all_plates = {}
    for sheetname in layout_sheets:
        df = pd.read_excel(filename, sheet_name=sheetname, header=None)
        all_plates[sheetname] = extract_sheet_plates(df)

    # 2. Find barcodes present in all sheets
    barcodes = set(all_plates[layout_sheets[0]].keys())
    for others in layout_sheets[1:]:
        barcodes &= set(all_plates[others].keys())
    barcodes = sorted(list(barcodes))

    # 3. Compose nested dict: barcode -> well -> list of values from all sheets
    result = {}
    for barcode in barcodes:
        dfs = [all_plates[s][barcode] for s in layout_sheets]
        result[barcode] = {}
        for row in dfs[0].index:
            for col in dfs[0].columns:
                well = f"{row}{col}"
                values = [df.at[row, col] if (row in df.index and col in df.columns) else np.nan for df in dfs]
                arr = np.array(values, dtype=object)
                if np.all(pd.isna(arr)):
                    continue
                if np.any(pd.isna(arr)) and not np.all(pd.isna(arr)):
                    raise ValueError(
                        f"Mixture of nan/values in {barcode} {well}: {values}. "
                        "Make sure all used wells have information across sheets."
                    )
                result[barcode][well] = values

    display_experiment_setup(result)
    barcodes = list(result.keys())
    
    print(f"Found {len(barcodes)} barcodes in experiment setup:")
    for bc in barcodes:
        print(bc)
    return result, barcodes

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

def estimate_staining_thresholds(ome_zarr_df, ome_zarr_dict, stainings, experiment_setup, table_name, label_name, pyramid_level = 0, control_condition = None, n = 20, seed = 0, sigma = 3, q = 0.5):
    """
    Estimate staining thresholds for a set of stainings based on organoid images.

    Parameters:
    - ome_zar_df (dict): Dictionary containing OME-Zarr wrappers for whole experiment with barcode as keys.
    - stainings (dict): Dictionary mapping antibody mix names to staining names.
    - experiment_setup (dict): Experiment setup dictionary.
    - table_name (str): Name of the table containing organoid segmentations.
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
            img, _ = load_img_mask_by_UID(fyle, ome_zarr_dict, table_name, label_name, pyramid_level, channel = thresholds[ab][0]-1)
            
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

def plot_thresholds(thresholds, dict_org, timepoints_lst, ome_zarr_dict, table_name, label_name, seed = 0, pyramid_level = 0):
    """
    Plot threshold distributions and organoid stainings based on set threshold.

    Parameters:
    - thresholds (dict): Dictionary containing threshold information.
    - dict_org (dict): Dictionary containing original organoid image paths.
    - timepoints_lst (list): List of timepoints.
    - ome_zar_df (dict): Dictionary containing OME-Zarr wrappers for whole experiment with barcode as keys.
    - table_name (str): Name of the table containing organoid segmentations.
    - label_name (string): Name of table which contains organoid segmentations.
    - seed (int): Random seed for reproducibility (default is 0).
    - pyramid_level (int): Pyramid level for image processing (default is 0).
    """

    # Set random seed
    random.seed(seed)

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

                img, _ = load_img_mask_by_UID(fyle, ome_zarr_dict, table_name, label_name, pyramid_level, channel = thresholds[ab][0]-1)

                
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
    
    dist_map = cv2.distanceTransform(image.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
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
    center_mask_dilated = cv2.dilate(center_mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations = 1)

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

def skeleton_feats(images, row_data, OID, spacing, sigma_skeleton, radius_multiplier=0.5):
    """
    Extracts skeleton-based features from the binary mask image and updates the results dictionary.

    Parameters:
    - images (dict): Dictionary containing the binary mask under the key "Mask".
      The mask should be a 2D numpy array representing the segmented object.
    - row_data (dict): Dictionary to store extracted features. This dict is updated and returned.
    - OID (str): Object identifier, used for bookkeeping; not directly used in this function.
    - spacing (float): Physical pixel size to convert pixel distances to real units.
    - sigma_skeleton (float): Sigma for Gaussian blur prior to skeletonization.
    - radius_multiplier (float, optional): Multiplier applied to the max inscribed circle radius 
      for feature calculation. Default is 0.5.
    """
    mask = images["Mask"]
    _, _, _, _, crypt_number, crypt_length_total, longest_crypt = extract_skeleton_features(mask, spacing, sigma_skeleton, radius_multiplier)
    row_data["crypt_count"] = crypt_number
    row_data["crypt_length_total"] = crypt_length_total
    row_data["crypt_length_max"] = longest_crypt
    return row_data

def extract_skeleton_features(image, spacing, sigma_skeleton=3, radius_multiplier=0.5, n_angle_determination=50):
    """
    Extract skeleton features and optionally elongate the skeleton from a binary mask image.

    Parameters:
    - image (numpy.ndarray): 2D binary mask image (e.g., uint8 or bool) representing the segmented object.
    - spacing (float): Physical pixel spacing to convert pixel distances to real-world units.
    - sigma_skeleton (float, optional): Sigma for Gaussian blur before skeletonization to smooth the mask. Default is 3.
    - radius_multiplier (float, optional): Multiplier for radius when computing the maximum inscribed circle. Default is 0.5.
    - n_angle_determination (int, optional): Number of pixels used to determine the elongation angle of skeleton endpoints. Default is 50.
    """
    blurred_image = filters.gaussian(image, sigma=sigma_skeleton)
    skeleton = morphology.skeletonize(blurred_image, method="lee")
    mask_circle, radius, center = get_max_inscribed_circle(image, radius_multiplier=radius_multiplier, value=255)
    skeleton[mask_circle.astype(bool)] = 0

    if np.sum(skeleton.astype(bool)) > 2:
        skeleton_data = summarize(Skeleton(skeleton))
        skeleton_data = skeleton_data[skeleton_data["branch-type"] != 2]
        elongated_skeleton = np.copy(skeleton)
        skeleton_int = np.copy(skeleton)

        skeleton_data = define_center_touching_branches(skeleton_data, mask_circle)
        skeleton_data = skeleton_data[~((skeleton_data["branch-type"] == 1) & (skeleton_data["Touching"] == 1))]
        skeleton_data = determine_skeleton_endpoints(skeleton_data, center, spacing)

        for row in skeleton_data.index:
            y, x = int(skeleton_data.loc[row,"endpoint-y"]), int(skeleton_data.loc[row,"endpoint-x"])
            adj_y, adj_x = y, x
            for _ in range(n_angle_determination):
                skeleton_int[adj_y, adj_x] = 0
                adj_y, adj_x, _ = find_adjacent_nonzero_pixel((adj_y, adj_x), skeleton_int)
            deg = calculate_angle((adj_x, adj_y), (x, y))
            x_end, y_end = calculate_endpoint((x, y), deg, image)
            skeleton_data.loc[row, "x_end"] = x_end
            skeleton_data.loc[row, "y_end"] = y_end
            elongated_skeleton = draw_line(elongated_skeleton, (x, y), (x_end, y_end))

        # Re-calculate features after elongation
        skeleton_data = summarize(Skeleton(elongated_skeleton))
        skeleton_data = skeleton_data[skeleton_data["branch-type"] != 2]
        skeleton_data = define_center_touching_branches(skeleton_data, mask_circle)
        skeleton_data = skeleton_data[~((skeleton_data["branch-type"] == 1) & (skeleton_data["Touching"] == 1))]
        crypt_number = len(skeleton_data)
        crypt_length_total = np.sum(skeleton_data["branch-distance"]) * spacing if crypt_number > 0 else 0
        longest_crypt = np.max(skeleton_data["branch-distance"]) * spacing if crypt_number > 0 else 0

        return elongated_skeleton, mask_circle, radius, center, crypt_number, crypt_length_total, longest_crypt

    return skeleton, mask_circle, radius, center, 0, 0, 0

def pad_to_aspect_ratio(image, target_aspect_ratio = 1.0):
    """
    Pad a 2D image with black pixels to achieve a target aspect ratio without distortion.

    Parameters:
    - image (numpy.ndarray): 2D array representing a grayscale or binary image.
    - target_aspect_ratio (float, optional): Desired width-to-height ratio for the output image.
      Default is 1.0 (square).
    """
    height, width = image.shape
    current_aspect = width / height

    if np.isclose(current_aspect, target_aspect_ratio):
        return image

    if current_aspect < target_aspect_ratio:
        # Pad width
        new_width = int(np.ceil(target_aspect_ratio * height))
        pad_total = new_width - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded_image = np.pad(image, ((0, 0), (pad_left, pad_right)), mode='constant', constant_values=0)
    else:
        # Pad height
        new_height = int(np.ceil(width / target_aspect_ratio))
        pad_total = new_height - height
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        padded_image = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=0)

    return padded_image

def test_skeletonization(barcodes, ome_zarrs_dict, ome_zarr_df, table_name, label_name, pyramid_level=0,
                        n=5, seed=0, sigma_skeleton=3, n_angle_determination=50, radius_multiplier=0.5):
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
    random.seed(seed)
    for barcode in barcodes:
        masks_lst = ome_zarr_df[ome_zarr_df["Barcode"] == barcode]["UID"].tolist()
        sampled_masks = []
        n_attempts = 0
        while len(sampled_masks) < n and n_attempts < 5*n:
            mask_candidate = random.choice(masks_lst)
            _, mask_img = load_img_mask_by_UID(mask_candidate, ome_zarrs_dict, table_name, label_name,
                                               pyramid_level, 0)
            if np.max(measure.label(mask_img.astype(bool))) == 1:
                if mask_candidate not in sampled_masks:
                    sampled_masks.append(mask_candidate)
            n_attempts += 1

        well_names = ome_zarrs_dict[barcode].get_names()
        ncols = max(len(sampled_masks), 1)
        fig, axarr = plt.subplots(1, ncols, figsize=(5*ncols, 8), squeeze=False)
        plt.suptitle(barcode, fontsize=20)

        for i, mask in enumerate(sampled_masks):
            obj_id, well_id = mask.split("-")[-1], mask.split("-")[-2]
            spacing = ome_zarrs_dict[barcode][well_names.index(well_id)].get_scale(pyramid_level)[-1]
            _, image = load_img_mask_by_UID(mask, ome_zarrs_dict, table_name, label_name, pyramid_level, 0)
            image = image.astype(np.uint8)
            image[image != 0] = 255
            skeleton, mask_circle, radius, center, crypt_count, crypt_length_total, longest_crypt = \
                extract_skeleton_features(image, spacing, sigma_skeleton, radius_multiplier, n_angle_determination)
            image_plot = np.copy(image)
            image_plot[skeleton.astype(bool)] = 0
            img_plot = cv2.circle(image_plot, tuple(center), int(radius*radius_multiplier), 0) # Draw max inscribed circle
            img_plot = pad_to_aspect_ratio(img_plot)
            ax = axarr[0][i] if ncols > 1 else axarr[0]
            ax.imshow(img_plot, cmap=plt.cm.gray, aspect="auto")
            ax.set_axis_off()
            ax.set_title(
                f"{well_id} - {obj_id}\nNumber of crypts: {crypt_count}\nCrypt length: {int(crypt_length_total)} µm\nLongest crypt: {int(longest_crypt)} µm",
                fontsize=12)
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

    for channel in range(len(stainings[experiment_setup["-".join(OID.split("-")[:-2])][OID.split("-")[-2]][1]])):
        


        # Load image as deepcopy
        image = copy.deepcopy(images["C0"+str(channel+1)])

        # Gaussian blur
        image = filters.gaussian(image, sigma = sigma, preserve_range=True)

        # Set values outside mask to 0
        image[~images["Mask"].astype(bool)] = 0

        # Get threshold for staining
        staining = stainings[experiment_setup["-".join(OID.split("-")[:-2])][OID.split("-")[-2]][1]][channel]
        threshold = thresholds[staining][2]

        # Set everything under stain-specific threshold to 0
        image[image<=threshold] = 0

        # Save as mask
        images[f"C0{channel+1}_Mask"] = image.astype(bool)
        
    return images

def get_border_fraction(mask, row_data, OID):
    """
    Calculate the largest fraction of an object that has a straight edge.

    Parameters:
    - mask (ndarray): Binary mask representing the segmented object.
    - row_data (dict): Dictionary to store the calculated features.
    - OID (str): Object ID (for API consistency).

    Returns:
    - row_data (dict): Updated with straight edge features.
    """
    # Ensure binary mask
    mask = mask > 0

    # Get object bounding box in mask
    indices = np.argwhere(mask)
    if indices.size == 0:
        # Empty mask: set features to zero and return
        row_data["StraightEdge_longest"] = 0
        row_data["StraightEdge_longest_fraction"] = 0
        return row_data

    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)

    # Compute straight edge lengths for bounding box edges
    v_left = np.sum(mask[:, x_min])
    v_right = np.sum(mask[:, x_max])
    h_top = np.sum(mask[y_min, :])
    h_bottom = np.sum(mask[y_max, :])

    # Candidate max straight edge length (in pixels)
    max_length = max(v_left, v_right, h_top, h_bottom)

    # Normalize by respective dimension lengths to get fraction
    fractions = [
        v_left / mask.shape[0],    # vertical edges normalized by height
        v_right / mask.shape[0],
        h_top / mask.shape[1],     # horizontal edges normalized by width
        h_bottom / mask.shape[1]
    ]
    max_fraction = max(fractions)

    # Store in row_data dict
    row_data["StraightEdge_longest"] = max_length
    row_data["StraightEdge_longest_fraction"] = max_fraction

    return row_data

def shape_calc_mask(mask, row_data, OID, spacing):
    """
    Calculate various shape features for a segmented object.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - row_data (dict): Dictionary to store the calculated features.
    - OID (str): Object ID (kept for API consistency).
    - spacing (float): Pixel spacing.

    Returns:
    - row_data (dict): Updated with shape features.
    """

    # Define properties to calculate
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

    # Compute region properties table (returns dict of arrays)
    props = measure.regionprops_table(mask.astype(np.uint8), properties=features, spacing=(spacing, spacing), cache=True)

    # Convert to DataFrame for easier access
    df_props = pd.DataFrame(props, index=[OID])

    # Update row_data dictionary with all returned features
    for col in df_props.columns:
        row_data[col] = df_props.at[OID, col]

    # Calculate additional features safely
    perimeter = row_data.get("perimeter", np.nan)
    area = row_data.get("area", np.nan)
    axis_major = row_data.get("axis_major_length", np.nan)
    axis_minor = row_data.get("axis_minor_length", np.nan)
    equiv_diameter = row_data.get("equivalent_diameter_area", np.nan)

    # Circularity: avoid division by zero
    if np.isfinite(perimeter) and perimeter > 0 and np.isfinite(area):
        row_data["circularity"] = 4 * math.pi * (area / (perimeter ** 2))
    else:
        row_data["circularity"] = np.nan

    # Axis ratio: minor/major lengths
    if np.isfinite(axis_minor) and np.isfinite(axis_major) and axis_major != 0:
        row_data["AxisRatio"] = axis_minor / axis_major
    else:
        row_data["AxisRatio"] = np.nan

    # Aspect ratio equivalent diameter
    if np.isfinite(axis_major) and np.isfinite(equiv_diameter) and equiv_diameter != 0:
        row_data["aspectRatio_equivalentDiameter"] = axis_major / equiv_diameter
    else:
        row_data["aspectRatio_equivalentDiameter"] = np.nan

    return row_data

def channel_mask_feat_calc(mask, mask_channel, staining, row_data, OID, spacing):
    """
    Calculate staining features related to the whole segmentation mask of an object.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - mask_channel (numpy.ndarray): Binary mask for the specific staining channel.
    - staining (str): Name of the staining channel.
    - row_data (dict): Row dictionary to update with calculated features.
    - OID (str): Object ID (not used here but kept for API consistency).
    - spacing (float): Pixel spacing.

    Returns:
    - row_data (dict): Updated with new features.
    """
    # Only calculate features if mask_channel contains any positive pixel
    if np.max(mask_channel.astype(np.uint8)) > 0:
        props_mask_channel = measure.regionprops(mask_channel.astype(np.uint8), spacing=(spacing, spacing))[0]
        props_mask = measure.regionprops(mask.astype(np.uint8), spacing=(spacing, spacing))[0]

        row_data[f"{staining}_area_T"] = props_mask_channel.area
        row_data[f"{staining}_area_T_ratio"] = props_mask_channel.area / props_mask.area if props_mask.area != 0 else 0

        # Calculate (normalized) asymmetry as Euclidean distance between centroids, divided by sqrt(area)
        centroid_dist = calculate_eucl_distance(
            props_mask_channel.centroid[1], props_mask_channel.centroid[0],
            props_mask.centroid[1], props_mask.centroid[0],
            spacing
        )
        row_data[f"{staining}_asymmetry"] = centroid_dist / np.sqrt(props_mask.area) if props_mask.area != 0 else 1
    else:
        row_data[f"{staining}_area_T"] = 0
        row_data[f"{staining}_area_T_ratio"] = 0
        row_data[f"{staining}_asymmetry"] = 1

    return row_data

def convex_hull_features(mask, row_data, OID, spacing, min_area_fraction=0.005):
    """
    Calculate the number of concavities in an object above a minimum size,
    and shape-related convex hull features.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - row_data (dict): Dictionary to store the calculated features.
    - OID (str): Object ID (kept for API consistency; not used here).
    - spacing (float): Pixel spacing.
    - min_area_fraction (float): Minimum concavity area fraction.

    Returns:
    - row_data (dict): Updated dict with calculated features.
    """
    prop_2D = measure.regionprops(mask, spacing=(spacing, spacing))[0]

    object_image = prop_2D.image
    convex_image = prop_2D.convex_image
    object_area = prop_2D.area

    diff_img = convex_image ^ object_image  # True where object is concave

    # Count concavities larger than min_area_fraction
    concavity_cnt = 0
    if np.sum(diff_img) > 0:
        labeled_diff_img = measure.label(diff_img, connectivity=1)
        concavity_props = measure.regionprops(labeled_diff_img, spacing=(spacing, spacing))
        for concavity in concavity_props:
            if (concavity.area / object_area) > min_area_fraction:
                concavity_cnt += 1
    row_data["concavity_count"] = concavity_cnt

    # Calculate centroids from raw image moments for object and convex hull
    object_moments = measure.moments(object_image)
    object_centroid = np.array([
        object_moments[1, 0] / object_moments[0, 0],
        object_moments[0, 1] / object_moments[0, 0]
    ])

    convex_moments = measure.moments(convex_image)
    convex_centroid = np.array([
        convex_moments[1, 0] / convex_moments[0, 0],
        convex_moments[0, 1] / convex_moments[0, 0]
    ])

    # Normalized Euclidean distance between centroids as asymmetry measure
    centroid_dist = np.linalg.norm(object_centroid - convex_centroid) / np.sqrt(object_area)
    row_data["asymmetry"] = centroid_dist

    # Normalized area difference between convex hull and object as concavity metric
    row_data["concavity"] = (prop_2D.convex_area - object_area) / prop_2D.convex_area

    return row_data

def moments_channel_mask(mask, int_image, row_data, OID, staining, spacing):
    """
    Calculate moments-based features for a staining channel within the segmented object.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - int_image (numpy.ndarray): Intensity image corresponding to the staining channel.
    - row_data (dict): Dict to update with calculated features.
    - OID (str): Object ID, for API consistency.
    - staining (str): Name of the staining channel.
    - spacing (float): Pixel spacing.

    Returns:
    - row_data (dict): Updated with moments-based features.
    """
    features = ('moments_weighted',)
    
    # Compute regionprops_table returns a dict of arrays
    props = measure.regionprops_table(
        mask.astype(np.uint8), 
        intensity_image=int_image, 
        properties=features, 
        spacing=(spacing, spacing)
    )
    
    # Convert to DataFrame with single-row indexed by OID for convenience
    df_props = pd.DataFrame(props, index=[OID])
    
    # Rename columns to prefix with staining name
    df_props.columns = [f"{staining}_{col}" for col in df_props.columns]
    
    # Update row_data dict with these features (extracting first row)
    for col in df_props.columns:
        row_data[col] = df_props.at[OID, col]

    return row_data

def intensity_feat_calc(img, mask, mask_channel, row_data, staining, OID, quantiles_to_calc):
    """
    Calculate intensity-based features for a staining channel within a segmented object.

    Parameters:
    - img (numpy.ndarray): Original intensity image.
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - mask_channel (numpy.ndarray): Binary mask representing a specific staining channel.
    - row_data (dict): Dictionary to store the calculated features.
    - staining (str): Name of the staining channel.
    - OID (str): Object ID (kept for API consistency).
    - quantiles_to_calc (list): List of quantiles to calculate (values between 0 and 1).
    
    Returns:
    - row_data (dict): Updated with intensity-based features.
    """

    # Mask indices with bool arrays once for efficiency
    mask_bool = mask.astype(bool)
    mask_chan_bool = mask_channel.astype(bool)

    img_masked = img[mask_bool]

    # Basic intensity features in full mask
    row_data[f"{staining}_min"] = np.min(img_masked) if img_masked.size > 0 else 0
    row_data[f"{staining}_mean"] = np.mean(img_masked) if img_masked.size > 0 else 0
    row_data[f"{staining}_max"] = np.max(img_masked) if img_masked.size > 0 else 0
    row_data[f"{staining}_std"] = np.std(img_masked) if img_masked.size > 0 else 0

    if np.any(mask_chan_bool):
        img_mask_chan = img[mask_chan_bool]
        row_data[f"{staining}_T_min"] = np.min(img_mask_chan)
        row_data[f"{staining}_T_mean"] = np.mean(img_mask_chan)
        row_data[f"{staining}_T_max"] = np.max(img_mask_chan)
        row_data[f"{staining}_T_std"] = np.std(img_mask_chan)
    else:
        row_data[f"{staining}_T_min"] = 0
        row_data[f"{staining}_T_mean"] = 0
        row_data[f"{staining}_T_max"] = 0
        row_data[f"{staining}_T_std"] = 0

    # Quantiles for full mask and threshold mask_channel
    for q in quantiles_to_calc:
        q_name = str(int(q * 100))
        row_data[f"{staining}_Q{q_name}"] = np.quantile(img_masked, q=q) if img_masked.size > 0 else 0

        if np.any(mask_chan_bool):
            row_data[f"{staining}_T_Q{q_name}"] = np.quantile(img[mask_chan_bool], q=q)
        else:
            row_data[f"{staining}_T_Q{q_name}"] = 0

    # Potency: mean intensity * area of T (thresholded mask)
    area_T = np.sum(mask_chan_bool)
    row_data[f"{staining}_area_T"] = area_T
    row_data[f"{staining}_potency"] = row_data[f"{staining}_mean"] * area_T

    # Substructure analysis

    # Threshold image using Otsu method
    try:
        otsu_thresh = filters.threshold_otsu(img)
    except Exception:
        # fallback if Otsu fails (e.g. uniform image)
        otsu_thresh = np.median(img)
    binary_image = img > otsu_thresh

    # Mask outside the segmented object: set to False
    binary_image = np.logical_and(binary_image, mask_bool)

    # Label connected components in binary image
    labeled_image, num_speckles = label(binary_image)

    # Filter out small speckles (< 15 pixels)
    min_size = 15
    filtered_speckles = np.zeros_like(labeled_image, dtype=int)
    current_label = 1

    for i in range(1, num_speckles + 1):
        speckle_mask = labeled_image == i
        if np.sum(speckle_mask) >= min_size:
            filtered_speckles[speckle_mask] = current_label
            current_label += 1

    # Collect speckle sizes and intensities
    remaining_labels = np.unique(filtered_speckles)
    remaining_labels = remaining_labels[remaining_labels != 0]  # exclude background

    speckle_sizes = []
    speckle_intensities = []

    for label_id in remaining_labels:
        speckle_mask = filtered_speckles == label_id
        speckle_sizes.append(np.sum(speckle_mask))
        speckle_intensities.append(np.mean(img[speckle_mask]))

    # Store substructure features, fallback 0 if empty
    row_data[f"{staining}_substructures_size"] = np.mean(speckle_sizes) if speckle_sizes else 0
    row_data[f"{staining}_substructures_intensity"] = np.mean(speckle_intensities) if speckle_intensities else 0
    row_data[f"{staining}_substructures_count"] = len(speckle_sizes)

    return row_data

def intensity_pearsonR(stainings, experiment_setup, images, barcode, well, OID, row_data):
    """
    Calculate Pearson correlation coefficients between staining channels within a segmented object.

    Parameters:
    - stainings (dict): Mapping of staining IDs to staining names.
    - experiment_setup (dict): Experimental setup info.
    - images (dict): Dictionary of staining channel images, including 'Mask'.
    - barcode (str): Barcode ID.
    - well (str): Well ID.
    - OID (str): Object ID (for API consistency).
    - row_data (dict): Dictionary to store the calculated features.

    Returns:
    - row_data (dict) with PearsonR features added.
    """
    # Cache the staining key list to avoid repeated indexing and splitting
    staining_key_list = stainings[experiment_setup[barcode][well][1]]
    n_channels = len(staining_key_list)

    mask_bool = images["Mask"].astype(bool)

    seen_pairs = set()

    for ch1 in range(n_channels):
        stain1 = staining_key_list[ch1]
        img1 = images[f"C0{ch1 + 1}"][mask_bool]

        for ch2 in range(ch1 + 1, n_channels):
            stain2 = staining_key_list[ch2]
            pair = frozenset([stain1, stain2])

            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)

            img2 = images[f"C0{ch2 + 1}"][mask_bool]

            # Handle edge cases: if either array is empty or constant, set correlation to NaN
            if img1.size == 0 or img2.size == 0 or np.std(img1) == 0 or np.std(img2) == 0:
                r = np.nan
            else:
                r = np.corrcoef(img1, img2)[0, 1]

            # Sort stains alphabetically in feature name for consistency
            key = f"{min(stain1, stain2)}-{max(stain1, stain2)}_PearsonR"
            row_data[key] = r

    return row_data

def image_analysis(images, OID, row_data, quantiles_to_calc, sigma_skeleton, spacing, stainings, experiment_setup,radius_multiplier):
    """
    Perform image analysis tasks and accumulate features in a row_data dict.

    Parameters:
    - images (dict): Dictionary of images.
    - OID (str): Object ID for dict storage.
    - row_data (dict): Dict where calculated features will be stored.
    - quantiles_to_calc (list): Quantiles to calculate for intensity features.
    - sigma_skeleton (int): Sigma param for Gaussian blurring before skeletonization.
    - spacing (float): Pixel spacing.
    - stainings (dict): Mapping from staining IDs to staining names.
    - experiment_setup (dict): Experimental setup info.
    - radius_multiplier (float): Multiplier for skeleton features.

    Returns:
    - row_data (dict): Updated dictionary including all calculated features.
    """
    # Parse barcode and well from OID just once
    barcode = "-".join(OID.split("-")[:-2])
    well = OID.split("-")[-2]

    # Shape features from mask
    row_data = shape_calc_mask(images["Mask"], row_data, OID, spacing)

    # Convex hull features (area, solidity, etc.)
    row_data = convex_hull_features(images["Mask"], row_data, OID, spacing, min_area_fraction=0.005)

    # Border fraction for filtering
    row_data = get_border_fraction(images["Mask"], row_data, OID)

    # Skeleton features (e.g., skeleton length, branch points, etc.)
    row_data = skeleton_feats(images, row_data, OID, spacing, sigma_skeleton, radius_multiplier)

    # Quickly resolve channel count and stainings for this well
    staining_key = experiment_setup[barcode][well][1]
    stain_names = stainings[staining_key]

    # Loop through channels
    for ch_idx, stain in enumerate(stain_names):
        channel = f"C0{ch_idx+1}"
        # Prepare copies for safety (should be only if needed)
        image = copy.deepcopy(images[channel])
        mask = copy.deepcopy(images["Mask"])
        mask_channel = copy.deepcopy(images.get(channel+"_Mask", mask))

        # Calculate channel-specific features
        row_data = channel_mask_feat_calc(mask, mask_channel, stain, row_data, OID, spacing)
        row_data = intensity_feat_calc(image, mask, mask_channel, row_data, stain, OID, quantiles_to_calc)
        row_data = moments_channel_mask(mask, image, row_data, OID, stain, spacing)

    # Pearson correlation coefficients for all combinations of intensities (across stains/channels)
    row_data = intensity_pearsonR(stainings, experiment_setup, images, barcode, well, OID, row_data)

    return row_data

def build_row(OID, bc, well, row, exp_info, PATH, stainings, experiment_ID):
    row_dict = {
        "Organoid_ID": OID,
        "Barcode": bc,
        "Well": well,
        "Object": OID.split("-")[-1],
        "Medium": exp_info[0],
        "ABs": exp_info[1],
        "PATH": PATH,
        "y_micrometer": row.y_micrometer,
        "x_micrometer": row.x_micrometer,
        "len_y_micrometer": row.len_y_micrometer,
        "len_x_micrometer": row.len_x_micrometer,
        "centroid_y_micrometer": row.y_micrometer + row.len_y_micrometer / 2.0,
        "centroid_x_micrometer": row.x_micrometer + row.len_x_micrometer / 2.0,
        "Cell_line": exp_info[2],
        "Other": exp_info[3],
        "Experiment_ID": experiment_ID
    }
    ab_key = exp_info[1]  # e.g. 'ABs'
    for ch_idx, ch_name in enumerate(stainings.get(ab_key, [])):
        row_dict[f"Staining_Ch{ch_idx+1}"] = ch_name
    return row_dict
    
def extract_features(
    ome_zarrs_dict,
    table_name,
    label_name,
    pyramid_level,
    source,
    folder,
    analysis_dir,
    barcodes,
    experiment_setup,
    thresholds,
    stainings,
    result_file_name,
    radius_multiplier,
    sigma_skeleton,
    quantiles_to_calc,
    experiment_ID
):
    """
    Optimized: Extract features from organoid images and save results in CSV and AnnData format.
    """
    # Prepare timestamp string once
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
    df_total_list = []

    # Create required output folders (robust to existing dirs)
    for subfolder in ["1_Scripts", "2_Tables", "3_Plots", "4_Results"]:
        os.makedirs(os.path.join(analysis_dir, subfolder), exist_ok=True)

    # -------- Main barcode loop ----------
    for i, bc in enumerate(tqdm(ome_zarrs_dict.keys(), desc="Barcodes")):
        plate = ome_zarrs_dict[bc]
        well_names = plate.get_names()
        wells_in_exp = experiment_setup[bc]
        # For efficient row accumulation:
        rows = []

        for well_idx, well in enumerate(tqdm(well_names, desc=f"Wells ({bc})", leave=False)):
            if well not in wells_in_exp:
                continue

            exp_info = wells_in_exp[well]
            well_ov = plate.images[well_idx]
            table = well_ov.get_table(table_name)
            pixel_spacing = well_ov.get_scale(pyramid_level=pyramid_level)[-1]  # Cache per well

            if table.empty:
                print(f"No table {table_name} for well {well} in barcode {bc}. Skipping.")
                continue

            for row_idx, row in enumerate(table.itertuples()):
                # --------- Smart array loading: ---------
                try:
                    img, mask = well_ov.get_array_pair_by_coordinate(
                        label_name=label_name,
                        pyramid_level=pyramid_level,
                        upper_left_yx=(row.y_micrometer, row.x_micrometer),
                        lower_right_yx=(row.y_micrometer+row.len_y_micrometer, row.x_micrometer+row.len_x_micrometer)
                    )
                except Exception as e:
                    print(f"Error retrieving arrays for {bc}, well {well}, organoid {row_idx}: {e}")
                    continue

                OID = f"{bc}-{well}-{row_idx}"
                PATH = well_ov.get_path()
                # --------- Prepare images dict ---------
                images = {f"C0{channel+1}": img[channel, 0] for channel in range(img.shape[0])}
                mask_arr = mask[label_name][0]
                mask_for_oid = (mask_arr == int(row_idx)+1).astype(mask_arr.dtype)
                images["Mask"] = mask_for_oid

                # --- Label check, only process single-labeled organoids ---
                n_labels = np.max(measure.label(mask_for_oid.astype(bool)))
                if n_labels != 1:
                    print(f"Skipping {getattr(row, 'UID', OID)} because it contains multiple labels. Expected one label per organoid.")
                    continue

                # --- Build observation row ---
                row_dict = build_row(OID, bc, well, row, exp_info, PATH, stainings, experiment_ID)

                # --- Pre-process images ---
                try:
                    images = image_preprocessing(
                        stainings, experiment_setup, images, OID, thresholds, sigma=3
                    )
                except Exception as e:
                    print(f"Error in image_preprocessing for {OID}: {e}")
                    continue

                # --- Calculate features and update row dict ---
                try:
                    row_dict = image_analysis(
                        images=images,
                        OID=OID,
                        row_data=row_dict,
                        quantiles_to_calc=quantiles_to_calc,
                        sigma_skeleton=sigma_skeleton,
                        spacing=pixel_spacing,
                        stainings=stainings,
                        experiment_setup=experiment_setup,
                        radius_multiplier=radius_multiplier
                        )
                except Exception as e:
                    print(f"Error in image_analysis for {OID}: {e}")
                    continue

                rows.append(row_dict)

        # ---- Build DataFrame for this plate ----
        if not rows:
            print(f"No valid rows for barcode {bc}.")
            continue
        df = pd.DataFrame(rows)

        # --- Remove moments-0-0 columns ---
        df = df.drop(columns=[col for col in df.columns if "moments-0-0" in col or "moments_weighted-0-0" in col], errors="ignore")

        # --- Save per-plate DF ---
        df = df.sort_values(by="Organoid_ID", key=natsort_keygen())
        save_path = os.path.join(source, folder[i], f"{result_file_name}_{barcodes[i]}_{timestamp}.csv")
        df.to_csv(save_path)
        print(f"Saved {bc} results as {save_path}.")

        df_total_list.append(df)

    # --- Merge all plates ---
    if not df_total_list:
        raise ValueError("No features extracted for any barcode; check your input data.")
    df_total = pd.concat(df_total_list, axis=0)

    # --- Save merged DataFrame and AnnData ---
    total_save_path = os.path.join(analysis_dir, "2_Tables", f"{result_file_name}_{timestamp}.csv")
    df_total = df_total.set_index("Organoid_ID", drop=False)
    df_total.to_csv(total_save_path)
    print(f"Saved merged csv as {total_save_path}.")

    # --- Compose AnnData ---
    feats_categorical = [
        "Organoid_ID", "Barcode", "Well", "Object", "Medium",
        "ABs", "Cell_line", "Other", "PATH", "Experiment_ID"
    ]
    feats_categorical += [col for col in df_total.columns if col.startswith("Staining_")]
    feats_numerical = [col for col in df_total.columns if col not in feats_categorical]

    ad = anndata.AnnData(
        X=df_total[feats_numerical].values,
        obs=df_total.loc[:, feats_categorical],
        var=pd.DataFrame(index=feats_numerical)
    )

    ad.uns["stainings"] = stainings
    ad.uns["experiment_setup"] = experiment_setup
    ad.uns["pixel_spacing"] = pixel_spacing
    ad.uns["pyramid_level"] = pyramid_level
    ad.uns["folders"] = folder
    ad.uns["source_dir"] = source
    ad.uns["table_dir"] = os.path.join(analysis_dir, "2_Tables")
    ad.uns["plot_dir"] = os.path.join(analysis_dir, "3_Plots")
    ad.uns["table_name"] = table_name
    ad.uns["label_name"] = label_name
    ad.uns["thresholds"] = {k: v[2] for k, v in thresholds.items()}

    save_adata(
        ad,
        f"{result_file_name}_{timestamp}",
    )
    return ad