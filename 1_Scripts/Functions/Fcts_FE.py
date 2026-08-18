import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

from skimage import filters, measure, morphology, segmentation
from natsort import natsorted
from skan import Skeleton, summarize
from IPython.display import display
import matplotlib.pyplot as plt
from scipy.ndimage import label
from tqdm.notebook import tqdm
import seaborn as sns
import pandas as pd
from ez_zarr import ome_zarr
import numpy as np
import itertools
import datetime
import anndata
import random
import glob
import copy
import math
import cv2
import os

from Functions.Fcts_Base import find_staining_in_ABs, find_barcodes_with_day, save_adata
from Functions.Fcts_Plotting import load_img_mask_by_UID

# Disable pandas performance warnings
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)



"""
***
FEATURE EXTRACTION (MULTICYCLE COMPATIBLE)
***

Implements multi-round feature extraction for datasets where the OME-Zarr structure is:
  <plate>.zarr/<row>/<col>/<round>/...

Important note for ez_zarr:
- ez_zarr.import_plate(..., image_name="0") selects the image group within each well (often "0").
- For multiplexing cycles stored as "0", "1", ... image groups, each round must be imported
  separately with image_name=str(round) to access that data.

Assumptions:
- ROI table exists in segmentation_round (default 0)
- labels exist in segmentation_round (default 0)
- additional rounds can be missing labels and tables
- segmentation is identical across rounds

Strategy:
- morphology features: computed once from segmentation_round labels
- intensity features: computed per round and prefixed with R{round}__
- Pearson correlations: computed across all (round, stain) vectors (within + between rounds)
"""

# NOTE: This file is a direct copy of the former top-level Fcts_FE.py with only import paths updated.


def make_experiment(source, folder, layout_sheets = ["MediumLayout", "StainingLayout", "LineLayout", "OtherLayout"]):
    
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
    barcodes = list(barcodes)

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

    display_experiment_setup(result, folder)
    barcodes = list(result.keys())
    
    return result, barcodes


def display_experiment_setup(experiment, folder):
    """
    Display the experiment setup for each barcode.

    Parameters:
    - experiment (dict): Experiment setup dictionary.
    """
    # Extract row and col names
    for i, barcode in enumerate(experiment.keys()):
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
        print("Source folder: %s"%folder[i])
        display(df_hm)
    
    return None


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

def apply_threshold_changes(thresholds_raw_by_round, thresholds_by_round, thresholds_to_change):
    """
    Apply multiple threshold updates from a dict like:
    {"R0__DAPI": 1000, "R0__KRT7": 500}

    Parameters:
    - thresholds_raw_by_round (dict): Raw threshold structure by round.
    - thresholds_by_round (dict): Simplified threshold structure by round.
    - thresholds_to_change (dict): Mapping of 'R{round}__{staining}' -> new threshold

    Returns:
    - thresholds_raw_by_round (dict)
    - thresholds_by_round (dict)
    """
    for key, new_threshold in thresholds_to_change.items():
        round_to_change_str, staining = key.split("__")
        round_to_change = int(round_to_change_str.replace("R", ""))

        thresholds_raw_by_round[round_to_change] = threshold_change(
            threshold=thresholds_raw_by_round[round_to_change],
            staining=staining,
            new_threshold=new_threshold,
        )

        thresholds_by_round[round_to_change] = {
            k: v[2] for k, v in thresholds_raw_by_round[round_to_change].items()
        }

    return thresholds_raw_by_round, thresholds_by_round

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

        # Check if the current pixel is within the image bounds and non-zero
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


def test_skeletonization(barcodes, stainings, experiment_setup, ome_zarrs_dict, ome_zarr_df, table_name, label_name, pyramid_level=0,
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
            bc, well, _ = mask_candidate.rsplit("-", 2)
            staining_dummy = "R0__"+stainings[experiment_setup[bc][well][1]]["0"][0]
            _, mask_img = load_img_mask_by_UID(mask_candidate, stainings, experiment_setup, ome_zarrs_dict, table_name, label_name,
                                               pyramid_level, staining_dummy)
            if np.max(measure.label(mask_img.astype(bool))) == 1:
                if mask_candidate not in sampled_masks:
                    sampled_masks.append(mask_candidate)
            n_attempts += 1

        well_names = ome_zarrs_dict[barcode].get_names()
        ncols = max(len(sampled_masks), 1)
        fig, axarr = plt.subplots(1, ncols, figsize=(5*ncols, 8), squeeze=False)
        plt.suptitle(barcode, fontsize=20)

        for i, mask in enumerate(sampled_masks):
            bc, well_id, obj_id = mask.rsplit("-", 2)
            staining_dummy = "R0__"+stainings[experiment_setup[bc][well_id][1]]["0"][0]
            spacing = ome_zarrs_dict[barcode][well_names.index(well_id)].get_scale(pyramid_level)[-1]
            _, image = load_img_mask_by_UID(mask, stainings, experiment_setup, ome_zarrs_dict, table_name, label_name,
                                               pyramid_level, staining_dummy)
            image = image.astype(np.uint8)
            image[image != 0] = 255
            skeleton, mask_circle, radius, center, crypt_count, crypt_length_total, longest_crypt = \
                extract_skeleton_features(image, spacing, sigma_skeleton, radius_multiplier, n_angle_determination)
            image_plot = np.copy(image)
            image_plot[skeleton.astype(bool)] = 0
            img_plot = cv2.circle(image_plot, tuple(center), int(radius*radius_multiplier), 0) # Draw max inscribed circle
            img_plot = pad_to_aspect_ratio(img_plot)
            axes = np.atleast_1d(axarr[0])
            ax = axes[i]
            ax.imshow(img_plot, cmap=plt.cm.gray, aspect="auto")
            ax.set_axis_off()
            ax.set_title(
                f"{well_id} - {obj_id}\nNumber of crypts: {crypt_count}\nCrypt length: {int(crypt_length_total)} µm\nLongest crypt: {int(longest_crypt)} µm",
                fontsize=12)
        plt.tight_layout()
        plt.show()


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


def shape_calc_mask(mask, row_data, OID, spacing, include_moments: bool = True):
    """
    Calculate various shape features for a segmented object.

    Parameters:
    - mask (numpy.ndarray): Binary mask representing the segmented object.
    - row_data (dict): Dictionary to store the calculated features.
    - OID (str): Object ID (kept for API consistency).
    - spacing (float): Pixel spacing.
    - include_moments (bool): If False, do not compute regionprops 'moments' (saves time + columns).

    Returns:
    - row_data (dict): Updated with shape features.
    """

    # Define properties to calculate
    features = [
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
        'perimeter',
        'solidity',
    ]
    if include_moments:
        features.append('moments')

    # Compute region properties table (returns dict of arrays)
    props = measure.regionprops_table(
        mask.astype(np.uint8),
        properties=tuple(features),
        spacing=(spacing, spacing),
        cache=True,
    )

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
    convex_image = prop_2D.image_convex
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
    row_data["concavity"] = (prop_2D.area_convex - object_area) / prop_2D.area_convex

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


def intensity_feat_calc(
    img,
    mask,
    mask_channel,
    row_data,
    staining,
    OID,
    quantiles_to_calc,
    include_thresholded: bool = True,
    include_substructure: bool = True,
):
    """
    Calculate intensity-based features for a staining channel within a segmented object.

    Parameters:
    - include_thresholded: if False, do NOT compute any _T_ features (incl. T-quantiles, potency, area_T).
    - include_substructure: if False, do NOT run Otsu + connected component substructure analysis.
    """

    mask_bool = mask.astype(bool)
    mask_chan_bool = mask_channel.astype(bool)

    img_masked = img[mask_bool]

    # --- Always compute full-mask intensity stats ---
    row_data[f"{staining}_min"] = np.min(img_masked) if img_masked.size > 0 else 0
    row_data[f"{staining}_mean"] = np.mean(img_masked) if img_masked.size > 0 else 0
    row_data[f"{staining}_max"] = np.max(img_masked) if img_masked.size > 0 else 0
    row_data[f"{staining}_std"] = np.std(img_masked) if img_masked.size > 0 else 0

    # Full-mask quantiles
    for q in quantiles_to_calc:
        q_name = str(int(q * 100))
        row_data[f"{staining}_Q{q_name}"] = np.quantile(img_masked, q=q) if img_masked.size > 0 else 0

    # --- Optional: thresholded (_T_) features ---
    if include_thresholded:
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

        for q in quantiles_to_calc:
            q_name = str(int(q * 100))
            if np.any(mask_chan_bool):
                row_data[f"{staining}_T_Q{q_name}"] = np.quantile(img[mask_chan_bool], q=q)
            else:
                row_data[f"{staining}_T_Q{q_name}"] = 0

        # Potency: mean intensity * area of T (thresholded mask)
        area_T = int(np.sum(mask_chan_bool))
        row_data[f"{staining}_area_T"] = area_T
        row_data[f"{staining}_potency"] = row_data[f"{staining}_mean"] * area_T

    # --- Optional: substructure analysis ---
    if include_substructure:
        img_speckles = np.copy(img)
        img_speckles[mask_bool] = 0
        labeled_image, num_speckles = label(img_speckles)

        filtered_speckles = np.zeros_like(labeled_image, dtype=int)
        current_label = 1

        for i in range(1, num_speckles + 1):
            speckle_mask = labeled_image == i
            filtered_speckles[speckle_mask] = current_label
            current_label += 1

        remaining_labels = np.unique(filtered_speckles)
        remaining_labels = remaining_labels[remaining_labels != 0]

        speckle_sizes = []
        speckle_intensities = []

        for label_id in remaining_labels:
            speckle_mask = filtered_speckles == label_id
            speckle_sizes.append(int(np.sum(speckle_mask)))
            speckle_intensities.append(float(np.mean(img[speckle_mask])))

        row_data[f"{staining}_substructures_size"] = float(np.mean(speckle_sizes)) if speckle_sizes else 0
        row_data[f"{staining}_substructures_intensity"] = float(np.mean(speckle_intensities)) if speckle_intensities else 0
        row_data[f"{staining}_substructures_count"] = int(len(speckle_sizes))

    return row_data




def _as_path_str(p) -> str:
    if isinstance(p, (list, tuple)) and len(p) > 0:
        p = p[0]
    return str(p)



def _parse_well_round_from_path(path: str) -> tuple[str, int] | tuple[None, None]:
    # Expect .../<row>/<col>/<round>
    parts = _as_path_str(path).replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 3:
        return None, None
    row = parts[-3]
    col = parts[-2]
    rnd = parts[-1]
    try:
        rnd_int = int(rnd)
    except Exception:
        return None, None
    return f"{row}{col}", rnd_int



def build_well_round_map(plate) -> dict[tuple[str, int], object]:
    """Map (well, round) -> ez_zarr Image object (for the images contained in this plate object)."""
    m = {}
    for img in plate.images:
        well, rnd = _parse_well_round_from_path(img.get_path())
        if well is None:
            continue
        m[(well, rnd)] = img
    return m


# --- Round-aware plate loading (ez_zarr image_name) ---
_PLATE_CACHE: dict[tuple[str, int], object] = {}



def _infer_plate_root_from_image_path(img_path: str) -> str:
    """Infer the plate root (.zarr) path from an image path ending with /<row>/<col>/<image_name>."""
    parts = _as_path_str(img_path).replace("\\", "/").rstrip("/").split("/")
    if len(parts) < 4:
        return _as_path_str(img_path)
    # remove last 3: row/col/image_name
    return "/".join(parts[:-3])



def _default_round_from_plate(plate) -> int:
    """Attempt to infer which image_name was used to import this plate (usually 0)."""
    try:
        if hasattr(plate, "images") and len(plate.images) > 0:
            _, rnd = _parse_well_round_from_path(plate.images[0].get_path())
            if rnd is not None:
                return int(rnd)
    except Exception:
        pass
    return 0



def _plate_root_from_plate(plate) -> str:
    """Infer plate root path from the first available image path."""
    if hasattr(plate, "images") and len(plate.images) > 0:
        return _infer_plate_root_from_image_path(plate.images[0].get_path())
    # Fallback: try plate.paths if available
    if hasattr(plate, "paths") and plate.paths:
        return _infer_plate_root_from_image_path(plate.paths[0])
    raise ValueError("Cannot infer plate root path from plate object")



def get_plate_for_round(plate, round_id: int):
    """Return a plate object for a given round (image_name), caching imports."""
    round_id = int(round_id)
    default_r = _default_round_from_plate(plate)
    if round_id == default_r:
        return plate

    root = _plate_root_from_plate(plate)
    key = (root, round_id)
    if key not in _PLATE_CACHE:
        _PLATE_CACHE[key] = ome_zarr.import_plate(root, image_name=str(round_id))
    return _PLATE_CACHE[key]



def build_well_round_map_multi(plate, rounds: set[int]) -> dict[tuple[str, int], object]:
    """Build a (well, round)->image map across multiple rounds by importing each round plate."""
    m_all: dict[tuple[str, int], object] = {}
    for r in sorted({int(x) for x in rounds}):
        pr = get_plate_for_round(plate, r)
        m_all.update(build_well_round_map(pr))
    return m_all



def get_stains_for_round(stainings: dict, ab_key: str, round_id: int) -> list[str]:
    v = stainings.get(ab_key, [])
    round_id = str(round_id)
    if isinstance(v, dict):
        return v.get(round_id, [])
    return v



def normalize_thresholds(thresholds):
    """Normalize thresholds into dict[int, dict[str, float]]."""
    if thresholds is None:
        return {}

    # thresholds provided as dict[stain] -> [ch, ..., value] (legacy)
    try:
        any_v = next(iter(thresholds.values()))
    except StopIteration:
        return {}

    if isinstance(any_v, (list, tuple)) and len(any_v) >= 3:
        return {0: {k: float(v[2]) for k, v in thresholds.items()}}

    # thresholds provided as dict[round] -> dict[stain] -> value
    if isinstance(any_v, dict):
        out = {}
        for r, d in thresholds.items():
            out[int(r)] = {k: float(v) for k, v in d.items()}
        return out

    # thresholds provided as dict[stain] -> value
    if isinstance(any_v, (int, float, np.number)):
        return {0: {k: float(v) for k, v in thresholds.items()}}

    raise ValueError("Unrecognized thresholds format")



def morphology_features(
    mask,
    row_data,
    OID,
    spacing,
    sigma_skeleton,
    radius_multiplier,
    include_moments: bool = True,
    include_skeleton: bool = True,
):
    row_data = shape_calc_mask(mask, row_data, OID, spacing, include_moments=include_moments)
    row_data = convex_hull_features(mask, row_data, OID, spacing, min_area_fraction=0.005)
    row_data = get_border_fraction(mask, row_data, OID)

    if include_skeleton:
        row_data = skeleton_feats({"Mask": mask}, row_data, OID, spacing, sigma_skeleton, radius_multiplier)
    else:
        # Ensure skeleton-derived features are absent
        row_data.pop("crypt_count", None)
        row_data.pop("crypt_length_total", None)
        row_data.pop("crypt_length_max", None)

    return row_data




def intensity_features_for_round(
    img_stack,
    mask,
    stain_names,
    thresholds_round: dict[str, float],
    quantiles_to_calc,
    round_id: int,
    spacing: float,
    OID: str,
    sigma: float = 3,
    include_thresholded: bool = True,
    include_substructure: bool = True,
    include_moments: bool = True,
):
    """
    Compute intensity features for a single round.

    Returns
    -------
    row_data_round : dict
        Feature names are prefixed via staining name: R{round}__{stain}
    vectors : dict[str, np.ndarray]
        Masked intensity vectors per (round, stain) for Pearson correlations.
    """

    row_data_round = {}
    vectors = {}

    mask_bool = mask.astype(bool)

    for ch_idx, stain in enumerate(stain_names):
        if ch_idx >= img_stack.shape[0]:
            continue

        staining_key = f"R{int(round_id)}__{stain}"

        img = img_stack[ch_idx, 0]
        img = np.pad(img, pad_width=20, mode="constant", constant_values=0)
        
        # Smooth and mask
        img_proc = filters.gaussian(img, sigma=sigma, preserve_range=True)
        img_proc[~mask_bool] = 0

        # Threshold mask for this stain/round (only used if thresholded features requested)
        thr = thresholds_round.get(stain, 0)
        mask_channel = (img_proc > thr) & mask_bool

        # Threshold-derived geometry features (area_T, asymmetry, etc.)
        if include_thresholded:
            row_data_round = channel_mask_feat_calc(
                mask,
                mask_channel.astype(bool),
                staining_key,
                row_data_round,
                OID=OID,
                spacing=spacing,
            )

        # Intensity features
        row_data_round = intensity_feat_calc(
            img,
            mask,
            mask_channel.astype(bool),
            row_data_round,
            staining_key,
            OID,
            quantiles_to_calc=quantiles_to_calc,
            include_thresholded=include_thresholded,
            include_substructure=include_substructure,
        )

        # Channel moments (moments_weighted)
        if include_moments:
            row_data_round = moments_channel_mask(
                mask,
                img,
                row_data_round,
                OID=OID,
                staining=staining_key,
                spacing=spacing,
            )

        # Pearson vectors (raw intensities within mask) – keep regardless of flags
        vectors[staining_key] = img[mask_bool]

    return row_data_round, vectors




def pearson_features_from_vectors(vectors: dict[str, np.ndarray]) -> dict:
    out = {}
    keys = sorted(vectors.keys())

    for k1, k2 in itertools.combinations(keys, 2):
        v1 = vectors[k1]
        v2 = vectors[k2]

        if v1.size == 0 or v2.size == 0 or np.std(v1) == 0 or np.std(v2) == 0:
            r = np.nan
        else:
            r = float(np.corrcoef(v1, v2)[0, 1])

        out[f"{k1}--{k2}_PearsonR"] = r

    return out



def estimate_staining_thresholds_all_rounds(
    rounds: list[int],
    ome_zarr_df,
    ome_zarr_dict,
    stainings,
    experiment_setup,
    table_name,
    label_name,
    pyramid_level=0,
    segmentation_round: int = 0,
    control_condition=None,
    n=20,
    seed=0,
    sigma=3,
    q=0.5,
):
    """Run threshold estimation across all rounds and aggregate results.

    Wraps `estimate_staining_thresholds_multicycle` for each round in `rounds`
    and collects results into per-round dictionaries.

    Returns
    -------
    thresholds_raw_by_round : dict
        {round_id: {stain: [channel_idx, raw_values, final_threshold]}}
    thresholds_by_round : dict
        {round_id: {stain: final_threshold}}
    dict_org_by_round : dict
        {round_id: {stain: {day: [UIDs]}}}
    timepoints_by_round : dict
        {round_id: [days]}
    """
    thresholds_raw_by_round = {}
    thresholds_by_round = {}
    dict_org_by_round = {}
    timepoints_by_round = {}

    for r in rounds:
        thr_r, dict_org, timepoints = estimate_staining_thresholds_multicycle(
            ome_zarr_df=ome_zarr_df,
            ome_zarr_dict=ome_zarr_dict,
            stainings=stainings,
            experiment_setup=experiment_setup,
            table_name=table_name,
            label_name=label_name,
            pyramid_level=pyramid_level,
            control_condition=control_condition,
            n=n,
            seed=seed,
            sigma=sigma,
            q=q,
            round_id=r,
            segmentation_round=segmentation_round,
        )
        thresholds_raw_by_round[r] = thr_r
        thresholds_by_round[r] = {stain: vals[2] for stain, vals in thr_r.items()}
        dict_org_by_round[r] = dict_org
        timepoints_by_round[r] = timepoints

    return thresholds_raw_by_round, thresholds_by_round, dict_org_by_round, timepoints_by_round

def estimate_staining_thresholds_multicycle(
    ome_zarr_df,
    ome_zarr_dict,
    stainings,
    experiment_setup,
    table_name,
    label_name,
    pyramid_level=0,
    control_condition=None,
    n=20,
    seed=0,
    sigma=3,
    q=0.5,
    round_id: int = 0,
    segmentation_round: int = 0,
):
    """Estimate staining thresholds for a specific round using segmentation_round ROI table coordinates.

    For each staining, randomly samples up to `n` organoids per day from control wells
    (or all wells if control_condition is None), extracts the bounding-box image crop,
    applies Gaussian blur, computes the triangle threshold, and returns the q-quantile
    over all samples as the final threshold.

    Returns
    -------
    thresholds : dict
        {stain: [channel_index, [raw_threshold_values], final_threshold]}
    dict_org : dict
        {stain: {day: [UIDs]}}
    timepoints_lst : list
        natsorted list of days present in the experiment.
    """
    random.seed(seed)
    rounds_needed = {round_id, segmentation_round}
    # Build stain → [channel_idx, raw_values, final_threshold] per AB mix for this round
    thresholds = {}
    for ab_mix in stainings:
        for i, stain in enumerate(get_stains_for_round(stainings, ab_mix, round_id)):
            thresholds[stain] = [i + 1, [], np.nan]

    # Collect unique days from experiment setup (index 3 = OtherLayout value)
    days = natsorted({
        experiment_setup[plate][well][3]
        for plate in experiment_setup
        for well in experiment_setup[plate]
    })

    dict_org = {stain: {day: [] for day in days} for stain in thresholds}
    dict_org_lst = {stain: [] for stain in thresholds}

    # Populate dict_org: map each stain × day to a list of UIDs
    for stain in thresholds:
        ab_mixes = find_staining_in_ABs(stainings, stain)
        for day in days:
            barcodes_day = find_barcodes_with_day(experiment_setup, day)
            mask = (
                ome_zarr_df["Barcode"].isin(barcodes_day)
                & ome_zarr_df["AB"].isin(ab_mixes)
                & (ome_zarr_df["Day"] == day)
            )
            if control_condition is not None:
                mask &= ome_zarr_df["Medium"] == control_condition
            dict_org[stain][day] = list(ome_zarr_df.loc[mask, "UID"])

    # Randomly sample up to n organoids per day; take all if fewer than n available
    for stain in thresholds:
        sampled = []
        for day in days:
            pool = dict_org[stain][day]
            sampled.extend(pool if len(pool) <= n else random.sample(pool, n))
        dict_org_lst[stain] = sampled

    # Cache well-round maps per barcode to avoid repeated imports
    maps_by_barcode: dict[str, dict[tuple[str, int], object]] = {}

    # Compute per-organoid triangle thresholds
    for stain, uids in dict_org_lst.items():
        ch = thresholds[stain][0] - 1  # 0-based channel index

        for uid in uids:
            bc, well, index_str = uid.rsplit("-", 2)
            idx = int(index_str)

            plate = ome_zarr_dict[bc]
            if bc not in maps_by_barcode:
                maps_by_barcode[bc] = build_well_round_map_multi(plate, rounds=rounds_needed)
            m = maps_by_barcode[bc]

            img_seg = m.get((well, segmentation_round))
            img_r = m.get((well, round_id))
            if img_seg is None or img_r is None:
                thresholds[stain][1].append(np.nan)
                continue

            table = img_seg.get_table(table_name, as_AnnData=True)
            if table is None:
                thresholds[stain][1].append(np.nan)
                continue

            if "label" in table.obs.columns:
                table.obs_names = table.obs["label"].astype(str)
            table = table.to_df()

            if table.empty or str(idx) not in table.index:
                thresholds[stain][1].append(np.nan)
                continue

            entry = table.loc[str(idx)]
            ul_y, ul_x = entry["y_micrometer"], entry["x_micrometer"]
            lr_y = ul_y + entry["len_y_micrometer"]
            lr_x = ul_x + entry["len_x_micrometer"]

            try:
                img = img_r.get_array_by_coordinate(
                    pyramid_level=pyramid_level,
                    upper_left_yx=(ul_y, ul_x),
                    lower_right_yx=(lr_y, lr_x),
                )[ch, 0]
            except Exception as e:
                warnings.warn(f"Image load failed for {uid} ({stain}): {e}")
                thresholds[stain][1].append(np.nan)
                continue

            img = filters.gaussian(img, sigma=sigma, preserve_range=True)
            raw = filters.threshold_triangle(img) if np.max(img) != 0 else np.nan
            thresholds[stain][1].append(raw)

        raw_vals = thresholds[stain][1]
        thresholds[stain][2] = float(np.nanquantile(raw_vals, q=q)) if raw_vals else np.nan

    return thresholds, dict_org, days



def plot_thresholds_multicycle(
    thresholds,
    dict_org,
    timepoints_lst,
    ome_zarr_dict,
    table_name,
    label_name,
    seed=0,
    pyramid_level=0,
    round_id: int = 0,
    segmentation_round: int = 0,
):
    """A minimal round-aware replacement for plot_thresholds (supports Round>0 without labels)."""

    import random
    import matplotlib.pyplot as plt
    import seaborn as sns

    random.seed(seed)

    round_id = int(round_id)
    segmentation_round = int(segmentation_round)
    rounds_needed = {round_id, segmentation_round}

    rows = len(dict_org[list(dict_org.items())[0][0]].keys()) + 1
    cols = len(dict_org.keys())

    fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(3 * cols, 3 * rows))

    for i, stain in enumerate(thresholds.keys()):
        if len(thresholds[stain][1]) > 1:
            sns.kdeplot(
                ax=ax[0, i],
                x=thresholds[stain][1],
                color="green",
                cut=0,
                fill=True,
                linewidth=1,
            )
            ax[0, i].axes.get_yaxis().set_visible(False)
            ax[0, i].set_title(stain, fontsize=12)
            ax[0, i].axvline(thresholds[stain][2], color="red")
            ax[0, i].text(x=0.9, y=0.9, s="T: " + str(int(thresholds[stain][2])), transform=ax[0, i].transAxes, ha="right")
        else:
            ax[0, i].imshow(np.zeros((200, 200)), aspect="auto", cmap="binary")
            ax[0, i].set_title(stain, fontsize=12)
            ax[0, i].set_axis_off()
            ax[0, i].text(x=0.5, y=0.5, s="No threshold found", transform=ax[0, i].transAxes, ha="center")

    # Cache per-barcode maps
    maps_by_barcode: dict[str, dict[tuple[str, int], object]] = {}

    # Show one random ROI per timepoint
    for col, stain in enumerate(thresholds.keys()):
        for row in range(1, len(timepoints_lst) + 1):
            uids = dict_org[stain][timepoints_lst[row - 1]]
            if len(uids) == 0:
                ax[row, col].imshow(np.zeros((200, 200)), aspect="auto", cmap="binary")
                ax[row, col].set_axis_off()
                ax[row, col].text(x=0.5, y=0.5, s="No image found", transform=ax[row, col].transAxes, ha="center")
                continue

            UID = random.sample(uids, 1)[0]
            bc, well, index_str = UID.rsplit("-", 2)   # split from right into 3 parts
            idx = int(index_str)
            
            plate = ome_zarr_dict[bc]
            if bc not in maps_by_barcode:
                maps_by_barcode[bc] = build_well_round_map_multi(plate, rounds=rounds_needed)
            m = maps_by_barcode[bc]

            img_seg = m.get((well, segmentation_round))
            img_r = m.get((well, round_id))

            if img_seg is None or img_r is None:
                ax[row, col].imshow(np.zeros((200, 200)), aspect="auto", cmap="binary")
                ax[row, col].set_axis_off()
                continue

            table = img_seg.get_table(table_name, as_AnnData = True)

            if "label" in table.obs.columns:
                table.obs_names = table.obs["label"]
            table = table.to_df()

            entry = table.loc[str(idx), :]
            ul_y, ul_x = entry["y_micrometer"], entry["x_micrometer"]
            lr_y = ul_y + entry["len_y_micrometer"]
            lr_x = ul_x + entry["len_x_micrometer"]

            ch = thresholds[stain][0] - 1
            try:
                img = img_r.get_array_by_coordinate(
                    pyramid_level=pyramid_level,
                    upper_left_yx=(ul_y, ul_x),
                    lower_right_yx=(lr_y, lr_x),
                )
                img = img[ch, 0]

                _, mask0 = img_seg.get_array_pair_by_coordinate(
                    label_name=label_name,
                    pyramid_level=pyramid_level,
                    upper_left_yx=(ul_y, ul_x),
                    lower_right_yx=(lr_y, lr_x),
                )

                mask_arr = mask0[label_name][0]
                mask_for_oid = (mask_arr == int(idx)).astype(mask_arr.dtype).astype(bool)
                img[~mask_for_oid] = 0

                boundary = segmentation.find_boundaries(mask_for_oid, mode="outer")

            except Exception:
                img = np.zeros((200, 200))
                mask_for_oid = np.zeros_like(img, dtype=bool)

            vmin = thresholds[stain][2]
            vmax = max(np.percentile(img[mask_for_oid], 99), vmin+1)

            if np.sum(img) > 0:
                img[boundary] = max(vmax, img.max())

            ax[row, col].imshow(img, vmin=vmin, vmax=vmax, aspect="auto", cmap="inferno")
            ax[row, col].set_axis_off()

    plt.tight_layout()
    plt.show()



def extract_features_multicycle(
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
    experiment_ID,
    rounds_to_extract: list[int] | None = None,
    segmentation_round: int = 0,
    sigma_intensity: float = 3,
    compute_cross_round_pearson: bool = True,
    do_moments_features: bool = True,
    do_skeleton_features: bool = True,
    do_thresholded_features: bool = True,
    do_substructure_features: bool = True,
    ):
    """Extract features with multi-round intensity support."""

    thresholds_by_round = normalize_thresholds(thresholds)

    if rounds_to_extract is None:
        rounds_to_extract = [0]
    rounds_to_extract = [int(r) for r in rounds_to_extract]
    segmentation_round = int(segmentation_round)

    rounds_needed = set(rounds_to_extract + [segmentation_round])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")

    for subfolder in ["1_Scripts", "2_Tables", "3_Plots", "4_Results"]:
        os.makedirs(os.path.join(analysis_dir, subfolder), exist_ok=True)

    df_total_list = []

    for i, bc in enumerate(ome_zarrs_dict.keys()):
        print(f"Processing barcode: {bc}")
        plate = ome_zarrs_dict[bc]
        m = build_well_round_map_multi(plate, rounds=rounds_needed)
        wells_in_exp = experiment_setup[bc]
        rows = []    
        for well in wells_in_exp.keys():
            print(f"  Processing well: {well}")
            img_seg = m.get((well, segmentation_round))
            if img_seg is None:
                continue

            exp_info = wells_in_exp[well]
            ab_key = exp_info[1]

            table = img_seg.get_table(table_name, as_AnnData = True)

            if table is None:
                continue

            if "label" in table.obs.columns:
                table.obs_names = table.obs["label"]
            table = table.to_df()

            if table.empty:
                continue

            pixel_spacing = img_seg.get_scale(pyramid_level=pyramid_level)[-1]

            for row in table.itertuples():
                row_idx = row.Index
                OID = f"{bc}-{well}-{row_idx}"
                ul_y, ul_x = row.y_micrometer, row.x_micrometer
                lr_y = row.y_micrometer + row.len_y_micrometer
                lr_x = row.x_micrometer + row.len_x_micrometer

                # Load segmentation round pair for mask
                try:
                    _, mask0 = img_seg.get_array_pair_by_coordinate(
                        label_name=label_name,
                        pyramid_level=pyramid_level,
                        upper_left_yx=(ul_y, ul_x),
                        lower_right_yx=(lr_y, lr_x),
                    )
                except Exception:
                    continue

                mask_arr = mask0[label_name][0]
                mask_arr  = np.pad(mask_arr, pad_width=20, mode="constant", constant_values=0)
                mask_for_oid = (mask_arr == int(row_idx)).astype(mask_arr.dtype)

                # Skip multi-label ROIs
                from skimage import measure
                if np.max(measure.label(mask_for_oid.astype(bool))) != 1:
                    continue

                row_data = {
                    "Organoid_ID": OID,
                    "Barcode": bc,
                    "Well": well,
                    "Object": str(row_idx),
                    "Medium": exp_info[0],
                    "ABs": exp_info[1],
                    "Cell_line": exp_info[2],
                    "Other": exp_info[3],
                    "PATH": img_seg.get_path(),
                    "y_micrometer": row.y_micrometer,
                    "x_micrometer": row.x_micrometer,
                    "len_y_micrometer": row.len_y_micrometer,
                    "len_x_micrometer": row.len_x_micrometer,
                    "centroid_y_micrometer": row.y_micrometer + row.len_y_micrometer / 2.0,
                    "centroid_x_micrometer": row.x_micrometer + row.len_x_micrometer / 2.0,
                    "Experiment_ID": experiment_ID,
                }

                # Add staining columns for each round
                for r in sorted(rounds_needed):
                    stains_r = get_stains_for_round(stainings, ab_key, r)
                    for ch_i, stain in enumerate(stains_r):
                        row_data[f"Staining_R{r}_Ch{ch_i+1}"] = stain

                # Morphology once (segmentation round)
                row_data = morphology_features(
                    mask_for_oid,
                    row_data,
                    OID,
                    spacing=pixel_spacing,
                    sigma_skeleton=sigma_skeleton,
                    radius_multiplier=radius_multiplier,
                    include_moments=do_moments_features,
                    include_skeleton=do_skeleton_features,
                )

                # Intensity per round
                pearson_vectors = {}
                for r in rounds_to_extract:
                    img_r_obj = m.get((well, int(r)))
                    if img_r_obj is None:
                        continue

                    stains_r = get_stains_for_round(stainings, ab_key, r)
                    if len(stains_r) == 0:
                        continue

                    try:
                        img_r = img_r_obj.get_array_by_coordinate(
                            pyramid_level=pyramid_level,
                            upper_left_yx=(ul_y, ul_x),
                            lower_right_yx=(lr_y, lr_x),
                        )
                    except Exception:
                        continue

                    thr_r = thresholds_by_round.get(int(r), {})

                    feats_r, vecs_r = intensity_features_for_round(
                    img_stack=img_r,
                    mask=mask_for_oid,
                    stain_names=stains_r,
                    thresholds_round=thr_r,
                    quantiles_to_calc=quantiles_to_calc,
                    round_id=int(r),
                    spacing=pixel_spacing,
                    OID=OID,
                    sigma=sigma_intensity,
                    include_thresholded=do_thresholded_features,
                    include_substructure=do_substructure_features,
                    include_moments=do_moments_features,
                    )

                    row_data.update(feats_r)
                    pearson_vectors.update(vecs_r)

                # Pearson across all rounds/channels
                if compute_cross_round_pearson and len(pearson_vectors) >= 2:
                    row_data.update(pearson_features_from_vectors(pearson_vectors))

                rows.append(row_data)

        if not rows:
            continue

        df = pd.DataFrame(rows)
        df = df.drop(columns=[col for col in df.columns if "moments-0-0" in col or "moments_weighted-0-0" in col], errors="ignore")

        # Save per-plate
        save_path = os.path.join(source, folder[i], f"{result_file_name}_{barcodes[i]}_{timestamp}.csv")
        df.to_csv(save_path)
        print(f"Saved {bc} results as {save_path}.")

        df_total_list.append(df)

    if not df_total_list:
        raise ValueError("No features extracted for any barcode; check your input data.")

    df_total = pd.concat(df_total_list, axis=0)
    df_total = df_total.set_index("Organoid_ID", drop=False)

    total_save_path = os.path.join(analysis_dir, "2_Tables", f"{result_file_name}_{timestamp}.csv")
    df_total.to_csv(total_save_path)
    print(f"Saved merged csv as {total_save_path}.")

    feats_categorical = [
        "Organoid_ID", "Barcode", "Well", "Object", "Medium",
        "ABs", "Cell_line", "Other", "PATH", "Experiment_ID",
        'y_micrometer', 'x_micrometer', 'len_y_micrometer',
        'len_x_micrometer', 'centroid_y_micrometer', 'centroid_x_micrometer'
        ]
    
    feats_categorical += [col for col in df_total.columns if col.startswith("Staining_")]
    feats_numerical = [col for col in df_total.columns if col not in feats_categorical]

    ad = anndata.AnnData(
        X=df_total[feats_numerical].values,
        obs=df_total.loc[:, feats_categorical],
        var=pd.DataFrame(index=feats_numerical),
    )

    ad.uns["stainings"] = _stringify_dict_keys(stainings)
    ad.uns["thresholds_by_round"] = _stringify_dict_keys(thresholds_by_round)
    ad.uns["experiment_setup"] = experiment_setup
    ad.uns["pixel_spacing"] = pixel_spacing
    ad.uns["pyramid_level"] = pyramid_level
    ad.uns["folders"] = folder
    ad.uns["source_dir"] = source
    ad.uns["table_dir"] = os.path.join(analysis_dir, "2_Tables")
    ad.uns["plot_dir"] = os.path.join(analysis_dir, "3_Plots")
    ad.uns["table_name"] = table_name
    ad.uns["label_name"] = label_name
    ad.uns["rounds_to_extract"] = rounds_to_extract
    ad.uns["segmentation_round"] = int(segmentation_round)

    save_adata(ad, f"{result_file_name}_{timestamp}")
    return ad


def _stringify_dict_keys(obj):
    if isinstance(obj, dict):
        return {str(k): _stringify_dict_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_stringify_dict_keys(x) for x in obj]
    return obj



def _prefix_vars_with_round(ad_t, round_id: int):
    """Prefix AnnData var_names with R{round}__ to match 1_FeatureExtraction naming."""
    ad_cp = ad_t.copy()
    ad_cp.var_names = [f"R{int(round_id)}__{v}" for v in ad_cp.var_names]
    return ad_cp



def merge_feature_tables_from_zarr(
    source: str,
    folder: list[str],
    experiment_setup: dict,
    stainings: dict | None,
    experiment_ID: str,
    result_file_name: str = "1_FeatureLoading",
    analysis_dir: str | None = None,
    feature_table_names: str | list[str] = "features",
    roi_table_name: str = "nuclei_ROI_table",
    label_name: str = "nuclei",
    multiplexing_round: int = 0,
    file_ending: str = ".zarr",
):
    """Load and merge precomputed AnnData feature tables stored in OME-Zarr under <round>/tables/<name>."""

    import os
    import anndata as ad
    import pandas as pd
    from anndata.io import read_zarr
    from ez_zarr import ome_zarr

    if analysis_dir is None:
        analysis_dir = source

    multiplexing_round = int(multiplexing_round)

    if isinstance(feature_table_names, str):
        feature_table_names = [feature_table_names]
    if len(feature_table_names) == 0:
        raise ValueError("feature_table_names is empty. Provide at least one table name.")

    def _load_plate_for_round(plate_path: str, round_id: int):
        return ome_zarr.import_plate(plate_path, image_name=str(int(round_id)))

    def _read_table_anndata(table_zarr_path: str):
        return read_zarr(table_zarr_path)

    tables_all = []

    for plate_folder in folder:
        plate_path = os.path.join(source, plate_folder)
        plate = _load_plate_for_round(plate_path, multiplexing_round)

        wells = plate.get_names()
        well_paths = plate.paths

        barcode_guess = None
        for bc in experiment_setup.keys():
            if bc in plate_folder:
                barcode_guess = bc
                break
        if barcode_guess is None:
            barcode_guess = list(experiment_setup.keys())[folder.index(plate_folder)]

        meta = experiment_setup[barcode_guess]

        for well, path_in_plate in zip(wells, well_paths):

            if well in meta.keys():
                ad_list = []
                for tname in feature_table_names:
                    table_path = os.path.join(plate_path, path_in_plate, "tables", tname)
                    if not os.path.exists(table_path):
                        raise FileNotFoundError(f"Cannot find table: {table_path}")
                    ad_list.append(_read_table_anndata(table_path))

                obs0 = ad_list[0].obs_names
                for j, ad_t in enumerate(ad_list[1:], start=1):
                    if ad_t.n_obs != ad_list[0].n_obs:
                        raise ValueError(
                            f"n_obs mismatch in well {well} between table {feature_table_names[0]} and {feature_table_names[j]}"
                        )
                    if not obs0.equals(ad_t.obs_names):
                        raise ValueError(
                            f"obs_names order mismatch in well {well} between table {feature_table_names[0]} and {feature_table_names[j]}"
                        )

                ad_list_pref = []
                for tname, ad_t in zip(feature_table_names, ad_list):
                    ad_cp = ad_t.copy()
                    ad_cp.var_names = [f"{tname}__{v}" for v in ad_cp.var_names]
                    ad_list_pref.append(ad_cp)

                ad_well = ad.concat(ad_list_pref, axis=1, merge="same", join="outer")
                ad_well = _prefix_vars_with_round(ad_well, multiplexing_round)

                ad_well.obs = ad_well.obs.copy()
                ad_well.obs["Barcode"] = barcode_guess
                ad_well.obs["Well"] = well
                ad_well.obs["PATH"] = path_in_plate
                ad_well.obs["Multiplexing_Round"] = multiplexing_round

                idx_in_well = pd.Series(range(ad_well.n_obs), index=ad_well.obs_names)
                ad_well.obs["Organoid_ID"] = barcode_guess + "-" + well + "-" + idx_in_well.astype(str).values

                exp_info = meta.get(well, [None, None, None, None])
                ad_well.obs["Medium"] = exp_info[0]
                ad_well.obs["ABs"] = exp_info[1]
                ad_well.obs["Cell_line"] = exp_info[2]
                ad_well.obs["Other"] = exp_info[3] if len(exp_info) > 3 else None
                ad_well.obs["Experiment_ID"] = experiment_ID

                tables_all.append(ad_well)

    if len(tables_all) == 0:
        raise RuntimeError("No feature tables loaded. Check table names and OME-Zarr structure.")

    ad_all = ad.concat(tables_all, axis=0, merge="same", join="outer", index_unique=None)
    ad_all.obs = ad_all.obs.copy()
    ad_all.obs_names = ad_all.obs["Organoid_ID"].astype(str)

    ad_all.uns["stainings"] = _stringify_dict_keys(stainings) if stainings is not None else {}
    ad_all.uns["experiment_setup"] = _stringify_dict_keys(experiment_setup)
    ad_all.uns["folders"] = folder
    ad_all.uns["source_dir"] = source
    ad_all.uns["table_name"] = roi_table_name
    ad_all.uns["label_name"] = label_name
    ad_all.uns["feature_table_names"] = feature_table_names
    ad_all.uns["multiplexing_round"] = multiplexing_round
    ad_all.uns["experiment_ID"] = experiment_ID
    ad_all.uns["table_dir"] = os.path.join(analysis_dir, "2_Tables")

    save_adata(ad_all, f"{result_file_name}_R{multiplexing_round}")

    return ad_all

def merge_feature_tables_from_zarr_rounds(
    source: str,
    folder: list[str],
    experiment_setup: dict,
    stainings: dict | None,
    experiment_ID: str,
    multiplexing_rounds: int | list[int] = 0,
    result_file_name: str = "1_FeatureLoading",
    analysis_dir: str | None = None,
    feature_table_names: str | list[str] = "features",
    roi_table_name: str = "nuclei_ROI_table",
    label_name: str = "nuclei",
    file_ending: str = ".zarr",
    validate_obs_names: bool = False,
    save_merged: bool = True,
    join: str = "outer",
):
    """Load one merged AnnData per round, then concatenate rounds along vars.

    Default behavior matches 1_FeatureExtraction: keep all objects across rounds and fill missing
    round-specific features with NaN.

    Parameters
    ----------
    join
        How to combine objects across rounds.
        - 'outer' (default): union of Organoid_IDs across rounds (missing features become NaN)
        - 'inner': keep only Organoid_IDs present in all rounds

    validate_obs_names
        When True, requires identical obs_names (only sensible with join='inner').
    """

    import anndata as ad

    if isinstance(multiplexing_rounds, int):
        rounds = [multiplexing_rounds]
    else:
        rounds = list(multiplexing_rounds)

    if len(rounds) == 0:
        raise ValueError("multiplexing_rounds is empty. Provide at least one round id.")

    ad_list = []
    for r in rounds:
        ad_r = merge_feature_tables_from_zarr(
            source=source,
            folder=folder,
            experiment_setup=experiment_setup,
            stainings=stainings,
            experiment_ID=experiment_ID,
            result_file_name=result_file_name,
            analysis_dir=analysis_dir,
            feature_table_names=feature_table_names,
            roi_table_name=roi_table_name,
            label_name=label_name,
            multiplexing_round=int(r),
            file_ending=file_ending,
        )
        ad_list.append(ad_r)

    if len(ad_list) == 1:
        ad_all = ad_list[0]
    else:
        if join not in {"inner", "outer"}:
            raise ValueError("join must be 'inner' or 'outer'")

        if join == "inner":
            common = ad_list[0].obs_names
            for ad_r in ad_list[1:]:
                common = common.intersection(ad_r.obs_names)
            if len(common) == 0:
                raise ValueError("No overlapping Organoid_IDs across requested rounds. Use join='outer' to keep union.")
            ad_list = [ad_r[common, :].copy() for ad_r in ad_list]

        if validate_obs_names:
            obs0 = ad_list[0].obs_names
            for j, ad_r in enumerate(ad_list[1:], start=1):
                if not obs0.equals(ad_r.obs_names):
                    raise ValueError(f"obs_names mismatch between rounds {rounds[0]} and {rounds[j]}")

        # join='outer' unions obs_names and fills missing values with NaN
        ad_all = ad.concat(ad_list, axis=1, merge="same", join="outer")

    ad_all.uns = ad_list[0].uns.copy()
    ad_all.uns["multiplexing_rounds"] = [int(r) for r in rounds]
    ad_all.uns["round_join"] = join

    if save_merged:
        rtag = "-".join([str(int(r)) for r in rounds])
        save_adata(ad_all, f"{result_file_name}_merged")

    return ad_all