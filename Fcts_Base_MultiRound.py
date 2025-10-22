from natsort import natsorted
from ez_zarr import ome_zarr
import pandas as pd
import datetime
import anndata
import glob
import os

from Fcts_Base import *  # Import all existing functions

"""
***
MULTI-ROUND EXTENSION FUNCTIONS
***
"""

def extract_ome_zarr_tables_multiround(experiment_setup, source, folder, table_name, max_rounds=None):
    """
    Enhanced version of extract_ome_zarr_tables that loads multiple multiplexing rounds.
    
    The mask for features is always only found in image_name = "0", thus shape features 
    stay constant between rounds and only staining-associated features need to be 
    extracted from subsequent rounds.
    
    Parameters:
    - experiment_setup: dict
        Nested dictionary with barcode as keys and well-specific metadata as values.
    - source: str
        Root directory path containing the OME-Zarr folders.
    - folder: list of str
        List of folder names corresponding to each barcode.
    - table_name: str
        Name of the table to extract from each plate.
    - max_rounds: int or None
        Maximum number of rounds to load. If None, will auto-detect available rounds.
        
    Returns:
    - ome_zarr_dict_multiround: dict
        Dictionary mapping (barcode, round) tuples to loaded OME-Zarr plate objects.
        Structure: {(barcode, round): plate_object, ...}
    - ome_zarr_df: pd.DataFrame
        Combined DataFrame containing all extracted tables from round 0 only, 
        annotated with metadata.
    """
    
    if len(folder) != len(experiment_setup):
        raise ValueError("Length of folder list must match number of barcodes in experiment_setup.")
    
    ome_zarr_dict_multiround = {}
    all_plate_dfs = []
    
    for i, barcode in enumerate(experiment_setup):
        # First, load round 0 to get the basic table structure
        try:
            plate_round0 = ome_zarr.import_plate(os.path.join(source, folder[i]), image_name="0")
            ome_zarr_dict_multiround[(barcode, 0)] = plate_round0
            
            # Get table from round 0 (this contains the segmentation masks)
            df_lst = plate_round0.get_table(table_name)
            wells = plate_round0.get_names()
            paths = plate_round0.paths
            
            # Annotate DataFrames with well and path info
            for df, well, path in zip(df_lst, wells, paths):
                df['well'] = well
                df['path'] = path
                df['round'] = 0  # Add round information
            
            plate_df = pd.concat(df_lst, ignore_index=False)
            plate_df["Barcode"] = barcode
            plate_df["UID"] = barcode + "-" + plate_df["well"].astype(str) + "-" + plate_df.index.astype(str)
            
            # Extract metadata
            meta = experiment_setup[barcode]
            plate_df["Medium"] = plate_df["well"].map(lambda w: meta.get(w, [None, None, None])[0])
            plate_df["AB"] = plate_df["well"].map(lambda w: meta.get(w, [None, None, None])[1])
            plate_df["Day"] = plate_df["well"].map(lambda w: meta.get(w, [None, None, None])[-1])
            
            all_plate_dfs.append(plate_df)
            
        except Exception as e:
            print(f"Warning: Could not load round 0 for barcode {barcode}: {e}")
            continue
        
        # Auto-detect available rounds if max_rounds not specified
        if max_rounds is None:
            detected_rounds = detect_available_rounds(source, folder[i])
        else:
            detected_rounds = list(range(max_rounds))
        
        # Load additional rounds (1, 2, 3, etc.)
        for round_num in detected_rounds:
            if round_num == 0:
                continue  # Already loaded
            
            try:
                plate_round = ome_zarr.import_plate(
                    os.path.join(source, folder[i]), 
                    image_name=str(round_num)
                )
                ome_zarr_dict_multiround[(barcode, round_num)] = plate_round
                print(f"Successfully loaded {barcode} round {round_num}")
                
            except Exception as e:
                print(f"Warning: Could not load round {round_num} for barcode {barcode}: {e}")
                break  # Stop trying higher rounds if this one fails
    
    # Create combined DataFrame (only from round 0, since that's where the masks are)
    ome_zarr_df = pd.concat(all_plate_dfs, ignore_index=True)
    
    return ome_zarr_dict_multiround, ome_zarr_df


def detect_available_rounds(source, folder_name, max_check=10):
    """
    Auto-detect available multiplexing rounds in an OME-Zarr dataset.
    
    Parameters:
    - source: str
        Root directory path.
    - folder_name: str
        Name of the folder containing the OME-Zarr data.
    - max_check: int
        Maximum number of rounds to check for.
        
    Returns:
    - available_rounds: list
        List of available round numbers.
    """
    available_rounds = []
    
    for round_num in range(max_check):
        try:
            # Try to load this round
            plate = ome_zarr.import_plate(
                os.path.join(source, folder_name), 
                image_name=str(round_num)
            )
            available_rounds.append(round_num)
        except:
            # If this round fails, stop checking higher rounds
            break
    
    return available_rounds


def load_img_mask_by_UID_multiround(UID, ome_zarr_dict_multiround, table_name, label_name, pyramid_level, channel, round_num=0):
    """
    Enhanced version that can load images from different multiplexing rounds.
    
    The mask is always loaded from round 0, but the image can be from any round.
    
    Parameters:
    - UID: str
        Unique identifier formatted as 'barcode-well-index'.
    - ome_zarr_dict_multiround: dict
        Dictionary mapping (barcode, round) tuples to Plate objects.
    - table_name: str
        Name of the table to extract entries from.
    - label_name: str
        Name of the label within the mask.
    - pyramid_level: int
        Pyramid level to extract from.
    - channel: int
        Channel index to extract.
    - round_num: int
        Round number to load image from (default: 0).
        
    Returns:
    - img: numpy.ndarray
        Image from specified round and channel.
    - mask: numpy.ndarray
        Mask from round 0.
    """
    
    # Parse UID
    try:
        index_str = UID.split('-')[-1]
        bc = "-".join(UID.split('-')[:-2])
        well = UID.split('-')[-2]
        index = int(index_str)
    except Exception as e:
        raise ValueError(f"UID format error: {UID} should be 'barcode-well-index'") from e
    
    # Check if barcode and rounds exist
    if (bc, round_num) not in ome_zarr_dict_multiround:
        raise KeyError(f"Barcode {bc} round {round_num} not found in ome_zarr_dict_multiround.")
    if (bc, 0) not in ome_zarr_dict_multiround:
        raise KeyError(f"Barcode {bc} round 0 not found in ome_zarr_dict_multiround.")
    
    # Get plates for image (specified round) and mask (always round 0)
    plate_image = ome_zarr_dict_multiround[(bc, round_num)]
    plate_mask = ome_zarr_dict_multiround[(bc, 0)]
    
    # Get well information from round 0 (where the table is)
    well_names = plate_mask.get_names()
    if well not in well_names:
        raise KeyError(f"Well {well} not found in barcode {bc}.")
    
    well_idx = well_names.index(well)
    
    # Get table from round 0
    well_ov_mask = plate_mask.images[well_idx]
    table = well_ov_mask.get_table(table_name)
    
    if table.empty:
        raise ValueError(f"Table {table_name} is empty for well {well} in barcode {bc}.")
    
    if index < 0 or index >= len(table):
        raise IndexError(f"Index {index} out of bounds for table length {len(table)}")
    
    # Get coordinates from round 0 table
    entry = table.iloc[index]
    ul_y, ul_x = entry["y_micrometer"], entry["x_micrometer"]
    lr_y = ul_y + entry["len_y_micrometer"]
    lr_x = ul_x + entry["len_x_micrometer"]
    
    # Get image from specified round
    well_ov_image = plate_image.images[well_idx]
    img, _ = well_ov_image.get_array_pair_by_coordinate(
        label_name=label_name,
        pyramid_level=pyramid_level,
        upper_left_yx=(ul_y, ul_x),
        lower_right_yx=(lr_y, lr_x)
    )
    
    # Get mask from round 0
    _, mask_dict = well_ov_mask.get_array_pair_by_coordinate(
        label_name=label_name,
        pyramid_level=pyramid_level,
        upper_left_yx=(ul_y, ul_x),
        lower_right_yx=(lr_y, lr_x)
    )
    
    # Check if label exists in mask
    if label_name not in mask_dict:
        raise KeyError(f"Label {label_name} not found in mask for well {well} in barcode {bc}.")
    
    # Check if channel is valid for the image
    if channel < 0 or channel >= img.shape[0]:
        raise IndexError(f"Channel {channel} out of bounds for image with {img.shape[0]} channels.")
    
    # Extract image and mask
    img_channel = img[channel, 0]  # Select specific channel
    mask = mask_dict[label_name][0]
    mask[mask != index+1] = 0  # Remove other labels
    
    return img_channel, mask


def get_available_rounds_for_barcode(ome_zarr_dict_multiround, barcode):
    """
    Get all available rounds for a specific barcode.
    
    Parameters:
    - ome_zarr_dict_multiround: dict
        Dictionary with (barcode, round) keys.
    - barcode: str
        Barcode to check.
        
    Returns:
    - rounds: list
        List of available round numbers for this barcode.
    """
    rounds = []
    for (bc, round_num) in ome_zarr_dict_multiround.keys():
        if bc == barcode:
            rounds.append(round_num)
    return sorted(rounds)
