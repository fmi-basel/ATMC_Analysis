from Fcts_FE import *  # Import all existing functions
from Fcts_Base_MultiRound import *
import copy
from tqdm.notebook import tqdm

"""
***
MULTI-ROUND FEATURE EXTRACTION FUNCTIONS
***
"""

def image_preprocessing_multiround(stainings, experiment_setup, images_all_rounds, OID, thresholds, sigma=3, rounds_to_process=None):
    """
    Enhanced image preprocessing that handles multiple multiplexing rounds.
    
    Parameters:
    - stainings: dict
        Dictionary mapping staining indices to stain names.
    - experiment_setup: dict
        Dictionary containing experimental setup information.
    - images_all_rounds: dict
        Dictionary containing images from all rounds. 
        Structure: {round_num: {channel: image, 'Mask': mask}, ...}
    - OID: str
        Unique organoid ID.
    - thresholds: dict
        Dictionary containing staining-specific thresholds.
    - sigma: int
        Sigma parameter for Gaussian blur.
    - rounds_to_process: list or None
        List of round numbers to process. If None, processes all available rounds.
        
    Returns:
    - images_all_rounds: dict
        Updated dictionary with thresholded masks for each round.
    """
    
    if rounds_to_process is None:
        rounds_to_process = list(images_all_rounds.keys())
    
    # Get staining information
    barcode = "-".join(OID.split("-")[:-2])
    well = OID.split("-")[-2]
    staining_key = experiment_setup[barcode][well][1]
    stainings_list = stainings[staining_key]
    
    for round_num in rounds_to_process:
        if round_num not in images_all_rounds:
            continue
            
        images_round = images_all_rounds[round_num]
        
        # Process each channel in this round
        for channel in range(len(stainings_list)):
            channel_key = f"C0{channel+1}"
            
            if channel_key not in images_round:
                continue
                
            # Load image as deep copy
            image = copy.deepcopy(images_round[channel_key])
            
            # Gaussian blur
            image = filters.gaussian(image, sigma=sigma, preserve_range=True)
            
            # Set values outside mask to 0 (mask is always from round 0)
            mask = images_all_rounds[0]["Mask"]  # Always use mask from round 0
            image[~mask.astype(bool)] = 0
            
            # Get threshold for staining
            staining = stainings_list[channel]
            threshold = thresholds[staining][2]
            
            # Set everything under stain-specific threshold to 0
            image[image <= threshold] = 0
            
            # Save as mask with round prefix
            images_round[f"C0{channel+1}_Mask"] = image.astype(bool)
    
    return images_all_rounds


def image_analysis_multiround(images_all_rounds, OID, row_data, quantiles_to_calc, sigma_skeleton, spacing, stainings, experiment_setup, radius_multiplier, rounds_to_process=None):
    """
    Enhanced image analysis that handles multiple multiplexing rounds.
    
    Shape features are calculated only once from round 0 (where the mask is).
    Intensity features are calculated for each round and each staining channel.
    
    Parameters:
    - images_all_rounds: dict
        Dictionary containing images from all rounds.
    - OID: str
        Object ID.
    - row_data: dict
        Dictionary to store calculated features.
    - quantiles_to_calc: list
        Quantiles to calculate for intensity features.
    - sigma_skeleton: int
        Sigma parameter for skeleton analysis.
    - spacing: float
        Pixel spacing.
    - stainings: dict
        Staining information.
    - experiment_setup: dict
        Experimental setup.
    - radius_multiplier: float
        Multiplier for skeleton features.
    - rounds_to_process: list or None
        List of round numbers to process.
        
    Returns:
    - row_data: dict
        Updated dictionary with all calculated features.
    """
    
    if rounds_to_process is None:
        rounds_to_process = list(images_all_rounds.keys())
    
    # Parse barcode and well from OID
    barcode = "-".join(OID.split("-")[:-2])
    well = OID.split("-")[-2]
    staining_key = experiment_setup[barcode][well][1]
    stain_names = stainings[staining_key]
    
    # --- SHAPE FEATURES (calculated only once from round 0) ---
    if 0 in images_all_rounds:
        images_round0 = images_all_rounds[0]
        mask = images_round0["Mask"]
        
        # Shape features from mask (round 0 only)
        row_data = shape_calc_mask(mask, row_data, OID, spacing)
        
        # Convex hull features
        row_data = convex_hull_features(mask, row_data, OID, spacing, min_area_fraction=0.005)
        
        # Border fraction for filtering
        row_data = get_border_fraction(mask, row_data, OID)
        
        # Skeleton features (round 0 only)
        row_data = skeleton_feats(images_round0, row_data, OID, spacing, sigma_skeleton, radius_multiplier)
    
    # --- INTENSITY FEATURES (calculated for each round) ---
    for round_num in rounds_to_process:
        if round_num not in images_all_rounds:
            continue
            
        images_round = images_all_rounds[round_num]
        mask = images_all_rounds[0]["Mask"]  # Always use mask from round 0
        
        # Process each channel in this round
        for ch_idx, stain in enumerate(stain_names):
            channel_key = f"C0{ch_idx+1}"
            
            if channel_key not in images_round:
                continue
                
            # Get images for this channel
            image = copy.deepcopy(images_round[channel_key])
            mask_channel = copy.deepcopy(images_round.get(f"{channel_key}_Mask", mask))
            
            # Create round-specific staining name for features
            if round_num == 0:
                stain_round = stain  # No suffix for round 0
            else:
                stain_round = f"{stain}_R{round_num}"
            
            # Calculate channel-specific features
            row_data = channel_mask_feat_calc(mask, mask_channel, stain_round, row_data, OID, spacing)
            row_data = intensity_feat_calc(image, mask, mask_channel, row_data, stain_round, OID, quantiles_to_calc)
            row_data = moments_channel_mask(mask, image, row_data, OID, stain_round, spacing)
    
    # --- CORRELATION FEATURES ---
    # Calculate Pearson correlations within each round
    for round_num in rounds_to_process:
        if round_num not in images_all_rounds:
            continue
            
        images_round = images_all_rounds[round_num]
        
        # Create a temporary stainings dict for this round
        if round_num == 0:
            temp_stainings = stainings.copy()
        else:
            temp_stainings = {}
            for key, stain_list in stainings.items():
                temp_stainings[key] = [f"{s}_R{round_num}" for s in stain_list]
        
        # Create temporary experiment setup for correlation calculation
        temp_exp_setup = copy.deepcopy(experiment_setup)
        
        row_data = intensity_pearsonR(temp_stainings, temp_exp_setup, images_round, barcode, well, OID, row_data)
    
    # Calculate cross-round correlations (between rounds)
    if len(rounds_to_process) > 1:
        row_data = calculate_cross_round_correlations(
            images_all_rounds, rounds_to_process, stain_names, row_data, OID
        )
    
    return row_data


def calculate_cross_round_correlations(images_all_rounds, rounds_to_process, stain_names, row_data, OID):
    """
    Calculate Pearson correlations between the same staining across different rounds.
    
    Parameters:
    - images_all_rounds: dict
        Images from all rounds.
    - rounds_to_process: list
        List of rounds to process.
    - stain_names: list
        List of staining names.
    - row_data: dict
        Dictionary to update with correlation features.
    - OID: str
        Object ID.
        
    Returns:
    - row_data: dict
        Updated with cross-round correlation features.
    """
    
    mask_bool = images_all_rounds[0]["Mask"].astype(bool)
    
    for ch_idx, stain in enumerate(stain_names):
        channel_key = f"C0{ch_idx+1}"
        
        # Get images from different rounds for the same staining
        round_images = {}
        for round_num in rounds_to_process:
            if round_num in images_all_rounds and channel_key in images_all_rounds[round_num]:
                round_images[round_num] = images_all_rounds[round_num][channel_key][mask_bool]
        
        # Calculate correlations between rounds
        round_nums = list(round_images.keys())
        for i, round1 in enumerate(round_nums):
            for round2 in round_nums[i+1:]:
                img1 = round_images[round1]
                img2 = round_images[round2]
                
                # Handle edge cases
                if img1.size == 0 or img2.size == 0 or np.std(img1) == 0 or np.std(img2) == 0:
                    r = np.nan
                else:
                    r = np.corrcoef(img1, img2)[0, 1]
                
                # Create feature name
                if round1 == 0:
                    stain1_name = stain
                else:
                    stain1_name = f"{stain}_R{round1}"
                    
                if round2 == 0:
                    stain2_name = stain
                else:
                    stain2_name = f"{stain}_R{round2}"
                
                key = f"{stain1_name}-{stain2_name}_PearsonR"
                row_data[key] = r
    
    return row_data


def extract_features_multiround(
    ome_zarrs_dict_multiround,
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
    max_rounds=None
):
    """
    Enhanced feature extraction that handles multiple multiplexing rounds.
    
    Shape features are extracted only from round 0 (where masks are located).
    Intensity features are extracted from all available rounds.
    
    Parameters:
    - ome_zarrs_dict_multiround: dict
        Dictionary mapping (barcode, round) tuples to OME-Zarr plate objects.
    - max_rounds: int or None
        Maximum number of rounds to process. If None, processes all available rounds.
    - Other parameters: Same as original extract_features function.
        
    Returns:
    - ad: anndata.AnnData
        AnnData object containing extracted features.
    """
    
    # Prepare timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
    df_total_list = []
    
    # Create output directories
    for subfolder in ["1_Scripts", "2_Tables", "3_Plots", "4_Results"]:
        os.makedirs(os.path.join(analysis_dir, subfolder), exist_ok=True)
    
    # Main processing loop
    for i, bc in enumerate(tqdm(barcodes, desc="Barcodes")):
        if (bc, 0) not in ome_zarrs_dict_multiround:
            print(f"Warning: Barcode {bc} round 0 not found. Skipping.")
            continue
        
        # Get available rounds for this barcode
        available_rounds = get_available_rounds_for_barcode(ome_zarrs_dict_multiround, bc)
        if max_rounds is not None:
            available_rounds = [r for r in available_rounds if r < max_rounds]
        
        print(f"Processing {bc} with rounds: {available_rounds}")
        
        # Get basic plate info from round 0
        plate_round0 = ome_zarrs_dict_multiround[(bc, 0)]
        well_names = plate_round0.get_names()
        wells_in_exp = experiment_setup[bc]
        
        rows = []
        
        for well_idx, well in enumerate(tqdm(well_names, desc=f"Wells ({bc})", leave=False)):
            if well not in wells_in_exp:
                continue
            
            exp_info = wells_in_exp[well]
            well_ov_round0 = plate_round0.images[well_idx]
            table = well_ov_round0.get_table(table_name)
            pixel_spacing = well_ov_round0.get_scale(pyramid_level=pyramid_level)[-1]
            
            if table.empty:
                print(f"No table {table_name} for well {well} in barcode {bc}. Skipping.")
                continue
            
            for row_idx, row in enumerate(table.itertuples()):
                OID = f"{bc}-{well}-{row_idx}"
                PATH = well_ov_round0.get_path()
                
                # Load images and masks from all rounds
                images_all_rounds = {}
                
                for round_num in available_rounds:
                    try:
                        if (bc, round_num) not in ome_zarrs_dict_multiround:
                            continue
                            
                        plate_round = ome_zarrs_dict_multiround[(bc, round_num)]
                        well_ov_round = plate_round.images[well_idx]
                        
                        # Get array data
                        img, mask_dict = well_ov_round.get_array_pair_by_coordinate(
                            label_name=label_name,
                            pyramid_level=pyramid_level,
                            upper_left_yx=(row.y_micrometer, row.x_micrometer),
                            lower_right_yx=(row.y_micrometer+row.len_y_micrometer, row.x_micrometer+row.len_x_micrometer)
                        )
                        
                        # Store images for this round
                        images_round = {}
                        for channel in range(img.shape[0]):
                            images_round[f"C0{channel+1}"] = img[channel, 0]
                        
                        # For round 0, also store the mask
                        if round_num == 0:
                            mask_arr = mask_dict[label_name][0]
                            mask_for_oid = (mask_arr == int(row_idx)+1).astype(mask_arr.dtype)
                            images_round["Mask"] = mask_for_oid
                            
                            # Check for multiple labels
                            n_labels = np.max(measure.label(mask_for_oid.astype(bool)))
                            if n_labels != 1:
                                print(f"Skipping {OID} because it contains multiple labels.")
                                break
                        
                        images_all_rounds[round_num] = images_round
                        
                    except Exception as e:
                        print(f"Error loading round {round_num} for {OID}: {e}")
                        continue
                
                # Skip if we couldn't load round 0 (needed for mask)
                if 0 not in images_all_rounds:
                    print(f"Could not load round 0 for {OID}. Skipping.")
                    continue
                
                # Build observation row
                row_dict = build_row(OID, bc, well, row, exp_info, PATH, stainings, experiment_ID)
                
                # Add round information
                row_dict["available_rounds"] = ",".join(map(str, list(images_all_rounds.keys())))
                
                try:
                    # Pre-process images for all rounds
                    images_all_rounds = image_preprocessing_multiround(
                        stainings, experiment_setup, images_all_rounds, OID, thresholds, sigma=3
                    )
                    
                    # Calculate features
                    row_dict = image_analysis_multiround(
                        images_all_rounds=images_all_rounds,
                        OID=OID,
                        row_data=row_dict,
                        quantiles_to_calc=quantiles_to_calc,
                        sigma_skeleton=sigma_skeleton,
                        spacing=pixel_spacing,
                        stainings=stainings,
                        experiment_setup=experiment_setup,
                        radius_multiplier=radius_multiplier,
                        rounds_to_process=list(images_all_rounds.keys())
                    )
                    
                except Exception as e:
                    print(f"Error in analysis for {OID}: {e}")
                    continue
                
                rows.append(row_dict)
        
        # Build DataFrame for this plate
        if not rows:
            print(f"No valid rows for barcode {bc}.")
            continue
            
        df = pd.DataFrame(rows)
        
        # Remove moments-0-0 columns
        df = df.drop(columns=[col for col in df.columns if "moments-0-0" in col or "moments_weighted-0-0" in col], errors="ignore")
        
        # Save per-plate DataFrame
        df = df.sort_values(by="Organoid_ID", key=natsort_keygen())
        save_path = os.path.join(source, folder[i], f"{result_file_name}_multiround_{barcodes[i]}_{timestamp}.csv")
        df.to_csv(save_path)
        print(f"Saved {bc} results as {save_path}.")
        
        df_total_list.append(df)
    
    # Merge all plates
    if not df_total_list:
        raise ValueError("No features extracted for any barcode.")
        
    df_total = pd.concat(df_total_list, axis=0)
    
    # Save merged DataFrame
    total_save_path = os.path.join(analysis_dir, "2_Tables", f"{result_file_name}_multiround_{timestamp}.csv")
    df_total = df_total.set_index("Organoid_ID", drop=False)
    df_total.to_csv(total_save_path)
    print(f"Saved merged csv as {total_save_path}.")
    
    # Create AnnData object
    feats_categorical = [
        "Organoid_ID", "Barcode", "Well", "Object", "Medium",
        "ABs", "Cell_line", "Other", "PATH", "Experiment_ID", "available_rounds"
    ]
    feats_categorical += [col for col in df_total.columns if col.startswith("Staining_")]
    feats_numerical = [col for col in df_total.columns if col not in feats_categorical]
    
    ad = anndata.AnnData(
        X=df_total[feats_numerical].values,
        obs=df_total.loc[:, feats_categorical],
        var=pd.DataFrame(index=feats_numerical)
    )
    
    # Add metadata
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
    ad.uns["multiround"] = True
    ad.uns["max_rounds"] = max_rounds
    
    save_adata(
        ad,
        f"{result_file_name}_multiround_{timestamp}"
    )
    
    return ad
