from natsort import natsorted
from ez_zarr import ome_zarr
import pandas as pd
import datetime
import anndata
import glob
import os
"""
***
BASE FUNCTIONS
***
"""

def remove_uns(ad, keys_to_remove):

    for key in keys_to_remove:
        if key in ad.uns:
            ad.uns.pop(key)

    return ad

def add_zarr_uns(ad):

    ome_zarr_dict, ome_zarr_df =  extract_ome_zarr_tables(ad.uns["experiment_setup"],ad.uns["source_dir"], ad.uns["folders"], ad.uns["table_name"])

    # Attach in-memory ome_zarr_dict to AnnData .uns (DO NOT save this inside AnnData file!)
    ad.uns["ome_zarr_dict"] = ome_zarr_dict
    ad.uns["ome_zarr_df"] = ome_zarr_df
    return ad

def load_adata(adata_path, load_zarrs = False):
    """
    Load an AnnData object from disk, then reconstruct only the ome_zarr_dict from source folders,
    and attach the ome_zarr_dict to adata.uns for runtime use.

    Parameters:
    - adata_path (str): Path to the saved AnnData (.h5ad) file.

    Returns:
    - adata (anndata.AnnData): Loaded AnnData with ome_zarr_dict attached in adata.uns.
    """

    # Load AnnData from disk without ome_zarr_dict stored
    ad = anndata.read_h5ad(adata_path)

    if load_zarrs:
        # Reconstruct ome_zarr_dict from source folders and attach to ad.uns
        ad = add_zarr_uns(ad)
    
    return ad

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

def save_adata(adata, filename, keys_to_remove = ["ome_zarr_dict", "ome_zarr_df"]):
    """
    Save an AnnData object to an h5ad file with a unique filename,
    removing `ome_zarr_dict` (and optionally other keys) from .uns before saving.

    Parameters:
    - path (str): Directory path to save the file.
    - filename (str): Base filename (without extension).
    - adata (anndata.AnnData): AnnData object to save.
    - keys_to_remove (list or None): List of keys to remove from adata.uns before saving.
                                    Default ["ome_zarr_dict", "ome_zarr_df"], can be None to skip removal.
    """

    path = adata.uns["table_dir"]
    if not os.path.exists(path):
        os.makedirs(path)

    savepath = os.path.join(path, filename + ".h5ad")
    i = 0

    # Find unique save path to avoid overwriting
    while os.path.exists(savepath):
        i += 1
        savepath = os.path.join(path, f"{filename}{i}.h5ad")

    # Backup keys to remove to restore later
    backup = {}
    for key in keys_to_remove:
        if key in adata.uns:
            backup[key] = adata.uns.pop(key)

    try:
        adata.write(savepath, compression="gzip")
    finally:
        # Restore removed keys regardless of success or error to avoid mutation side effects
        for key, val in backup.items():
            adata.uns[key] = val

    print(f"Saved as:\n{savepath}")

def save_fig(fig, path, filename, dpi=300):
    """
    Save a Matplotlib figure to a PDF and PNG file with a unique filename.

    Parameters:
    - fig (matplotlib.figure.Figure): Matplotlib figure to be saved.
    - path (str): Directory path for saving the files.
    - filename (str): Base filename (without extension).
    - dpi (int): Dots per inch for the figure resolution (default is 300).
    """
    if not os.path.exists(path):
        os.makedirs(path)

    savepath_pdf = os.path.join(path, filename + ".pdf")
    savepath_png = savepath_pdf.replace(".pdf", ".png")
    i = 1

    # Find unique save path
    while os.path.exists(savepath_pdf) or os.path.exists(savepath_png):
        i += 1
        savepath_pdf = os.path.join(path, filename +"_"+str(i) + ".pdf")
        savepath_png = savepath_pdf.replace(".pdf", ".png")

    fig.savefig(savepath_pdf, transparent=True, dpi=dpi, bbox_inches="tight")
    fig.savefig(savepath_png, transparent=True, dpi=dpi, bbox_inches="tight")
    print(f"Saved as:\n{savepath_pdf}\nand\n{savepath_png}")

def save_after_filtering(df, df_raw, ad_raw, save_dir = None):
    """
    Save the filtered DataFrame, updated AnnData object, and a list of deleted organoid IDs.

    Parameters:
    - save_dir (str): Directory path to save the results.
    - df (pd.DataFrame): Filtered DataFrame containing organoid information.
    - ad_raw (anndata.AnnData): Original AnnData object.
    """

    save_dir = ad_raw.uns["table_dir"]

    # Save DF
    save_df(save_dir, "2_FeaturesFiltered"+"_{date:%Y-%m-%d_%Hh%Mmin%Ss}".format(date=datetime.datetime.now()), df)

    keys_to_remove = ["ome_zarr_dict", "ome_zarr_df"]
    for key in keys_to_remove:
        if key in ad_raw.uns:
            ad_raw.uns.pop(key)

    # Filter anndata
    ad = ad_raw[df.index,:].copy()

    # Save list of deleted organoid IDs
    ad.uns["deleted_IDs"] = list(set(df_raw.index.to_list()) - set(df.index.to_list()))

    # Save AnnData object
    save_adata(ad, "2_FeaturesFiltered")

def load_img_mask_by_UID(UID, ome_zarr_dict, table_name, label_name, pyramid_level, channel):
    """
    Retrieve an image and corresponding mask for a specific entry in an OME-Zarr plate dataset,
    based on a unique identifier (UID) and additional parameters.

    Parameters:
        UID (str): Unique identifier of the data instance, formatted as 'barcode-well-index'.
        ome_zarr_dict (dict): Dictionary mapping barcodes to Plate objects, assumed to adhere to an OME-Zarr interface.
        table_name (str): Name of the table to extract entries from for the well (e.g., 'organoids').
        label_name (str): Name of the label within the mask to extract (e.g., 'organoid').
        pyramid_level (int): Pyramid level to extract the image/mask from.
        channel (int): Index of the channel to extract from the image.
    """

    # UID format: 'barcode-well-index'
    try:
        bc, well, index_str = UID.rsplit("-", 2)   # split from right into 3 parts
        index = int(index_str)

    except Exception as e:
        raise ValueError(f"UID format error: {UID} should be 'barcode-well-index'") from e

    # Check if barcode exists
    if bc not in ome_zarr_dict:
        raise KeyError(f"Barcode {bc} not found in ome_zarr_dict.")
    plate = ome_zarr_dict[bc]

    # Check if well exists
    well_names = plate.get_names()
    if well not in well_names:
        raise KeyError(f"Well {well} not found in barcode {bc}.")

    # Retrieve well object
    well_idx = well_names.index(well)
    well_ov = plate.images[well_idx]

    # Retrieve table and check existence
    table = well_ov.get_table(table_name)
    if table.empty:
        raise ValueError(f"Table {table_name} is empty for well {well} in barcode {bc}.")

    # Check if index is valid
    if index < 0 or index >= len(table):
        raise IndexError(f"Index {index} out of bounds for table length {len(table)}")
    


    # Get row and coordinates
    entry = table.iloc[index]
    ul_y, ul_x = entry["y_micrometer"], entry["x_micrometer"]
    lr_y = ul_y + entry["len_y_micrometer"]
    lr_x = ul_x + entry["len_x_micrometer"]

    # Retrieve image and mask
    img, mask = well_ov.get_array_pair_by_coordinate(
        label_name=label_name,
        pyramid_level=pyramid_level,
        upper_left_yx=(ul_y, ul_x),
        lower_right_yx=(lr_y, lr_x)
    )

    # Check if label exists in mask
    if label_name not in mask:
        raise KeyError(f"Label {label_name} not found in mask for well {well} in barcode {bc}.")
    
    # Check if channel is valid
    if channel < 0 or channel >= img.shape[0]:
        raise IndexError(f"Channel {channel} out of bounds for image with {img.shape[0]} channels.")

    img = img[channel, 0]  # Select specific channel

    mask = mask[label_name][0]
    mask[mask != index+1] = 0  # Remove other labels than current organoid

    return img, mask

def find_zarr_dirs(root_dir, max_depth=None, file_ending=".zarr"):
    """
    Finds all OME-Zarr directories within a given root directory,
    stopping at the first depth where any are found.
    
    Parameters:
    - root_dir (str): The root directory to start the search from.
    - max_depth (int, optional): The maximum depth to search for OME-Zarr directories.
    """
    from collections import deque

    zarr_dirs = []
    min_depth_found = None

    # Use deque for efficient BFS traversal
    queue = deque([(root_dir, 0)])

    # Store all zarr dirs found at the minimal depth
    while queue:
        current_dir, current_depth = queue.popleft()
        # Depth-limiting, if requested
        if max_depth is not None and current_depth > max_depth:
            continue
        try:
            dirnames = [
                d for d in os.listdir(current_dir)
                if os.path.isdir(os.path.join(current_dir, d))
            ]
        except PermissionError:
            continue  # skip directories you can't access

        # Find .zarr folders at this level
        this_level_zarrs = [
            os.path.join(current_dir, d)
            for d in dirnames if d.endswith(file_ending)
        ]

        if this_level_zarrs:
            if min_depth_found is None or current_depth < min_depth_found:
                min_depth_found = current_depth
                zarr_dirs = this_level_zarrs
            elif current_depth == min_depth_found:
                zarr_dirs.extend(this_level_zarrs)
            # Do not enqueue subdirs once we've found zarrs at this (shallowest) depth
            continue

        # Only queue deeper dirs _if_ we have not found any .zarr yet
        for d in dirnames:
            queue.append((os.path.join(current_dir, d), current_depth + 1))

    if not zarr_dirs:
        print("No OME-Zarr directories found. Please check the root directory and depth.")
        return []
    print(f"Found {len(zarr_dirs)} OME-Zarr directories in {root_dir} at depth {min_depth_found}.")
    print("Directories found. Make sure that variable \033[3mbarcodes\033[0m is in the same order:")
    for zarr_dir in natsorted(zarr_dirs):
        print(f"- {zarr_dir}")

    return natsorted(zarr_dirs)

def find_staining_in_ABs(stainings, staining_to_find):
    """Find AB mixes that contain a staining, supporting both legacy and round-aware stainings."""

    found_in = []
    for mix, v in stainings.items():
        if isinstance(v, dict):
            # round-aware
            for _, stains in v.items():
                if staining_to_find in stains:
                    found_in.append(mix)
                    break
        else:
            # legacy list
            if staining_to_find in v:
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
        plate_df["Other"] = plate_df["well"].map(lambda w: meta.get(w, [None, None, None])[-1])
        all_plate_dfs.append(plate_df)

    ome_zarr_df = pd.concat(all_plate_dfs, ignore_index=True)

    return ome_zarr_dict, ome_zarr_df

def get_pipetting_info(source, layout_sheet="StainingLayout"):
    
    """
    Extracts and summarizes immunostaining mix preparation instructions from an experiment's
    layout Excel file. For each antibody mix, it reports which reagents at what volumes
    should be added into the specified staining solution volume.

    Parameters
    ----------
    source : str
        Path to the folder containing the 'Layout*.xlsx' file.
    layout_sheet : str, optional
        Name of the worksheet tab containing the immunostaining table (default: 'StainingLayout').
    """
      
    file = glob.glob(source+'/Layout*.xlsx')[0]
    
    df = pd.read_excel(file, sheet_name=layout_sheet, header=None)

    # Find real header row
    header_row = None
    for i in range(len(df)):
        if df.iloc[i, :].astype(str).str.strip().eq("UniqueID").any():
            header_row = i
            break
    if header_row is None:
        raise RuntimeError("Cannot find immunostaining table in StainingLayout")

    # Parse header (remove trailing blanks)
    header = [str(x).strip() for x in df.iloc[header_row, :]]
    last_col = max([i for i,v in enumerate(header) if v and v != 'nan' and v != 'None'] + [0])
    header = header[:last_col+1]

    # Collect rows below header
    data = []
    for i in range(header_row+1, min(header_row + 40, len(df))):
        row_raw = df.iloc[i, :len(header)].tolist()
        if all([(str(v).strip() == '' or pd.isna(v)) for v in row_raw]):
            continue
        data.append(row_raw)
    tab = pd.DataFrame(data, columns=header)

    # Clean up: drop blank/Brightfield
    tab = tab.loc[
        tab.Target.notna()
        & tab['Antibody Mix'].notna()
        & (~tab.Target.astype(str).str.lower().str.contains('brightfield'))
        & (tab.Target.astype(str).str.strip() != "")
    ]

    # Find all unique mixes
    mix_set = set()
    for mixentry in tab['Antibody Mix']:
        for m in str(mixentry).split(','):
            m = m.strip()
            if m: mix_set.add(m)
    mixes = sorted(list(mix_set))

    out_lines = []
    for mix in mixes:
        sub = tab[tab['Antibody Mix'].astype(str).str.contains(rf'\b{mix}\b', regex=True)]
        # Get total staining volume for this mix
        total_vol = None
        for tv in sub['Total Staining Vol. [µl]']:
            try:
                total_vol = float(tv)
                if total_vol: break
            except: pass

        ingrs = []
        for _, row in sub.iterrows():
            # -------- Add conjugation to target if exists!
            target = str(row['Target']).strip()
            conj = str(row['Conjugation']).strip() if 'Conjugation' in row and not pd.isna(row['Conjugation']) else ''
            if conj and conj != "-":
                target = f"{target}-{conj}"
            location = str(row.get('Location', '')).strip()
            try:
                dilution = float(row['Dilution'])
                if not (dilution > 0): continue
            except Exception:
                continue
            ul = total_vol / dilution if total_vol else None
            if ul:
                ingrs.append(f"{ul:.2f} ul {target} ({location})")
        if ingrs and total_vol:
            s = f"For mix {mix}, add {', '.join(ingrs)} into {total_vol:.0f} ul staining solution."
            out_lines.append(s)

    for l in out_lines:
        print(l)
    return None

def get_stainings(source, sheet="StainingLayout"):
    """
    Extract antibody staining information from an Excel file describing immunostaining layout.

    Supports both:
    - legacy layouts without 'Round' column (all channels are treated as round 0)
    - multicycle layouts with a 'Round' column (int or numeric string)

    Returns
    -------
    out : dict
        Mapping of antibody mix -> either:
        - dict(round -> list of stains by channel) if 'Round' column exists
        - list of stains by channel (legacy)

    Notes
    -----
    The returned structure is round-aware if 'Round' exists. Downstream code should handle both.
    """

    filename = glob.glob(source+'/Layout*.xlsx')[0]

    df = pd.read_excel(filename, sheet_name=sheet, header=None)
    # Find header row
    header_row = None
    for i in range(len(df)):
        if df.iloc[i, :].astype(str).str.strip().eq("UniqueID").any():
            header_row = i
            break
    if header_row is None:
        raise RuntimeError("Immunostaining table not found!")
    header = [str(x).strip() for x in df.iloc[header_row, :]]
    last_col = max(i for i,v in enumerate(header) if v and v.lower() != "nan" and v.lower() != "none")
    header = header[:last_col+1]
    # Read relevant data rows
    rows = []
    for i in range(header_row+1, len(df)):
        row = df.iloc[i, :len(header)].tolist()
        if all([pd.isna(x) or str(x).strip() == '' for x in row]):
            continue
        rows.append(row)
    tab = pd.DataFrame(rows, columns=header)
    # Filter rows
    tab = tab[tab["Target"].notna() & tab["Antibody Mix"].notna()]
    tab = tab[~tab["Target"].astype(str).str.lower().str.contains("brightfield")]
    tab = tab[tab["Imaging Channel"].notna()]
    tab = tab[~tab["Type"].astype(str).str.lower().str.contains("secondary")]
    tab["Imaging Channel"] = tab["Imaging Channel"].astype(str).str.strip()
    tab = tab[tab["Imaging Channel"] != ""]

    # Round handling (optional)
    has_round = "Round" in tab.columns
    if has_round:
        tab["Round"] = pd.to_numeric(tab["Round"], errors="coerce").fillna(0).astype(int)
    else:
        tab["Round"] = 0

    # Explode mixes: a row can list more than one (e.g 'AB1, AB2')
    mix_rows = []
    for _, row in tab.iterrows():
        mixentries = [m.strip() for m in str(row["Antibody Mix"]).split(",") if m.strip()]
        for mix in mixentries:
            data = row.to_dict()
            data["Antibody Mix"] = mix
            mix_rows.append(data)
    dfmix = pd.DataFrame(mix_rows)

    # Build output: mix -> round -> stains list
    out: dict[str, dict[int, list[str]]] = {}
    for (mix, rnd), g in dfmix.groupby(["Antibody Mix", "Round"]):
        g = g.copy()
        chlist = g["Imaging Channel"].astype(float)
        if chlist.duplicated().any():
            raise ValueError(f"Duplicate channel for mix {mix}, round {rnd}: {list(g['Imaging Channel'])}")
        stains = g.sort_values("Imaging Channel")["Target"].tolist()
        out.setdefault(mix, {})[int(rnd)] = stains

    # Print summary
    print(f"Found {len(out)} staining mixes in experiment setup:")
    for mix, rounds in out.items():
        rounds_str = ", ".join([f"R{r}({v})" for r, v in sorted(rounds.items())])
        print(f"  {mix}: {rounds_str}")

    get_pipetting_info(source)

    # Backward compatibility: if no Round column, collapse to legacy list
    if not has_round:
        legacy = {mix: rounds.get(0, []) for mix, rounds in out.items()}
        return legacy

    return out
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