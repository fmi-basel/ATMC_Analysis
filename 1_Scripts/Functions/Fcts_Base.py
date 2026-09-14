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

def infer_barcode_folder_map(ad):
    """
    Derive a {barcode: plate_root_folder} mapping from ad.obs["Barcode"]/ad.obs["PATH"].

    Current extractions store ad.uns["folders"] as a {barcode: folder} dict (see
    resolve_barcode_folders), so this is only needed for AnnData files written before
    that change, where "folders" was a plain list positionally matched against
    ad.uns["experiment_setup"]. A save/reload round-trip through .h5ad does not
    guarantee that a dict in .uns keeps its key order, which silently desynced the two.
    ad.obs["PATH"] is recorded per-organoid at extraction time and still reflects the
    true barcode -> folder pairing, so recover it from there rather than trusting order.
    """
    if "Barcode" not in ad.obs.columns or "PATH" not in ad.obs.columns:
        return None
    mapping = {}
    for bc, path in zip(ad.obs["Barcode"], ad.obs["PATH"]):
        if bc in mapping:
            continue
        # path looks like <plate_root>/<row>/<col>/<image_name>
        mapping[bc] = "/".join(str(path).rstrip("/").split("/")[:-3])
    return mapping

def add_zarr_uns(ad):

    folder_map = infer_barcode_folder_map(ad)
    folders = folder_map if folder_map else ad.uns["folders"]
    ome_zarr_dict, ome_zarr_df =  extract_ome_zarr_tables(ad.uns["experiment_setup"],ad.uns["source_dir"], folders, ad.uns["table_name"])

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
        savepath = os.path.join(path, f"{filename}-{i}.h5ad")

    # Backup keys to remove to restore later
    backup = {}
    for key in keys_to_remove:
        if key in adata.uns:
            backup[key] = adata.uns.pop(key)

    try:
        anndata.settings.allow_write_nullable_strings = True
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

def save_after_filtering(df, df_raw, ad_raw):
    """
    Save the filtered DataFrame, updated AnnData object, and a list of deleted organoid IDs.

    Parameters:
    - df (pd.DataFrame): Filtered DataFrame containing organoid information.
    - df_raw (pd.DataFrame): Unfiltered DataFrame, used to record which IDs were removed.
    - ad_raw (anndata.AnnData): Original AnnData object. Left unmodified: the runtime-only
      keys are stripped from the saved copy, so the filtering cells stay re-runnable.
    """

    save_dir = ad_raw.uns["table_dir"]

    # Save DF
    save_df(save_dir, "2_FeaturesFiltered"+"_{date:%Y-%m-%d_%Hh%Mmin%Ss}".format(date=datetime.datetime.now()), df)

    # Filter anndata, then strip runtime-only keys from the copy (not from ad_raw)
    ad = ad_raw[df.index,:].copy()
    for key in ["ome_zarr_dict", "ome_zarr_df"]:
        ad.uns.pop(key, None)

    # Save list of deleted organoid IDs
    ad.uns["deleted_IDs"] = list(set(df_raw.index.to_list()) - set(df.index.to_list()))

    # Save AnnData object
    save_adata(ad, "2_FeaturesFiltered")

def find_zarr_dirs(root_dir, max_depth=None, file_ending=".zarr", analysis_dir = None):

    """
    Finds all OME-Zarr directories within a given root directory,
    stopping at the first depth where any are found.
    
    Parameters:
    - root_dir (str): The root directory to start the search from.
    - max_depth (int, optional): The maximum depth to search for OME-Zarr directories.
    """
    from collections import deque

    if analysis_dir is None:
        analysis_dir = root_dir

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
        raise FileNotFoundError(
            f"No directories ending in '{file_ending}' found under {root_dir}"
            + (f" within depth {max_depth}." if max_depth is not None else ".")
            + " Check the experiment folder and the file_ending parameter."
        )
    print(f"Found {len(zarr_dirs)} OME-Zarr directories in {root_dir} at depth {min_depth_found}.")

    return natsorted(zarr_dirs), analysis_dir


def _layout_file(source):
    """
    Return the single Layout*.xlsx in `source`.

    All layout readers go through this so they can never disagree about which file
    they are reading when more than one is present.
    """
    matches = natsorted(glob.glob(os.path.join(source, "Layout*.xlsx")))
    if not matches:
        raise FileNotFoundError(f"No Layout*.xlsx found in {source}")
    if len(matches) > 1:
        print(
            f"Warning: found {len(matches)} Layout*.xlsx files in {source} "
            f"({[os.path.basename(m) for m in matches]}); using {os.path.basename(matches[0])}."
        )
    return matches[0]


def _norm_match_key(s):
    """Lowercase and drop every non-alphanumeric character, so '_', '-' and ' ' never matter."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def get_plate_folder_hints(source, layout_sheets=("MediumLayout", "StainingLayout",
                                                  "LineLayout", "OtherLayout")):
    """
    Read the optional 'Folder:' cells that declare which OME-Zarr folder each plate lives in.

    In any layout sheet, a plate block may carry a folder hint on the same row as its
    barcode:

        | Barcode: | 260429LG001ACAajACAakD10 | Folder: | acaak_d10 |

    The hint only has to be a substring that uniquely identifies one OME-Zarr folder
    (matched against the folder name first, then the full path); it does not have to be
    the complete folder name. Accepted labels are 'Folder:', 'Zarr:' and 'Plate Folder:'.

    Parameters:
    - source (str): Experiment folder containing the Layout*.xlsx file.
    - layout_sheets (iterable of str): Sheets to scan for plate blocks.

    Returns:
    - hints (dict): {barcode: folder hint}. Empty if no hints are present.
    """
    filename = _layout_file(source)

    hints = {}
    for sheet in layout_sheets:
        try:
            df = pd.read_excel(filename, sheet_name=sheet, header=None)
        except ValueError:
            continue  # sheet not present in this layout file
        nrows, ncols = df.shape
        for j in range(nrows):
            row = df.iloc[j, :]
            barcode = folder_hint = None
            for k, v in enumerate(row):
                if k + 1 >= ncols:
                    continue
                label = str(v).strip().lower()
                if label == "barcode:":
                    barcode = str(row[k + 1]).strip()
                elif label in ("folder:", "zarr:", "plate folder:"):
                    folder_hint = str(row[k + 1]).strip()
            if barcode and folder_hint and folder_hint.lower() not in ("nan", "none", ""):
                if barcode in hints and hints[barcode] != folder_hint:
                    raise ValueError(
                        f"Conflicting 'Folder:' hints for barcode {barcode}: "
                        f"'{hints[barcode]}' vs '{folder_hint}'. Make them identical across sheets."
                    )
                hints[barcode] = folder_hint
    return hints


def resolve_barcode_folders(barcodes, folders, hints=None, verbose=True):
    """
    Map every barcode to exactly one OME-Zarr folder, explicitly and reproducibly.

    Resolution order per barcode:
      1. the 'Folder:' hint declared in the layout file, if present
      2. otherwise the barcode itself, matched against the folder name

    In both cases the key is matched as a normalised substring (case-, '_'- and
    '-'-insensitive), first against the folder name and then against the full path, so a
    hint such as 'output/2' can disambiguate identical folder names in different parents.

    There is deliberately no positional fallback. Pairing a barcode with a plate by sort
    order is unverifiable and, when wrong, mislabels every object on that plate instead of
    failing; so anything that cannot be resolved unambiguously raises here.

    Parameters:
    - barcodes (list of str): Barcodes from the layout file.
    - folders (list of str, or dict): Candidate OME-Zarr folders (e.g. from find_zarr_dirs).
      An already-resolved {barcode: folder} dict is validated and returned unchanged.
    - hints (dict, optional): {barcode: folder hint} from get_plate_folder_hints.
    - verbose (bool): Print the resolved pairings for visual confirmation.

    Returns:
    - folder_map (dict): {barcode: folder path}, in the order of `barcodes`.
    """
    barcodes = list(barcodes)

    # Already resolved (e.g. reloaded from ad.uns["folders"]): validate and pass through.
    if isinstance(folders, dict):
        missing = [bc for bc in barcodes if bc not in folders]
        if missing:
            raise ValueError(f"Folder mapping is missing entries for barcodes: {missing}")
        return {bc: folders[bc] for bc in barcodes}

    hints = dict(hints or {})
    folders = [str(f) for f in folders]
    names = [os.path.basename(f.rstrip("/")) for f in folders]

    mapping, how, problems = {}, {}, []

    for bc in barcodes:
        key = hints.get(bc, bc)
        nkey = _norm_match_key(key)
        if not nkey:
            problems.append(f"  {bc}: empty match key.")
            continue

        candidates = [i for i, n in enumerate(names) if nkey in _norm_match_key(n)]
        if not candidates:
            candidates = [i for i, f in enumerate(folders) if nkey in _norm_match_key(f)]

        if len(candidates) == 1:
            mapping[bc] = folders[candidates[0]]
            how[bc] = f"Folder: '{key}'" if bc in hints else "barcode in folder name"
        elif not candidates:
            problems.append(
                f"  {bc}: no folder matches '{key}'. Add a 'Folder:' cell next to this "
                f"barcode in the layout file."
            )
        else:
            problems.append(
                f"  {bc}: '{key}' matches {len(candidates)} folders "
                f"({[names[i] for i in candidates]}). Make the 'Folder:' hint more specific."
            )

    claimed = {}
    for bc, f in mapping.items():
        if f in claimed:
            problems.append(f"  {claimed[f]} and {bc} both resolve to {os.path.basename(f)}.")
        claimed[f] = bc

    if problems:
        raise ValueError(
            "Could not resolve barcode -> OME-Zarr folder unambiguously:\n"
            + "\n".join(problems)
            + "\n\nAvailable folders:\n"
            + "\n".join(f"  {n}   ({f})" for n, f in zip(names, folders))
        )

    if verbose:
        width = max((len(b) for b in barcodes), default=0)
        print(f"Resolved {len(mapping)} barcode -> OME-Zarr folder pairings:")
        for bc in barcodes:
            print(f"  {bc:<{width}}  ->  {os.path.basename(mapping[bc])}   [{how[bc]}]")
        unused = [f for f in folders if f not in claimed]
        if unused:
            print(
                f"  Note: {len(unused)} OME-Zarr folder(s) not claimed by any barcode: "
                f"{[os.path.basename(f) for f in unused]}"
            )

    return {bc: mapping[bc] for bc in barcodes}


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

def find_barcodes_with_condition(experiment_setup, value):
    """
    Find the barcodes of plates that use a given condition value in any well.

    Parameters:
    - experiment_setup (dict): {barcode: {well: [Medium, AB, Cell_line, Other]}}.
    - value: The condition value to search for, in any of the four layout slots.
    """
    found_barcodes = []
    for barcode, wells in experiment_setup.items():
        for _, treatments in wells.items():
            if value in treatments:
                found_barcodes.append(barcode)
                break
    return found_barcodes

def extract_ome_zarr_tables(experiment_setup, source, folder, table_name):
    """
    Loads OME-Zarr plates for each barcode, extracts tables, and combines them into a single DataFrame.

    Parameters:
    - experiment_setup: dict
        Nested dictionary with barcode as keys and well-specific metadata as values.
        Example: {barcode: {well: [Medium, AB, Cell_line, Other], ...}, ...}
    - source: str
        Root directory path containing the OME-Zarr folders.
    - folder: list of str, or dict {barcode: str}
        Folder name for each barcode. If a list, folder[i] is paired
        positionally with the i-th key of experiment_setup (only safe when
        experiment_setup's key order is known to match folder's order, e.g.
        right after both are freshly built in the same session). If a dict,
        each barcode's folder is looked up by key, which is robust to
        experiment_setup's key order changing (e.g. after an .h5ad reload).
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

    if not isinstance(folder, dict):
        print(
            "Warning: pairing barcodes to OME-Zarr folders by position. This is only correct "
            "if both happen to sort identically. Pass the {barcode: folder} mapping returned "
            "by make_experiment instead."
        )

    ome_zarr_dict = {}
    all_plate_dfs = []

    for i, barcode in enumerate(experiment_setup):
        folder_path = folder[barcode] if isinstance(folder, dict) else folder[i]
        plate = ome_zarr.import_plate(os.path.join(source, folder_path))
        ome_zarr_dict[barcode] = plate
        df_lst = plate.get_table(table_name, as_AnnData = True)

        # Prepare well and path lists
        wells = plate.get_names()
        paths = plate.paths

        # Only use wells present in experiment_setup[barcode].keys()
        valid_wells = set(experiment_setup[barcode].keys())
        filtered = [
            (df, well, path)
            for df, well, path in zip(df_lst, wells, paths)
            if well in valid_wells
        ]

        if not filtered:
            continue

        dfs, wells_filtered, paths_filtered = zip(*filtered)

        # Annotate each DataFrame with well and path
        
        converted_dfs = []

        for df, well, path in zip(dfs, wells_filtered, paths_filtered):
            if df is not None:
                if "label" in df.obs.columns:
                    df.obs_names = df.obs["label"]
                df = df.to_df()
                df['well'] = well
                df['path'] = path
                converted_dfs.append(df)

        plate_df = pd.concat(converted_dfs, ignore_index=False)
        plate_df["Barcode"] = barcode
        plate_df["UID"] = barcode + "-" + plate_df["well"].astype(str) + "-" + plate_df.index.astype(str)

        # Extract metadata with safe access
        meta = experiment_setup[barcode]
        # Layout order is [Medium, AB, Cell_line, Other]; index each explicitly. Using [-1]
        # for both "Day" and "Other" made them duplicates and silently dropped Cell_line.
        empty = [None, None, None, None]
        plate_df["Medium"] = plate_df["well"].map(lambda w: meta.get(w, empty)[0])
        plate_df["AB"] = plate_df["well"].map(lambda w: meta.get(w, empty)[1])
        plate_df["Cell_line"] = plate_df["well"].map(lambda w: meta.get(w, empty)[2])
        plate_df["Other"] = plate_df["well"].map(lambda w: meta.get(w, empty)[3])
        all_plate_dfs.append(plate_df)

    ome_zarr_df = pd.concat(all_plate_dfs, ignore_index=True)
    ome_zarr_df.index = ome_zarr_df.UID
    return ome_zarr_dict, ome_zarr_df


def get_stainings(source, sheet="StainingLayout"):
    """
    Extract antibody staining information from an Excel file describing immunostaining layout.

    Returns
    -------
    out : dict
        Mapping of antibody mix -> either:
        - dict(round -> list of stains by channel) if 'Round' column exists
        - list of stains by channel (legacy)
    """

    from Functions.Fcts_FE import _stringify_dict_keys

    filename = _layout_file(source)

    df = pd.read_excel(filename, sheet_name=sheet, header=None)

    # Find header row — anchor on 'UniqueID' (legacy) or 'Target' (new)
    header_row = None
    anchor_values = {"uniqueid", "target"}
    for i in range(len(df)):
        row_vals = df.iloc[i, :].astype(str).str.strip().str.lower()
        if row_vals.isin(anchor_values).any():
            header_row = i
            break
    if header_row is None:
        raise RuntimeError("Immunostaining table not found!")

    header = [str(x).strip() for x in df.iloc[header_row, :]]
    last_col = max(i for i, v in enumerate(header) if v and v.lower() not in ("nan", "none"))
    header = header[:last_col + 1]

    # Read relevant data rows
    rows = []
    for i in range(header_row + 1, len(df)):
        row = df.iloc[i, :len(header)].tolist()
        if all(pd.isna(x) or str(x).strip() == '' for x in row):
            continue
        rows.append(row)
    tab = pd.DataFrame(rows, columns=header)

    # Filter rows
    tab = tab[tab["Target"].notna() & tab["Antibody Mix"].notna()]
    tab = tab[~tab["Target"].astype(str).str.lower().str.contains("brightfield")]
    tab = tab[tab["Imaging Channel"].notna()]

    # Exclude secondary antibodies only if 'Type' column exists (legacy layout)
    if "Type" in tab.columns:
        tab = tab[~tab["Type"].astype(str).str.lower().str.contains("secondary")]

    # "Imaging Channel" is a 1-based channel number, NOT a position: channel 1 is index 0
    # of the OME-Zarr channel axis. Parse it as a number so that channels are ordered
    # numerically (string sorting puts "10" before "2") and gaps stay visible.
    tab["Imaging Channel"] = pd.to_numeric(tab["Imaging Channel"], errors="coerce")
    unparsable = tab["Imaging Channel"].isna()
    if unparsable.any():
        raise ValueError(
            "Non-numeric 'Imaging Channel' values in the staining layout for targets: "
            f"{list(tab.loc[unparsable, 'Target'])}"
        )
    tab["Imaging Channel"] = tab["Imaging Channel"].astype(int)
    if (tab["Imaging Channel"] < 1).any():
        raise ValueError(
            "'Imaging Channel' must be 1-based (channel 1 = first channel of the image). "
            f"Found: {sorted(set(tab['Imaging Channel']))}"
        )

    # Round handling (optional)
    has_round = "Round" in tab.columns
    if has_round:
        tab["Round"] = pd.to_numeric(tab["Round"], errors="coerce").fillna(0).astype(int)
    else:
        tab["Round"] = 0

    # Explode mixes: a row can list more than one (e.g. 'AB1, AB2')
    mix_rows = []
    for _, row in tab.iterrows():
        mixentries = [m.strip() for m in str(row["Antibody Mix"]).split(",") if m.strip()]
        for mix in mixentries:
            data = row.to_dict()
            data["Antibody Mix"] = mix
            mix_rows.append(data)
    dfmix = pd.DataFrame(mix_rows)

    # Build output: mix -> round -> stains list, indexed by channel.
    #
    # Position i of the returned list is the stain imaged on "Imaging Channel" i+1, i.e.
    # it is directly usable as an index into the OME-Zarr channel axis. Channels with no
    # stain of their own (brightfield and secondary rows are filtered out above, and a
    # layout may simply skip a channel) are filled with "" so every later stain keeps its
    # true channel. Consumers must skip falsy entries.
    out: dict[str, dict[int, list[str]]] = {}
    gap_notes = []
    for (mix, rnd), g in dfmix.groupby(["Antibody Mix", "Round"]):
        channels = g["Imaging Channel"].astype(int)
        if channels.duplicated().any():
            dupes = sorted(channels[channels.duplicated()].unique())
            raise ValueError(
                f"Duplicate imaging channel(s) {dupes} for mix {mix}, round {rnd}: "
                f"{list(zip(g['Target'], channels))}"
            )
        stains = [""] * int(channels.max())
        for channel, target in zip(channels, g["Target"].astype(str)):
            stains[channel - 1] = target
        missing = [i + 1 for i, s in enumerate(stains) if not s]
        if missing:
            gap_notes.append(f"  {mix} round {rnd}: no stain on channel(s) {missing}; they will be skipped.")
        out.setdefault(mix, {})[int(rnd)] = stains

    # Print summary
    print(f"Found {len(out)} staining mixes in experiment setup:")
    for mix, rounds in out.items():
        rounds_str = ", ".join(
            f"R{r}(" + ", ".join(f"ch{i + 1}:{s}" for i, s in enumerate(v) if s) + ")"
            for r, v in sorted(rounds.items())
        )
        print(f"  {mix}: {rounds_str}")
    for note in gap_notes:
        print(note)

    # Backward compatibility: if no Round column, collapse to legacy list
    if not has_round:
        legacy = {mix: rounds.get(0, []) for mix, rounds in out.items()}
        return legacy

    return _stringify_dict_keys(out)
