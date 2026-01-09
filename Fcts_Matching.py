import numpy as np
import pandas as pd
import anndata as ad

from scipy.spatial import cKDTree
from scipy import sparse


"""***
MATCHING FUNCTIONS
***

Utility functions to match two AnnData objects 1:1 by centroid proximity within wells.
Designed for the ATMC_Analysis workflow, where each object has:
- Barcode
- Well
- centroid_x_micrometer
- centroid_y_micrometer
"""


def _prefix_index(idx: pd.Index, prefix: str) -> pd.Index:
    """Prefix a pandas Index with '{prefix}__' """
    return pd.Index([f"{prefix}__{x}" for x in idx], dtype="object")


def _hstack_X(X1, X2):
    """Column-wise stack dense or sparse matrices."""
    if sparse.issparse(X1) or sparse.issparse(X2):
        X1 = sparse.csr_matrix(X1)
        X2 = sparse.csr_matrix(X2)
        return sparse.hstack([X1, X2]).tocsr()
    return np.concatenate([np.asarray(X1), np.asarray(X2)], axis=1)


def validate_required_obs(adata: ad.AnnData, required_cols: list[str], name: str = "AnnData") -> None:
    missing = [c for c in required_cols if c not in adata.obs.columns]
    if missing:
        raise ValueError(f"{name} is missing required obs columns: {missing}")


def coerce_numeric_coords(adata: ad.AnnData, x_col: str, y_col: str, name: str = "AnnData") -> None:
    adata.obs[x_col] = pd.to_numeric(adata.obs[x_col], errors="coerce")
    adata.obs[y_col] = pd.to_numeric(adata.obs[y_col], errors="coerce")
    if adata.obs[[x_col, y_col]].isna().any().any():
        raise ValueError(f"{name} contains NaN centroid coordinates in {x_col}/{y_col}.")


def match_1to1_by_centroid(
    ad1: ad.AnnData,
    ad2: ad.AnnData,
    max_distance_um: float,
    barcode_col: str = "Barcode",
    well_col: str = "Well",
    x_col: str = "centroid_x_micrometer",
    y_col: str = "centroid_y_micrometer",
) -> tuple[pd.DataFrame, list[str]]:
    """Greedy per-(Barcode,Well) 1:1 matching between AD1 and AD2.

    Workflow per well:
    1) Enumerate all candidate pairs (ad1_i, ad2_j) within max_distance_um.
    2) Sort candidate pairs by distance.
    3) Greedily assign smallest distances while enforcing 1:1.

    Returns
    -------
    match_df : pd.DataFrame
        Indexed by AD1 obs_names, with columns:
        - ad2_id
        - match_distance_um
    unmatched_ad1 : list[str]
        AD1 obs_names that had no AD2 match within max_distance_um.
    """

    required = [barcode_col, well_col, x_col, y_col]
    validate_required_obs(ad1, required, name="AD1")
    validate_required_obs(ad2, required, name="AD2")

    coerce_numeric_coords(ad1, x_col, y_col, name="AD1")
    coerce_numeric_coords(ad2, x_col, y_col, name="AD2")

    matches: list[tuple[str, str, float]] = []
    unmatched: list[str] = []

    ad2_groups = dict(tuple(ad2.obs.groupby([barcode_col, well_col], sort=False)))

    for (bc, well), g1 in ad1.obs.groupby([barcode_col, well_col], sort=False):
        if (bc, well) not in ad2_groups:
            unmatched.extend(g1.index.tolist())
            continue

        g2 = ad2_groups[(bc, well)]

        coords1 = g1[[x_col, y_col]].to_numpy(dtype=float)
        coords2 = g2[[x_col, y_col]].to_numpy(dtype=float)

        tree2 = cKDTree(coords2)

        pairs: list[tuple[float, int, int]] = []
        for i, p in enumerate(coords1):
            js = tree2.query_ball_point(p, r=max_distance_um)
            if not js:
                continue
            dists = np.linalg.norm(coords2[js] - p, axis=1)
            for j, d in zip(js, dists):
                pairs.append((float(d), i, j))

        pairs.sort(key=lambda t: t[0])
        used_i: set[int] = set()
        used_j: set[int] = set()
        local_map: dict[int, tuple[int, float]] = {}

        for d, i, j in pairs:
            if i in used_i or j in used_j:
                continue
            used_i.add(i)
            used_j.add(j)
            local_map[i] = (j, d)

        g1_obs = g1.index.to_list()
        g2_obs = g2.index.to_list()

        for i, ad1_id in enumerate(g1_obs):
            if i not in local_map:
                unmatched.append(ad1_id)
                continue
            j, d = local_map[i]
            matches.append((ad1_id, g2_obs[j], d))

    if len(matches) == 0:
        raise ValueError(
            "No matches found at all. Increase max_distance_um or check centroids / wells / barcodes."
        )

    match_df = (
        pd.DataFrame(matches, columns=["ad1_id", "ad2_id", "match_distance_um"]).set_index("ad1_id")
    )

    return match_df, unmatched


def merge_anndata_by_matches(
    ad1: ad.AnnData,
    ad2: ad.AnnData,
    match_df: pd.DataFrame,
    prefix_ad1: str,
    prefix_ad2: str,
    prefix_ad2_obs: str | None = None,
    keep_uns_from: str = "ad1",
    matching_uns: dict | None = None,
) -> ad.AnnData:
    """Merge AD1 and AD2 given a 1:1 match_df.

    - Keeps rows: matched AD1 rows (in AD1 order).
    - Adds columns: AD1 features + AD2 features (after prefixing var_names).
    - Merges obs: AD1 obs + prefixed AD2 obs + matched_ad2_id + match_distance_um.

    Parameters
    ----------
    match_df : DataFrame
        index = AD1 obs_names, column 'ad2_id' contains AD2 obs_names.
    """

    if prefix_ad2_obs is None:
        prefix_ad2_obs = prefix_ad2

    ad1_m = ad1[match_df.index].copy()
    ad2_m = ad2[match_df["ad2_id"].values].copy()

    ad1_m.var_names = _prefix_index(ad1_m.var_names, prefix_ad1)
    ad2_m.var_names = _prefix_index(ad2_m.var_names, prefix_ad2)

    X = _hstack_X(ad1_m.X, ad2_m.X)
    var = pd.concat([ad1_m.var.copy(), ad2_m.var.copy()], axis=0)

    obs = ad1_m.obs.copy()
    for col in ad2_m.obs.columns:
        new_col = f"{prefix_ad2_obs}__{col}"
        if new_col in obs.columns:
            raise ValueError(f"Obs column collision after prefixing: {new_col}")
        obs[new_col] = ad2_m.obs[col].values

    obs["matched_ad2_id"] = match_df["ad2_id"].values
    obs["match_distance_um"] = match_df["match_distance_um"].values

    ad_merged = ad.AnnData(X=X, obs=obs, var=var)

    if keep_uns_from == "ad1":
        ad_merged.uns = dict(ad1.uns)
    elif keep_uns_from == "ad2":
        ad_merged.uns = dict(ad2.uns)
    else:
        ad_merged.uns = {}

    if matching_uns is not None:
        ad_merged.uns["matching"] = matching_uns

    return ad_merged


def save_matching_outputs(
    ad_merged: ad.AnnData,
    unmatched_ad1: list[str],
    out_dir: str,
    out_basename: str = "1b_Matched_1to1",
    error_on_unmatched: bool = True,
) -> None:
    """Save merged AnnData using the project's save_adata() helper, plus an unmatched report.

    Notes
    -----
    - save_adata() writes into ad_merged.uns['table_dir'].
    - This function sets that key to out_dir.
    - Unmatched AD1 IDs are written as a CSV next to the merged h5ad.
    """

    import os
    import datetime
    from Fcts_Base import save_adata, save_df

    os.makedirs(out_dir, exist_ok=True)

    # Ensure save_adata knows where to write
    ad_merged.uns["table_dir"] = out_dir

    # Save AnnData
    save_adata(ad_merged, out_basename)

    # Save unmatched report
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mmin%Ss")
    df_unmatched = pd.DataFrame({"ad1_id": unmatched_ad1})
    save_df(out_dir, f"{out_basename}_unmatched_ad1_{timestamp}", df_unmatched)

    if error_on_unmatched and len(unmatched_ad1) > 0:
        raise ValueError(
            f"{len(unmatched_ad1)} AD1 objects had no AD2 match within the selected max distance. "
            "Merged output was written, but matching is incomplete (see unmatched CSV)."
        )
