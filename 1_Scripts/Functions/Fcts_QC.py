import matplotlib.pyplot as plt
import re
import scipy.ndimage
import scipy.sparse
import scipy.stats
import pandas as pd
import numpy as np
import warnings
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator
from scipy.optimize import minimize_scalar
from Functions.Fcts_Base import remove_organoids

"""
***
QUALITY CONTROL FUNCTIONS
***
"""


def normalize_groups(adata, group_by, control=None, transform_features=None, transform="log1p",
                     skew_threshold=0.5):
    """
    Normalize data in an AnnData object by groups, applying a log1p or Yeo-Johnson transformation
    to skewed features (see transform_features), and then creating separate layers for robust scaling,
    z-normalization and min-max scaling.

    Parameters:
    adata (anndata.AnnData): The AnnData object containing the data to be normalized.
    group_by (str or list of str): The column name(s) in adata.obs to group the data by.
    control (dict, optional): A dictionary specifying the control group for normalization.
        Each key is a column name in adata.obs, and each value is the control value for
        that column. Multiple key-value pairs are combined with AND, e.g.
        {"Medium": "Control", "Cell_line": "WT"} selects cells that are both Medium ==
        "Control" and Cell_line == "WT". If None, each group is normalized independently.
    transform_features (None, "auto" or list of str, optional): Features to transform before scaling.
        None: features whose name contains "area", "moments", "potency" or "substructures_size"
        (but not "ratio"). "auto": skewed features in the control cells that become more symmetric
        after the transformation, found by find_skewed_features, which also plots an overview.
        A list: exactly these features.
    transform (str): "log1p" or "yeo-johnson"; see find_skewed_features. log1p keeps differences
        interpretable as fold changes but only fixes right skew of non-negative features.
        Yeo-Johnson makes more features symmetric, including left-skewed ones and ones with
        negative values, at the cost of a feature-specific, less interpretable scale.
    skew_threshold (float): Only used with transform_features="auto"; see find_skewed_features.

    The transformed data are stored in adata.layers["transformed"], the transformed features in
    adata.uns["transformed_features"], the transformation in adata.uns["transform"] and, for
    Yeo-Johnson, its parameters in adata.uns["transform_params"]. With "auto", the skewness table
    is stored in adata.uns["skewness"].

    Notes:
    - Cells with NaN in a feature are ignored when fitting scalers for that feature,
      but are NOT dropped. Each feature is scaled independently using its own
      non-NaN control (or group) cells.
    - NaN values in the output layers remain NaN.
    - Features that are NaN for every cell in a group (e.g. a staining not used on
      that plate) are skipped for that group and stay NaN.
    - A ValueError is raised if no control cells are found in a group, or if a
      feature has values in a group but zero non-NaN control cells.
    """

    if isinstance(group_by, str):
        group_by = [group_by]

    # Fix pandas grouping observed warning by specifying observed=True
    groups = adata.obs.groupby(group_by, observed=True).groups

    # Handle sparse matrix: convert to dense if needed
    if scipy.sparse.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = adata.X.copy()

    df = pd.DataFrame(X, index=adata.obs_names, columns=adata.var_names)

    _check_transform(transform)
    for key in ["skewness", "transform_params"]:
        adata.uns.pop(key, None)  # set again below where they apply

    params = None
    if transform_features is None:
        lower_var_names = [v.lower() for v in adata.var_names]
        to_transform = [
            adata.var_names[i] for i, v in enumerate(lower_var_names)
            if any(key in v for key in ["area", "moments", "potency", "substructures_size"])
            and "ratio" not in v
        ]
    elif isinstance(transform_features, str) and transform_features == "auto":
        to_transform, params = find_skewed_features(
            adata, group_by=group_by, control=control, transform=transform, skew_threshold=skew_threshold
        )
        adata.uns["skewness"] = params
    else:
        if isinstance(transform_features, str):
            to_transform = [transform_features]
        else:
            to_transform = list(transform_features)
        unknown = [f for f in to_transform if f not in adata.var_names]
        if unknown:
            raise ValueError(f"transform_features not found in adata.var_names: {unknown}")

    if transform == "yeo-johnson" and to_transform:
        if params is None:
            params = _fit_yeo_johnson(adata, np.asarray(df[to_transform], dtype=np.float64), to_transform,
                                      group_by, control, trim=0.005, min_cells=20)
        params = params.loc[to_transform, ["lambda", "center", "scale"]]
        not_fitted = list(params.index[params["lambda"].isna()])
        if not_fitted:
            raise ValueError(f"Yeo-Johnson could not be fitted (too few or constant control values): {not_fitted}")
        adata.uns["transform_params"] = params
    adata.uns["transformed_features"] = list(to_transform)
    adata.uns["transform"] = transform

    df_transformed = df.copy()
    if to_transform:
        df_transformed[to_transform] = _apply_transform(
            np.asarray(df[to_transform], dtype=np.float64), transform, params
        )

    adata.layers["transformed"] = df_transformed.values

    robust_normalized_data = np.full_like(df_transformed.values, np.nan, dtype=np.float64)
    z_normalized_data      = np.full_like(df_transformed.values, np.nan, dtype=np.float64)
    minmax_normalized_data = np.full_like(df_transformed.values, np.nan, dtype=np.float64)

    if control is not None:
        if len(control) == 0:
            raise ValueError("Control dictionary must have at least one key-value pair")
        control_mask_all = _control_mask(adata.obs, control)
        control_label = _control_label(control)

    for group, indices in groups.items():
        int_indices = adata.obs_names.get_indexer(indices)
        group_data = adata.layers["transformed"][int_indices]
        n_features = group_data.shape[1]

        # Determine which cells to fit scalers on (NaNs handled per-feature below)
        if control is None:
            fit_data = group_data
        else:
            control_mask = control_mask_all[int_indices]
            fit_data = group_data[control_mask]

            if fit_data.shape[0] == 0:
                raise ValueError(
                    f"No control cells ({control_label}) found in group '{group}'. "
                    f"Cannot fit scalers."
                )

            # Features that are NaN in the whole group (e.g. staining not used on this plate)
            # are skipped and stay NaN. Raise only if the group has values for a feature but
            # its controls don't, since those values could not be normalized.
            n_valid_ctrl  = (~np.isnan(fit_data)).sum(axis=0)
            n_valid_group = (~np.isnan(group_data)).sum(axis=0)
            absent_features = [
                adata.var_names[i] for i in range(n_features) if n_valid_group[i] == 0
            ]
            missing_features = [
                adata.var_names[i] for i in range(n_features)
                if n_valid_ctrl[i] == 0 and n_valid_group[i] > 0
            ]
            if absent_features:
                print(f"Group '{group}': {len(absent_features)} features have no values "
                      f"in this group and are left as NaN.")
            if missing_features:
                raise ValueError(
                    f"Group '{group}': the following features have values in the group but "
                    f"NO non-NaN control cells and cannot be normalized:\n{missing_features}"
                )

        # --- Per-feature scaler fitting (NaN-aware, no row dropping) ---
        robust_params = []   # (center, scale)  per feature
        z_params      = []   # (mean,   std)     per feature
        minmax_params = []   # (min,    range)   per feature

        for j in range(n_features):
            col = fit_data[:, j]
            valid = col[~np.isnan(col)]

            if valid.shape[0] == 0:
                # No non-NaN values in this feature for this group — keep NaN in output
                robust_params.append((np.nan, 1.0))
                z_params.append((np.nan, 1.0))
                minmax_params.append((np.nan, 1.0))
                continue

            # RobustScaler: center = median, scale = IQR
            center = np.median(valid)
            q75, q25 = np.percentile(valid, [75, 25])
            iqr = q75 - q25
            robust_params.append((center, iqr if iqr != 0 else 1.0))

            # StandardScaler: mean, std
            mu    = valid.mean()
            sigma = valid.std()
            z_params.append((mu, sigma if sigma != 0 else 1.0))

            # MinMaxScaler: min, range
            vmin, vmax = valid.min(), valid.max()
            vrange = vmax - vmin
            minmax_params.append((vmin, vrange if vrange != 0 else 1.0))

        # --- Transform all group cells per-feature, preserving NaNs ---
        robust_scaled = np.full_like(group_data, np.nan, dtype=np.float64)
        z_scaled      = np.full_like(group_data, np.nan, dtype=np.float64)
        minmax_scaled = np.full_like(group_data, np.nan, dtype=np.float64)

        for j in range(n_features):
            col        = group_data[:, j]
            valid_mask = ~np.isnan(col)

            r_center, r_scale  = robust_params[j]
            z_mu,     z_sigma  = z_params[j]
            mm_min,   mm_range = minmax_params[j]

            if np.isnan(r_center):
                continue  # feature had no valid fit data — leave as NaN

            robust_scaled[valid_mask, j] = (col[valid_mask] - r_center) / r_scale
            z_scaled[valid_mask, j]      = (col[valid_mask] - z_mu)     / z_sigma
            minmax_scaled[valid_mask, j] = (col[valid_mask] - mm_min)   / mm_range

        robust_normalized_data[int_indices] = robust_scaled
        z_normalized_data[int_indices]      = z_scaled
        minmax_normalized_data[int_indices] = minmax_scaled

    adata.layers["robust_scaled"] = robust_normalized_data
    adata.layers["z_scaled"]      = z_normalized_data
    adata.layers["minmax_scaled"] = minmax_normalized_data

    if control is not None:
        check_control_normalization(adata, control=control, group_by=group_by)
    return adata


def _control_mask(obs, control):
    """
    Boolean mask over the rows of obs that match ALL key-value pairs in control.
    """
    missing_keys = [k for k in control if k not in obs.columns]
    if missing_keys:
        raise KeyError(f"Control key(s) not found in adata.obs: {missing_keys}")

    mask = np.ones(len(obs), dtype=bool)
    for key, value in control.items():
        mask &= (obs[key] == value).values
    return mask


def _control_label(control):
    """
    Readable description of a control dict for log/error messages, e.g. "Medium=Control, Cell_line=WT".
    """
    return ", ".join(f"{k}={v}" for k, v in control.items())


def check_control_normalization(adata, control, group_by, tol=0.5):
    """
    Sanity check: for the control group, 
    - z_scaled features should have mean ≈ 0
    - robust_scaled features should have median ≈ 0
    
    Parameters:
        adata: AnnData object after normalization
        control: dict, e.g. {"Cell_line": "WT_11"} or {"Medium": "Control", "Cell_line": "WT"}
            (multiple key-value pairs are combined with AND)
        group_by: str or list of str — same as used in normalize_groups
        tol: tolerance threshold to flag deviations (default 0.5)
    """
    if isinstance(group_by, str):
        group_by = [group_by]

    control_mask_all = _control_mask(adata.obs, control)
    control_label = _control_label(control)

    results = {}
    for layer, metric_name, metric_fn in [
        ("z_scaled",      "mean",   lambda x: np.nanmean(x, axis=0)),
        ("robust_scaled", "median", lambda x: np.nanmedian(x, axis=0)),
    ]:
        # Check per group_by group
        for group, indices in adata.obs.groupby(group_by, observed=True).groups.items():
            int_idx = adata.obs_names.get_indexer(indices)
            ctrl_int_idx = int_idx[control_mask_all[int_idx]]

            if len(ctrl_int_idx) == 0:
                print(f"[{layer}] Group '{group}': no control samples found, skipping.")
                continue

            data = adata.layers[layer][ctrl_int_idx]
            # Features absent from this group are all-NaN; silence the empty-slice warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                stat = metric_fn(data)
            max_dev = np.nanmax(np.abs(stat))
            status = "✅ OK" if max_dev < tol else "⚠️  WARNING"

            print(f"[{layer}] Group '{group}' | control '{control_label}' | "
                  f"max |{metric_name}| = {max_dev:.4f}  {status}")

            results[(layer, str(group))] = {"metric": metric_name, "max_deviation": max_dev}

    return results


# Plot colors: blue marks the version used for normalization (transformed or raw), gray the other one
_BLUE, _BLUE_LIGHT = "#2a78d6", "#9ec5f4"
_GRAY, _GRAY_LIGHT = "#898781", "#c3c2b7"
_INK, _INK_SECONDARY = "#0b0b0b", "#52514e"
_GRID = "#e1e0d9"

_TRANSFORM_NAMES = {"log1p": "log1p", "yeo-johnson": "Yeo-Johnson"}


def find_skewed_features(adata, group_by, control=None, transform="log1p", skew_threshold=0.5,
                         trim=0.005, min_cells=20, plot=True):
    """
    Find skewed features that become more symmetric after a log1p or Yeo-Johnson transformation.

    Skewness is measured in the reference cells (the control cells, or all cells if control is
    None) of each group (e.g. plate) separately, and the median over groups is used, so plate
    shifts and treatment effects do not show up as skew. Within each group, the most extreme
    `trim` fraction of values on each side is ignored, so a few segmentation artifacts do not
    decide (one artifact at +10 SD among 500 cells alone gives a skewness of ~1.5). Trimming
    lowers the skewness of genuinely skewed features a little.

    transform="log1p" only fixes right skew. A feature is selected if it has no negative values
    in any cell, its skewness is above skew_threshold, and log1p brings it closer to 0.

    transform="yeo-johnson" fixes right and left skew and allows negative values. A feature is
    selected if its absolute skewness is above skew_threshold and the transformation brings it
    closer to 0. The power lambda is fitted by maximum likelihood on the reference cells of each
    group (median over groups) and restricted to [0, 2] (0 = log1p, 1 = no change): only in this
    range is the transformation unbounded in both directions, so cells far outside the control
    range (strong treatment effects) are not squashed against a ceiling. Two placements are tried
    per feature and the more symmetric one is kept: the original values (best for multiplicative
    features such as areas and moments), and values robustly standardized with the reference
    median and IQR (best for bounded features far from 0, such as solidity).

    Parameters:
    adata (anndata.AnnData): The AnnData object; adata.X holds the untransformed features.
    group_by (str or list of str): The column name(s) in adata.obs to group the data by,
        as in normalize_groups.
    control (dict, optional): The reference cells, as in normalize_groups. If None, all cells are used.
    transform (str): "log1p" or "yeo-johnson".
    skew_threshold (float): Skewness above which a feature counts as skewed. Rule of thumb:
        below 0.5 roughly symmetric, 0.5-1 moderately skewed, above 1 highly skewed.
    trim (float): Fraction of values ignored on each side within each group (0 uses all values).
    min_cells (int): A group is skipped for a feature if it has fewer reference values for it.
    plot (bool): Plot the skewness before and after the transformation of every feature.

    Returns:
    features (list of str): The features to transform.
    stats (pd.DataFrame): Per feature: skew_raw and skew_transformed (median over groups),
        n_groups (groups with enough values) and decision: "transform", "not skewed",
        "no improvement", "not testable" (too few or constant values in every group), and for
        log1p also "left-skewed" and "negative values". For Yeo-Johnson also lambda, center and
        scale: the transformation is yeo_johnson((x - center) / scale, lambda).
    """
    _check_transform(transform)
    if isinstance(group_by, str):
        group_by = [group_by]

    X = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
    # float64: cubed deviations of raw moments overflow float32
    X = np.asarray(X, dtype=np.float64)

    stats = _skew_stats(adata, X, adata.var_names, group_by, control, transform, skew_threshold, trim, min_cells)
    features = list(stats.index[stats.decision == "transform"])

    name = _TRANSFORM_NAMES[transform]
    criterion = "skewness" if transform == "log1p" else "|skewness|"
    left_as_is = ", ".join(
        f"{n} {d}" for d, n in stats.decision[stats.decision != "transform"].value_counts().items()
    )
    print(f"{len(features)} of {len(stats)} features selected for {name} ({criterion} > {skew_threshold:g} "
          f"and closer to 0 after {name}). Left as is: {left_as_is or 'none'}.")
    still_skewed = list(stats.index[(stats.decision == "transform")
                                    & (stats.skew_transformed.abs() > skew_threshold)])
    if still_skewed:
        reason = "No transformation fixes "
        if transform == "log1p":
            reason = "log1p only acts like a log for values well above 1, and no transformation fixes "
        print(f"Still skewed after {name}, check with plot_transform_check: {still_skewed}. {reason}"
              f"discrete features or features where 0 means 'not defined' (e.g. _T_ intensities).")

    if plot:
        reference = "control cells" if control is not None else "all cells"
        _plot_skew_overview(stats, transform, skew_threshold, reference, group_by)
    return features, stats


def plot_transform_check(adata, features, group_by, control=None, transform=None, skew_threshold=0.5,
                         trim=0.005, bins=30):
    """
    Plot the distributions of features in the reference cells before and after the transformation,
    with one outline per group (e.g. plate), to check the transformation by eye.

    The version used for normalization is drawn in blue, the other one in gray. After
    normalize_groups this follows adata.uns["transformed_features"]; before, the rule of
    find_skewed_features with skew_threshold decides.

    Parameters:
    adata (anndata.AnnData): The AnnData object; adata.X holds the untransformed features.
    features (str or list of str): The features to plot.
    group_by (str or list of str): The column name(s) in adata.obs to group the data by,
        as in normalize_groups.
    control (dict, optional): The reference cells, as in normalize_groups. If None, all cells are used.
    transform (str or None): "log1p" or "yeo-johnson". None uses the one normalize_groups used
        (adata.uns["transform"]), or log1p before normalize_groups has run.
    skew_threshold (float): Only used before normalize_groups has run; see find_skewed_features.
    trim (float): Fraction of values ignored on each side within each group, as in find_skewed_features.
    bins (int): Number of histogram bins.
    """
    if isinstance(features, str):
        features = [features]
    unknown = [f for f in features if f not in adata.var_names]
    if unknown:
        raise ValueError(f"Feature(s) not found in adata.var_names: {unknown}")
    if isinstance(group_by, str):
        group_by = [group_by]
    if transform is None:
        transform = adata.uns.get("transform", "log1p")
    _check_transform(transform)
    name = _TRANSFORM_NAMES[transform]

    X = adata[:, features].X
    X = np.asarray(X.toarray() if scipy.sparse.issparse(X) else X, dtype=np.float64)
    stats = _skew_stats(adata, X, features, group_by, control, transform, skew_threshold, trim, min_cells=20)
    groups = [values for _, values in _reference_values(adata, X, group_by, control, trim)]

    if "transformed_features" in adata.uns and adata.uns.get("transform") == transform:
        transformed = set(adata.uns["transformed_features"])
    else:
        transformed = set(stats.index[stats.decision == "transform"])

    fig, axes = plt.subplots(len(features), 2, figsize=(5, 0.5 + 1.2 * len(features)),
                             squeeze=False, layout="constrained")
    for j, feat in enumerate(features):
        params = stats.iloc[[j]]
        for ax, version in zip(axes[j], ["raw", "transformed"]):
            used = (version == "transformed") == (feat in transformed)
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)

            values = [g[:, [j]] for g in groups]
            label = "raw"
            if version == "transformed":
                if params.decision.iloc[0] == "negative values":
                    ax.set_xticks([])
                    ax.text(0.5, 0.5, "log1p not applicable:\nnegative values", transform=ax.transAxes,
                            ha="center", va="center", color=_INK_SECONDARY)
                    continue
                values = [_apply_transform(v, transform, params) for v in values]
                label = name
                if transform == "yeo-johnson" and np.isfinite(params["lambda"].iloc[0]):
                    standardized = params.center.iloc[0] != 0 or params.scale.iloc[0] != 1
                    label += f" λ={params['lambda'].iloc[0]:.2f}" + (" standardized" if standardized else "")
            values = [v[~np.isnan(v)] for v in values]
            values = [v for v in values if v.size > 0]
            if not values:
                ax.set_xticks([])
                ax.text(0.5, 0.5, "no values", transform=ax.transAxes,
                        ha="center", va="center", color=_INK_SECONDARY)
                continue

            edges = np.histogram_bin_edges(np.concatenate(values), bins=bins)
            for v in values:
                ax.hist(v, bins=edges, density=True, histtype="step",
                        color=_BLUE if used else _GRAY, linewidth=1, alpha=0.8)
            skew = params[f"skew_{version}"].iloc[0]
            skew = f"{skew:.2f}" if np.isfinite(skew) else "n/a"
            ax.set_xlabel(f"{label} · skew {skew}" + (" · used" if used else ""))
        axes[j, 0].set_title(feat, loc="left")

    reference = "control cells" if control is not None else "all cells"
    fig.suptitle(f"{reference.capitalize()}, one outline per {' × '.join(group_by)}. "
                 f"Blue: version used for normalization.")
    plt.show()


def _check_transform(transform):
    if transform not in _TRANSFORM_NAMES:
        raise ValueError(f"transform must be one of {list(_TRANSFORM_NAMES)}, not {transform!r}")


def _apply_transform(values, transform, params=None):
    """
    Transform the columns of values: log1p (negative values clipped to 0), or Yeo-Johnson with
    the lambda, center and scale columns of params (one row per column of values).
    """
    if transform == "log1p":
        return np.log1p(np.clip(values, 0, None))
    standardized = (values - params["center"].to_numpy()) / params["scale"].to_numpy()
    return _yeo_johnson(standardized, params["lambda"].to_numpy())


def _yeo_johnson(values, lmbda):
    """Yeo-Johnson transformation of each column with its own lambda; NaN stays NaN."""
    lmbda = np.broadcast_to(lmbda, values.shape)
    out = np.full(values.shape, np.nan)
    pos, neg = values >= 0, values < 0
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x, lam = values[pos], lmbda[pos]
        log_case = np.abs(lam) < 1e-8
        out[pos] = np.where(log_case, np.log1p(x), ((x + 1) ** lam - 1) / np.where(log_case, 1, lam))
        x, lam = values[neg], lmbda[neg]
        log_case = np.abs(lam - 2) < 1e-8
        out[neg] = np.where(log_case, -np.log1p(-x), -((1 - x) ** (2 - lam) - 1) / np.where(log_case, 1, 2 - lam))
    return out


def _fit_lambda(values, min_cells):
    """Maximum-likelihood Yeo-Johnson lambda in [0, 2] for one column (NaN if too few or constant values)."""
    values = values[~np.isnan(values)]
    if values.size < min_cells or np.ptp(values) == 0:
        return np.nan
    return minimize_scalar(lambda lam: -scipy.stats.yeojohnson_llf(lam, values),
                           bounds=(0, 2), method="bounded").x


def _fit_yeo_johnson(adata, X, var_names, group_by, control, trim, min_cells):
    """
    Yeo-Johnson lambda, center and scale for each column of X, and the skewness after the
    transformation (median over groups); see find_skewed_features for the procedure.
    """
    ref_mask = _control_mask(adata.obs, control) if control is not None else np.ones(adata.n_obs, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # features without any values
        center = np.nanmedian(X[ref_mask], axis=0)
        q75, q25 = np.nanpercentile(X[ref_mask], [75, 25], axis=0)
        std = np.nanstd(X[ref_mask], axis=0)
    scale = (q75 - q25) / 1.349  # the IQR of a normal distribution is 1.349 SD
    scale = np.where(scale > 0, scale, std)
    scale = np.where(scale > 0, scale, np.nan)

    n_features = X.shape[1]
    fits = []
    for c, s in [(np.zeros(n_features), np.ones(n_features)), (center, scale)]:
        groups = [values for _, values in _reference_values(adata, (X - c) / s, group_by, control, trim)]
        if not groups:
            raise ValueError("No reference cells found in any group.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # features without any values
            lmbda = np.nanmedian([[_fit_lambda(v[:, j], min_cells) for j in range(n_features)] for v in groups],
                                 axis=0)
            skew = np.nanmedian([_skewness(_yeo_johnson(v, lmbda), min_cells) for v in groups], axis=0)
        fits.append((c, s, lmbda, skew))

    # Keep the placement that gives the more symmetric distribution
    (c0, s0, l0, k0), (c1, s1, l1, k1) = fits
    standardized = np.abs(k1) < np.where(np.isnan(k0), np.inf, np.abs(k0))
    return pd.DataFrame({
        "lambda": np.where(standardized, l1, l0),
        "center": np.where(standardized, c1, c0),
        "scale": np.where(standardized, s1, s0),
        "skew_transformed": np.where(standardized, k1, k0),
    }, index=pd.Index(var_names))


def _reference_values(adata, X, group_by, control, trim):
    """
    Yield (group, values) per group: the rows of X for the group's reference cells (control
    cells, or all cells if control is None), with the `trim` extremes of each column set to NaN.
    """
    if control is not None:
        ref_mask = _control_mask(adata.obs, control)
    else:
        ref_mask = np.ones(adata.n_obs, dtype=bool)
    for group, indices in adata.obs.groupby(group_by, observed=True).groups.items():
        int_idx = adata.obs_names.get_indexer(indices)
        int_idx = int_idx[ref_mask[int_idx]]
        if len(int_idx) > 0:
            yield group, _trim_extremes(X[int_idx], trim)


def _trim_extremes(values, trim):
    """Set values below the `trim` and above the `1 - trim` quantile of their column to NaN."""
    if trim <= 0:
        return values
    values = values.copy()
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
        lower, upper = np.nanquantile(values, [trim, 1 - trim], axis=0)
        values[(values < lower) | (values > upper)] = np.nan
    return values


def _skewness(values, min_cells):
    """
    Skewness of each column, ignoring NaN. NaN for columns with fewer than min_cells values
    or only one distinct value.
    """
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns
        deviation = values - np.nanmean(values, axis=0)
        skew = np.nanmean(deviation ** 3, axis=0) / np.nanmean(deviation ** 2, axis=0) ** 1.5
        constant = np.nanmax(values, axis=0) == np.nanmin(values, axis=0)
    too_few = (~np.isnan(values)).sum(axis=0) < min_cells
    skew[too_few | constant | ~np.isfinite(skew)] = np.nan
    return skew


def _skew_stats(adata, X, var_names, group_by, control, transform, skew_threshold, trim, min_cells):
    """Skewness before and after the transformation per feature, and the decision; see find_skewed_features."""
    groups = [values for _, values in _reference_values(adata, X, group_by, control, trim)]
    if not groups:
        raise ValueError("No reference cells found in any group.")
    skew_raw = np.vstack([_skewness(v, min_cells) for v in groups])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # features without any values
        stats = pd.DataFrame({
            "skew_raw": np.nanmedian(skew_raw, axis=0),
            "skew_transformed": np.nan,
            "n_groups": (~np.isnan(skew_raw)).sum(axis=0),
        }, index=pd.Index(var_names))
        if transform == "log1p":
            has_negative = np.nanmin(X, axis=0) < 0
            stats["skew_transformed"] = np.nanmedian(
                [_skewness(np.log1p(np.clip(v, 0, None)), min_cells) for v in groups], axis=0
            )
            # log1p would clip these features, so their log1p skewness means nothing
            stats.loc[has_negative, "skew_transformed"] = np.nan
        else:
            fit = _fit_yeo_johnson(adata, X, var_names, group_by, control, trim, min_cells)
            for col in fit.columns:
                stats[col] = fit[col].to_numpy()

    raw, after = stats.skew_raw, stats.skew_transformed
    if transform == "log1p":
        conditions = [raw.isna(), has_negative, raw < -skew_threshold, raw <= skew_threshold]
        labels = ["not testable", "negative values", "left-skewed", "not skewed"]
    else:
        conditions = [raw.isna(), raw.abs() <= skew_threshold]
        labels = ["not testable", "not skewed"]
    decision = np.select(conditions + [after.abs() < raw.abs()], labels + ["transform"], default="no improvement")
    stats.insert(3, "decision", decision)
    return stats


def _plot_skew_overview(stats, transform, skew_threshold, reference, group_by):
    """
    Dumbbell plot of each feature's skewness before (hollow dot) and after the transformation
    (filled dot), most skewed on top. Blue rows are transformed, gray rows are left as is.
    """
    df = stats[stats.decision != "not testable"].sort_values("skew_raw", ascending=False)
    if df.empty:
        return
    name = _TRANSFORM_NAMES[transform]
    y = np.arange(len(df))
    raw, after = df.skew_raw.to_numpy(), df.skew_transformed.to_numpy()
    has_after = ~np.isnan(after)
    selected = (df.decision == "transform").to_numpy()
    dot_color = np.where(selected, _BLUE, _GRAY)
    line_color = np.where(selected, _BLUE_LIGHT, _GRAY_LIGHT)
    # log1p only fixes right skew; Yeo-Johnson both directions
    thresholds = [skew_threshold] if transform == "log1p" else [-skew_threshold, skew_threshold]

    fig, ax = plt.subplots(figsize=(4.5, 1.4 + 0.12 * len(df)), layout="constrained")
    ax.axvline(0, color=_GRAY_LIGHT, linewidth=0.8, zorder=1)
    for t in thresholds:
        ax.axvline(t, color=_INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.hlines(y[has_after], raw[has_after], after[has_after], colors=line_color[has_after],
              linewidth=1.5, zorder=2)
    ax.scatter(raw, y, s=20, facecolors="white", edgecolors=list(dot_color), linewidths=1.2, zorder=3)
    ax.scatter(after[has_after], y[has_after], s=20, c=list(dot_color[has_after]),
               edgecolors="white", linewidths=0.5, zorder=4)

    # Raw moments can be skewed by 10-100, intensities by ~1: linear within ±1, log beyond
    ax.set_xscale("symlog", linthresh=1)
    ax.xaxis.set_major_locator(FixedLocator([-100, -30, -10, -3, -1, 0, 1, 3, 10, 30, 100, 300, 1000]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    labels = [f"{f} (negative values)" if d == "negative values" else f
              for f, d in zip(df.index, df.decision)]
    ax.set_yticks(y, labels)
    for label, sel in zip(ax.get_yticklabels(), selected):
        label.set_color(_INK if sel else _GRAY)
    ax.set_ylim(len(df) - 0.5, -0.5)  # most skewed on top
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)

    threshold_label = f"±{skew_threshold:g}" if len(thresholds) == 2 else f"{skew_threshold:g}"
    handles = [
        Line2D([], [], linestyle="", marker="o", markerfacecolor="white",
               markeredgecolor=_INK_SECONDARY, label="raw"),
        Line2D([], [], linestyle="", marker="o", markerfacecolor=_INK_SECONDARY,
               markeredgecolor="white", label=f"after {name}"),
        Patch(color=_BLUE, label="transformed"),
        Patch(color=_GRAY, label="left as is"),
        Line2D([], [], color=_INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)),
               label=f"threshold ({threshold_label})"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    ax.set_xlabel(f"Skewness of {reference} (median over {' × '.join(group_by)}, symmetric log axis)")
    fig.suptitle(f"Skewness before and after {name}")
    plt.show()


# Plate maps: diverging blue - gray - red for signed scores, a one-hue blue ramp for magnitudes
_DIVERGING = LinearSegmentedColormap.from_list(
    "qc_diverging", ["#184f95", "#6da7ec", "#e1e0d9", "#e66767", "#a32a2a"]
)
_SEQUENTIAL = LinearSegmentedColormap.from_list("qc_sequential", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

# Wells per plate -> (rows, columns)
_PLATE_FORMATS = {6: (2, 3), 12: (3, 4), 24: (4, 6), 48: (6, 8), 96: (8, 12), 384: (16, 24), 1536: (32, 48)}


def plate_qc(adata, condition_cols=("Medium", "Cell_line", "ABs", "Other"), plate_col="Barcode", well_col="Well",
             control=None, features=None, features_to_plot=None, intensity_features=None, z_threshold=3.5,
             anomaly_fraction=0.1, dropout_delta=-0.8, min_organoids=5, plate_format=None, remove_flagged=False,
             plot=True):
    """
    Find problematic wells (e.g. a forgotten staining mix, seeding errors) and position effects
    (e.g. at plate edges from evaporation, or gradients) on each plate.

    Wells are only compared with wells of the same condition on the same plate, so the plate
    layout itself does not show up as bias. For each well and feature, the whole distribution of
    its organoids is compared with the organoids of the other wells of its condition by Cliff's
    delta: d = P(organoid of this well > organoid of the other wells) - P(<), from -1 to 1, with
    0 = no difference. Unlike a mean or median, d is not driven by single outliers, handles
    heterogeneous (e.g. bimodal) organoid populations, and does not change with log1p, Yeo-Johnson
    or any other monotone transformation. d is turned into a score z by dividing it by its
    expected spread: the sampling noise from the well's organoid count plus the normal
    well-to-well variation of that feature on the plate. Wells with few organoids are therefore
    not flagged just for being noisy.

    A well is flagged if
    - it has no organoids (wells listed in adata.uns["Outgrowth_DF"] by calculate_outgrowth),
    - its organoid count deviates from the other wells of its condition (|z| >= z_threshold;
      log counts, relative to the plate's well-to-well variation). The counts come from
      adata.uns["Outgrowth_DF"], so organoids removed as cut off in 2_Filtering still count;
      without it, the organoids in adata are counted,
    - a staining is much dimmer than in the other wells of its condition (stain dropout:
      z <= -z_threshold and d <= dropout_delta in an intensity feature), or
    - it deviates in at least anomaly_fraction of the features (|z| >= z_threshold).

    Position effects are estimated per feature by regressing d on being at the border of the used
    area (a used well next to the plate edge or to unused wells outside the layout; wells next to
    unused wells enclosed by the layout do not count, so any layout shape works), the row position
    and the column position, pooled over plates. Since d is relative
    to the same condition, they are only estimable where a condition has wells at different
    positions (e.g. at the edge and inside); a layout with each condition in one block on every
    plate leaves only the differences within blocks.

    Parameters:
    adata (anndata.AnnData): The AnnData object; adata.X holds the features.
    condition_cols (list of str): Columns in adata.obs that together define a condition; columns
        not in adata.obs are ignored.
    plate_col, well_col (str): Columns in adata.obs with the plate ID and the well name (e.g. "C03").
    control (dict, optional): Control cells as in normalize_groups. Their wells are marked in the
        plots and flagged control wells are listed separately, since normalization relies on them.
    features (list of str, optional): Features to analyze. None uses all.
    features_to_plot (list of str, optional): Features to show as plate maps of z.
    intensity_features (list of str, optional): Features checked for stain dropout. None uses the
        features ending in "_mean" except "_T_mean", i.e. the mean intensity of each staining.
    z_threshold (float): Score from which a deviation counts (3.5 is the usual cutoff for robust z-scores).
    anomaly_fraction (float): Share of deviating features from which a well is flagged.
    dropout_delta (float): Cliff's delta at or below which a dim staining counts as dropout
        (-0.8: ~90% of the well's organoids are dimmer than those of the other wells).
    min_organoids (int): Wells, and the rest of their condition, need this many organoids to be
        assessed per feature.
    plate_format (int, optional): Wells per plate (6, 12, 24, 48, 96, 384 or 1536). None infers the
        smallest standard format that contains all wells.
    remove_flagged (bool): Remove the organoids of flagged wells (recorded in
        adata.uns["removed_organoids"]). If False, they are only marked.
    plot (bool): Plot the plate overview, the plate maps of features_to_plot and the position effects.

    Returns:
    adata (anndata.AnnData): With adata.uns["well_qc"] (one row per well with flag and reasons),
        adata.uns["position_effects"] (one row per feature) and adata.obs["well_qc_pass"] (False
        for organoids in flagged wells; with remove_flagged, these organoids are removed).
    """
    if isinstance(condition_cols, str):
        condition_cols = [condition_cols]
    condition_cols = [c for c in condition_cols if c in adata.obs.columns]
    if isinstance(features_to_plot, str):
        features_to_plot = [features_to_plot]
    features_to_plot = list(features_to_plot or [])
    features = list(adata.var_names) if features is None else list(features)
    if intensity_features is None:
        intensity_features = [f for f in features if f.endswith("_mean") and not f.endswith("_T_mean")]
    intensity_features = list(intensity_features)
    features = list(dict.fromkeys(features + features_to_plot + intensity_features))
    unknown = [f for f in features if f not in adata.var_names]
    if unknown:
        raise ValueError(f"Feature(s) not found in adata.var_names: {unknown}")

    plate = adata.obs[plate_col].astype(str).to_numpy()
    well = adata.obs[well_col].astype(str).to_numpy()
    condition = _condition_labels(adata.obs, condition_cols)

    X = adata[:, features].X
    X = pd.DataFrame(np.asarray(X.toarray() if scipy.sparse.issparse(X) else X, dtype=np.float64),
                     columns=features)
    delta, var = _well_deltas(X, plate, condition, well, min_organoids)
    z = _deviation_scores(delta, var)

    wells = _well_table(adata, plate, well, condition, plate_col, well_col, control)
    wells["count_z"] = _count_scores(wells)
    assessed = z.notna().sum(axis=1)
    dropout = (z[intensity_features] <= -z_threshold) & (delta[intensity_features] <= dropout_delta)
    wells = wells.join(pd.DataFrame({
        "features_assessed": assessed,
        "deviating_fraction": (z.abs() >= z_threshold).sum(axis=1) / assessed.where(assessed > 0),
        "stain_dropout": [", ".join(f"{f} (d={delta.at[i, f]:+.2f})" for f in dropout.columns[row])
                          for i, row in zip(dropout.index, dropout.to_numpy())],
        "top_features": _top_features(z, delta),
    }, index=z.index))
    wells["features_assessed"] = wells["features_assessed"].fillna(0).astype(int)
    wells[["stain_dropout", "top_features"]] = wells[["stain_dropout", "top_features"]].fillna("")

    reasons = []
    for w in wells.itertuples():
        r = []
        if w.n_organoids == 0:
            r.append("no organoids")
        if abs(w.count_z) >= z_threshold:
            r.append(f"{'low' if w.count_z < 0 else 'high'} organoid count (z={w.count_z:+.1f})")
        if w.stain_dropout:
            r.append(f"stain dropout: {w.stain_dropout}")
        if w.deviating_fraction >= anomaly_fraction:
            r.append(f"deviates in {w.deviating_fraction:.0%} of features, most: {w.top_features}")
        reasons.append("; ".join(r))
    wells["reasons"] = reasons
    wells["flagged"] = wells["reasons"] != ""

    geometry, shape = _well_geometry(wells.index, plate_format)
    wells = wells.join(geometry[["row", "col"]])
    effects = _position_effects(delta, geometry.loc[delta.index])

    adata.obs["well_qc_pass"] = ~pd.MultiIndex.from_arrays([plate, well]).isin(wells.index[wells.flagged])
    adata.uns["well_qc"] = wells.reset_index()[[
        "plate", "well", "condition", "control", "n_organoids", "n_measured", "count_z", "features_assessed",
        "deviating_fraction", "stain_dropout", "top_features", "flagged", "reasons",
    ]]
    adata.uns["position_effects"] = effects

    _print_plate_qc(wells, effects, z_threshold, anomaly_fraction, len(features))
    if plot:
        lim = z_threshold + 1.5
        reference = f"|z| ≥ {z_threshold:g}"
        _plot_plates(wells, wells["deviating_fraction"], shape, _SEQUENTIAL, 0, max(0.3, 3 * anomaly_fraction),
                     f"Share of features with {reference}", wells["flagged"], "flagged well",
                     "Deviation from the other wells of the same condition, all features")
        _plot_plates(wells, wells["count_z"], shape, _DIVERGING, -lim, lim, "z (log organoid count)",
                     wells["flagged"], "flagged well", "Organoid count vs. the other wells of the same condition")
        for feat in features_to_plot:
            values = z[feat].reindex(wells.index)
            _plot_plates(wells, values, shape, _DIVERGING, -lim, lim, "z", values.abs() >= z_threshold, reference,
                         f"{feat}: deviation from the other wells of the same condition")
        _plot_position_effects(effects)

    if remove_flagged:
        keep = adata.obs["well_qc_pass"].to_numpy()
        print(f"Removed {(~keep).sum()} organoids in {wells['flagged'].sum()} flagged wells.")
        reasons = wells["reasons"].reindex(pd.MultiIndex.from_arrays([plate, well])).fillna("").to_numpy()
        adata = remove_organoids(adata, ~keep, "plate_qc", "flagged well: " + reasons.astype(object))
    return adata


def _condition_labels(obs, condition_cols):
    """One label per cell joining its condition columns, e.g. "Ctrl | UC_H | AB1 | Day7"."""
    if not condition_cols:
        return np.full(len(obs), "all")
    labels = obs[condition_cols[0]].astype(str)
    for col in condition_cols[1:]:
        labels = labels + " | " + obs[col].astype(str)
    return labels.to_numpy()


def _well_deltas(X, plate, condition, well, min_organoids):
    """
    Cliff's delta of each well against the other wells of its plate x condition, per feature
    (rows: (plate, well)), and its sampling variance. NaN if the well or the rest of its
    condition has fewer than min_organoids values.
    """
    group, by_well = [plate, condition], [plate, well]
    valid = X.notna()
    # Mann-Whitney U from the rank sum of the well within its whole plate x condition group
    rank_sum = X.groupby(group).rank(method="average").groupby(by_well).sum()
    n_well = valid.groupby(by_well).sum()
    n_rest = valid.groupby(group).transform("sum").groupby(by_well).first() - n_well
    with np.errstate(invalid="ignore", divide="ignore"):
        u = rank_sum - n_well * (n_well + 1) / 2
        delta = 2 * u / (n_well * n_rest) - 1
        var = (n_well + n_rest + 1) / (3 * n_well * n_rest)
    too_few = (n_well < min_organoids) | (n_rest < min_organoids)
    delta, var = delta.mask(too_few), var.mask(too_few)
    delta.index.names = var.index.names = ["plate", "well"]
    return delta, var


def _deviation_scores(delta, var):
    """
    z = d / sqrt(sampling variance of d + well-to-well variance of d), per plate and feature.
    The well-to-well variance is the robust (MAD) variance of d on the plate minus the typical
    sampling variance.
    """
    by_plate = delta.index.get_level_values("plate")
    spread2 = ((delta - delta.groupby(by_plate).transform("median")).abs()
               .groupby(by_plate).transform("median") * 1.4826) ** 2
    between = (spread2 - var.groupby(by_plate).transform("median")).clip(lower=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        z = delta / np.sqrt(var + between)
    return z.where(np.isfinite(z))


def _well_table(adata, plate, well, condition, plate_col, well_col, control, warn=True):
    """
    One row per (plate, well): condition, control well, organoid count and measured organoids (in
    adata). The count comes from adata.uns["Outgrowth_DF"] (calculate_outgrowth), which also adds
    the wells without measured organoids (condition "unknown"); otherwise the organoids in adata
    are counted.
    """
    is_control = _control_mask(adata.obs, control) if control is not None else np.zeros(adata.n_obs, dtype=bool)
    wells = pd.DataFrame({"plate": plate, "well": well, "condition": condition, "control": is_control})
    wells = wells.groupby(["plate", "well"]).agg(
        condition=("condition", "first"), control=("control", "any"), n_measured=("condition", "size")
    )
    outgrowth = adata.uns.get("Outgrowth_DF")
    if outgrowth is not None and {plate_col, well_col, "Organoid_No"} <= set(outgrowth.columns):
        counts = outgrowth.astype({plate_col: str, well_col: str}).groupby([plate_col, well_col])["Organoid_No"].sum()
        counts.index.names = ["plate", "well"]
        unmeasured = counts.index.difference(wells.index)
        if len(unmeasured):
            wells = pd.concat([wells, pd.DataFrame({"condition": "unknown", "control": False, "n_measured": 0},
                                                   index=unmeasured)])
        wells["n_organoids"] = counts.reindex(wells.index).fillna(wells["n_measured"]).astype(int)
    else:
        wells["n_organoids"] = wells["n_measured"]
        if warn:
            print("No adata.uns['Outgrowth_DF'] (run calculate_outgrowth in 2_Filtering first): wells without "
                  "any organoids cannot be detected, and removed organoids are not counted.")
    return wells.sort_index()


def _count_scores(wells):
    """
    Robust z of each well's log organoid count against the median of the other wells of its
    plate x condition, scaled by the plate's well-to-well spread (MAD).
    """
    residual = pd.Series(np.nan, index=wells.index)
    with_organoids = wells[(wells["n_organoids"] > 0) & (wells["condition"] != "unknown")]
    for _, group in with_organoids.groupby([with_organoids.index.get_level_values("plate"), "condition"]):
        if len(group) < 2:
            continue
        log_n = np.log1p(group["n_organoids"].to_numpy(dtype=float))
        others = [np.median(np.delete(log_n, i)) for i in range(len(log_n))]
        residual.loc[group.index] = log_n - others
    by_plate = residual.index.get_level_values("plate")
    spread = (residual - residual.groupby(by_plate).transform("median")).abs().groupby(by_plate).transform("median")
    return residual / (spread * 1.4826).where(spread > 0)


def _top_features(z, delta, n=3):
    """The n features with the largest |z| per well, e.g. "area (d=+0.62), ..."."""
    score = np.abs(z.to_numpy())
    score[np.isnan(score)] = -1
    top = np.argsort(-score, axis=1)[:, :n]
    return [", ".join(f"{z.columns[j]} (d={delta.iat[i, j]:+.2f})" for j in idx if score[i, j] >= 0)
            for i, idx in enumerate(top)]


def _row_index(letters):
    """Plate row letters to a 0-based index: A -> 0, Z -> 25, AA -> 26."""
    index = 0
    for ch in letters.upper():
        index = index * 26 + ord(ch) - ord("A") + 1
    return index - 1


def _row_name(index):
    """0-based row index to plate row letters (inverse of _row_index)."""
    name, index = "", index + 1
    while index > 0:
        index, rest = divmod(index - 1, 26)
        name = chr(ord("A") + rest) + name
    return name


def _well_geometry(index, plate_format):
    """
    Row, column and position terms for each (plate, well), and the plate shape (rows, columns).
    ring: the used well is at the outer border of the used area (see _border_wells);
    row_pos / col_pos: -0.5 (first row or column of the plate) to 0.5 (last).
    """
    parsed = index.get_level_values("well").str.extract(r"^\s*([A-Za-z]+)(\d+)\s*$")
    bad = list(index.get_level_values("well")[parsed.isna().any(axis=1).to_numpy()])
    if bad:
        raise ValueError(f"Cannot read well names (expected e.g. 'C03'): {bad[:5]}")
    geometry = pd.DataFrame({"row": parsed[0].map(_row_index).to_numpy(),
                             "col": parsed[1].astype(int).to_numpy() - 1}, index=index)

    n_rows, n_cols = geometry["row"].max() + 1, geometry["col"].max() + 1
    if plate_format is not None:
        if plate_format not in _PLATE_FORMATS:
            raise ValueError(f"plate_format must be one of {list(_PLATE_FORMATS)}, not {plate_format}")
        if _PLATE_FORMATS[plate_format][0] < n_rows or _PLATE_FORMATS[plate_format][1] < n_cols:
            raise ValueError(f"Wells up to row {_row_name(n_rows - 1)}, column {n_cols} do not fit a "
                             f"{plate_format}-well plate.")
        shape = _PLATE_FORMATS[plate_format]
    else:
        fits = [s for s in _PLATE_FORMATS.values() if s[0] >= n_rows and s[1] >= n_cols]
        shape = fits[0] if fits else (n_rows, n_cols)

    rows, cols = geometry["row"].to_numpy(), geometry["col"].to_numpy()
    plate = index.get_level_values("plate")
    geometry["ring"] = False
    for p in plate.unique():
        in_plate = np.asarray(plate == p)
        edge = _border_wells(rows[in_plate], cols[in_plate], shape)
        geometry.loc[in_plate, "ring"] = edge[rows[in_plate], cols[in_plate]]
    geometry["row_pos"] = rows / max(shape[0] - 1, 1) - 0.5
    geometry["col_pos"] = cols / max(shape[1] - 1, 1) - 0.5
    return geometry, shape


def _border_wells(rows, cols, shape):
    """
    Boolean plate grid: used wells at the outer border of the used area, i.e. with at least one
    of their 4 neighbors off the plate or unused and connected to the plate edge through unused
    wells. Wells next to interior holes (unused wells enclosed by used ones) do not count, so
    any layout shape works.
    """
    used = np.zeros(shape, dtype=bool)
    used[rows, cols] = True
    # Pad with one unused ring, so the off-plate area is part of the outside
    padded = np.pad(used, 1, constant_values=False)
    labels, _ = scipy.ndimage.label(~padded)  # 4-connected unused regions
    outside = labels == labels[0, 0]
    touches_outside = np.zeros_like(padded)
    touches_outside[1:, :] |= outside[:-1, :]
    touches_outside[:-1, :] |= outside[1:, :]
    touches_outside[:, 1:] |= outside[:, :-1]
    touches_outside[:, :-1] |= outside[:, 1:]
    return (padded & touches_outside)[1:-1, 1:-1]


def _position_effects(delta, geometry):
    """
    Per feature, OLS of d on border of the used area, row and column position: estimate, standard error,
    p-value and Benjamini-Hochberg FDR per term. Terms without variation are NaN.
    """
    terms = ["edge", "row", "col"]
    design = geometry[["ring", "row_pos", "col_pos"]].astype(float).to_numpy()
    results = {}
    for feat in delta.columns:
        y = delta[feat].to_numpy()
        ok = ~np.isnan(y)
        res = {f"{t}{s}": np.nan for t in terms for s in ["", "_se", "_p"]}
        res["n_wells"] = int(ok.sum())
        use = [j for j in range(len(terms)) if ok.sum() > 0 and np.ptp(design[ok, j]) > 0]
        n, p = int(ok.sum()), len(use) + 1
        if use and n > p + 2:
            A = np.column_stack([np.ones(n), design[ok][:, use]])
            beta = np.linalg.lstsq(A, y[ok], rcond=None)[0]
            resid = y[ok] - A @ beta
            se = np.sqrt(np.diag(np.linalg.pinv(A.T @ A)) * (resid @ resid) / (n - p))
            with np.errstate(invalid="ignore", divide="ignore"):
                pval = 2 * scipy.stats.t.sf(np.abs(beta / se), df=n - p)
            for k, j in enumerate(use, start=1):
                res[terms[j]], res[f"{terms[j]}_se"], res[f"{terms[j]}_p"] = beta[k], se[k], pval[k]
        results[feat] = res
    effects = pd.DataFrame.from_dict(results, orient="index")
    for t in terms:
        p = effects[f"{t}_p"]
        effects[f"{t}_fdr"] = np.nan
        if p.notna().any():
            effects.loc[p.notna(), f"{t}_fdr"] = scipy.stats.false_discovery_control(p[p.notna()])
    return effects


def _print_plate_qc(wells, effects, z_threshold, anomaly_fraction, n_features):
    n_plates = wells.index.get_level_values("plate").nunique()
    flagged = wells[wells["flagged"]]
    counts = {
        "no organoids": (flagged["n_organoids"] == 0).sum(),
        "organoid count": (flagged["count_z"].abs() >= z_threshold).sum(),
        "stain dropout": (flagged["stain_dropout"] != "").sum(),
        f"deviating in >= {anomaly_fraction:.0%} of features": (flagged["deviating_fraction"] >= anomaly_fraction).sum(),
    }
    print(f"Plate QC: {len(wells)} wells on {n_plates} plates, {n_features} features. {len(flagged)} wells flagged ("
          + ", ".join(f"{n} {k}" for k, n in counts.items() if n) + "). See adata.uns['well_qc'].")
    not_assessed = ((wells["n_organoids"] > 0) & (wells["features_assessed"] == 0)).sum()
    if not_assessed:
        print(f"{not_assessed} wells not assessed per feature (too few organoids, or no other wells of their "
              f"condition on the plate).")
    flagged_controls = flagged[flagged["control"]]
    if len(flagged_controls):
        print(f"Flagged CONTROL wells (normalization relies on them): "
              f"{[f'{p} {w}' for p, w in flagged_controls.index]}")
    for (p, w), row in flagged.head(40).iterrows():
        print(f"  {p} {w} [{row['condition']}, n={row['n_organoids']}]: {row['reasons']}")
    if len(flagged) > 40:
        print(f"  ... {len(flagged) - 40} more")
    significant = {t: int((effects[f"{t}_fdr"] < 0.05).sum()) for t in ["edge", "row", "col"]}
    print(f"Position effects (FDR < 5%): border of the used area in {significant['edge']}, row gradient in "
          f"{significant['row']}, column gradient in {significant['col']} of {len(effects)} features. "
          f"See adata.uns['position_effects'].")


def _plot_plates(wells, values, shape, cmap, vmin, vmax, cbar_label, outline, outline_label, title):
    """One plate map per plate (small multiples on a shared color scale)."""
    plates = list(wells.index.get_level_values("plate").unique())
    n_rows, n_cols = shape
    ncols = min(len(plates), 2)  # wider figures are only scaled down in a notebook
    nrows = -(-len(plates) // ncols)
    panel_w, panel_h = 0.17 * n_cols + 0.5, 0.17 * n_rows + 0.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols + 0.9, panel_h * nrows + 0.7),
                             squeeze=False, layout="constrained")
    for ax, plate in zip(axes.flat, plates):
        image = _draw_plate(ax, wells.loc[plate], values.loc[plate], outline.eq(True).loc[plate],
                            shape, cmap, vmin, vmax)
        ax.set_title(plate, loc="left")
    for ax in axes.flat[len(plates):]:
        ax.set_visible(False)
    fig.colorbar(image, ax=axes, shrink=0.7, label=cbar_label)

    handles = [Rectangle((0, 0), 1, 1, fill=False, edgecolor=_INK, linewidth=1.2, label=outline_label)]
    if outline_label is None:  # no outlined wells on this map
        handles = []
    handles += [Line2D([], [], color=_GRAY, linewidth=0.6, label="condition border"),
               Rectangle((0, 0), 1, 1, fill=False, edgecolor=_GRAY_LIGHT, linewidth=0.8, label="no value")]
    if (wells["n_organoids"] == 0).any():
        handles.append(Line2D([], [], linestyle="", marker="x", color=_INK_SECONDARY, label="no organoids"))
    if wells["control"].any():
        handles.append(Line2D([], [], linestyle="", marker="o", markersize=3, color=_INK, label="control well"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False)
    fig.suptitle(title)
    plt.show()


def _draw_plate(ax, wells, values, outline, shape, cmap, vmin, vmax):
    """Heatmap of one plate. Unused wells are blank, used wells without a value hollow."""
    n_rows, n_cols = shape
    rows, cols = wells["row"].to_numpy(), wells["col"].to_numpy()
    values = values.reindex(wells.index).to_numpy(dtype=float)
    grid = np.full(shape, np.nan)
    grid[rows, cols] = values
    image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")

    # White gaps between wells
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="both", length=0)

    for r, c in zip(rows[np.isnan(values)], cols[np.isnan(values)]):
        ax.add_patch(Rectangle((c - 0.4, r - 0.4), 0.8, 0.8, fill=False, edgecolor=_GRAY_LIGHT, linewidth=0.8))

    condition = dict(zip(zip(rows, cols), wells["condition"]))
    segments = []
    for (r, c), cond in condition.items():
        right, below = condition.get((r, c + 1)), condition.get((r + 1, c))
        if cond != "unknown" and right not in (None, "unknown", cond):
            segments.append([(c + 0.5, r - 0.5), (c + 0.5, r + 0.5)])
        if cond != "unknown" and below not in (None, "unknown", cond):
            segments.append([(c - 0.5, r + 0.5), (c + 0.5, r + 0.5)])
    ax.add_collection(LineCollection(segments, colors=_GRAY, linewidths=0.6, zorder=3))

    outline = outline.reindex(wells.index).eq(True).to_numpy()
    for r, c in zip(rows[outline], cols[outline]):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor=_INK, linewidth=1.2, zorder=4))
    empty = wells["n_organoids"].to_numpy() == 0
    ax.scatter(cols[empty], rows[empty], marker="x", s=28, color=_INK_SECONDARY, linewidths=1, zorder=5)
    control = wells["control"].to_numpy(dtype=bool)
    ax.scatter(cols[control], rows[control], marker="o", s=6, color=_INK, linewidths=0, zorder=5)

    col_step = 1 if n_cols <= 24 else 2
    row_step = 1 if n_rows <= 16 else 2
    ax.set_xticks(np.arange(0, n_cols, col_step), [str(c + 1) for c in range(0, n_cols, col_step)])
    ax.set_yticks(np.arange(0, n_rows, row_step), [_row_name(r) for r in range(0, n_rows, row_step)])
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(_GRAY_LIGHT)
        spine.set_linewidth(0.6)
    return image


def _plot_position_effects(effects, top_n=20):
    """Estimates with 95% CI of the edge, row and column effects for the top_n features (lowest FDR)."""
    terms = [("edge", "Border of the used area vs. inner"), ("row", "Rows: first → last"),
             ("col", "Columns: first → last")]
    terms = [(t, title) for t, title in terms if effects[t].notna().any()]
    if not terms:
        return
    score = effects[[f"{t}_fdr" for t, _ in terms]].min(axis=1)
    df = effects.loc[score.nsmallest(top_n).index]
    y = np.arange(len(df))

    fig, axes = plt.subplots(1, len(terms), sharey=True, squeeze=False, layout="constrained",
                             figsize=(2.8 * len(terms) + 2, 1.3 + 0.2 * len(df)))
    for ax, (t, title) in zip(axes[0], terms):
        est, se = df[t].to_numpy(), df[f"{t}_se"].to_numpy()
        color = np.where(df[f"{t}_fdr"].to_numpy() < 0.05, _BLUE, _GRAY)
        ax.axvline(0, color=_GRAY_LIGHT, linewidth=0.8, zorder=1)
        ax.hlines(y, est - 1.96 * se, est + 1.96 * se, colors=list(color), linewidth=1.2, zorder=2)
        ax.scatter(est, y, s=16, c=list(color), edgecolors="white", linewidths=0.5, zorder=3)
        ax.grid(axis="x", color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Δ Cliff's delta")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0, 0].set_yticks(y, df.index)
    axes[0, 0].set_ylim(len(df) - 0.5, -0.5)

    handles = [Line2D([], [], marker="o", color=_BLUE, markeredgecolor="white", linewidth=1.2, label="FDR < 5%"),
               Line2D([], [], marker="o", color=_GRAY, markeredgecolor="white", linewidth=1.2, label="not significant"),
               ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"Position effects within conditions, {len(df)} features with the lowest FDR "
                 f"(pooled over plates; dot: estimate, line: 95% CI)")
    plt.show()


# ── Assay-level QC: saturation, positions in the well images, outlier organoids, label
#    consistency, phenotypic activity and batch effects ─────────────────────────────────────

_DEFAULT_CONDITIONS = ("Medium", "Cell_line", "ABs", "Other")
# Categorical colors in fixed order (e.g. one per plate)
_CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def check_saturation(adata, plate_col="Barcode", well_col="Well", ceiling=None, tolerance=0.01, max_fraction=0.01,
                     plate_format=None, plot=True):
    """
    Find staining channels with saturated (clipped) intensities.

    Channels are the prefixes of intensity feature pairs "<channel>_max" and "<channel>_mean". An
    organoid counts as saturated in a channel if its brightest pixel (<channel>_max) is within
    `tolerance` of the channel's ceiling. Without saturation hardly any organoid reaches the
    brightest value of a channel; with saturation many pile up there.

    In a flagged channel, the features of the brightest pixels (_max and _Q95 to _Q100, also their
    _T_ versions) mostly measure the ceiling. They are listed in adata.uns["saturated_features"],
    which select_dimred_features excludes. Means and other intensities of the channel are
    compressed, so differences between conditions shrink.

    Parameters:
    adata (anndata.AnnData): The AnnData object; adata.X holds the raw features.
    plate_col, well_col (str): Columns in adata.obs with the plate ID and the well name.
    ceiling (float, optional): The detector maximum, e.g. 65535 for 16-bit or 4095 for 12-bit
        images. None uses the highest value of each channel in the data, which also works when
        background subtraction or illumination correction shifted the maximum.
    tolerance (float): Relative distance to the ceiling that still counts as saturated.
    max_fraction (float): Share of saturated organoids above which a channel is flagged.
    plate_format (int, optional): Wells per plate for the plate maps; None infers it.
    plot (bool): Plot the saturated share per channel, and plate maps of the flagged channels.

    Returns:
    adata (anndata.AnnData): With adata.uns["saturation"] (one row per channel),
        adata.uns["saturation_by_well"] (saturated share per well and channel) and
        adata.uns["saturated_features"].
    """
    channels = [f[:-4] for f in adata.var_names
                if f.endswith("_max") and not f.endswith("_T_max") and f"{f[:-4]}_mean" in adata.var_names]
    if not channels:
        print("No intensity channels found (features '<channel>_max' with '<channel>_mean').")
        return adata
    X = _feature_frame(adata, [f"{c}_max" for c in channels])
    X.columns = channels
    top = X.max() if ceiling is None else pd.Series(float(ceiling), index=channels)
    saturated = X.ge((1 - tolerance) * top)
    measured = X.notna()

    table = pd.DataFrame({
        "ceiling": top,
        "saturated_fraction": saturated.sum() / measured.sum().where(measured.sum() > 0),
        "organoids_measured": measured.sum(),
    })
    table["flagged"] = table["saturated_fraction"] > max_fraction
    flagged = list(table.index[table["flagged"]])
    brightest = re.compile(r"(_max|_Q(9[5-9]|100))$")
    affected = [f for f in adata.var_names if brightest.search(f) and any(f.startswith(c + "_") for c in flagged)]

    by_well = [adata.obs[plate_col].astype(str).to_numpy(), adata.obs[well_col].astype(str).to_numpy()]
    per_well = saturated.groupby(by_well).sum() / measured.groupby(by_well).sum().replace(0, np.nan)
    per_well.index.names = ["plate", "well"]
    adata.uns["saturation"] = table
    adata.uns["saturation_by_well"] = per_well.reset_index()
    adata.uns["saturated_features"] = affected

    print(f"Saturation: {len(flagged)} of {len(channels)} channels with more than {max_fraction:.0%} of the "
          f"organoids within {tolerance:.0%} of the ceiling"
          + (": " + ", ".join(f"{c} ({table.at[c, 'saturated_fraction']:.1%})" for c in flagged) if flagged else "")
          + ".")
    if affected:
        print(f"{len(affected)} features of the brightest pixels are listed in adata.uns['saturated_features'] "
              f"(excluded by select_dimred_features): {affected}. Other intensity features of these channels are "
              f"compressed; consider a shorter exposure in future experiments.")

    if plot:
        _plot_saturation(table, max_fraction)
        if flagged:
            wells, shape = _map_wells(adata, plate_col, well_col, plate_format=plate_format)
            no_outline = pd.Series(False, index=wells.index)
            for c in flagged:
                _plot_plates(wells, per_well[c].reindex(wells.index), shape, _SEQUENTIAL, 0, 1,
                             "share of saturated organoids", no_outline, None, f"{c}: saturated organoids per well")
    return adata


def check_well_positions(adata, x_col="centroid_x_micrometer", y_col="centroid_y_micrometer",
                         box_cols=("x_micrometer", "y_micrometer", "len_x_micrometer", "len_y_micrometer"),
                         plate_col="Barcode", well_col="Well", features=None, grid=3, gap_p=0.001,
                         border_tolerance=1.0, min_organoids=10, min_rho=0.1, plot=True):
    """
    Check where the organoids lie within their well images.

    - Cut-off organoids: organoids whose bounding box touches the border of the well image are only
      partly imaged, so their size, shape and intensities are wrong. They are removed in
      2_Filtering (remove_cut_organoids); any left are marked in adata.obs["touches_image_border"].
    - Empty regions: each well image is divided into grid x grid tiles. An empty tile is suspicious
      (missing imaging field, stitching error, detached dome) if, given the well's organoid count
      and the plate's typical distribution of organoids over the tiles, at least -ln(gap_p)
      organoids were expected there, so that an empty tile has a probability below gap_p (Poisson).
      Cut-off organoids removed in 2_Filtering (adata.uns["removed_organoids"]) count here too,
      since they still show where organoids are.
    - Radial effects: organoids near the edge of the well or dome can differ biologically (oxygen,
      nutrients, matrix) or technically (illumination, focus). Per well, the Spearman correlation
      between each feature and the organoid's distance from the image center is computed; it is
      averaged over wells and tested against 0 (t-test over wells, Benjamini-Hochberg FDR).
      Because it compares organoids within wells, condition and plate differences do not matter.
      Cut-off organoids are left out, since truncation alone changes size and shape near the border.
      With many wells even tiny effects are significant, so effects count as relevant from
      |rho| >= min_rho.

    The image size per plate comes from adata.uns["image_size"], stored by remove_cut_organoids
    before the cut-off organoids were removed; otherwise it is the largest extent of the organoid
    bounding boxes. The image center is its middle; adata.obs["radial_position"] is the distance
    from the center (0) relative to half the image size (1 = middle of an image edge).

    Parameters:
    adata (anndata.AnnData): The AnnData object; adata.X holds the features.
    x_col, y_col (str): Columns in adata.obs with the organoid centroid in the well image (µm).
    box_cols (tuple of str): Columns in adata.obs with the bounding box origin x, y and size x, y (µm).
        If missing, cut-off organoids are not checked.
    plate_col, well_col (str): Columns in adata.obs with the plate ID and the well name.
    features (list of str, optional): Features tested for radial effects. None uses all.
    grid (int): Tiles per image side for the empty-region check.
    gap_p (float): Probability below which an empty tile is flagged.
    border_tolerance (float): Distance to the image border (µm) that still counts as touching.
    min_organoids (int): Wells with fewer organoids are skipped for the radial effects.
    min_rho (float): Smallest |rho| of a radial effect that counts as relevant.
    plot (bool): Plot the organoid positions per plate, wells with empty regions and radial effects.

    Returns:
    adata (anndata.AnnData): With adata.obs["touches_image_border"] and ["radial_position"],
        adata.uns["well_positions"] (one row per well: organoids incl. cut-off ones, share cut off,
        expected organoids in the emptiest tile) and adata.uns["radial_effects"] (one row per feature).
    """
    obs = adata.obs
    plate = obs[plate_col].astype(str).to_numpy()
    well = obs[well_col].astype(str).to_numpy()
    x = pd.to_numeric(obs[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(obs[y_col], errors="coerce").to_numpy(dtype=float)
    has_box = all(c in obs.columns for c in box_cols)
    if has_box:
        bx, by, bw, bh = (pd.to_numeric(obs[c], errors="coerce").to_numpy(dtype=float) for c in box_cols)
        right, bottom = bx + bw, by + bh
    else:
        right, bottom = x, y
        print(f"Bounding box columns {list(box_cols)} not found: cut-off organoids are not checked.")
    width, height = _image_size(adata, plate, right, bottom)

    touches = np.zeros(len(obs), dtype=bool)
    if has_box:
        touches = _border_contacts(bx, by, right, bottom, width, height, border_tolerance).any(axis=1).to_numpy()
    radial = np.hypot((x - width / 2) / (width / 2), (y - height / 2) / (height / 2))
    valid = np.isfinite(radial)

    # Empty regions: observed vs. expected organoids per tile. Cut-off organoids removed in
    # 2_Filtering still show where organoids are, so they count here.
    positions = pd.DataFrame({"plate": plate, "well": well, "x": x, "y": y, "width": width, "height": height,
                              "cut": touches})[valid]
    positions = pd.concat([positions, _removed_cut_positions(adata, plate_col, well_col, x_col, y_col)],
                          ignore_index=True)
    px, py, pw, ph = (positions[c].to_numpy(dtype=float) for c in ["x", "y", "width", "height"])
    pplate, pwell = positions["plate"].to_numpy(), positions["well"].to_numpy()
    tile_x = np.clip((px / pw * grid).astype(int), 0, grid - 1)
    tile_y = np.clip((py / ph * grid).astype(int), 0, grid - 1)
    counts = pd.crosstab([pplate, pwell], tile_y * grid + tile_x).reindex(columns=range(grid * grid), fill_value=0)
    counts.index.names = ["plate", "well"]
    plate_share = counts.groupby(level="plate").sum()
    plate_share = plate_share.div(plate_share.sum(axis=1), axis=0)
    expected = counts.sum(axis=1).to_numpy()[:, None] * plate_share.reindex(
        counts.index.get_level_values("plate")).to_numpy()
    empty_expected = np.where(counts.to_numpy() == 0, expected, 0)
    cutoff = -np.log(gap_p)

    border = pd.Series(positions["cut"].to_numpy(dtype=bool)).groupby([pplate, pwell]).mean()
    border.index.names = ["plate", "well"]
    wells = pd.DataFrame({
        "n_organoids": counts.sum(axis=1),
        "border_fraction": border.reindex(counts.index),
        "empty_region_expected": empty_expected.max(axis=1),
    }, index=counts.index)
    wells["flagged"] = wells["empty_region_expected"] >= cutoff

    use = valid & ~touches
    effects = _radial_effects(adata, features, plate[use], well[use], radial[use], use, min_organoids)

    adata.obs["touches_image_border"] = touches
    adata.obs["radial_position"] = radial
    adata.uns["well_positions"] = wells.reset_index()
    adata.uns["radial_effects"] = effects

    if has_box:
        print(f"Cut-off organoids: {touches.sum()} ({touches.mean():.1%}) touch the border of their well image"
              + (" (adata.obs['touches_image_border']); they are left out of the radial effects. Remove them with "
                 "remove_cut_organoids in 2_Filtering." if touches.any() else "."))
    flagged = wells[wells["flagged"]]
    print(f"Empty regions: {len(flagged)} of {len(wells)} wells have an empty tile where >= {cutoff:.1f} organoids "
          f"were expected (p < {gap_p:g})"
          + (": " + ", ".join(f"{p} {w}" for p, w in flagged.index[:30]) if len(flagged) else "") + ".")
    relevant = effects[(effects["fdr"] < 0.05) & (effects["rho"].abs() >= min_rho)]
    relevant = relevant.sort_values("rho", key=np.abs, ascending=False)
    print(f"Radial effects (FDR < 5% and |rho| >= {min_rho:g}, cut-off organoids excluded): {len(relevant)} of "
          f"{len(effects)} features change with the distance from the image center"
          + (", strongest: " + ", ".join(f"{f} (rho={r:+.2f})" for f, r in relevant["rho"].head(5).items())
             if len(relevant) else "") + ". See adata.uns['radial_effects'].")

    if plot:
        _plot_position_density(px, py, pplate, pw, ph, grid)
        if len(flagged):
            _plot_empty_regions(px, py, pplate, pwell, pw, ph, grid, flagged, counts, expected, cutoff)
        top = effects.dropna(subset=["rho"]).sort_values("rho", key=np.abs, ascending=False).head(20)
        if len(top):
            _plot_estimates(top.index, top["rho"], top["low"], top["high"],
                            (top["fdr"] < 0.05) & (top["rho"].abs() >= min_rho),
                            f"FDR < 5% and |rho| >= {min_rho:g}", "not relevant",
                            "Spearman correlation with the distance from the center",
                            f"Radial effects within wells, {len(top)} strongest features (dot: mean over wells, "
                            f"line: 95% CI)")
    return adata


def _image_size(adata, plate, right, bottom):
    """
    Width and height of each organoid's well image: from adata.uns["image_size"] (stored by
    remove_cut_organoids) where available, otherwise the largest extent of the bounding boxes
    (right, bottom) on the plate.
    """
    width = pd.Series(right).groupby(plate).transform("max").to_numpy()
    height = pd.Series(bottom).groupby(plate).transform("max").to_numpy()
    stored = adata.uns.get("image_size")
    if stored is not None:
        stored = pd.DataFrame(stored).astype({"plate": str}).set_index("plate")
        known = np.isin(plate, stored.index)
        width[known] = stored["width"].reindex(plate[known]).to_numpy()
        height[known] = stored["height"].reindex(plate[known]).to_numpy()
    return width, height


def _removed_cut_positions(adata, plate_col, well_col, x_col, y_col):
    """
    Positions of the cut-off organoids removed by remove_cut_organoids (adata.uns["removed_organoids"])
    in wells that still have organoids in adata, or None.
    """
    removed = adata.uns.get("removed_organoids")
    if removed is None or not {plate_col, well_col, x_col, y_col, "step"} <= set(removed.columns):
        return None
    cut = removed[removed["step"] == "remove_cut_organoids"]
    plate, well = cut[plate_col].astype(str).to_numpy(), cut[well_col].astype(str).to_numpy()
    x = pd.to_numeric(cut[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(cut[y_col], errors="coerce").to_numpy(dtype=float)
    width, height = _image_size(adata, plate, x, y)
    present = pd.MultiIndex.from_arrays([adata.obs[plate_col].astype(str), adata.obs[well_col].astype(str)])
    keep = (pd.MultiIndex.from_arrays([plate, well]).isin(present)
            & np.isfinite(x) & np.isfinite(y) & np.isfinite(width) & np.isfinite(height))
    if not keep.any():
        return None
    return pd.DataFrame({"plate": plate, "well": well, "x": x, "y": y, "width": width, "height": height,
                         "cut": True})[keep]


def _border_contacts(bx, by, right, bottom, width, height, tolerance):
    """Whether each bounding box touches the left, top, right and bottom border of its image."""
    with np.errstate(invalid="ignore"):
        return pd.DataFrame({"left": bx <= tolerance, "top": by <= tolerance,
                             "right": right >= width - tolerance, "bottom": bottom >= height - tolerance})


def check_label_consistency(adata, condition_cols=_DEFAULT_CONDITIONS, plate_col="Barcode", well_col="Well",
                            control=None, features=None, min_wells=3, min_organoids=5, z_threshold=3.5,
                            distance_threshold=2.0, variance=0.9, plate_format=None, plot=True):
    """
    Find wells whose profile matches another condition better than their own (e.g. swapped wells,
    pipetting or layout errors).

    Each well's profile is the average rank-based normal score of its organoids in every feature
    (scores within the plate, robust to outliers and heterogeneous organoids), reduced to the
    principal components covering `variance` of the variance. Per plate, a well is compared with
    the centroid of the other wells of its condition (distance d_own) and with the centroids of all
    other conditions (closest: d_other, reported as the best match). Two robust z-scores within the
    plate (median/MAD) decide:
    - match_z, of log(d_own / d_other): how much more the well resembles another condition than
      its own; flagged from z_threshold.
    - distance_z, of log(d_own): how atypical the well is for its condition; flagged from
      distance_threshold. This second criterion keeps wells of conditions that are simply alike
      (e.g. a treatment without effect next to the control) from being flagged.
    Wells with fewer than min_organoids organoids are skipped, since their profiles are noisy.

    Parameters:
    adata (anndata.AnnData): The AnnData object; adata.X holds the features.
    condition_cols (list of str): Columns in adata.obs that together define a condition.
    plate_col, well_col (str): Columns in adata.obs with the plate ID and the well name.
    control (dict, optional): Control cells as in normalize_groups, only to mark control wells.
    features (list of str, optional): Features to use. None uses all features without missing values.
    min_wells (int): Conditions need this many wells on a plate for their wells to be checked.
    min_organoids (int): Wells with fewer organoids are not checked and not used for centroids.
    z_threshold (float): match_z from which a well is flagged.
    distance_threshold (float): distance_z a flagged well needs at least.
    variance (float): Share of the variance kept by the principal components.
    plate_format (int, optional): Wells per plate for the plate map; None infers it.
    plot (bool): Plot plate maps of the distance to the own condition.

    Returns:
    adata (anndata.AnnData): With adata.uns["label_consistency"] (one row per well).
    """
    cols = [c for c in condition_cols if c in adata.obs.columns]
    profiles, info = _well_profiles(adata, features, cols, plate_col, well_col, control)
    n_organoids = pd.Series(1, index=adata.obs_names).groupby(
        [adata.obs[plate_col].astype(str).to_numpy(), adata.obs[well_col].astype(str).to_numpy()]).sum()
    enough = (n_organoids.to_numpy() >= min_organoids)
    profiles, info = profiles[enough], info[enough]
    M = profiles.to_numpy()
    plates = info.index.get_level_values("plate")
    for p in plates.unique():
        rows = np.asarray(plates == p)
        M[rows] -= M[rows].mean(axis=0)
    coords = M @ _pca_basis(M, variance).T

    distance_own = np.full(len(M), np.nan)
    distance_other = np.full(len(M), np.nan)
    closest = np.full(len(M), "", dtype=object)
    for p in plates.unique():
        idx = np.flatnonzero(np.asarray(plates == p))
        labels = info["condition"].to_numpy()[idx]
        names, inverse, n = np.unique(labels, return_inverse=True, return_counts=True)
        sums = np.zeros((len(names), coords.shape[1]))
        np.add.at(sums, inverse, coords[idx])
        centroids = sums / n[:, None]
        for j, i in enumerate(idx):
            own = inverse[j]
            if n[own] < min_wells:
                continue
            loo = (sums[own] - coords[i]) / (n[own] - 1)
            others = [k for k in range(len(names)) if k != own and n[k] >= 2]
            distance_own[i] = np.linalg.norm(coords[i] - loo)
            if others:
                d = np.linalg.norm(centroids[others] - coords[i], axis=1)
                distance_other[i] = d.min()
                closest[i] = names[others[int(np.argmin(d))]]

    with np.errstate(invalid="ignore", divide="ignore"):
        distance_z = _robust_z_by_plate(pd.Series(np.log(distance_own), index=info.index))
        match_z = _robust_z_by_plate(pd.Series(np.log(distance_own / distance_other), index=info.index))
    table = info.assign(match_z=match_z.to_numpy(), distance_z=distance_z.to_numpy(),
                        closest_other_condition=closest, closer_to_other=distance_other < distance_own)
    table["flagged"] = ((table["match_z"] >= z_threshold) & (table["distance_z"] >= distance_threshold)
                        & table["closer_to_other"])
    adata.uns["label_consistency"] = table.reset_index()

    flagged = table[table["flagged"]]
    checked = table["distance_z"].notna().sum()
    print(f"Label consistency: {len(flagged)} of {checked} checked wells look more like another condition than "
          f"their own (match z >= {z_threshold:g}, distance z >= {distance_threshold:g}). "
          f"See adata.uns['label_consistency'].")
    for (p, w), row in flagged.iterrows():
        print(f"  {p} {w} [{row['condition']}] looks like [{row['closest_other_condition']}] "
              f"(match z={row['match_z']:+.1f}, distance z={row['distance_z']:+.1f})")

    if plot:
        wells, shape = _map_wells(adata, plate_col, well_col, cols, control, plate_format)
        lim = z_threshold + 1.5
        _plot_plates(wells, table["distance_z"].reindex(wells.index), shape, _DIVERGING, -lim, lim,
                     "z (distance to own condition)", table["flagged"].reindex(wells.index), "closer to another condition",
                     "Distance of each well to the other wells of its condition")
    return adata


def phenotypic_activity(adata, control, condition_cols=_DEFAULT_CONDITIONS, plate_col="Barcode", well_col="Well",
                        features=None, positive_control=None, layer="transformed", n_permutations=1000,
                        variance=0.9, seed=0, plot=True):
    """
    Which conditions differ reproducibly from the control, and (with a positive control) how large
    the assay window is.

    Phenotypic activity (mAP): each well's profile is the average rank-based normal score of its
    organoids in every feature (scores within the plate, robust to outliers and heterogeneous
    organoids), relative to the control wells of its plate and reduced to the principal components
    covering `variance` of the variance. For each well of a condition, the other wells of that
    condition and the control wells on the same plate are ranked by profile similarity (cosine);
    the average precision (AP) measures how well the condition's own wells rank on top. The mean AP
    over all wells of a condition (mAP; 1 = always on top) is compared with a permutation null
    (condition and control labels shuffled within each plate); p-values are FDR corrected over
    conditions. This is the mAP framework of image-based profiling (Kalinin et al., 2024).

    Assay window (only with positive_control): per feature and plate, the Z'-factor
    1 - 3 (SD_pos + SD_neg) / |mean_pos - mean_neg| and the SSMD
    (mean_pos - mean_neg) / sqrt(SD_pos^2 + SD_neg^2) between positive and negative control wells,
    using well values (20% trimmed means of the organoids in `layer`), median over plates.
    Z' > 0.5 is an excellent, 0 to 0.5 a usable window for phenotypic assays; |SSMD| >= 3 is a
    strong effect.

    Parameters:
    adata (anndata.AnnData): The AnnData object.
    control (dict): Negative control cells as in normalize_groups.
    condition_cols (list of str): Columns in adata.obs that together define a condition.
    plate_col, well_col (str): Columns in adata.obs with the plate ID and the well name.
    features (list of str, optional): Features to use. None uses all features without missing values.
    positive_control (dict, optional): Positive control cells, in the same format as control.
    layer (str): Layer for the assay window; adata.X if not present.
    n_permutations (int): Permutations for the mAP p-values.
    variance (float): Share of the variance kept by the principal components.
    seed (int): Random seed for the permutations.
    plot (bool): Plot the mAP per condition, and the assay window of the strongest features.

    Returns:
    adata (anndata.AnnData): With adata.uns["phenotypic_activity"] (one row per condition) and,
        with a positive control, adata.uns["assay_window"] (one row per feature).
    """
    cols = [c for c in condition_cols if c in adata.obs.columns]
    profiles, info = _well_profiles(adata, features, cols, plate_col, well_col, control)
    M = profiles.to_numpy()
    plates = info.index.get_level_values("plate")
    is_control = info["control"].to_numpy()
    for p in plates.unique():
        rows = np.asarray(plates == p)
        M[rows] = M[rows] - M[rows & is_control].mean(axis=0) if (rows & is_control).any() else np.nan
    usable = ~np.isnan(M).any(axis=1)
    coords = np.full((len(M), 0), np.nan)
    if usable.any():
        coords = M @ _pca_basis(M[usable], variance).T
    norms = np.linalg.norm(coords, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = coords / norms

    rng = np.random.default_rng(seed)
    results = {}
    labels = info["condition"].to_numpy()
    for cond in sorted(set(labels[~is_control])):
        obs_sum, obs_n = 0.0, 0
        null_sum, null_n = np.zeros(n_permutations), np.zeros(n_permutations)
        n_wells, n_plates = 0, 0
        for p in plates.unique():
            on_plate = np.asarray(plates == p) & usable
            cond_idx = np.flatnonzero(on_plate & (labels == cond) & ~is_control)
            ctrl_idx = np.flatnonzero(on_plate & is_control)
            if len(cond_idx) < 2 or len(ctrl_idx) < 1:
                continue
            idx = np.r_[cond_idx, ctrl_idx]
            similarity = unit[idx] @ unit[idx].T
            is_cond = np.r_[np.ones(len(cond_idx), dtype=bool), np.zeros(len(ctrl_idx), dtype=bool)]
            ap = _average_precisions(similarity, is_cond[None, :])
            ap_null = _average_precisions(similarity, rng.permuted(np.tile(is_cond, (n_permutations, 1)), axis=1))
            obs_sum += np.nansum(ap)
            obs_n += np.sum(~np.isnan(ap))
            null_sum += np.nansum(ap_null, axis=1)
            null_n += np.sum(~np.isnan(ap_null), axis=1)
            n_wells += len(cond_idx)
            n_plates += 1
        if obs_n == 0:
            continue
        observed, null = obs_sum / obs_n, null_sum / null_n
        results[cond] = {"mAP": observed, "chance_mAP": null.mean(), "chance_95": np.quantile(null, 0.95),
                         "p": (1 + np.sum(null >= observed)) / (1 + n_permutations),
                         "n_wells": n_wells, "n_plates": n_plates}
    activity = pd.DataFrame.from_dict(results, orient="index")
    if len(activity):
        activity["fdr"] = scipy.stats.false_discovery_control(activity["p"])
        activity["active"] = activity["fdr"] < 0.05
        activity = activity.sort_values("mAP", ascending=False)
    adata.uns["phenotypic_activity"] = activity
    print(f"Phenotypic activity: {int(activity['active'].sum()) if len(activity) else 0} of {len(activity)} "
          f"conditions differ reproducibly from the control (mAP, FDR < 5%). See adata.uns['phenotypic_activity'].")

    window = None
    if positive_control is not None:
        window = _assay_window(adata, control, positive_control, features, layer, plate_col, well_col)
        adata.uns["assay_window"] = window
        best = window.reindex(window["ssmd"].abs().sort_values(ascending=False).index).head(10)
        print(f"Assay window (positive vs. negative control, median over plates), strongest features: "
              + ", ".join(f"{f} (SSMD={r.ssmd:+.1f}, Z'={r.z_prime:.2f})" for f, r in best.iterrows())
              + ". See adata.uns['assay_window'].")

    if plot and len(activity):
        _plot_activity(activity)
    if plot and window is not None and len(window):
        top = window.reindex(window["ssmd"].abs().sort_values(ascending=False).index).head(20)
        _plot_estimates(top.index, top["ssmd"], top["ssmd_min"], top["ssmd_max"], top["ssmd"].abs() >= 3,
                        "|SSMD| >= 3", "|SSMD| < 3", "SSMD (positive vs. negative control)",
                        f"Assay window, {len(top)} features with the largest |SSMD| (dot: median over plates, "
                        f"line: range over plates)")
    return adata


def check_batch_effects(adata, control, condition_cols=_DEFAULT_CONDITIONS, plate_col="Barcode", well_col="Well",
                        before_layer="transformed", after_layer="z_scaled", features=None, plot=True):
    """
    Measure plate (batch) effects before and after normalization, for plates that are true replicates.

    Plates are only compared if their control wells have exactly the same condition (all
    condition_cols, e.g. same Medium, Cell_line, ABs and Other); otherwise plate and condition
    differences cannot be separated. Within such a plate group, every condition present on at least
    two of its plates is used. Per feature, the plate effect is the share of the variance between
    the condition's wells that comes from the plate (eta squared; well values are 20% trimmed means
    of the organoids), averaged over conditions.
    - Control: only "before" is informative, since normalization sets the control mean and spread
      of every plate to the same values by construction.
    - Other conditions shared by the plates: "before" vs. "after" shows whether normalizing to the
      control also removed the plate effect from them. A remaining effect means that the plates
      differ in a way that is not the same for all conditions.

    Parameters:
    adata (anndata.AnnData): The AnnData object after normalize_groups.
    control (dict): Control cells as in normalize_groups.
    condition_cols (list of str): Columns in adata.obs that together define a condition.
    plate_col, well_col (str): Columns in adata.obs with the plate ID and the well name.
    before_layer, after_layer (str): Layers before and after the per-plate scaling; adata.X is used
        if before_layer is missing.
    features (list of str, optional): Features to use. None uses all without missing values.
    plot (bool): Plot the plate effects across features and a PCA of the wells.

    Returns:
    adata (anndata.AnnData): With adata.uns["batch_effects"] (one row per feature and plate group),
        if any plates are comparable.
    """
    obs = adata.obs
    cols = [c for c in condition_cols if c in obs.columns]
    label = _condition_labels(obs, cols)
    plate = obs[plate_col].astype(str).to_numpy()
    well = obs[well_col].astype(str).to_numpy()
    is_control = _control_mask(obs, control)

    control_labels = pd.Series(label[is_control]).groupby(plate[is_control]).unique()
    groups = {}
    for p, labels in control_labels.items():
        if len(labels) == 1:
            groups.setdefault(labels[0], []).append(p)
    comparable = {k: v for k, v in groups.items() if len(v) >= 2}
    if not comparable:
        print(f"Batch effects: no plates whose control wells have exactly the same condition ({', '.join(cols)}), "
              f"so plate and condition differences cannot be separated; skipped. Control condition per plate:")
        for p, labels in control_labels.items():
            print(f"  {p}: {', '.join(labels)}")
        return adata
    if after_layer not in adata.layers:
        raise ValueError(f"Layer '{after_layer}' not found; run normalize_groups first.")
    if before_layer not in adata.layers:
        print(f"Layer '{before_layer}' not found; using adata.X as 'before'.")
        before_layer = None

    results = []
    for ref, group_plates in comparable.items():
        in_group = np.isin(plate, group_plates)
        plates_per_label = pd.Series(plate[in_group]).groupby(label[in_group]).nunique()
        shared = plates_per_label.index[plates_per_label >= 2]
        rows = in_group & np.isin(label, shared)
        before = _feature_frame(adata, features, layer=before_layer)[rows].reset_index(drop=True)
        after = _feature_frame(adata, features, layer=after_layer)[rows].reset_index(drop=True)
        complete = before.notna().all() & after.notna().all()
        before, after = before.loc[:, complete], after.loc[:, complete]
        by_well = [plate[rows], well[rows]]
        means_before = _trimmed_well_means(before, by_well)
        means_after = _trimmed_well_means(after, by_well)
        well_label = pd.Series(label[rows]).groupby(by_well).first()
        well_label.index.names = ["plate", "well"]

        eta = {"control_before": None, "others_before": [], "others_after": [], "weights": []}
        for cond in shared:
            sel = (well_label == cond).to_numpy()
            plates_sel = well_label.index.get_level_values("plate")[sel]
            e_before = _plate_eta_squared(means_before[sel], plates_sel)
            if cond == ref:
                eta["control_before"] = e_before
            else:
                eta["others_before"].append(e_before)
                eta["others_after"].append(_plate_eta_squared(means_after[sel], plates_sel))
                eta["weights"].append(sel.sum())
        table = pd.DataFrame({"control_before": eta["control_before"]})
        if eta["weights"]:
            w = np.array(eta["weights"], dtype=float)
            table["others_before"] = (pd.concat(eta["others_before"], axis=1) * w).sum(axis=1) / w.sum()
            table["others_after"] = (pd.concat(eta["others_after"], axis=1) * w).sum(axis=1) / w.sum()
        else:
            table["others_before"] = table["others_after"] = np.nan
        table["plates"] = ", ".join(group_plates)
        table["n_shared_conditions"] = len(shared)
        results.append(table.rename_axis("feature").reset_index())

        print(f"Batch effects for plates {', '.join(group_plates)} (control: {ref}), {len(shared)} shared conditions, "
              f"{complete.sum()} features. Median share of well variance from the plate: control before "
              f"normalization {table['control_before'].median():.2f}"
              + (f"; other conditions before {table['others_before'].median():.2f}, after "
                 f"{table['others_after'].median():.2f}" if eta["weights"] else "; no other shared conditions") + ".")
        if plot:
            _plot_batch_effects(table, means_after, well_label, ref, group_plates)

    adata.uns["batch_effects"] = pd.concat(results, ignore_index=True)
    return adata


def _robust_z_by_plate(values):
    """Robust z-score (median/MAD) within each plate (first index level)."""
    by_plate = values.index.get_level_values(0)
    centered = values - values.groupby(by_plate).transform("median")
    spread = centered.abs().groupby(by_plate).transform("median") * 1.4826
    return centered / spread.where(spread > 0)


def _feature_frame(adata, features=None, layer=None, complete=False):
    """Dense float64 DataFrame (RangeIndex) of the features; complete=True keeps features without NaN."""
    features = list(adata.var_names) if features is None else list(features)
    unknown = [f for f in features if f not in adata.var_names]
    if unknown:
        raise ValueError(f"Feature(s) not found in adata.var_names: {unknown}")
    sub = adata[:, features]
    X = sub.layers[layer] if layer is not None else sub.X
    X = pd.DataFrame(np.asarray(X.toarray() if scipy.sparse.issparse(X) else X, dtype=np.float64), columns=features)
    if complete:
        X = X.loc[:, X.notna().all().to_numpy()]
        if X.shape[1] == 0:
            raise ValueError("No feature without missing values.")
    return X


def _normal_scores(X, groups):
    """Rank-based inverse normal scores of each column within each group; NaN stays NaN."""
    ranks = X.groupby(groups).rank(method="average")
    n = X.notna().groupby(groups).transform("sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        return pd.DataFrame(scipy.stats.norm.ppf((ranks - 0.5) / n), index=X.index, columns=X.columns)


def _pca_basis(M, variance=0.9, max_components=30):
    """Principal axes (rows) of the centered rows of M that cover `variance` of the variance."""
    _, s, Vt = np.linalg.svd(M - M.mean(axis=0), full_matrices=False)
    explained = np.cumsum(s ** 2) / np.sum(s ** 2)
    return Vt[:min(int(np.searchsorted(explained, variance)) + 1, max_components, len(s))]


def _pca_scores(M, variance=0.9, max_components=30):
    """Principal component scores of the rows of M covering `variance` of the variance."""
    return (M - M.mean(axis=0)) @ _pca_basis(M, variance, max_components).T


def _well_profiles(adata, features, condition_cols, plate_col, well_col, control):
    """
    Well profiles (average rank-based normal score within the plate per feature; index (plate,
    well)) and per well its condition and whether it holds control organoids.
    """
    X = _feature_frame(adata, features, complete=True)
    plate = adata.obs[plate_col].astype(str).to_numpy()
    well = adata.obs[well_col].astype(str).to_numpy()
    profiles = _normal_scores(X, plate).groupby([plate, well]).mean()
    is_control = _control_mask(adata.obs, control) if control is not None else np.zeros(len(X), dtype=bool)
    info = pd.DataFrame({"condition": _condition_labels(adata.obs, condition_cols), "control": is_control})
    info = info.groupby([plate, well]).agg(condition=("condition", "first"), control=("control", "any"))
    profiles.index.names = info.index.names = ["plate", "well"]
    return profiles, info


def _map_wells(adata, plate_col, well_col, condition_cols=_DEFAULT_CONDITIONS, control=None, plate_format=None):
    """Wells (index (plate, well)) with condition, organoid count and position, and the plate shape."""
    plate = adata.obs[plate_col].astype(str).to_numpy()
    well = adata.obs[well_col].astype(str).to_numpy()
    condition = _condition_labels(adata.obs, [c for c in condition_cols if c in adata.obs.columns])
    wells = _well_table(adata, plate, well, condition, plate_col, well_col, control, warn=False)
    geometry, shape = _well_geometry(wells.index, plate_format)
    return wells.join(geometry[["row", "col"]]), shape


def _trimmed_well_means(X, by_well, proportion=0.2):
    """Mean per well (index (plate, well)) without the lowest and highest `proportion` of each feature."""
    keys = pd.MultiIndex.from_arrays(by_well)
    grouped = X.groupby(by_well)
    lower = grouped.quantile(proportion).reindex(keys).to_numpy()
    upper = grouped.quantile(1 - proportion).reindex(keys).to_numpy()
    values = X.to_numpy()
    means = X.where((values >= lower) & (values <= upper)).groupby(by_well).mean()
    means.index.names = ["plate", "well"]
    return means


def _plate_eta_squared(means, plates):
    """Share of the variance between wells (rows) explained by the plate, per feature."""
    centered = means - means.mean()
    between = (means.groupby(np.asarray(plates)).transform("mean") - means.mean()) ** 2
    total = (centered ** 2).sum()
    return between.sum() / total.where(total > 0)


def _average_precisions(similarity, labels):
    """
    Average precision of each item (query) for retrieving the other items of its label (True)
    among all other items ranked by similarity, for every labeling (rows of `labels`).
    Returns an array shaped like `labels`, NaN for items labeled False.
    """
    n = similarity.shape[0]
    out = np.full(labels.shape, np.nan)
    ranks = np.arange(1, n)
    for i in range(n):
        others = np.delete(np.arange(n), i)
        order = others[np.argsort(-similarity[i, others], kind="stable")]
        hits = labels[:, order].astype(float)
        n_pos = hits.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            ap = (np.cumsum(hits, axis=1) / ranks * hits).sum(axis=1) / n_pos
        out[:, i] = np.where(labels[:, i] & (n_pos > 0), ap, np.nan)
    return out


def _assay_window(adata, control, positive_control, features, layer, plate_col, well_col):
    """Z'-factor and SSMD per feature between positive and negative control wells (median over plates)."""
    X = _feature_frame(adata, features, layer=layer if layer in adata.layers else None)
    plate = adata.obs[plate_col].astype(str).to_numpy()
    by_well = [plate, adata.obs[well_col].astype(str).to_numpy()]
    is_neg = pd.Series(_control_mask(adata.obs, control))
    is_pos = pd.Series(_control_mask(adata.obs, positive_control))
    keep = (is_neg | is_pos).to_numpy()
    means = _trimmed_well_means(X[keep], [k[keep] for k in by_well])
    neg = is_neg[keep].groupby([k[keep] for k in by_well]).any().to_numpy()
    pos = is_pos[keep].groupby([k[keep] for k in by_well]).any().to_numpy()
    plates = means.index.get_level_values("plate")
    z_prime, ssmd = [], []
    for p in plates.unique():
        on_plate = np.asarray(plates == p)
        a, b = means[on_plate & pos], means[on_plate & neg]
        if len(a) < 2 or len(b) < 2:
            continue
        diff = a.mean() - b.mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            z_prime.append(1 - 3 * (a.std() + b.std()) / diff.abs())
            ssmd.append(diff / np.sqrt(a.var() + b.var()))
    if not ssmd:
        print("Assay window: no plate with at least 2 positive and 2 negative control wells.")
        return pd.DataFrame(columns=["z_prime", "ssmd", "ssmd_min", "ssmd_max", "n_plates"])
    ssmd = pd.concat(ssmd, axis=1)
    return pd.DataFrame({"z_prime": pd.concat(z_prime, axis=1).median(axis=1), "ssmd": ssmd.median(axis=1),
                         "ssmd_min": ssmd.min(axis=1), "ssmd_max": ssmd.max(axis=1),
                         "n_plates": ssmd.notna().sum(axis=1)})


def _radial_effects(adata, features, plate, well, radial, valid, min_organoids):
    """Per feature: mean over wells of the within-well Spearman correlation with the radial position."""
    X = _feature_frame(adata, features)[valid].reset_index(drop=True)
    keys = [plate, well]
    feature_ranks = X.groupby(keys).rank()
    radial_ranks = pd.Series(radial).groupby(keys).rank()
    f = feature_ranks - feature_ranks.groupby(keys).transform("mean")
    r = radial_ranks - radial_ranks.groupby(keys).transform("mean")
    covariance = f.mul(r, axis=0).groupby(keys).sum(min_count=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = covariance / np.sqrt((f ** 2).groupby(keys).sum(min_count=1).mul((r ** 2).groupby(keys).sum(), axis=0))
    rho = rho.where(X.notna().groupby(keys).sum() >= min_organoids)
    z = np.arctanh(rho.clip(-0.999, 0.999))
    n = z.notna().sum()
    mean, se = z.mean(), z.std() / np.sqrt(n)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = mean / se
    df = (n - 1).where(n > 1)
    p = pd.Series(2 * scipy.stats.t.sf(np.abs(t), df), index=t.index)
    half = scipy.stats.t.ppf(0.975, df) * se
    effects = pd.DataFrame({"rho": np.tanh(mean), "low": np.tanh(mean - half), "high": np.tanh(mean + half),
                            "n_wells": n, "p": p})
    effects["fdr"] = np.nan
    ok = effects["p"].notna()
    if ok.any():
        effects.loc[ok, "fdr"] = scipy.stats.false_discovery_control(effects.loc[ok, "p"])
    return effects


def _plot_estimates(names, est, low, high, highlight, highlight_label, other_label, xlabel, title):
    """Dot plot of estimates with intervals, one row per name; highlighted rows in blue."""
    names, est, low, high = list(names), np.asarray(est), np.asarray(low), np.asarray(high)
    color = np.where(np.asarray(highlight), _BLUE, _GRAY)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6, 1.3 + 0.2 * len(names)), layout="constrained")
    ax.axvline(0, color=_GRAY_LIGHT, linewidth=0.8, zorder=1)
    ax.hlines(y, low, high, colors=list(color), linewidth=1.2, zorder=2)
    ax.scatter(est, y, s=18, c=list(color), edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_yticks(y, names)
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    handles = [Line2D([], [], marker="o", color=_BLUE, markeredgecolor="white", linewidth=1.2, label=highlight_label),
               Line2D([], [], marker="o", color=_GRAY, markeredgecolor="white", linewidth=1.2, label=other_label)]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(title)
    plt.show()


def _plot_saturation(table, max_fraction):
    df = table.sort_values("saturated_fraction")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5, 1.2 + 0.28 * len(df)), layout="constrained")
    ax.barh(y, df["saturated_fraction"] * 100, height=0.6, color=list(np.where(df["flagged"], _BLUE, _GRAY)))
    ax.axvline(max_fraction * 100, color=_INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)))
    for yi, v in zip(y, df["saturated_fraction"]):
        ax.text(v * 100, yi, f" {v:.1%}", va="center", color=_INK_SECONDARY)
    ax.set_yticks(y, df.index)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Organoids whose brightest pixel is at the ceiling (%)")
    handles = [Patch(color=_BLUE, label="flagged"), Patch(color=_GRAY, label="not flagged"),
               Line2D([], [], color=_INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 2)),
                      label=f"threshold ({max_fraction:.0%})")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Saturation per channel")
    plt.show()


def _plot_position_density(x, y, plate, width, height, grid):
    """Organoid positions in the well images, all wells of a plate pooled, with the tile grid."""
    plates = list(pd.unique(plate))
    ncols = min(len(plates), 2)
    nrows = -(-len(plates) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols + 0.8, 3.3 * nrows + 0.5), squeeze=False,
                             layout="constrained")
    for ax, p in zip(axes.flat, plates):
        m = (plate == p) & np.isfinite(x) & np.isfinite(y)
        w, h = width[m].max(), height[m].max()
        counts, _, _ = np.histogram2d(y[m], x[m], bins=40, range=[[0, h], [0, w]])
        image = ax.imshow(np.ma.masked_equal(counts, 0), extent=(0, w, h, 0), cmap=_SEQUENTIAL,
                          interpolation="nearest")
        for k in range(1, grid):
            ax.axvline(w * k / grid, color=_GRAY, linewidth=0.6)
            ax.axhline(h * k / grid, color=_GRAY, linewidth=0.6)
        ax.set_title(p, loc="left")
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        fig.colorbar(image, ax=ax, shrink=0.8, label="organoids")
    for ax in axes.flat[len(plates):]:
        ax.set_visible(False)
    fig.suptitle("Organoid positions in the well images, all wells of a plate pooled (lines: tiles)")
    plt.show()


def _plot_empty_regions(x, y, plate, well, width, height, grid, flagged, counts, expected, cutoff, max_wells=12):
    """Organoid positions of wells with an unexpectedly empty tile (shaded)."""
    show = list(flagged.index[:max_wells])
    ncols = min(len(show), 4)
    nrows = -(-len(show) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(max(2.4 * ncols + 0.4, 6), 2.4 * nrows + 0.6), squeeze=False,
                             layout="constrained")
    row_of = {key: i for i, key in enumerate(counts.index)}
    for ax, (p, w) in zip(axes.flat, show):
        m = (plate == p) & (well == w)
        on_plate = plate == p
        W, H = width[on_plate].max(), height[on_plate].max()
        i = row_of[(p, w)]
        for t in np.flatnonzero((counts.iloc[i].to_numpy() == 0) & (expected[i] >= cutoff)):
            ty, tx = divmod(t, grid)
            ax.add_patch(Rectangle((tx * W / grid, ty * H / grid), W / grid, H / grid, color=_GRID, zorder=0))
        ax.scatter(x[m], y[m], s=8, color=_BLUE, edgecolors="white", linewidths=0.4, zorder=2)
        for k in range(1, grid):
            ax.axvline(W * k / grid, color=_GRAY_LIGHT, linewidth=0.6)
            ax.axhline(H * k / grid, color=_GRAY_LIGHT, linewidth=0.6)
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(_GRAY_LIGHT)
        ax.set_title(f"{p} {w} (n={int(m.sum())})", loc="left")
    for ax in axes.flat[len(show):]:
        ax.set_visible(False)
    fig.suptitle("Wells with an empty region (shaded tile: no organoids, but many expected)")
    plt.show()


def _plot_activity(activity):
    df = activity.sort_values("mAP")
    y = np.arange(len(df))
    color = np.where(df["active"], _BLUE, _GRAY)
    row_height = 0.2 if len(df) <= 30 else 0.13
    fig, ax = plt.subplots(figsize=(6, 1.3 + row_height * len(df)), layout="constrained")
    ax.scatter(df["chance_95"], y, marker="|", s=60, color=_INK_SECONDARY, zorder=2)
    ax.scatter(df["mAP"], y, s=22, c=list(color), edgecolors="white", linewidths=0.5, zorder=3)
    ax.set_yticks(y, df.index)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(-0.5, len(df) - 0.5)
    ax.grid(axis="x", color=_GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("mAP (retrieving the condition's own wells among its wells and the control wells)")
    handles = [Line2D([], [], linestyle="", marker="o", color=_BLUE, label="active (FDR < 5%)"),
               Line2D([], [], linestyle="", marker="o", color=_GRAY, label="not active"),
               Line2D([], [], linestyle="", marker="|", markersize=8, color=_INK_SECONDARY,
                      label="95th percentile by chance")]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle("Phenotypic activity: which conditions differ reproducibly from the control")
    plt.show()


def _plot_batch_effects(table, means_after, well_label, ref, group_plates):
    """Distribution of plate effects over features (left) and a PCA of the wells after normalization (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained")
    series = [("control_before", "control, before", _CATEGORICAL[0]),
              ("others_before", "other conditions, before", _CATEGORICAL[1]),
              ("others_after", "other conditions, after", _CATEGORICAL[2])]
    for col, name, color in series:
        values = np.sort(table[col].dropna().to_numpy())
        if len(values):
            ax1.plot(values, np.arange(1, len(values) + 1) / len(values), color=color, linewidth=2, label=name)
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("Share of well variance from the plate (eta squared)")
    ax1.set_ylabel("Share of features")
    ax1.grid(color=_GRID, linewidth=0.6)
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_title("Plate effect per feature", loc="left")

    M = means_after.to_numpy()
    M = (M - M.mean(axis=0)) / np.where(M.std(axis=0) > 0, M.std(axis=0), 1)
    coords = _pca_scores(M, variance=1.0, max_components=2)
    plates = means_after.index.get_level_values("plate")
    is_ref = (well_label.reindex(means_after.index) == ref).to_numpy()
    for k, p in enumerate(group_plates):
        m = np.asarray(plates == p)
        color = _CATEGORICAL[k % len(_CATEGORICAL)]
        ax2.scatter(coords[m & ~is_ref, 0], coords[m & ~is_ref, 1], s=14, color=color, edgecolors="white",
                    linewidths=0.4, label=p)
        ax2.scatter(coords[m & is_ref, 0], coords[m & is_ref, 1], s=30, marker="D", color=color,
                    edgecolors=_INK, linewidths=0.6)
    ax2.scatter([], [], s=30, marker="D", color="white", edgecolors=_INK, linewidths=0.6, label="control wells")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    ax2.legend(frameon=False, loc="best")
    ax2.set_title("Wells after normalization", loc="left")
    fig.suptitle(f"Batch effects between plates {', '.join(group_plates)}")
    plt.show()
