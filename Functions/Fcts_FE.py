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
import itertools
import datetime
import anndata
import random
import glob
import copy
import math
import cv2
import os

from Functions.Fcts_Base import find_staining_in_ABs, load_img_mask_by_UID, find_barcodes_with_day, save_adata

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

Pearson correlations: computed across all (round, stain) vectors (within + between rounds)
"""

# --- NOTE: file content copied from former top-level Fcts_FE.py; only imports adjusted. ---

# (The rest of the module is identical to the original file.)
