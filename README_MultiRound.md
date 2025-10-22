# Multi-Round Multiplexing Support for ATMC Analysis

This extension adds support for analyzing multiple multiplexing rounds in your OME-Zarr data while maintaining compatibility with the existing single-round workflow.

## Key Features

### 🎯 **Smart Feature Extraction**
- **Shape features** (area, perimeter, skeleton, convex hull, etc.) are extracted **only once from Round 0** where the segmentation masks are located
- **Intensity features** (mean, std, quantiles, correlations, etc.) are extracted from **all available rounds**
- **Cross-round correlations** are automatically calculated between the same stainings across different rounds

### 🔄 **Automatic Round Detection**
- Automatically detects available multiplexing rounds in your OME-Zarr data
- No manual specification needed - just point to your data directory
- Graceful handling of missing rounds (continues with available ones)

### 📊 **Enhanced Feature Naming**
- Round 0 features use standard names: `DAPI_mean`, `Ki67_std`, etc.
- Additional rounds get round suffixes: `DAPI_R1_mean`, `Ki67_R2_std`, etc.
- Cross-round correlations: `DAPI-DAPI_R1_PearsonR`

## Quick Start

### 1. Use the Multi-Round Notebook
Start with `1_FeatureExtraction_MultiRound.ipynb` instead of the original notebook.

### 2. Updated Import Statements
```python
# Import original functions
from Fcts_FE import make_experiment, estimate_staining_thresholds
from Fcts_Base import get_stainings, find_zarr_dirs

# Import multi-round extensions
from Fcts_Base_MultiRound import extract_ome_zarr_tables_multiround
from Fcts_FE_MultiRound import extract_features_multiround
```

### 3. Load Multi-Round Data
```python
# This automatically detects and loads all available rounds
ome_zarr_dict_multiround, ome_zarr_df = extract_ome_zarr_tables_multiround(
    experiment_setup, 
    source, 
    folder, 
    table_name,
    max_rounds=None  # None = auto-detect all rounds
)
```

### 4. Extract Features from All Rounds
```python
ad = extract_features_multiround(
    ome_zarrs_dict_multiround=ome_zarr_dict_multiround,
    # ... other parameters same as before ...
    max_rounds=None  # Process all detected rounds
)
```

## How It Works

### Data Structure
The multi-round system uses a dictionary structure:
```python
ome_zarr_dict_multiround = {
    ('barcode1', 0): plate_round0_object,
    ('barcode1', 1): plate_round1_object, 
    ('barcode1', 2): plate_round2_object,
    ('barcode2', 0): plate_round0_object,
    # ...
}
```

### Feature Extraction Logic

1. **Shape Features (Round 0 Only)**
   - Segmentation masks are always in Round 0 (`image_name="0"`)
   - All morphological features extracted once from Round 0
   - Includes: area, perimeter, skeleton analysis, convex hull features

2. **Intensity Features (All Rounds)**
   - For each round, extract intensity-based features from each channel
   - Thresholds from Round 0 are applied to corresponding channels in other rounds
   - Same mask (from Round 0) used for all rounds

3. **Correlation Analysis**
   - Within-round correlations calculated for each round separately
   - Cross-round correlations calculated between same stainings across rounds

### Example Output Features

For a 3-round experiment with DAPI and Ki67 stainings:

**Shape Features (Round 0 only):**
- `area`, `perimeter`, `crypt_count`, `crypt_length_total`, etc.

**Intensity Features (per round):**
- Round 0: `DAPI_mean`, `DAPI_std`, `Ki67_mean`, `Ki67_std`
- Round 1: `DAPI_R1_mean`, `DAPI_R1_std`, `Ki67_R1_mean`, `Ki67_R1_std`
- Round 2: `DAPI_R2_mean`, `DAPI_R2_std`, `Ki67_R2_mean`, `Ki67_R2_std`

**Correlation Features:**
- Within-round: `DAPI-Ki67_PearsonR`, `DAPI_R1-Ki67_R1_PearsonR`
- Cross-round: `DAPI-DAPI_R1_PearsonR`, `DAPI-DAPI_R2_PearsonR`, `Ki67-Ki67_R1_PearsonR`

## Technical Implementation

### Key Functions

#### `extract_ome_zarr_tables_multiround()`
- Enhanced version of `extract_ome_zarr_tables()` 
- Loads multiple rounds using `ome_zarr.import_plate(path, image_name=str(round_num))`
- Auto-detects available rounds by trying sequential loading

#### `load_img_mask_by_UID_multiround()`
- Enhanced version that can load images from any round
- Always loads masks from Round 0
- Maintains same coordinates across all rounds

#### `extract_features_multiround()`
- Main orchestration function for multi-round analysis
- Handles round detection, image loading, and feature calculation
- Maintains backward compatibility with single-round data

### Error Handling
- Graceful handling of missing rounds (continues with available ones)
- Validates that Round 0 exists (required for masks)
- Comprehensive error messages for debugging

### Memory Efficiency
- Processes one organoid at a time across all rounds
- Uses deep copying only when necessary
- Efficient numpy operations for correlations

## Configuration Options

### Round Selection
```python
# Process all detected rounds
max_rounds = None

# Process only first 3 rounds (0, 1, 2)
max_rounds = 3

# The system will automatically handle missing intermediate rounds
```

### Threshold Application
- Thresholds estimated from Round 0 are applied to all rounds
- Same staining in different rounds uses the same threshold
- Manual threshold adjustments affect all rounds

## Backward Compatibility

The multi-round system is fully backward compatible:
- Single-round data works without any changes
- Original functions remain unchanged
- Can mix single-round and multi-round analyses in the same project

## File Naming Conventions

### Output Files
- Multi-round CSV: `1_FeatureExtraction_multiround_BARCODE_TIMESTAMP.csv`
- Multi-round AnnData: `1_FeatureExtraction_multiround_TIMESTAMP.h5ad`
- Metadata includes `multiround=True` flag

### OME-Zarr Structure Expected
Your OME-Zarr files should support the `image_name` parameter:
```python
# Round 0 (contains masks)
plate_r0 = ome_zarr.import_plate(path, image_name="0")

# Round 1 (additional stainings)
plate_r1 = ome_zarr.import_plate(path, image_name="1")

# Round 2 (additional stainings)  
plate_r2 = ome_zarr.import_plate(path, image_name="2")
```

## Troubleshooting

### Common Issues

1. **"Round 0 not found"**
   - Ensure your OME-Zarr data has `image_name="0"` 
   - Check that segmentation masks are in Round 0

2. **"No features extracted"**
   - Verify that at least Round 0 loads successfully
   - Check that segmentation masks are properly labeled

3. **Memory issues with many rounds**
   - Reduce `max_rounds` to process fewer rounds
   - The system processes organoids sequentially to minimize memory usage

### Debug Mode
Add debug prints to see which rounds are being processed:
```python
print("Available rounds per barcode:")
for barcode in barcodes:
    rounds = get_available_rounds_for_barcode(ome_zarr_dict_multiround, barcode)
    print(f"  {barcode}: {rounds}")
```

## Performance Considerations

- **Loading time**: Scales linearly with number of rounds
- **Memory usage**: Processes one organoid at a time across all rounds
- **Feature computation**: Shape features computed once, intensity features per round
- **File size**: Output files grow with number of rounds and extracted features

## Future Extensions

The multi-round framework is designed for extensibility:
- Easy to add new cross-round analysis methods
- Support for round-specific thresholds if needed
- Integration with time-series analysis workflows

---

**Need Help?** 
If you encounter issues with the multi-round functionality, please check:
1. Your OME-Zarr data structure supports `image_name` parameter
2. Round 0 contains the segmentation masks
3. All rounds have consistent spatial coordinates
4. The debug output showing detected rounds per barcode
