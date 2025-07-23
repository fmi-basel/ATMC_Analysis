import random
from ez_zarr import plotting

def segmentation_fidelity_check(ome_zarrs_dict, channels, channel_colors, channel_ranges, n, label_name, pyramid_lvl_plot=4):
    """
    Plot randomly selected images and their corresponding masks from OME-ZARR files.

    Parameters:
    - ome_zarrs_dict (dict): Dictionary containing OME-ZARR files.
    - channels (list): List of channels to plot.
    - channel_colors (list): List of colors for each channel.
    - channel_ranges (list): List of ranges for each channel.
    - n (int): Number of images to plot per barcode.
    - label_name (str): Name of the label to plot.
    - pyramid_lvl_plot (int): Pyramid level to plot.
    """
    # Loop over barcodes
    for barcode in ome_zarrs_dict:

        # Get random wells
        wells = random.sample(ome_zarrs_dict[barcode].names, n)

        # Loop over wells
        for well in wells:

            # Load image and corresponding mask
            img, msk = ome_zarrs_dict[barcode][well].get_array_pair_by_coordinate(label_name, pyramid_level=pyramid_lvl_plot)
            msk = msk[label_name]
            
            # Plot image and mask
            plotting.plot_image(im=img[0], msk=msk[0], msk_alpha=0.3,
                                channels=channels, channel_colors=channel_colors, channel_ranges=channel_ranges,
                                fig_width_inch=15, fig_height_inch=15,
                                title=f"{barcode} - {well}")