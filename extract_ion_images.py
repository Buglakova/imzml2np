import argparse
from pathlib import Path
import pandas as pd
from imzml2np import extract_peaks, peaks_df_to_images
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Parse parameters
    parser = argparse.ArgumentParser(
        description="""Script to extract ion images from imzml.

        The script takes as input path to imzml and list of ions to annotate,
        then saves corresponding ion images as a numpy array.
        In addition, calculate and plot sum of given ions as an example in the end.
        """
    )
    parser.add_argument(
        "imzml", type=str, help="imzml file to process",
    )
    parser.add_argument(
        "metadata", type=str, help="list of ions to analyze in metaspace output format",
    )

    parser.add_argument(
        "output_path", type=str, help="path to .npz file",
    )

    parser.add_argument(
        "--tol", type=float, default=5, help="Tolerance in ppm at the base mz",
    )

    parser.add_argument(
        "--base_mz", type=float, default=200, help="Base mz for mass tolerance",
    )

    args = parser.parse_args()

    imzml_path = args.imzml
    tol = args.tol
    base_mz = args.base_mz

    # Open metadata
    metadata = pd.read_csv(args.metadata)
    print(metadata)

    coords_df, peaks = extract_peaks(
        imzml_path, metadata, tol_ppm=tol, tol_mode="orbitrap", base_mz=base_mz
    )

    # Extract images
    images = []
    for peak in peaks:
        image = peaks_df_to_images(coords_df, peak["peaks_df"])
        images.append(image[1].T)
    images = np.array(images)

    print(f"Extracted {images.shape[0]} ion images of shape {images.shape[1:]}")

    # Save as a numpy object
    np.save(args.output_path, images, allow_pickle=True)

    # Open numpy object
    images = np.load(args.output_path)
    print(f"Read numpy array of shape {images.shape}")

    # Calculate and plot sum or do whatever
    sum_image = np.sum(images, axis=0)

    sns.heatmap(
        sum_image,
        cmap="magma",
        cbar=True,
        linewidths=0,
        xticklabels=False,
        yticklabels=False,
        square=True,
    )
    plt.tight_layout()
    plt.savefig("sum_image.png", dpi=300, bbox_inches="tight")
