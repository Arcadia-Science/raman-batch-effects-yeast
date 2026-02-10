import numpy as np

from raman_batch_effects import analysis_utils
from raman_batch_effects.datasets import RamanDataset


def identify_outlier_spectra(dataset: RamanDataset, group_by: list[str]) -> RamanDataset:
    """
    Identify outlier spectra within groups defined by metadata columns.

    This function adds an "is_outlier" column to the dataset metadata by computing
    the distance of each spectrum from the median spectrum within its group.
    Outliers are identified using elbow detection on the sorted distances.

    The algorithm:
    1. Groups spectra by the specified metadata columns.
    2. Within each group, computes the median spectrum.
    3. For each spectrum, computes a distance as the standard deviation of the differences
        between it and the median spectrum of its group.
    4. Sorts spectra by distance and uses elbow detection to identify outliers.

    Parameters
    ----------
    dataset : RamanDataset
        Dataset to annotate with outlier information
    group_by : list[str]
        List of metadata column names to group by.

    Returns
    -------
    RamanDataset
        The input dataset with "is_outlier" column added to metadata
    """
    min_group_size = 3
    dataset.metadata["is_outlier"] = False

    for group_keys in dataset.metadata.groupby(group_by).groups.keys():
        # Handle single-column groupby (returns scalars not tuples).
        if len(group_by) == 1:
            group_keys = (group_keys,)

        # Create mask for this group.
        mask = np.ones(len(dataset.metadata), dtype=bool)
        for col, val in zip(group_by, group_keys, strict=True):
            mask &= dataset.metadata[col] == val

        X = np.array(
            [
                dataset.get_spectrum(row).spectral_data
                for _, row in dataset.metadata[mask].iterrows()
            ]
        )

        if len(X) < min_group_size:
            continue

        # Compute distance as the standard deviation of differences from median spectrum.
        median_spectrum = np.median(X, axis=0, keepdims=True)
        dists = np.std(X - median_spectrum, axis=1)
        ordered_inds = np.argsort(dists)

        # Use elbow detection to identify outliers.
        elbow_index = analysis_utils.find_elbow_by_max_distance(dists[ordered_inds])
        outlier_inds = ordered_inds[elbow_index:]
        mask_inds = np.argwhere(mask).flatten()

        dataset.metadata.loc[mask_inds[outlier_inds], "is_outlier"] = True

    return dataset
