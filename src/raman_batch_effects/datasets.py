from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterator

import numpy as np
import pandas as pd
import ramanspy

KEY_COLUMN = "__key__"


class RamanDataset:
    """
    A base class for Raman datasets (collections of spectra).
    """

    def __init__(self):
        self._spectra = {}
        self.metadata = pd.DataFrame()

    def _normalize_metadata_for_hashing(self, metadata: dict) -> dict:
        """
        Normalize metadata values to reduce the fragility of hashing.
        """
        normalized_metadata = {}
        for key, value in metadata.items():
            if value is None or (isinstance(value, float) and np.isnan(value)):
                normalized_metadata[key] = None

            # Round floats to reasonable precision to avoid hash differences
            # from tiny floating-point variations.
            elif isinstance(value, float):
                rounded = round(value, 10)
                if rounded == int(rounded):
                    normalized_metadata[key] = int(rounded)
                else:
                    normalized_metadata[key] = rounded
            elif isinstance(value, str):
                normalized_metadata[key] = value.strip().lower()
            else:
                normalized_metadata[key] = value

        return normalized_metadata

    def add_spectrum(self, spectrum: ramanspy.Spectrum, **kwargs) -> None:
        """
        Add a spectrum to the dataset.

        Args:
            spectrum: The spectrum to add.
            **kwargs: The metadata for the spectrum. Must *not* include a "key" column,
            as this will be generated automatically.

        Raises:
            ValueError: If a spectrum with identical metadata already exists, or if
                wavenumbers don't match existing spectra.
        """
        if self._spectra and not np.array_equal(spectrum.spectral_axis, self.wavenumbers):
            raise ValueError("Spectrum wavenumbers don't match existing spectra")

        kwargs.pop(KEY_COLUMN, None)

        normalized_metadata = self._normalize_metadata_for_hashing(kwargs)

        key = hashlib.sha256(json.dumps(normalized_metadata, sort_keys=True).encode()).hexdigest()
        if key in self._spectra:
            raise ValueError("Spectrum with this metadata already exists")

        metadata_row = pd.DataFrame([kwargs])
        metadata_row[KEY_COLUMN] = key

        self._spectra[key] = spectrum
        self.metadata = pd.concat([self.metadata, metadata_row], ignore_index=True)

    def get_spectrum(self, row_or_mask: pd.Series) -> ramanspy.Spectrum:
        """
        Get the spectrum corresponding to the given row of metadata.
        """
        if len(row_or_mask) == len(self.metadata):
            row = self.metadata.loc[row_or_mask].iloc[0]
        else:
            row = row_or_mask

        if row[KEY_COLUMN] not in self._spectra:
            raise ValueError(f"Spectrum with metadata {row} not found in dataset")

        return self._spectra[row[KEY_COLUMN]]

    def __iter__(self) -> Iterator[tuple[ramanspy.Spectrum, pd.Series]]:
        """
        Iterate over the spectra in the dataset, returning tuples of (spectrum, metadata).
        """
        # We need to access KEY_COLUMN by position because its name starts with underscores
        # and gets replaced by pandas.
        key_col_idx = self.metadata.columns.get_loc(KEY_COLUMN)
        for row in self.metadata.itertuples():
            # row[0] is Index, so data starts at row[1].
            key = row[key_col_idx + 1]
            yield self._spectra[key], self.metadata.iloc[row.Index]

    def __len__(self) -> int:
        """
        Return the number of spectra in the dataset.
        """
        return len(self.metadata)

    @property
    def wavenumbers(self) -> np.ndarray:
        """
        Return the wavenumbers of the spectra in the dataset.
        """
        return self._spectra[next(iter(self._spectra.keys()))].spectral_axis

    def copy(self) -> RamanDataset:
        """
        Return a deep copy of the dataset.
        """
        new_dataset = RamanDataset()
        new_dataset._spectra = {
            k: ramanspy.Spectrum(v.spectral_data.copy(), v.spectral_axis.copy())
            for k, v in self._spectra.items()
        }
        new_dataset.metadata = self.metadata.copy()
        return new_dataset

    def apply(
        self,
        steps: ramanspy.preprocessing.PreprocessingStep
        | list[ramanspy.preprocessing.PreprocessingStep],
        inplace: bool = False,
    ) -> RamanDataset | None:
        """
        Apply one or more preprocessing steps to the spectra in the dataset.
        """
        if isinstance(steps, ramanspy.preprocessing.PreprocessingStep):
            steps = [steps]

        pipeline = ramanspy.preprocessing.Pipeline(steps)

        output_dataset = RamanDataset()
        for spectrum, metadata in self:
            processed_spectrum = pipeline.apply(spectrum)
            output_dataset.add_spectrum(processed_spectrum, **metadata.to_dict())

        if inplace:
            self._spectra = output_dataset._spectra
            self.metadata = output_dataset.metadata
        else:
            return output_dataset

    def loc(self, mask: pd.Series) -> RamanDataset:
        """
        Filter the dataset by the given mask.
        """
        filtered_dataset = RamanDataset()
        filtered_metadata = self.metadata[mask].reset_index(drop=True)
        filtered_dataset._spectra = {
            row[KEY_COLUMN]: self._spectra[row[KEY_COLUMN]]
            for _, row in filtered_metadata.iterrows()
        }
        filtered_dataset.metadata = filtered_metadata
        return filtered_dataset

    def filter(self, **kwargs: str | int | float | bool | Callable) -> RamanDataset:
        """
        Convenience function to filter the dataset by the given keyword arguments,
        which must match column names in the metadata dataframe.

        Values can be:
        - Literal values (str, int, float, bool) for exact matching.
        - Callable/lambda functions that take a column value and return bool.

        Examples:
        - filter(line="mCherry")
        - filter(line=lambda x: x in ["EGFP", "mCherry"])
        """
        mask = self._construct_mask_for_filtering(**kwargs)
        return self.loc(mask)

    def _construct_mask_for_filtering(
        self, **kwargs: str | int | float | bool | Callable
    ) -> pd.Series:
        """
        Construct a binary metadata mask from the given keyword arguments.
        Keyword argument names must match column names in the metadata dataframe.

        Values can be:
        - Literal values (str, int, float, bool) for exact matching
        - Callable/lambda functions that take a column value and return bool

        Examples:
            dataset.filter(line="mCherry")  # exact match
            dataset.filter(strain=lambda x: x.lower().startswith("cdc"))  # prefix match
        """
        mask = pd.Series(True, index=self.metadata.index)
        for key, value in kwargs.items():
            if callable(value):
                # Apply the callable to the column values
                mask &= self.metadata[key].apply(value)
            else:
                mask &= self.metadata[key] == value
        return mask

    def subsample(
        self, n: int | None = None, frac: float | None = None, random_state: int | None = None
    ) -> RamanDataset:
        """
        Randomly subsample the dataset without replacement.

        Args:
            n: Number of samples to select. If None, frac must be provided.
            frac: Fraction of samples to select (between 0 and 1). If None, n must be provided.
            random_state: Random seed for reproducibility.

        Returns:
            A new RamanDataset with the subsampled spectra and metadata.

        Raises:
            ValueError: If both n and frac are None, or if both are provided.
        """
        if n is None and frac is None:
            raise ValueError("Either n or frac must be provided")
        if n is not None and frac is not None:
            raise ValueError("Only one of n or frac can be provided")

        sampled_metadata = self.metadata.sample(
            n=n, frac=frac, random_state=random_state
        ).reset_index(drop=True)

        subsampled_dataset = RamanDataset()
        subsampled_dataset._spectra = {
            row[KEY_COLUMN]: self._spectra[row[KEY_COLUMN]]
            for _, row in sampled_metadata.iterrows()
        }
        subsampled_dataset.metadata = sampled_metadata
        return subsampled_dataset

    def to_matrix(self) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Convert the spectra to a data matrix and labels dataframe.
        """
        X = []
        labels = []
        for spectrum, metadata in self:
            X.append(spectrum.spectral_data)
            labels.append(metadata.to_dict())

        X = np.array(X)
        labels = pd.DataFrame(labels)
        return X, labels

    @classmethod
    def from_matrix(
        cls, X: np.ndarray, metadata: pd.DataFrame, wavenumbers: np.ndarray
    ) -> RamanDataset:
        """
        Create a RamanDataset from a data matrix and metadata (or labels) dataframe.

        The rows of the metadata dataframe are assumed to correspond to the rows of the data matrix.
        """
        if len(X) != len(metadata):
            raise ValueError(
                "The number of rows in the data matrix and the metadata dataframe must match."
            )

        if X.shape[1] != len(wavenumbers):
            raise ValueError(
                "The number of columns in the data matrix must match the length of the wavenumbers."
            )

        dataset = cls()
        for ind in range(X.shape[0]):
            spectral_data = X[ind, :].flatten()
            dataset.add_spectrum(
                ramanspy.Spectrum(spectral_data, wavenumbers), **metadata.iloc[ind].to_dict()
            )
        return dataset

    def concat(self, other: RamanDataset) -> RamanDataset:
        """
        Concatenate two datasets.
        """
        concatenated_dataset = self.copy()
        for spectrum, metadata in other:
            concatenated_dataset.add_spectrum(spectrum, **metadata.to_dict())
        return concatenated_dataset
