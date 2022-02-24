import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Splitter:
    """Class to split an input data frame into two different sets using different strategies."""

    def __init__(self, dataframe: pd.DataFrame, obsCol: str, respCol: str, mode: str):
        # do input checks
        if dataframe.empty:
            raise ValueError("Provided dataframe is empty.")
        if mode not in ["regression", "classification"]:
            raise ValueError(
                "Parameter mode needs to be either classification or regression, see respective Enum."
            )
        if obsCol not in dataframe.columns or respCol not in dataframe.columns:
            raise ValueError(
                "One or both columns are not part of the supplied dataframe."
            )

        # set the response column to the appropriate type
        dataframe[obsCol] = dataframe[obsCol].astype("object")
        if mode == "regression":
            dataframe[respCol] = dataframe[respCol].astype("float")
        else:
            dataframe[respCol] = dataframe[respCol].astype("object")

        # store stuff internally
        self._df = dataframe
        self._mode = mode
        self._obsCol = obsCol
        self._respCol = respCol

    def split_randomly(self, fraction=0.2, seed=42):
        """Function that will return two randomly split datasets, the first with "1-fraction" and the second with
        "fraction" observations."""
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError(
                "Parameter fraction must be a number between 0 and 1 (exclusively)."
            )

        # Note: new version of sklearn.model_selection.train_test_split is able to return pandas data frames
        training_set, test_set = train_test_split(
            self._df, shuffle=True, test_size=fraction, random_state=seed
        )
        return training_set, test_set

    def split_temporal(self, time_column, threshold):
        """Function that will return two temporally split datasets, the the first (up and including the threshold) is
        the training, the remaining observations constitute the test sets, respectively."""

        # set the type of the column to be numeric for ordering
        self._df[time_column] = self._df[time_column].astype("float")

        # split the sets
        training_set = self._df[self._df[time_column] <= threshold]
        test_set = self._df[self._df[time_column] > threshold]

        # do warn, if any of the sets are empty
        if len(training_set) == 0 or len(test_set) == 0:
            warnings.warn("One or both sets are empty.")

        return training_set, test_set

    def split_stratified(self, fraction=0.2, bins="fd", seed=42):
        """Function that will return two stratified split datasets, i.e. the input distribution will be binned by the
        specification used and for each bin a fraction is sampled randomly. This ensures that both sets resemble
        a similar distribution."""
        if self._mode == "classification":
            raise NotImplementedError(
                "The stratefication strategy is not implemented for classification tasks yet. One should use sklearn's StratifiedKfold when doing the implementation."
            )
        if fraction <= 0.0 or fraction >= 1.0:
            raise ValueError(
                "Parameter fraction must be a number between 0 and 1 (exclusively)."
            )

        # bin the values and return two sets
        samples_per_bin, bins = np.histogram(self._df[self._respCol], bins=bins)

        # set the bins to be just a little larger at the ends to include every observation
        bins[0] = np.nextafter(bins[0], -np.inf)
        bins[-1] = np.nextafter(bins[-1], np.inf)

        # introduce safe-guard for cases, where too few observations were present in a bin
        bins = np.delete(bins, np.flatnonzero(samples_per_bin < 10))

        # calculate the indices (bin-IDs) for the response values
        bin_idxs = np.digitize(x=self._df[self._respCol], bins=bins)

        # split and return the sets
        train, test = train_test_split(
            self._df,
            stratify=bin_idxs,
            shuffle=True,
            test_size=fraction,
            random_state=seed,
        )
        return train, test
