import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from inmoose.pycombat import pycombat_norm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from tqdm import tqdm

from raman_batch_effects.datasets import RamanDataset


def correct_batch_effects_for_single_feature_lmm(
    df: pd.DataFrame,
    feature_column: str,
    batch_column: str,
    fixed_effect_column: str | None = None,
    additional_covariate_column: str | None = None,
) -> pd.Series:
    """
    Correct a single feature for batch effects using a linear mixed model (LMM).

    The model models the batch effect as a random effect, while optionally including
    a fixed effect to preserve biologically relevant variation
    (e.g. strain- or condition-level effects).

    We assume that, if provided, the fixed effect is fully confounded with the batch effect,
    so we model the batch effect as a random effect to make it identifiable.

    Because the LMM assumes homoscedasticity, it only directly estimates
    batch-specific biases (or offsets). To crudely correct for batch-specific scales,
    we then scale the residuals by the within-batch standard deviation.

    We use a naive estimate of the within-batch standard deviation (i.e. without shrinkage),
    so this method assumes that there are no small batches or batches with very little variation.

    Args:
        df: DataFrame containing the feature to correct and the batch and fixed effect columns.
        feature_column: The name of the column containing the feature to correct.
        batch_column: The name of the column containing the batch effect.
        fixed_effect_column: The name of the column containing the fixed effect. If None,
            no fixed effect is included in the model.
        additional_covariate_column: The name of a column containing an additional
            quantitative covariate to include in the model.

    Returns:
        The corrected feature.
    """
    if fixed_effect_column is not None:
        formula = f"{feature_column} ~ C({fixed_effect_column})"
    else:
        formula = f"{feature_column} ~ 1"

    if additional_covariate_column is not None:
        formula += f" + {additional_covariate_column}"

    model = smf.mixedlm(formula, df, groups=df[batch_column])

    # Suppress common benign warnings during fitting.
    # TODO: The convergence warning is not necessary benign; it could indicate that the model
    # is close to non-identifiability and the fit is not reliable.
    with warnings.catch_warnings(record=True):
        warnings.filterwarnings("ignore", "Random effects covariance is singular")
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        fit_results = model.fit(method="lbfgs", reml=True)

    residuals = df[feature_column] - fit_results.fittedvalues

    # This is the random intercept for each sample's batch.
    uhat = df[batch_column].map(fit_results.random_effects).apply(lambda d: d.values[0])

    all_fixed_effects = fit_results.fittedvalues - uhat

    if additional_covariate_column is not None:
        fixed_effect_of_interest = (
            all_fixed_effects
            - df[additional_covariate_column] * fit_results.params[additional_covariate_column]
        )
    else:
        fixed_effect_of_interest = all_fixed_effects.copy()

    # The standard deviation of the residuals within each batch.
    batch_sd = df[batch_column].map(
        df.groupby(batch_column).apply(
            lambda group: np.std(residuals[group.index], ddof=1), include_groups=False
        )
    )

    global_sd = np.std(residuals, ddof=1)

    value_corrected = fixed_effect_of_interest + residuals / batch_sd * global_sd

    return value_corrected


def correct_batch_effects_combat(
    dataset: RamanDataset,
    batch_column: str,
    fixed_effect_column: str | None = None,
) -> RamanDataset:
    """
    Correct batch effects using ComBat from the inmoose package.

    ComBat (Johnson et al. 2007) uses empirical Bayes to estimate batch effects
    and adjust for them while preserving biological variation specified by covariates.

    Args:
        dataset: RamanDataset containing the spectra to correct.
        batch_column: The name of the metadata column containing the batch effect.
        fixed_effect_column: The name of the metadata column containing the fixed effect.
            If None, no fixed effect is included in the model.

    Returns:
        A new RamanDataset with batch-corrected spectra and the same metadata as the input.
    """
    X, labels = dataset.to_matrix()

    batch = labels[batch_column].values

    if fixed_effect_column is not None:
        covariate_df = pd.DataFrame({fixed_effect_column: labels[fixed_effect_column].values})
    else:
        covariate_df = None

    # pycombat_norm returns a numpy matrix, which behaves differently from an array.
    X_corrected = np.asarray(pycombat_norm(X.T, batch, covariate_df).T)

    corrected_dataset = RamanDataset.from_matrix(X_corrected, labels, dataset.wavenumbers)

    return corrected_dataset


def correct_batch_effects_lmm(
    dataset: RamanDataset,
    batch_column: str,
    fixed_effect_column: str | None = None,
) -> RamanDataset:
    """
    Correct batch effects for a RamanDataset using linear mixed models.

    Args:
        dataset: RamanDataset containing the spectra to correct.
        batch_column: The name of the metadata column containing the batch effect.
        fixed_effect_column: The name of the metadata column containing the fixed effect.
            If None, no fixed effect is included in the model.

    Returns:
        A new RamanDataset with batch-corrected spectra and the same metadata as the input.
    """
    X, labels = dataset.to_matrix()

    # Matrix of corrected spectra.
    X_corrected = np.zeros_like(X)

    for ind in tqdm(range(X.shape[1]), total=X.shape[1], desc="Correcting batch effects"):
        df = pd.DataFrame(
            {
                "value": X[:, ind],
                "batch": labels[batch_column],
                "fixed_effect": labels[fixed_effect_column]
                if fixed_effect_column is not None
                else None,
            }
        )
        X_corrected[:, ind] = correct_batch_effects_for_single_feature_lmm(
            df,
            feature_column="value",
            batch_column="batch",
            fixed_effect_column=("fixed_effect" if fixed_effect_column is not None else None),
        )

    corrected_dataset = RamanDataset.from_matrix(X_corrected, labels, dataset.wavenumbers)

    return corrected_dataset
