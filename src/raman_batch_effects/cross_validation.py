from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import Pool

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, LeaveOneOut
from tqdm import tqdm

# Named tuples for worker function results
RocFoldResult = namedtuple(
    "RocFoldResult",
    ["y_true", "probas", "importances"],
)
ConfusionMatrixFoldResult = namedtuple(
    "ConfusionMatrixFoldResult",
    ["y_true", "y_pred", "y_proba", "importances"],
)


def default_model_factory(random_state):
    return RandomForestClassifier(
        n_estimators=100, random_state=random_state, class_weight="balanced"
    )


def model_factory_from_model(model):
    return lambda: clone(model)


def _calc_roc_loo_fold(args):
    """
    Worker function for parallel CV fold computation for calculating an ROC curve.

    Handles both LOO (single sample) and k-fold (multiple samples) cases.
    """
    train_idx, test_idx, X, Y, model_factory = args

    model = model_factory()
    model.fit(X[train_idx], Y[train_idx])
    probas = model.predict_proba(X[test_idx])[:, 1]
    y_true = Y[test_idx]

    importances = None
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_

    return RocFoldResult(y_true=y_true, probas=probas, importances=importances)


def calc_roc_loo(
    X,
    Y,
    show_progress: bool = True,
    n_threads=None,
    model=None,
    model_factory=None,
    n_folds: int | None = None,
):
    """
    Calculate ROC curve using Leave-One-Out or k-fold cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Array of binary labels (n_samples,)
        show_progress: Whether to show a progress bar
        n_threads: Number of threads to use. If None, uses all available CPUs.
        model: Optional sklearn model to use. If provided, will be cloned for each fold.
        model_factory: Optional factory function that returns a model. Ignored if model is provided.
        n_folds: Number of folds for k-fold CV. If None (default), uses Leave-One-Out CV.

    Returns:
        fpr: False positive rate
        tpr: True positive rate
        mean_importances: Mean feature importances (if applicable)
    """

    if model is not None:
        model_factory = model_factory_from_model(model)
    elif model_factory is None:
        random_state = 42
        model_factory = partial(default_model_factory, random_state=random_state)

    if n_folds is None:
        cv_splitter = LeaveOneOut()
    else:
        cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    args_for_loo_folds = [
        (train_idx, test_idx, X, Y, model_factory) for train_idx, test_idx in cv_splitter.split(X)
    ]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(
            tqdm(
                executor.map(_calc_roc_loo_fold, args_for_loo_folds),
                total=len(args_for_loo_folds),
                leave=False,
                disable=not show_progress,
            )
        )

    # Flatten results: for LOO, each result is a scalar; for k-fold, each result is an array.
    y_true = np.concatenate([np.atleast_1d(result.y_true) for result in results])
    y_scores = np.concatenate([np.atleast_1d(result.probas) for result in results])
    all_importances = np.array([result.importances for result in results])

    mean_importances = None
    if np.all(all_importances is not None):
        mean_importances = all_importances.mean(axis=0)

    fpr, tpr, _ = roc_curve(y_true, y_scores)

    return fpr, tpr, mean_importances


def _calc_confusion_matrix_loo_fold(args):
    """
    Worker function for parallel CV fold computation for calculating a confusion matrix.

    Handles both LOO (single sample) and k-fold (multiple samples) cases.
    """
    train_idx, test_idx, X, Y, model_factory = args
    model = model_factory()
    model.fit(X[train_idx], Y[train_idx])
    y_pred = model.predict(X[test_idx])
    y_true = Y[test_idx]

    # Get predicted probabilities for AUC calculation.
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X[test_idx])

    importances = None
    if isinstance(model, RandomForestClassifier):
        importances = model.feature_importances_

    return ConfusionMatrixFoldResult(
        y_true=y_true, y_pred=y_pred, y_proba=y_proba, importances=importances
    )


def calc_confusion_matrix_loo(
    X, Y, n_threads=None, model=None, model_factory=None, n_folds: int | None = None
):
    """
    Calculate confusion matrix using Leave-One-Out or k-fold cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Labels (n_samples,)
        n_threads: Number of threads to use. If None, uses all available CPUs.
        model: Optional sklearn model to use. If provided, will be cloned for each fold.
        model_factory: Optional factory function that returns a model. Ignored if model is provided.
        n_folds: Number of folds for k-fold CV. If None (default), uses Leave-One-Out CV.

    Returns:
        cm: Confusion matrix
        unique_labels: Sorted list of unique labels
        ovr_auc: One-vs-Rest macro-averaged AUC score (None if unavailable)
        mean_importances: Mean feature importances (if applicable)
    """

    unique_labels = sorted(set(Y))

    if model is not None:
        model_factory = model_factory_from_model(model)
    elif model_factory is None:
        model_factory = partial(default_model_factory, random_state=42)

    if n_folds is None:
        cv_splitter = LeaveOneOut()
    else:
        cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    args_for_loo_folds = [
        (train_idx, test_idx, X, Y, model_factory) for train_idx, test_idx in cv_splitter.split(X)
    ]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(
            tqdm(
                executor.map(_calc_confusion_matrix_loo_fold, args_for_loo_folds),
                total=len(args_for_loo_folds),
                leave=False,
            )
        )

    # Flatten results: for LOO, each result is a scalar; for k-fold, each result is an array.
    Y_true = np.concatenate([np.atleast_1d(result.y_true) for result in results])
    Y_pred = np.concatenate([np.atleast_1d(result.y_pred) for result in results])

    # Handle probabilities - can be None, scalar, or array per fold.
    Y_proba_list = [
        np.atleast_2d(result.y_proba) for result in results if result.y_proba is not None
    ]

    all_importances = np.array([result.importances for result in results])

    mean_importances = None
    if np.all([importances is not None for importances in all_importances]):
        mean_importances = all_importances.mean(axis=0)

    # Calculate OVR AUC if probabilities are available.
    ovr_auc = _calculate_ovr_auc(Y_proba_list, [Y_true], unique_labels)

    confusion_matrix = metrics.confusion_matrix(Y_true, Y_pred, labels=list(unique_labels))
    return confusion_matrix, unique_labels, ovr_auc, mean_importances


def calc_roc_lobo(X, Y, batch_labels, model=None):
    """
    Calculate ROC curve using Leave-One-Batch-Out cross-validation.

    X: Feature matrix (n_samples, n_features)
    Y: Array of class labels (n_samples,)
    batch_labels: Array of batch labels (n_samples,)

    Returns:
        fpr: False positive rate
        tpr: True positive rate
        importances: Mean feature importances
    """

    if model is None:
        model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")

    y_true = []
    y_scores = []
    importances = []

    for batch_label in tqdm(sorted(set(batch_labels))):
        mask = np.array(batch_labels) == batch_label
        X_train = X[~mask]
        X_test = X[mask]
        Y_train = Y[~mask]
        Y_test = Y[mask]
        model.fit(X_train, Y_train)

        # Handle case where only one class is present in training data.
        probas = model.predict_proba(X_test)
        if probas.shape[1] == 2:
            # Both classes present in training.
            probas = probas[:, 1]
        else:
            # Only one class present in training.
            if model.classes_[0] == 1:
                # Only positive class in training, probability for positive class is in column 0.
                probas = probas[:, 0]
            else:
                # Only negative class (0) in training, probability for positive class is 0.
                probas = np.zeros(len(X_test))

        y_scores.extend(probas)
        y_true.extend(Y_test)
        if isinstance(model, RandomForestClassifier):
            importances.append(model.feature_importances_)
        elif isinstance(model, LogisticRegression):
            importances.append(model.coef_)

    if len(importances) > 0:
        importances = np.array(importances).mean(axis=0)

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return fpr, tpr, importances


def _calculate_ovr_auc(all_Y_proba: list, all_Y_true: list, unique_labels: list) -> float | None:
    """
    Calculate one-vs-rest AUC if all conditions are met, otherwise return None.

    Args:
        all_Y_proba: List of probability arrays from each fold
        all_Y_true: List of true label arrays from each fold
        unique_labels: Sorted list of unique class labels

    Returns:
        OVR AUC score, or None if calculation is not possible
    """
    if not all_Y_proba:
        return None

    # Check if all probability arrays have the same shape matching unique labels.
    # In LOBO CV, some folds may exclude entire classes from training,
    # resulting in probability arrays with different numbers of columns.
    proba_shapes = [arr.shape[1] for arr in all_Y_proba]
    if len(set(proba_shapes)) != 1 or proba_shapes[0] != len(unique_labels):
        return None

    all_Y_true = np.concatenate(all_Y_true)
    all_Y_proba = np.vstack(all_Y_proba)

    # Only calculate AUC if all classes are present in test sets.
    if len(np.unique(all_Y_true)) != len(unique_labels):
        return None

    # Calculate OVR AUC.
    if len(unique_labels) == 2:
        return roc_auc_score(all_Y_true, all_Y_proba[:, 1])
    else:
        return roc_auc_score(all_Y_true, all_Y_proba, multi_class="ovr", average="macro")


def calc_confusion_matrix_lobo(
    X: np.ndarray, Y: np.ndarray, batch_labels: np.ndarray, model: BaseEstimator | None = None
) -> tuple[np.ndarray, list[str], np.ndarray | None, float | None]:
    """
    Calculate confusion matrices using Leave-One-Batch-Out cross-validation.

    One confusion matrix is calculated for each batch.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Binary labels (n_samples,)
        batch_labels: Batch labels for each sample (n_samples,)

    Returns:
        confusion_matrices: Summed confusion matrix across all batches
        unique_labels: Sorted list of unique class labels
        ovr_auc: One-vs-Rest macro-averaged AUC score (None if unavailable)
        feature_importances: Dictionary mapping batch labels to feature importances
    """
    if model is None:
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, class_weight="balanced"
        )

    unique_labels = sorted(set(Y))
    confusion_matrices = {}
    feature_importances = {}

    # Collect all predictions and true labels for OVR AUC calculation.
    all_Y_true = []
    all_Y_proba = []

    for batch_label in tqdm(sorted(set(batch_labels))):
        mask = np.array(batch_labels) == batch_label
        X_train = X[~mask]
        X_test = X[mask]
        Y_train = Y[~mask]
        Y_test = Y[mask]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        cm = metrics.confusion_matrix(Y_test, Y_pred, labels=list(unique_labels))
        confusion_matrices[batch_label] = cm
        if isinstance(model, RandomForestClassifier):
            feature_importances[batch_label] = model.feature_importances_.copy()

        # Collect probabilities for OVR AUC.
        if hasattr(model, "predict_proba"):
            Y_proba = model.predict_proba(X_test)
            all_Y_true.append(Y_test)
            all_Y_proba.append(Y_proba)

    # Sum the confusion matrices across all batches.
    summed_confusion_matrix = np.zeros_like(list(confusion_matrices.values())[0])
    for value in confusion_matrices.values():
        summed_confusion_matrix += value

    # Average the feature importances across all batches.
    mean_feature_importances = None
    if len(feature_importances) > 0:
        mean_feature_importances = np.array(list(feature_importances.values())).mean(axis=0)

    # Calculate OVR AUC if probabilities are available.
    ovr_auc = _calculate_ovr_auc(all_Y_proba, all_Y_true, unique_labels)

    return summed_confusion_matrix, unique_labels, ovr_auc, mean_feature_importances


def _calc_roc_loo_for_mask(args):
    """Wrapper function for parallel execution."""
    X, Y, mask, label = args
    fpr, tpr, _ = calc_roc_loo(X[:, mask], Y, show_progress=False)
    return label, fpr, tpr


def _calc_roc_lobo_for_mask(args):
    """Wrapper function for parallel execution."""
    X, Y, mask, batch_labels, label = args
    fpr, tpr, _ = calc_roc_lobo(X[:, mask], Y, batch_labels)
    return label, fpr, tpr


def calc_loo_rocs_for_intensity_windows(X, Y, percentiles):
    """
    Calculate ROC curves for different intensity windowed thresholds in parallel.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Labels (n_samples,)
        percentiles: List of percentile thresholds to test

    Returns:
        rocs: List of (label, fpr, tpr) tuples for each window.
    """

    X_mean = X.mean(axis=0)
    percentile_values = np.percentile(X_mean, percentiles)

    args = []
    for ind, (percentile, percentile_value) in enumerate(
        zip(percentiles[:-1], percentile_values[:-1], strict=True)
    ):
        mask = (X_mean > percentile_value) & (X_mean < percentile_values[ind + 1])
        args.append((X, Y, mask, f">{percentile:.1f}-{percentiles[ind + 1]:.1f}"))

    print(f"Computing {len(args)} ROC curves for windowed thresholds...")
    with Pool() as pool:
        results = list(tqdm(pool.imap(_calc_roc_loo_for_mask, args), total=len(args)))

    rocs = [(label, fpr, tpr) for label, fpr, tpr in results]

    return rocs


def calc_loo_rocs_for_spectral_windows(X, Y, window_inds):
    """
    Calculate ROC curves for different spectral windowed thresholds in parallel.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Labels (n_samples,)
        window_inds: List of tuples of start and end position of the windows, in wavenumber indices.

    Returns:
        rocs: List of (label, fpr, tpr) tuples for each window.
    """

    args = []
    for start_ind, end_ind in window_inds:
        mask = slice(start_ind, end_ind)
        args.append((X, Y, mask, f"window-{start_ind:.1f}-{end_ind:.1f}"))

    print(f"Computing {len(args)} ROC curves for spectral windows...")
    with Pool() as pool:
        results = list(tqdm(pool.imap(_calc_roc_loo_for_mask, args), total=len(args)))

    rocs = [(label, fpr, tpr) for label, fpr, tpr in results]

    return rocs


def calc_lobo_rocs_for_spectral_windows(X, Y, window_inds, batch_labels):
    """
    Calculate ROC curves for different spectral windowed thresholds in parallel.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Labels (n_samples,)
        window_inds: List of tuples of start and end position of the windows, in wavenumber indices.

    Returns:
        rocs: List of (label, fpr, tpr) tuples for each window.
    """

    args = []
    for start_ind, end_ind in window_inds:
        mask = slice(start_ind, end_ind)
        args.append((X, Y, mask, batch_labels, f"window-{start_ind:.1f}-{end_ind:.1f}"))

    print(f"Computing {len(args)} ROC curves for spectral windows...")
    with Pool() as pool:
        results = list(tqdm(pool.imap(_calc_roc_lobo_for_mask, args), total=len(args)))

    rocs = [(label, fpr, tpr) for label, fpr, tpr in results]

    return rocs


def calc_regression_lobo(
    X: np.ndarray, Y: np.ndarray, batch_labels: np.ndarray, model: BaseEstimator | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray | None]:
    """
    Calculate regression predictions using Leave-One-Batch-Out cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features).
        Y: Target values (n_samples,).
        batch_labels: Batch labels for each sample (n_samples,).
        model: Optional sklearn regression model to use.

    Returns:
        y_true: True target values.
        y_pred: Predicted target values.
        metrics_dict: Dictionary with regression metrics (r2, rmse, mae).
        feature_importances: Mean feature importances (if applicable).
    """
    if model is None:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    y_true = []
    y_pred = []
    feature_importances = []

    for batch_label in tqdm(sorted(set(batch_labels))):
        mask = np.array(batch_labels) == batch_label
        X_train = X[~mask]
        X_test = X[mask]
        Y_train = Y[~mask]
        Y_test = Y[mask]

        model.fit(X_train, Y_train)
        predictions = model.predict(X_test)

        y_pred.extend(predictions)
        y_true.extend(Y_test)

        if isinstance(model, RandomForestRegressor) and hasattr(model, "feature_importances_"):
            feature_importances.append(model.feature_importances_)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mean_feature_importances = None
    if len(feature_importances) > 0:
        mean_feature_importances = np.array(feature_importances).mean(axis=0)

    metrics_dict = {
        "r2": r2_score(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
    }

    return y_true, y_pred, metrics_dict, mean_feature_importances
