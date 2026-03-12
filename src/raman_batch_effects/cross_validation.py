from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial

import numpy as np
from sklearn import metrics
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, LeaveOneOut
from tqdm import tqdm


@dataclass
class CVResults:
    confusion_matrix: np.ndarray
    unique_labels: list[str]
    mean_feature_importances: np.ndarray | None
    per_fold_accuracy: list[float]
    per_fold_macro_precision: list[float]
    per_fold_macro_recall: list[float]
    per_fold_macro_f1: list[float]
    per_fold_weighted_f1: list[float]
    per_fold_ovr_auc: list[float]
    per_fold_ovr_mcc: list[float]
    per_fold_mcc: list[float]
    aggregate_y_true: np.ndarray
    aggregate_y_pred: np.ndarray


def _per_fold_metrics(y_true, y_pred, unique_labels, y_proba=None):
    result = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=unique_labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=unique_labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=unique_labels, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0
        ),
        "ovr_auc": _per_fold_ovr_auc(y_true, y_proba, unique_labels),
        "ovr_mcc": _per_fold_ovr_mcc(y_true, y_pred, unique_labels),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    return result


def _per_fold_ovr_mcc(y_true, y_pred, unique_labels):
    """Compute macro-averaged one-vs-rest MCC for a single fold."""
    mccs = []
    for label in unique_labels:
        y_true_binary = (np.asarray(y_true) == label).astype(int)
        y_pred_binary = (np.asarray(y_pred) == label).astype(int)
        # MCC is undefined if either column is constant — skip those classes.
        if len(np.unique(y_true_binary)) < 2 or len(np.unique(y_pred_binary)) < 2:
            continue
        mccs.append(matthews_corrcoef(y_true_binary, y_pred_binary))
    if not mccs:
        return None
    return float(np.mean(mccs))


def _per_fold_ovr_auc(y_true, y_proba, unique_labels):
    """Compute OVR AUC for a single fold, returning None if not possible."""
    if y_proba is None:
        return None
    y_proba = np.atleast_2d(y_proba)
    if y_proba.shape[1] != len(unique_labels):
        return None
    if len(np.unique(y_true)) < 2:
        return None
    try:
        if len(unique_labels) == 2:
            return roc_auc_score(y_true, y_proba[:, 1])
        else:
            return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        return None


# Named tuples for worker function results.
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


def _calc_confusion_matrix_single_fold(args):
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


def calc_confusion_matrix_kfold(
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
        CVResults with confusion matrix, labels, AUC, importances, per-fold metrics,
        and aggregate predictions.
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

    args_for_folds = [
        (train_idx, test_idx, X, Y, model_factory) for train_idx, test_idx in cv_splitter.split(X)
    ]

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        results = list(
            tqdm(
                executor.map(_calc_confusion_matrix_single_fold, args_for_folds),
                total=len(args_for_folds),
                leave=False,
            )
        )

    # Compute per-fold metrics.
    fold_metrics = []
    for result in results:
        y_true_fold = np.atleast_1d(result.y_true)
        y_pred_fold = np.atleast_1d(result.y_pred)
        y_proba_fold = result.y_proba
        fold_metrics.append(
            _per_fold_metrics(y_true_fold, y_pred_fold, unique_labels, y_proba=y_proba_fold)
        )

    # Flatten results: for LOO, each result is a scalar; for k-fold, each result is an array.
    Y_true = np.concatenate([np.atleast_1d(result.y_true) for result in results])
    Y_pred = np.concatenate([np.atleast_1d(result.y_pred) for result in results])

    all_importances = np.array([result.importances for result in results])

    mean_importances = None
    if np.all([importances is not None for importances in all_importances]):
        mean_importances = all_importances.mean(axis=0)

    confusion_matrix = metrics.confusion_matrix(Y_true, Y_pred, labels=list(unique_labels))
    return CVResults(
        confusion_matrix=confusion_matrix,
        unique_labels=unique_labels,
        mean_feature_importances=mean_importances,
        per_fold_accuracy=[m["accuracy"] for m in fold_metrics],
        per_fold_macro_precision=[m["macro_precision"] for m in fold_metrics],
        per_fold_macro_recall=[m["macro_recall"] for m in fold_metrics],
        per_fold_macro_f1=[m["macro_f1"] for m in fold_metrics],
        per_fold_weighted_f1=[m["weighted_f1"] for m in fold_metrics],
        per_fold_ovr_auc=[m["ovr_auc"] for m in fold_metrics if m["ovr_auc"] is not None],
        per_fold_ovr_mcc=[m["ovr_mcc"] for m in fold_metrics if m["ovr_mcc"] is not None],
        per_fold_mcc=[m["mcc"] for m in fold_metrics],
        aggregate_y_true=Y_true,
        aggregate_y_pred=Y_pred,
    )


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


def calc_confusion_matrix_lobo(
    X: np.ndarray, Y: np.ndarray, batch_labels: np.ndarray, model: BaseEstimator | None = None
) -> CVResults:
    """
    Calculate confusion matrices using Leave-One-Batch-Out cross-validation.

    One confusion matrix is calculated for each batch.

    Args:
        X: Feature matrix (n_samples, n_features)
        Y: Binary labels (n_samples,)
        batch_labels: Batch labels for each sample (n_samples,)

    Returns:
        CVResults with summed confusion matrix, labels, AUC, importances, per-batch metrics,
        and aggregate predictions.
    """
    if model is None:
        model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10, class_weight="balanced"
        )

    unique_labels = sorted(set(Y))
    confusion_matrices = {}
    feature_importances = {}

    # Collect per-batch metrics and aggregate predictions.
    fold_metrics = []
    agg_y_true = []
    agg_y_pred = []

    for batch_label in tqdm(sorted(set(batch_labels))):
        mask = np.array(batch_labels) == batch_label
        X_train = X[~mask]
        X_test = X[mask]
        Y_train = Y[~mask]
        Y_test = Y[mask]

        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        Y_proba = None
        if hasattr(model, "predict_proba"):
            Y_proba = model.predict_proba(X_test)

        fold_metrics.append(_per_fold_metrics(Y_test, Y_pred, unique_labels, y_proba=Y_proba))
        agg_y_true.extend(Y_test)
        agg_y_pred.extend(Y_pred)

        cm = metrics.confusion_matrix(Y_test, Y_pred, labels=list(unique_labels))
        confusion_matrices[batch_label] = cm
        if isinstance(model, RandomForestClassifier):
            feature_importances[batch_label] = model.feature_importances_.copy()

    # Sum the confusion matrices across all batches.
    summed_confusion_matrix = np.zeros_like(list(confusion_matrices.values())[0])
    for value in confusion_matrices.values():
        summed_confusion_matrix += value

    # Average the feature importances across all batches.
    mean_feature_importances = None
    if len(feature_importances) > 0:
        mean_feature_importances = np.array(list(feature_importances.values())).mean(axis=0)

    return CVResults(
        confusion_matrix=summed_confusion_matrix,
        unique_labels=unique_labels,
        mean_feature_importances=mean_feature_importances,
        per_fold_accuracy=[m["accuracy"] for m in fold_metrics],
        per_fold_macro_precision=[m["macro_precision"] for m in fold_metrics],
        per_fold_macro_recall=[m["macro_recall"] for m in fold_metrics],
        per_fold_macro_f1=[m["macro_f1"] for m in fold_metrics],
        per_fold_weighted_f1=[m["weighted_f1"] for m in fold_metrics],
        per_fold_ovr_auc=[m["ovr_auc"] for m in fold_metrics if m["ovr_auc"] is not None],
        per_fold_ovr_mcc=[m["ovr_mcc"] for m in fold_metrics if m["ovr_mcc"] is not None],
        per_fold_mcc=[m["mcc"] for m in fold_metrics],
        aggregate_y_true=np.array(agg_y_true),
        aggregate_y_pred=np.array(agg_y_pred),
    )
