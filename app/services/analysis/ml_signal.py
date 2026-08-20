from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "RSI14",
    "MACDHist",
    "BBPercentB",
    "BBBandwidth",
    "ADX14",
    "PlusDI",
    "MinusDI",
    "VolumeRatio",
    "TrendScore",
    "RelativeStrength",
]


@dataclass(frozen=True)
class WalkForwardResult:
    fold_scores: pd.DataFrame
    out_of_sample_accuracy: float
    out_of_sample_auc: float
    n_folds: int
    n_samples: int
    baseline_accuracy: float
    note: str


@dataclass(frozen=True)
class LiveModelSignal:
    probability_up: float
    model_name: str
    trained_samples: int
    out_of_sample_accuracy: float
    out_of_sample_auc: float
    beats_baseline: bool
    note: str


def build_feature_target_frame(frame: pd.DataFrame, horizon_days: int = 10) -> pd.DataFrame:
    """Build a model-ready table: engineered features + forward return label.

    Label is 1 if the close `horizon_days` ahead is higher than today's close,
    else 0. Rows near the end of the frame won't have a forward label yet and
    are dropped here but kept separately for live (unlabeled) prediction.
    """

    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing required indicator columns: {missing}. Run add_indicators() first.")

    df = frame.copy().reset_index(drop=True)
    df["ForwardReturn"] = df["Close"].shift(-horizon_days) / df["Close"] - 1
    df["Target"] = (df["ForwardReturn"] > 0).astype(int)
    return df


def _clean_features(df: pd.DataFrame, fallback_medians: pd.Series | None = None) -> pd.DataFrame:
    features = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    medians = features.median(numeric_only=True)
    if fallback_medians is not None:
        medians = medians.fillna(fallback_medians)
    features = features.fillna(medians)
    # A feature that's NaN for every row (e.g. RelativeStrength with no
    # benchmark supplied) has an undefined median too; neutralize it to 0
    # rather than leaving NaNs that would crash sklearn.
    return features.fillna(0.0)


def walk_forward_validate(
    frame: pd.DataFrame,
    horizon_days: int = 10,
    n_splits: int = 5,
    model: str = "logistic",
) -> WalkForwardResult:
    """Time-series cross-validate a direction-prediction model.

    Uses scikit-learn's TimeSeriesSplit so every fold trains only on the past
    and tests only on a later, unseen block — no shuffling, which would leak
    future information into training and inflate accuracy. Reports
    out-of-sample accuracy/AUC pooled across folds, plus the naive baseline
    (always predict the majority class) so a model that's barely better than
    "always guess up" is visible as such rather than looking impressive on
    its own.
    """

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return WalkForwardResult(
            pd.DataFrame(), float("nan"), float("nan"), 0, 0, float("nan"),
            "scikit-learn is not installed; ML signal unavailable (rule-based signal still works).",
        )

    labeled = build_feature_target_frame(frame, horizon_days=horizon_days).dropna(subset=["ForwardReturn"])
    labeled = labeled.dropna(subset=FEATURE_COLUMNS, how="all")
    min_samples = (n_splits + 1) * 30
    if len(labeled) < min_samples:
        return WalkForwardResult(
            pd.DataFrame(), float("nan"), float("nan"), 0, len(labeled), float("nan"),
            f"Only {len(labeled)} labeled rows available; need at least {min_samples} for a "
            f"{n_splits}-fold walk-forward test. Use a longer history window.",
        )

    X = _clean_features(labeled)
    y = labeled["Target"].to_numpy()

    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    all_true: list[int] = []
    all_pred: list[int] = []
    all_proba: list[float] = []

    for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model == "gradient_boosting":
            estimator = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=2)
        else:
            estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

        estimator.fit(X_train, y_train)
        proba = estimator.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        fold_acc = float(accuracy_score(y_test, pred))
        try:
            fold_auc = float(roc_auc_score(y_test, proba))
        except ValueError:
            fold_auc = float("nan")  # only one class present in this fold's test set

        fold_rows.append(
            {
                "fold": fold_id,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "accuracy": fold_acc,
                "auc": fold_auc,
                "positive_rate_test": float(y_test.mean()),
            }
        )
        all_true.extend(y_test.tolist())
        all_pred.extend(pred.tolist())
        all_proba.extend(proba.tolist())

    fold_frame = pd.DataFrame(fold_rows)
    pooled_accuracy = float(accuracy_score(all_true, all_pred))
    try:
        pooled_auc = float(roc_auc_score(all_true, all_proba))
    except ValueError:
        pooled_auc = float("nan")

    baseline_accuracy = float(max(np.mean(all_true), 1 - np.mean(all_true)))

    note = (
        f"Out-of-sample accuracy of {pooled_accuracy:.1%} vs. a naive "
        f"'always predict majority class' baseline of {baseline_accuracy:.1%}. "
    )
    if pooled_accuracy <= baseline_accuracy + 0.03:
        note += "The model is providing little to no edge over the naive baseline at this horizon."
    else:
        note += "The model shows some edge over the naive baseline, but treat this as noisy on a single stock."

    return WalkForwardResult(
        fold_scores=fold_frame,
        out_of_sample_accuracy=pooled_accuracy,
        out_of_sample_auc=pooled_auc,
        n_folds=n_splits,
        n_samples=len(labeled),
        baseline_accuracy=baseline_accuracy,
        note=note,
    )


def latest_model_signal(
    frame: pd.DataFrame,
    horizon_days: int = 10,
    n_splits: int = 5,
    model: str = "logistic",
) -> LiveModelSignal:
    """Validate out-of-sample, then fit on all labeled history and score the latest bar.

    The probability returned is only as trustworthy as `out_of_sample_accuracy`
    reported alongside it — always surface both together in the UI rather than
    the probability alone, which invites over-interpreting a coin-flip-level
    model as a confident forecast.
    """

    validation = walk_forward_validate(frame, horizon_days=horizon_days, n_splits=n_splits, model=model)
    if validation.n_samples == 0 or validation.fold_scores.empty:
        return LiveModelSignal(float("nan"), model, 0, float("nan"), float("nan"), False, validation.note)

    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return LiveModelSignal(float("nan"), model, 0, float("nan"), float("nan"), False, "scikit-learn unavailable.")

    labeled = build_feature_target_frame(frame, horizon_days=horizon_days).dropna(subset=["ForwardReturn"])
    labeled = labeled.dropna(subset=FEATURE_COLUMNS, how="all")
    X_train = _clean_features(labeled)
    y_train = labeled["Target"].to_numpy()

    full_frame = build_feature_target_frame(frame, horizon_days=horizon_days)
    latest_row = full_frame.iloc[[-1]]
    # Use the training set's medians as the fallback for the live row, since
    # a single row has no meaningful median of its own to fall back on.
    X_latest = _clean_features(latest_row, fallback_medians=X_train.median(numeric_only=True))

    if model == "gradient_boosting":
        estimator = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=2)
    else:
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    estimator.fit(X_train, y_train)
    probability_up = float(estimator.predict_proba(X_latest)[:, 1][0])

    beats_baseline = validation.out_of_sample_accuracy > validation.baseline_accuracy + 0.03
    return LiveModelSignal(
        probability_up=probability_up,
        model_name=model,
        trained_samples=len(labeled),
        out_of_sample_accuracy=validation.out_of_sample_accuracy,
        out_of_sample_auc=validation.out_of_sample_auc,
        beats_baseline=beats_baseline,
        note=validation.note,
    )
