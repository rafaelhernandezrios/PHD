from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_window_features() -> pd.DataFrame:
    csv_path = PROJECT_ROOT / "csv" / "eeg_window_features.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run 'eeg_make_window_features.py' first."
        )

    df = pd.read_csv(csv_path)

    required_cols = {"file_name", "subject_id", "task_type", "condition_4", "load_3"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Drop unknown labels if any
    df = df[df["condition_4"] != "unknown"].copy()
    df = df[df["load_3"] != "unknown"].copy()

    return df.reset_index(drop=True)


def split_by_file(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split at file level to avoid leakage, stratifying by the dominant class
    of each file for the chosen target column.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not present in DataFrame.")

    if "file_name" not in df.columns:
        raise ValueError("Expected 'file_name' column for file-level split.")

    file_level = (
        df.groupby("file_name")[target_col]
        .agg(lambda s: s.mode().iloc[0])
        .reset_index()
    )

    train_files, test_files = train_test_split(
        file_level["file_name"],
        test_size=test_size,
        random_state=random_state,
        stratify=file_level[target_col],
    )

    train_mask = df["file_name"].isin(train_files)
    test_mask = df["file_name"].isin(test_files)

    df_train = df[train_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    return df_train, df_test


def build_feature_matrix(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract X (features), y (labels) and feature names from the window-level DataFrame.
    """
    y = df[target_col].astype(str).to_numpy()

    non_feature_cols = {
        "file_name",
        "subject_id",
        "task_type",
        "condition_4",
        "load_3",
        "window_start_idx",
        "window_end_idx",
    }

    feature_cols = [
        c
        for c in df.columns
        if c not in non_feature_cols and pd.api.types.is_numeric_dtype(df[c])
    ]

    X = df[feature_cols].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, feature_cols


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str,
) -> None:
    print(f"\n=== Training for target: {target_col} ===")

    df_train, df_test = split_by_file(df, target_col=target_col)

    X_train, y_train, feature_cols = build_feature_matrix(df_train, target_col)
    X_test, y_test, _ = build_feature_matrix(df_test, target_col)

    print(
        f"Train files: {df_train['file_name'].nunique()}, "
        f"test files: {df_test['file_name'].nunique()} -> "
        f"train windows: {len(X_train)}, test windows: {len(X_test)}, "
        f"features: {len(feature_cols)}"
    )

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_depth=None,
            learning_rate=0.1,
            max_iter=300,
            random_state=42,
            class_weight="balanced",
        ),
    }

    for name, clf in models.items():
        print(f"\n--- Model: {name} ---")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        labels = sorted(set(y_train) | set(y_test))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred, labels=labels))

        # Feature importance only for models that expose it
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            idx_sorted = np.argsort(importances)[::-1]
            top_k = min(20, len(feature_cols))

            print("\nTop feature importances:")
            for i in range(top_k):
                idx = idx_sorted[i]
                print(f"{i+1:2d}. {feature_cols[idx]}: {importances[idx]:.4f}")


def main() -> None:
    df = load_window_features()

    # 1) Train for original 4-level stress condition
    train_and_evaluate(df, target_col="condition_4")

    # 2) Train for 3-level coarse load (your project)
    train_and_evaluate(df, target_col="load_3")


if __name__ == "__main__":
    main()

