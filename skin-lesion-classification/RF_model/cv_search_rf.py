"""
Problem 1, step 2: 5-fold cross-validated hyperparameter search for the
random forest.

Each config in rf_grid is evaluated with 5-fold StratifiedGroupKFold
(grouped by patient, stratified on fine_label). Within each fold, rare
classes are augmented in the training fold only, then the RF is scored on
the validation fold by MINIMUM age-group macro AUROC: macro one-vs-rest
AUROC is computed separately for each age group and the lowest one is
kept, so a config can't score well by fitting only the dominant class or
age group. Train-fold scores are also recorded for the train vs.
validation curves.

Search history (greedy, one hyperparameter at a time; each sweep was run
by editing rf_grid and fixing the best value before the next sweep):
  1. max_depth          10, 12, 15, 50, 100       -> 15   (report Figure 1)
  2. min_samples_leaf   1, 3, 5, 7, 9, 10         -> 5    (Figure 2)
  3. n_estimators       200, 400, 600, 800        -> 600  (Figure 3)
  4. min_samples_split  12, 15, 20, 30            -> 12   (Figure 4)
max_features was fixed at sqrt. The rf_grid below is the final sweep
(4), which is why its outputs are named *_rf4.csv.

Inputs: x_train.csv, y_train.csv, x_train_img_features.csv (from
make_img_features.py), x_train_img.npz.

Writes:
  cv_fold_results_rf4.csv     per-fold, per-config metrics
  cv_summary_results_rf4.csv  mean and std per config

Usage (from data/):
    python ../RF_model/cv_search_rf.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


TARGET_COL = "fine_label"

# Augmented copies per training image, by fine label (training folds only).
# Same values as train_rf.py and test_predictions_part1.py.
AUG_COUNTS_BY_CLASS = {
    "MEL": 6,
    "SEK": 6,
    "NEV": 4,
    "ACK": 1,
    "SCC": 1,
    "BCC": 0,
}


# ----------------------------
# Feature engineering
# ----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "age" in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 30, 60, 120],
            labels=["Young (<30)", "Adult (30-60)", "Senior (60+)"],
        )

    if "diameter_1" in df.columns and "diameter_2" in df.columns:
        df["diameter_mean"] = df[["diameter_1", "diameter_2"]].mean(axis=1)
        df["diameter_diff"] = (df["diameter_1"] - df["diameter_2"]).abs()

    return df


def normalize_binary_df(df):
    """Map true/false, 1/0, yes/no to 'True'/'False'; everything else to 'Missing'."""
    df = pd.DataFrame(df).copy()
    mapping = {
        "true": "True",
        "false": "False",
        "1": "True",
        "0": "False",
        "yes": "True",
        "no": "False",
    }
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .where(lambda s: s.notna(), other="Missing")
        )
    return df


def split_column_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    object_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    binary_like_cols = []
    categorical_cols = []

    for col in object_cols:
        vals = set(
            X[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
            .unique()
        )
        if vals and vals.issubset({"true", "false", "0", "1", "yes", "no"}):
            binary_like_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, binary_like_cols, categorical_cols


def build_pipeline(X: pd.DataFrame, rf_params: dict, random_state: int = 42):
    numeric_cols, binary_like_cols, categorical_cols = split_column_types(X)

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Missing values become an explicit "Missing" category: in this dataset
    # a missing field often means "unverified", not "negative".
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("normalize", FunctionTransformer(normalize_binary_df, validate=False)),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("bin", binary_pipe, binary_like_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
        **rf_params,
    )

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])


# ----------------------------
# Augmentation helpers
# ----------------------------
def augment_image(pil_img):
    """Random rotation, flips, and mild brightness/contrast changes."""
    img = pil_img.copy()

    angle = random.uniform(-15, 15)
    img = img.rotate(angle)

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    brightness_factor = 1 + random.uniform(-0.1, 0.1)
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    contrast_factor = 1 + random.uniform(-0.1, 0.1)
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    return img


def compute_edge_density(img_rgb: np.ndarray) -> float:
    """Fraction of pixels with grayscale gradient magnitude above 20."""
    gray = (
        0.299 * img_rgb[..., 0]
        + 0.587 * img_rgb[..., 1]
        + 0.114 * img_rgb[..., 2]
    ).astype(np.float32)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])

    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    edges = grad_mag > 20.0
    return float(edges.mean())


def extract_image_features_from_array(img_rgb: np.ndarray) -> dict:
    img_rgb = np.asarray(img_rgb)
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    return {
        "img_mean_color": float(img_rgb.mean()),
        "img_color_variance": float(img_rgb.var()),
        "img_edge_density": compute_edge_density(img_rgb),
    }


def load_npz_image_mapping(npz_path: str):
    npz_obj = np.load(npz_path, allow_pickle=True)

    image_key = None
    id_key = None
    for k in npz_obj.files:
        arr = npz_obj[k]
        if arr.ndim == 4 and arr.shape[-1] == 3:
            image_key = k
        elif arr.ndim == 1:
            id_key = k

    if image_key is None or id_key is None:
        raise ValueError(f"Could not infer keys from {npz_path}. Keys: {npz_obj.files}")

    images = npz_obj[image_key]
    img_ids = [str(x) for x in npz_obj[id_key]]
    return {img_id: img for img_id, img in zip(img_ids, images)}


def augment_training_rows(X_train, y_train, img_map, aug_counts_by_class):
    """
    Append augmented copies of training rows. Each copy keeps the clinical
    features and gets image features recomputed from an augmented image.
    """
    X_aug_rows = []
    y_aug_rows = []

    train_df = X_train.copy()
    train_df["_target_"] = y_train.values

    eligible_df = train_df[train_df["_target_"].isin(aug_counts_by_class.keys())].copy()

    for _, row in eligible_df.iterrows():
        img_id = str(row["img_id"])
        label = row["_target_"]

        n_aug = aug_counts_by_class.get(label, 0)
        if n_aug <= 0 or img_id not in img_map:
            continue

        img_array = img_map[img_id]
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img_array)

        for aug_idx in range(1, n_aug + 1):
            aug_np = np.array(augment_image(pil_img))

            new_row = row.drop(labels=["_target_"]).copy()
            new_row["img_id"] = f"{img_id}_aug{aug_idx}"

            for k, v in extract_image_features_from_array(aug_np).items():
                new_row[k] = v

            X_aug_rows.append(dict(new_row))
            y_aug_rows.append(label)

    if len(X_aug_rows) == 0:
        return X_train.copy(), y_train.copy()

    X_aug = pd.DataFrame(X_aug_rows)
    y_aug = pd.Series(y_aug_rows, name=y_train.name)

    X_train_final = pd.concat([X_train.reset_index(drop=True), X_aug], ignore_index=True)
    y_train_final = pd.concat([y_train.reset_index(drop=True), y_aug], ignore_index=True)

    return X_train_final, y_train_final


# ----------------------------
# Metrics
# ----------------------------
def assign_age_group(age):
    # Note: age 30 falls in the youngest group here (<= 30). The fusion CNN
    # scripts put age 30 in the 30-60 group (< 30), so subgroup metrics
    # differ slightly at that boundary between the two models.
    if pd.isna(age):
        return "Missing"
    if age <= 30:
        return "le_30"
    elif age <= 60:
        return "30_60"
    else:
        return "gt_60"


def safe_multiclass_auc(y_true, y_proba):
    """Macro one-vs-rest AUROC, or NaN if undefined (e.g. only one class present)."""
    y_true = pd.Series(y_true)
    if y_true.nunique() < 2:
        return np.nan
    try:
        return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        return np.nan


def compute_fold_metrics(y_val, y_proba, class_names, X_val_with_age):
    """
    Overall macro/weighted AUROC, macro AUROC within each age group, and
    the minimum across age groups (the model-selection metric).

    class_names is not used in the computation; it was kept for tracking
    the model's class order while debugging.
    """
    metrics = {}

    metrics["overall_macro_auroc"] = safe_multiclass_auc(y_val, y_proba)
    try:
        metrics["overall_weighted_auroc"] = roc_auc_score(
            y_val, y_proba, multi_class="ovr", average="weighted"
        )
    except ValueError:
        metrics["overall_weighted_auroc"] = np.nan

    age_groups = X_val_with_age["age"].apply(assign_age_group)

    age_metric_names = {
        "le_30": "age_le_30_macro_auroc",
        "30_60": "age_30_60_macro_auroc",
        "gt_60": "age_gt_60_macro_auroc",
    }

    subgroup_scores = []

    for grp, metric_name in age_metric_names.items():
        mask = age_groups == grp
        y_sub = y_val[mask]
        proba_sub = y_proba[mask]

        metrics[f"val_n_{grp}"] = int(mask.sum())

        if mask.sum() == 0:
            metrics[metric_name] = np.nan
        else:
            score = safe_multiclass_auc(y_sub, proba_sub)
            metrics[metric_name] = score
            if not np.isnan(score):
                subgroup_scores.append(score)

    metrics["min_age_group_macro_auroc"] = (
        float(np.min(subgroup_scores)) if len(subgroup_scores) > 0 else np.nan
    )

    metrics["n_valid_age_groups"] = len(subgroup_scores)
    metrics["val_size"] = len(y_val)

    return metrics


# ----------------------------
# CV search
# ----------------------------
def run_cv_search(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    img_map: dict,
    rf_grid: list[dict],
    aug_counts_by_class: dict,
    fold_records=None,
    n_splits: int = 5,
    random_state: int = 42,
):
    if fold_records is None:
        fold_records = []

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    results = []

    for config_idx, rf_params in enumerate(rf_grid, start=1):
        fold_scores = []

        print(f"\n=== Config {config_idx}/{len(rf_grid)} ===")
        print(rf_params)

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups), start=1):
            # Per-fold seed so augmentation is reproducible and identical across configs
            random.seed(random_state + fold_idx)
            np.random.seed(random_state + fold_idx)

            X_train = X.iloc[train_idx].copy()
            X_val = X.iloc[val_idx].copy()
            y_train = y.iloc[train_idx].copy()
            y_val = y.iloc[val_idx].copy()

            # Augment the training fold only
            X_train, y_train = augment_training_rows(
                X_train=X_train,
                y_train=y_train,
                img_map=img_map,
                aug_counts_by_class=aug_counts_by_class,
            )
            X_val_with_age = X_val.copy()
            X_train_with_age = X_train.copy()

            drop_cols = ["patient_id", "patient_id_x", "patient_id_y", "lesion_id", "img_id"]
            drop_train = [c for c in drop_cols if c in X_train.columns]
            drop_val = [c for c in drop_cols if c in X_val.columns]

            X_train = X_train.drop(columns=drop_train)
            X_val = X_val.drop(columns=drop_val)

            pipeline = build_pipeline(X_train, rf_params=rf_params, random_state=random_state)
            pipeline.fit(X_train, y_train)

            y_proba = pipeline.predict_proba(X_val)
            # Train-fold predictions, for the train vs. validation curves.
            # Note: these are scored on the AUGMENTED training fold.
            y_proba_train = pipeline.predict_proba(X_train)
            class_names = pipeline.named_steps["model"].classes_

            train_metrics = compute_fold_metrics(
                y_val=y_train.reset_index(drop=True),
                y_proba=y_proba_train,
                class_names=class_names,
                X_val_with_age=X_train_with_age.reset_index(drop=True),
            )

            fold_metrics = compute_fold_metrics(
                y_val=y_val.reset_index(drop=True),
                y_proba=y_proba,
                class_names=class_names,
                X_val_with_age=X_val_with_age.reset_index(drop=True),
            )

            fold_record = {
                "config_id": config_idx,
                "fold": fold_idx,
                **rf_params,
                **fold_metrics,  # validation
                "train_min_age_group_macro_auroc": train_metrics["min_age_group_macro_auroc"],
            }
            fold_records.append(fold_record)

            score = fold_metrics["min_age_group_macro_auroc"]
            fold_scores.append(score)

            print(f"  Fold {fold_idx:02d}: min age-group macro AUROC = {score:.4f}")

        # np.nanstd is the population standard deviation (ddof=0)
        mean_score = float(np.nanmean(fold_scores))
        std_score = float(np.nanstd(fold_scores))

        print(f"Mean min age-group macro AUROC: {mean_score:.4f} ± {std_score:.4f}")

        results.append({
            "rf_params": rf_params,
            "fold_scores": fold_scores,
            "mean_min_age_group_macro_auroc": mean_score,
            "std_min_age_group_macro_auroc": std_score,
        })

    results = sorted(results, key=lambda d: d["mean_min_age_group_macro_auroc"], reverse=True)
    return results


def main():
    random_state = 42
    random.seed(random_state)
    np.random.seed(random_state)

    X = pd.read_csv(Path("x_train.csv"))
    y_df = pd.read_csv(Path("y_train.csv"))
    img_feat_df = pd.read_csv(Path("x_train_img_features.csv"))

    Xy = X.merge(
        y_df[["patient_id", "img_id", TARGET_COL]],
        on=["patient_id", "img_id"],
        how="inner",
        validate="one_to_one",
    )

    y = Xy[TARGET_COL]
    X = Xy.drop(columns=[TARGET_COL])

    X = X.merge(img_feat_df, on="img_id", how="left", validate="one_to_one")
    X = add_features(X)

    groups = X["patient_id"].copy()
    img_map = load_npz_image_mapping("x_train_img.npz")

    # Sweep 4 (final): min_samples_split, with the earlier choices fixed
    rf_grid = [
        {"n_estimators": 600, "max_depth": 15, "min_samples_split": 12, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 600, "max_depth": 15, "min_samples_split": 15, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 600, "max_depth": 15, "min_samples_split": 20, "min_samples_leaf": 5, "max_features": "sqrt"},
        {"n_estimators": 600, "max_depth": 15, "min_samples_split": 30, "min_samples_leaf": 5, "max_features": "sqrt"},
    ]

    fold_records = []

    results = run_cv_search(
        X=X,
        y=y,
        groups=groups,
        img_map=img_map,
        rf_grid=rf_grid,
        aug_counts_by_class=AUG_COUNTS_BY_CLASS,
        n_splits=5,
        random_state=random_state,
        fold_records=fold_records,
    )

    fold_df = pd.DataFrame(fold_records)
    fold_df.to_csv("cv_fold_results_rf4.csv", index=False)

    summary_df = (
        fold_df.groupby("config_id", as_index=False)
        .agg({
            "n_estimators": "first",
            "max_depth": "first",
            "min_samples_split": "first",
            "min_samples_leaf": "first",
            "max_features": "first",
            "overall_macro_auroc": ["mean", "std"],
            "overall_weighted_auroc": ["mean", "std"],
            "age_le_30_macro_auroc": ["mean", "std"],
            "age_30_60_macro_auroc": ["mean", "std"],
            "age_gt_60_macro_auroc": ["mean", "std"],
            "min_age_group_macro_auroc": ["mean", "std"],
        })
    )

    summary_df.columns = [
        "_".join(col).strip("_") for col in summary_df.columns.to_flat_index()
    ]
    summary_df.to_csv("cv_summary_results_rf4.csv", index=False)

    print("\n=== Final ranking ===")
    for i, r in enumerate(results, start=1):
        print(
            f"{i}. mean_min_age_group_macro_auroc={r['mean_min_age_group_macro_auroc']:.4f} "
            f"std={r['std_min_age_group_macro_auroc']:.4f} "
            f"params={r['rf_params']}"
        )


if __name__ == "__main__":
    main()
