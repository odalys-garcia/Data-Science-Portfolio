"""
Problem 1: final random forest, test-set predictions, and held-out
confidence grid.

1. Trains the RF on the FULL training set (with rare-class augmentation)
   using the hyperparameters chosen by cv_search_rf.py, and writes the
   test-set submission: P(needs treatment) = P(BCC) + P(MEL) + P(SCC) + P(ACK).
   This part is unchanged from the run that produced the Gradescope result
   (test AUROC 0.901).

2. Trains a SECOND RF, with the same hyperparameters and augmentation, on
   only the 80% training portion of a patient-grouped split, and plots its
   3 most confidently-correct and 3 most confidently-incorrect predictions
   on the remaining 20%. Those patients were never seen by this model, so
   the figure is on genuinely held-out data.

   The split matches the fusion CNN's (train_final.py): StratifiedGroupKFold,
   5 splits, stratified on coarse_label, grouped by patient_id, seed 42,
   first fold. Both models' figures therefore come from the same
   validation patients. Correct/incorrect is judged on the coarse label
   (needs treatment or not), the same task the submission is scored on.

Outputs:
  yproba1_test_part1.txt         test-set probabilities (submission)
  rf_confidence_grid_heldout.png held-out confidence grid (report Figure 5)
  rf_heldout_val_img_ids.csv     image IDs in the held-out split, to check
                                 it matches the CNN's validation set

Usage (from data/):
    python ../RF_model/test_predictions_part1.py
"""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder


TARGET_COL = "fine_label"       # the RF is trained on the 6 fine labels
COARSE_COL = "coarse_label"     # binary task: needs treatment (1) or not (0)
RANDOM_STATE = 42

# Final hyperparameters from cv_search_rf.py
FINAL_RF_PARAMS = {
    "n_estimators": 600,
    "max_depth": 15,
    "max_features": "sqrt",
    "min_samples_split": 12,
    "min_samples_leaf": 5,
}

# Fine labels that need treatment; their probabilities are summed into
# the "needs treatment" score
POSITIVE_CLASSES = ["ACK", "BCC", "MEL", "SCC"]

# Augmented copies per training image, by fine label (training data only).
# Scaled to class rarity; BCC (~57% of training data) is not augmented.
AUG_COUNTS_BY_CLASS = {
    "MEL": 6,
    "SEK": 6,
    "NEV": 4,
    "ACK": 1,
    "SCC": 1,
    "BCC": 0,
}

ID_COLS = ["patient_id", "patient_id_x", "patient_id_y", "lesion_id", "img_id"]


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

    print("Numeric columns:", numeric_cols)
    print("Binary-like columns:", binary_like_cols)
    print("Categorical columns:", categorical_cols)

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


def drop_id_cols(X):
    return X.drop(columns=[c for c in ID_COLS if c in X.columns])


def positive_probability(pipeline, X_model):
    """P(needs treatment) = sum of the positive fine-label probabilities."""
    y_proba = pipeline.predict_proba(X_model)
    class_names = list(pipeline.named_steps["model"].classes_)
    idx = [class_names.index(c) for c in POSITIVE_CLASSES]
    return y_proba[:, idx].sum(axis=1)


# ----------------------------
# Held-out confidence grid
# ----------------------------
def display_age_group(age):
    """Same bins and labels as the fusion CNN's figure (train_final.py)."""
    if pd.isna(age):
        return "missing"
    age = float(age)
    if age < 30:
        return "0-29"
    elif age <= 60:
        return "30-60"
    return "61+"


def save_rf_confidence_grid(pipeline, X_val, y_val_coarse, y_val_fine, img_map,
                            output_path="rf_confidence_grid_heldout.png", n=3):
    """
    Plot the n most confident correct (top row) and incorrect (bottom row)
    held-out predictions on the binary needs-treatment task.
    """
    img_ids = X_val["img_id"].astype(str).values
    age_groups = X_val["age"].apply(display_age_group).values

    pos_probs = positive_probability(pipeline, drop_id_cols(X_val))
    preds = (pos_probs >= 0.5).astype(int)
    confidence = np.where(preds == 1, pos_probs, 1 - pos_probs)

    y_true = np.asarray(y_val_coarse).astype(int)
    fine_labels = np.asarray(y_val_fine)
    correct = (preds == y_true)

    correct_idx = np.where(correct)[0]
    incorrect_idx = np.where(~correct)[0]

    top_correct = correct_idx[np.argsort(confidence[correct_idx])[::-1][:n]]
    top_incorrect = incorrect_idx[np.argsort(confidence[incorrect_idx])[::-1][:n]]

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    fig.suptitle("RF: Most Confident Predictions on Held-Out Data",
                 fontsize=14, fontweight="bold")

    row_meta = [
        ("Correct", top_correct, "#2ecc71"),
        ("Incorrect", top_incorrect, "#e74c3c"),
    ]

    for row, (row_label, indices, color) in enumerate(row_meta):
        for col, idx in enumerate(indices):
            ax = axes[row, col]

            img_np = img_map.get(img_ids[idx])
            if img_np is not None:
                if img_np.dtype != np.uint8:
                    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                ax.imshow(img_np)
            else:
                ax.text(0.5, 0.5, f"No image\n{img_ids[idx]}",
                        ha="center", va="center", transform=ax.transAxes)
                ax.set_facecolor("#dddddd")

            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(6)
                spine.set_visible(True)

            ax.set_title(
                f"True: {y_true[idx]} ({fine_labels[idx]})  Pred: {preds[idx]}\n"
                f"Age group: {age_groups[idx]}\n"
                f"Confidence: {confidence[idx]:.3f}",
                fontsize=9,
            )

        axes[row, 0].set_ylabel(row_label, fontsize=12,
                                fontweight="bold", color=color, labelpad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved RF confidence grid to {output_path}")


def main():
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    # ---------------------------
    # Load training data
    # ---------------------------
    X_train_raw = pd.read_csv(Path("x_train.csv"))
    y_train_df = pd.read_csv(Path("y_train.csv"))
    train_img_feat_df = pd.read_csv(Path("x_train_img_features.csv"))

    Xy = X_train_raw.merge(
        y_train_df[["patient_id", "img_id", TARGET_COL, COARSE_COL]],
        on=["patient_id", "img_id"],
        how="inner",
        validate="one_to_one",
    )

    y_fine = Xy[TARGET_COL].copy()
    y_coarse = Xy[COARSE_COL].copy()
    X_all = Xy.drop(columns=[TARGET_COL, COARSE_COL])

    X_all = X_all.merge(
        train_img_feat_df,
        on="img_id",
        how="left",
        validate="one_to_one",
    )
    X_all = add_features(X_all)

    img_map = load_npz_image_mapping("x_train_img.npz")

    # =========================================================
    # 1. Full-training-set model -> test-set submission
    #    (same steps, in the same order, as the original run)
    # =========================================================
    X_full, y_full = augment_training_rows(
        X_train=X_all,
        y_train=y_fine,
        img_map=img_map,
        aug_counts_by_class=AUG_COUNTS_BY_CLASS,
    )
    print("Full training set class counts after augmentation:")
    print(y_full.value_counts().sort_index())

    X_full = drop_id_cols(X_full)
    full_pipeline = build_pipeline(X_full, rf_params=FINAL_RF_PARAMS, random_state=RANDOM_STATE)
    full_pipeline.fit(X_full, y_full)

    X_test = pd.read_csv(Path("x_test.csv"))
    test_img_feat_df = pd.read_csv(Path("x_test_img_features.csv"))
    X_test = X_test.merge(test_img_feat_df, on="img_id", how="left", validate="one_to_one")
    X_test = add_features(X_test)

    test_pos_probs = positive_probability(full_pipeline, drop_id_cols(X_test))
    np.savetxt("yproba1_test_part1.txt", test_pos_probs, fmt="%.10f")
    print(f"Saved yproba1_test_part1.txt with {len(test_pos_probs)} rows "
          f"(P(needs treatment) = sum of {POSITIVE_CLASSES}).")

    # =========================================================
    # 2. Held-out model -> confidence grid
    #    Same split as the fusion CNN: stratified on coarse_label,
    #    grouped by patient, seed 42, first fold.
    # =========================================================
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, val_idx = next(iter(sgkf.split(
        X_all,
        y_coarse.astype(str).values,
        X_all["patient_id"].astype(str).values,
    )))

    X_tr = X_all.iloc[train_idx].reset_index(drop=True)
    y_tr = y_fine.iloc[train_idx].reset_index(drop=True)
    X_val = X_all.iloc[val_idx].reset_index(drop=True)
    y_val_fine = y_fine.iloc[val_idx].reset_index(drop=True)
    y_val_coarse = y_coarse.iloc[val_idx].reset_index(drop=True)

    overlap = set(X_tr["patient_id"]) & set(X_val["patient_id"])
    print(f"\nHeld-out split: {len(X_tr)} train rows, {len(X_val)} val rows, "
          f"patient overlap = {len(overlap)}")
    print("Val coarse-label counts:")
    print(y_val_coarse.value_counts().sort_index())

    # Save the held-out image IDs so the split can be checked against the CNN's
    X_val[["img_id", "patient_id"]].assign(coarse_label=y_val_coarse.values) \
        .to_csv("rf_heldout_val_img_ids.csv", index=False)

    # Augment the 80% only, then fit
    X_tr_aug, y_tr_aug = augment_training_rows(
        X_train=X_tr,
        y_train=y_tr,
        img_map=img_map,
        aug_counts_by_class=AUG_COUNTS_BY_CLASS,
    )
    heldout_pipeline = build_pipeline(
        drop_id_cols(X_tr_aug), rf_params=FINAL_RF_PARAMS, random_state=RANDOM_STATE
    )
    heldout_pipeline.fit(drop_id_cols(X_tr_aug), y_tr_aug)

    save_rf_confidence_grid(
        pipeline=heldout_pipeline,
        X_val=X_val,
        y_val_coarse=y_val_coarse,
        y_val_fine=y_val_fine,
        img_map=img_map,
        output_path="rf_confidence_grid_heldout.png",
        n=3,
    )


if __name__ == "__main__":
    main()
