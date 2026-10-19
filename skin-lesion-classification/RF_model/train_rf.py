"""
Problem 1, step 1: single-split exploration and per-class threshold analysis
for the random forest.

Fits an RF on one patient-grouped 80/20 split (the first fold of a 5-fold
StratifiedGroupKFold, stratified on fine_label) and reports, on the 20%:
  - a classification report under custom per-class thresholds
    (MEL >= 0.10, NEV >= 0.25)
  - macro, weighted, and per-class one-vs-rest AUROC

The thresholds were explored here to understand the precision/recall
trade-off for MEL (rare, malignant: missing one is costly) and NEV
(benign: over-flagging leads to overtreatment). Model selection in
cv_search_rf.py and the test submission use threshold-free probabilities.

The RF settings in build_pipeline() (400 trees, no depth limit,
min_samples_split=10, min_samples_leaf=3) are the starting values used
before the hyperparameter search, not the final model. The final
hyperparameters are in test_predictions_part1.py.

Inputs: x_train.csv, y_train.csv, x_train_img_features.csv (from
make_img_features.py), x_train_img.npz.

Usage (from data/; as run via RF_model/train_rf.sh):
    python ../RF_model/train_rf.py --img-features x_train_img_features.csv --augment-rare
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

TARGET_COL = 'fine_label'

# Augmented copies per training image, by fine label (training split only).
# Scaled to class rarity; BCC (~57% of training data) is not augmented.
# Same values as cv_search_rf.py and test_predictions_part1.py.
AUG_COUNTS_BY_CLASS = {
    'MEL': 6,
    'SEK': 6,
    'NEV': 4,
    'ACK': 1,
    'SCC': 1,
    'BCC': 0,
}

# Per-class probability thresholds to evaluate. Other values were tried
# by editing this list; these are the ones reported.
THRESHOLD_SETS = [
    {"MEL": 0.10, "NEV": 0.25},
]

ID_COLS = ['patient_id', 'patient_id_x', 'patient_id_y', 'lesion_id', 'img_id']


# ----------------------------
# Feature engineering
# ----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 30, 60, 120],
            labels=['Young (<30)', 'Adult (30-60)', 'Senior (60+)']
        )

    if 'diameter_1' in df.columns and 'diameter_2' in df.columns:
        df['diameter_mean'] = df[['diameter_1', 'diameter_2']].mean(axis=1)
        df['diameter_diff'] = (df['diameter_1'] - df['diameter_2']).abs()

    return df


def normalize_binary_df(df):
    """Map true/false, 1/0, yes/no to 'True'/'False'; everything else to 'Missing'."""
    df = pd.DataFrame(df).copy()
    mapping = {
        'true': 'True',
        'false': 'False',
        '1': 'True',
        '0': 'False',
        'yes': 'True',
        'no': 'False',
    }
    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(mapping)
            .where(lambda s: s.notna(), other='Missing')
        )
    return df


def split_column_types(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    object_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

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
        if vals and vals.issubset({'true', 'false', '0', '1', 'yes', 'no'}):
            binary_like_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, binary_like_cols, categorical_cols


def build_pipeline(X: pd.DataFrame, random_state: int = 42):
    numeric_cols, binary_like_cols, categorical_cols = split_column_types(X)

    print('Numeric columns:', numeric_cols)
    print('Binary-like columns:', binary_like_cols)
    print('Categorical columns:', categorical_cols)

    numeric_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
    ])

    # Missing values become an explicit "Missing" category: in this dataset
    # a missing field often means "unverified", not "negative".
    binary_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('normalize', FunctionTransformer(normalize_binary_df, validate=False)),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipe, numeric_cols),
        ('bin', binary_pipe, binary_like_cols),
        ('cat', categorical_pipe, categorical_cols),
    ])

    # Starting hyperparameters (before the CV search)
    rf_pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=10,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    return rf_pipeline


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
        0.299 * img_rgb[..., 0] +
        0.587 * img_rgb[..., 1] +
        0.114 * img_rgb[..., 2]
    ).astype(np.float32)

    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    gx[:, 1:] = np.abs(gray[:, 1:] - gray[:, :-1])
    gy[1:, :] = np.abs(gray[1:, :] - gray[:-1, :])

    grad_mag = np.sqrt(gx**2 + gy**2)
    edges = grad_mag > 20.0
    return float(edges.mean())


def extract_image_features_from_array(img_rgb: np.ndarray) -> dict:
    img_rgb = np.asarray(img_rgb)

    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)

    return {
        'img_mean_color': float(img_rgb.mean()),
        'img_color_variance': float(img_rgb.var()),
        'img_edge_density': compute_edge_density(img_rgb),
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
        raise ValueError(f"Could not infer image/id keys from {npz_path}. Found keys: {npz_obj.files}")

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
    train_df['_target_'] = y_train.values

    eligible_df = train_df[train_df['_target_'].isin(aug_counts_by_class.keys())].copy()

    for _, row in eligible_df.iterrows():
        img_id = str(row['img_id'])
        label = row['_target_']

        if img_id not in img_map:
            continue

        n_aug = aug_counts_by_class.get(label, 0)
        if n_aug <= 0:
            continue

        img_array = img_map[img_id]
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(img_array)

        for aug_idx in range(1, n_aug + 1):
            aug_np = np.array(augment_image(pil_img))

            new_row = row.drop(labels=['_target_']).copy()
            new_row['img_id'] = f"{img_id}_aug{aug_idx}"

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


def predict_with_thresholds(y_proba, class_names, thresholds):
    """
    For each row, among the classes whose probability clears their own
    threshold, predict the one with the highest probability. If no class
    clears its threshold, fall back to the overall argmax.
    """
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}
    y_pred = []

    for p in y_proba:
        chosen_class = None
        chosen_score = -1.0

        for cls, thr in thresholds.items():
            idx = class_to_idx[cls]
            if p[idx] >= thr and p[idx] > chosen_score:
                chosen_class = cls
                chosen_score = p[idx]

        if chosen_class is None:
            chosen_class = class_names[np.argmax(p)]

        y_pred.append(chosen_class)

    return y_pred


def main():
    parser = argparse.ArgumentParser(description='Single-split RF exploration and per-class threshold analysis.')
    parser.add_argument('--x', default='x_train.csv', help='Training features CSV')
    parser.add_argument('--y', default='y_train.csv', help='Training labels CSV')
    parser.add_argument('--img-features', default=None, help='Optional CSV with img_id and extracted image features')
    # test_size=0.2 is implemented as the first fold of a 5-fold StratifiedGroupKFold
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation fraction')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--drop-diagnostic', action='store_true', help='Drop diagnostic column if present')
    parser.add_argument('--drop-biopsed', action='store_true', help='Drop biopsed column if present')
    parser.add_argument('--train-img-npz', default='x_train_img.npz', help='Training image NPZ')
    parser.add_argument('--augment-rare', action='store_true', help='Augment rare classes in training split only')
    args = parser.parse_args()

    random.seed(args.random_state)
    np.random.seed(args.random_state)

    # ---------------------------
    # Load and merge (by keys, not row order)
    # ---------------------------
    X = pd.read_csv(Path(args.x))
    y_df = pd.read_csv(Path(args.y))

    Xy = X.merge(
        y_df[['patient_id', 'img_id', TARGET_COL]],
        on=['patient_id', 'img_id'],
        how='inner',
        validate='one_to_one'
    )

    y = Xy[TARGET_COL]
    X = Xy.drop(columns=[TARGET_COL])

    if args.img_features:
        img_feat_df = pd.read_csv(Path(args.img_features))
        if 'img_id' not in X.columns or 'img_id' not in img_feat_df.columns:
            raise ValueError('Both X and img-features CSV must contain img_id for merging.')
        X = X.merge(img_feat_df, on='img_id', how='left', validate='one_to_one')
        print(f'Merged image features from {args.img_features}')

    X = add_features(X)

    # ---------------------------
    # Patient-grouped split, done BEFORE augmentation so augmented copies
    # never reach the validation set
    # ---------------------------
    if 'patient_id' in X.columns:
        groups = X['patient_id']
    elif 'patient_id_x' in X.columns:
        groups = X['patient_id_x']
    else:
        raise ValueError("No patient_id column found for grouped split.")

    n_splits = int(round(1 / args.test_size))
    if n_splits < 2:
        n_splits = 5

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=args.random_state,
    )

    train_idx, val_idx = next(sgkf.split(X, y, groups=groups))

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()

    train_patients = set(groups.iloc[train_idx])
    val_patients = set(groups.iloc[val_idx])
    overlap = train_patients & val_patients

    print(f"Train rows: {len(X_train)}")
    print(f"Val rows:   {len(X_val)}")
    print(f"Train patients: {len(train_patients)}")
    print(f"Val patients:   {len(val_patients)}")
    print(f"Patient overlap: {len(overlap)}")

    print("\nTrain class counts:")
    print(y_train.value_counts().sort_index())

    print("\nVal class counts:")
    print(y_val.value_counts().sort_index())

    # ---------------------------
    # Augment the training split only
    # ---------------------------
    if args.augment_rare:
        img_map = load_npz_image_mapping(args.train_img_npz)

        X_train, y_train = augment_training_rows(
            X_train=X_train,
            y_train=y_train,
            img_map=img_map,
            aug_counts_by_class=AUG_COUNTS_BY_CLASS,
        )
        print(f'After augmentation: {len(X_train)} training rows')
        print("\nTraining class counts after augmentation:")
        print(y_train.value_counts().sort_index())

    # ---------------------------
    # Drop non-feature columns
    # ---------------------------
    non_feature_cols = list(ID_COLS)
    if args.drop_diagnostic:
        non_feature_cols.append('diagnostic')
    if args.drop_biopsed:
        non_feature_cols.append('biopsed')

    X_train = X_train.drop(columns=[c for c in non_feature_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in non_feature_cols if c in X_val.columns])

    # ---------------------------
    # Fit and evaluate
    # ---------------------------
    rf_pipeline = build_pipeline(X_train, random_state=args.random_state)
    rf_pipeline.fit(X_train, y_train)

    y_proba = rf_pipeline.predict_proba(X_val)
    class_names = rf_pipeline.named_steps['model'].classes_

    for thresholds in THRESHOLD_SETS:
        y_pred_custom = predict_with_thresholds(y_proba, class_names, thresholds)

        print(f"\n=== Thresholds: {thresholds} ===")
        print(len(y_val), len(y_pred_custom))
        print(classification_report(y_val, y_pred_custom))

    auc_macro = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
    auc_weighted = roc_auc_score(y_val, y_proba, multi_class='ovr', average='weighted')
    auc_per_class = roc_auc_score(y_val, y_proba, multi_class='ovr', average=None)

    print(f'Macro AUROC:    {auc_macro:.6f}')
    print(f'Weighted AUROC: {auc_weighted:.6f}')

    print('Per-class AUROC:')
    for cls, auc in zip(class_names, auc_per_class):
        print(f'  {cls}: {auc:.6f}')


if __name__ == '__main__':
    main()
