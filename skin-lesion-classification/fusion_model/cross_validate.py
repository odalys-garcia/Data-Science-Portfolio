"""
Cross-validation for the EfficientNet-B0 + tabular MLP fusion model.

Runs 5-fold StratifiedGroupKFold (grouped by patient, stratified on the
coarse label) and reports, per fold, the minimum age-group AUROC and the
overall AUROC at the epoch selected by minimum age-group AUROC.

Produced:
  - Table 5 of the report (default configuration, sampler power 0.15),
    via fusion_model/run_default_cv.sh
  - hyperparam_search_summary.csv (8-run fractional-factorial
    search), via fusion_model/run_factorial_search.sh

Usage (from data/):
    python ../fusion_model/cross_validate.py --head-lr 1e-3 --head-weight-decay 1e-4 \
        --dropout 0.3 --batch-size 16 --sampler-power 0.15 \
        --run-name default_run

Note: this code was cleaned for readability after the results were
produced. Changes are limited to comments, naming, logging, and removal
of unused code; model training behavior is unchanged.
"""

import argparse
import copy
import csv
import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# =========================================================
# CONFIG
# =========================================================
X_TRAIN_CSV = "x_train.csv"
Y_TRAIN_CSV = "y_train.csv"
X_TRAIN_IMG_NPZ = "x_train_img.npz"

IMAGE_KEY = "images"
IMAGE_ID_KEY = "image_ids"

TARGET_COL = "coarse_label"   # binary: needs treatment ("1") or not ("0")
GROUP_COL = "patient_id"      # CV groups, so no patient spans train and val
IMAGE_ID_COL = "img_id"

N_SPLITS = 5
NUM_EPOCHS = 10
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RANDOM_SEED = 42

# Early stopping: validation performance peaked within the first few
# epochs and then degraded, so training stops after 3 epochs without
# improvement in minimum age-group AUROC.
PATIENCE = 3
MIN_DELTA = 1e-4

# Architecture of the trainable layers (the backbone is frozen)
IMAGE_EMBED_DIM = 256
TAB_HIDDEN_DIMS = (128, 64)
FUSION_HIDDEN_DIMS = (128, 64)

# The pretrained EfficientNet-B0 backbone is frozen to limit overfitting
# on a small dataset (~1,200 images). Only the image projection, tabular
# MLP, and fusion head are trained.
FREEZE_BACKBONE = True

# Identifier columns: never used as features.
ID_COLS = ["patient_id", "img_id"]

# Engineered features excluded after the feature ablation (report Table 4).
# Diameter mean + diameter > 6mm + edge density gave the best minimum
# age-group AUROC; adding color features or the extra diameter features
# lowered it. The raw diameters are replaced by diameter_mean.
EXCLUDED_FEATURES = [
    "diameter_1",
    "diameter_2",
    "diameter_diff",
    "diameter_ratio",
    "diameter_area_ellipse",
    "img_color_entropy",
    "img_var_r",
    "img_var_g",
    "img_var_b",
]

NON_FEATURE_COLS = ID_COLS + EXCLUDED_FEATURES

# Weighted sampling over (class, age group) cells.
# The model over-predicted for negative senior patients, so that cell is
# sampled 1.5x more. Applying the same boost to adult negatives hurt
# performance, so only seniors are boosted.
NEGATIVE_LABEL = "0"
SAMPLER_MULTIPLIERS = {(NEGATIVE_LABEL, "61+"): 1.5}


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# IMAGE UTILS
# =========================================================
def np_image_to_pil(img_np):
    img_np = np.nan_to_num(img_np, nan=0.0)

    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        else:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return Image.fromarray(img_np)


def rgb_float_image(img_np):
    img = np.nan_to_num(img_np.astype(np.float32), nan=0.0)

    if img.max() <= 1.0:
        img = img * 255.0

    img = np.clip(img, 0, 255)

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected RGB image of shape (H, W, 3), got {img.shape}")

    return img


def grayscale_image(img_rgb):
    return 0.2989 * img_rgb[..., 0] + 0.5870 * img_rgb[..., 1] + 0.1140 * img_rgb[..., 2]


# =========================================================
# ENGINEERED FEATURES (ABCDE-inspired)
# =========================================================
# Edge density targets Border irregularity; color entropy and per-channel
# variance target Color variation; diameter features target Diameter.
# Only edge density and the diameter mean / > 6mm features are used in
# the final model (see EXCLUDED_FEATURES).

def compute_edge_density(img_np):
    """
    Fraction of pixels whose grayscale gradient magnitude exceeds
    mean + 1 std (an adaptive threshold). Proxy for border complexity.
    """
    img = np.nan_to_num(img_np.astype(np.float32), nan=0.0)

    if img.max() <= 1.0:
        img = img * 255.0

    if img.ndim == 3 and img.shape[2] == 3:
        gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    else:
        gray = img.astype(np.float32)

    # Finite-difference gradients, trimmed to a common shape
    gx = np.abs(gray[:, 1:] - gray[:, :-1])
    gy = np.abs(gray[1:, :] - gray[:-1, :])
    gx = gx[:-1, :]
    gy = gy[:, :-1]
    grad_mag = np.sqrt(gx**2 + gy**2)

    thresh = grad_mag.mean() + grad_mag.std()
    edges = grad_mag > thresh

    return float(edges.mean())


def compute_color_entropy(img_np, bins=32):
    """
    Shannon entropy of the grayscale intensity histogram.
    Higher values mean more intensity heterogeneity.
    """
    img_rgb = rgb_float_image(img_np)
    gray = grayscale_image(img_rgb)

    hist, _ = np.histogram(gray.ravel(), bins=bins, range=(0, 255), density=False)
    hist = hist.astype(np.float64)

    if hist.sum() == 0:
        return 0.0

    p = hist / hist.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def compute_rgb_variances(img_np):
    """Variance of each RGB channel."""
    img_rgb = rgb_float_image(img_np)
    return {
        "img_var_r": float(img_rgb[..., 0].var()),
        "img_var_g": float(img_rgb[..., 1].var()),
        "img_var_b": float(img_rgb[..., 2].var()),
    }


def build_image_feature_df(images, image_ids):
    rows = []

    for img_id, img_np in zip(image_ids, images):
        feat = {
            IMAGE_ID_COL: str(img_id),
            "img_color_entropy": compute_color_entropy(img_np),
            "img_edge_density": compute_edge_density(img_np),
        }
        feat.update(compute_rgb_variances(img_np))
        rows.append(feat)

    return pd.DataFrame(rows)


def add_diameter_engineered_features(df):
    df = df.copy()

    d1 = pd.to_numeric(df["diameter_1"], errors="coerce")
    d2 = pd.to_numeric(df["diameter_2"], errors="coerce")

    diameter_mean = (d1 + d2) / 2.0
    df["diameter_mean"] = diameter_mean
    # Melanomas are typically larger than 6mm (ABCDE "D")
    df["diameter_gt_6mm"] = (diameter_mean > 6.0).astype(float)

    df["diameter_diff"] = np.abs(d1 - d2)

    dmax = np.maximum(d1, d2)
    dmin = np.minimum(d1, d2)
    df["diameter_ratio"] = np.where(dmax > 0, dmin / dmax, np.nan)

    df["diameter_area_ellipse"] = np.pi * (d1 / 2.0) * (d2 / 2.0)

    return df


def make_age_group(age_value):
    if pd.isna(age_value):
        return "missing"

    age_value = float(age_value)

    if age_value <= 29:
        return "0-29"
    elif age_value <= 60:
        return "30-60"
    else:
        return "61+"


# =========================================================
# IMAGE TRANSFORMS
# =========================================================
# Augmentation is done online: each time a training image is drawn, it
# gets a fresh random flip / rotation / color jitter. These transforms
# preserve lesion morphology.
def build_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# =========================================================
# DATASET
# =========================================================
class SkinFusionDataset(Dataset):
    """Pairs each image with its preprocessed tabular row (and label, if given)."""

    def __init__(self, image_array, tabular_array, labels_int=None, transform=None):
        self.image_array = image_array
        self.tabular_array = tabular_array
        self.labels_int = labels_int
        self.transform = transform

    def __len__(self):
        return len(self.image_array)

    def __getitem__(self, idx):
        img = np_image_to_pil(self.image_array[idx])

        if self.transform is not None:
            img = self.transform(img)

        tab = torch.tensor(self.tabular_array[idx], dtype=torch.float32)

        if self.labels_int is None:
            return img, tab

        y = torch.tensor(self.labels_int[idx], dtype=torch.long)
        return img, tab, y


# =========================================================
# MODEL
# =========================================================
class EfficientNetTabularFusion(nn.Module):
    """
    Image branch:   frozen EfficientNet-B0 -> 1280-d features
                    -> trainable projection to a 256-d embedding
    Tabular branch: trainable MLP (128, 64)
    Fusion head:    concatenated embeddings -> MLP (128, 64) -> logits
    """

    def __init__(
        self,
        num_tabular_features,
        num_classes,
        image_embed_dim=256,
        tab_hidden_dims=(128, 64),
        fusion_hidden_dims=(128, 64),
        dropout=0.3,
        freeze_backbone=True,
    ):
        super().__init__()

        # Pretrained backbone with its classifier removed
        self.image_backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        backbone_out_dim = self.image_backbone.classifier[1].in_features
        self.image_backbone.classifier = nn.Identity()

        if freeze_backbone:
            for p in self.image_backbone.parameters():
                p.requires_grad = False

        # Trainable image projection
        self.image_projection = nn.Sequential(
            nn.Linear(backbone_out_dim, image_embed_dim),
            nn.ReLU(),
            nn.BatchNorm1d(image_embed_dim),
            nn.Dropout(dropout),
        )

        # Trainable tabular MLP
        tab_layers = []
        in_dim = num_tabular_features
        for h in tab_hidden_dims:
            tab_layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            in_dim = h
        self.tabular_mlp = nn.Sequential(*tab_layers)
        tab_out_dim = in_dim

        # Trainable fusion head
        fusion_layers = []
        in_dim = image_embed_dim + tab_out_dim
        for h in fusion_hidden_dims:
            fusion_layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.BatchNorm1d(h),
                nn.Dropout(dropout),
            ])
            in_dim = h
        fusion_layers.append(nn.Linear(in_dim, num_classes))
        self.fusion_head = nn.Sequential(*fusion_layers)

    def forward(self, images, tabular):
        img_embed = self.image_projection(self.image_backbone(images))
        tab_embed = self.tabular_mlp(tabular)
        fused = torch.cat([img_embed, tab_embed], dim=1)
        return self.fusion_head(fused)


# =========================================================
# SAMPLER
# =========================================================
def make_class_age_sampler(label_names, age_groups, power=1.0, combo_multipliers=None):
    """
    Per-sample weights for WeightedRandomSampler:

        w(c, a) = (1 / count(c, a)) ** power * multiplier(c, a)

    where (c, a) is the sample's (class, age group) cell. Rare cells are
    drawn more often; `power` controls how strongly (0 = uniform,
    1 = fully inverse-frequency).
    """
    combo_df = pd.DataFrame({
        "label": pd.Series(label_names).astype(str),
        "age_group": pd.Series(age_groups).astype(str),
    })

    combo_counts = combo_df.value_counts(["label", "age_group"]).to_dict()

    if combo_multipliers is None:
        combo_multipliers = {}

    weights = []
    for lbl, age in zip(combo_df["label"], combo_df["age_group"]):
        count = combo_counts[(lbl, age)]
        w = (1.0 / count) ** power
        w *= combo_multipliers.get((lbl, age), 1.0)
        weights.append(w)

    return torch.DoubleTensor(weights)


# =========================================================
# DATA LOADING
# =========================================================
def load_and_merge_data():
    """Merge tabular features, labels, images, and engineered image features."""
    x_df = pd.read_csv(X_TRAIN_CSV)
    y_df = pd.read_csv(Y_TRAIN_CSV)

    df = x_df.merge(
        y_df[[GROUP_COL, IMAGE_ID_COL, TARGET_COL]],
        on=[GROUP_COL, IMAGE_ID_COL],
        how="inner",
    )

    if "age" not in df.columns:
        raise ValueError("Expected 'age' column in x_train.csv")

    df["age_group"] = df["age"].apply(make_age_group)
    df = add_diameter_engineered_features(df)

    npz = np.load(X_TRAIN_IMG_NPZ, allow_pickle=True)
    images = npz[IMAGE_KEY]
    image_ids = npz[IMAGE_ID_KEY]

    img_lookup = {str(img_id): img for img_id, img in zip(image_ids, images)}
    img_feat_df = build_image_feature_df(images, image_ids)

    # Merge keys as strings
    df[IMAGE_ID_COL] = df[IMAGE_ID_COL].astype(str)
    img_feat_df[IMAGE_ID_COL] = img_feat_df[IMAGE_ID_COL].astype(str)

    # Keep only rows with a matching image
    df = df[df[IMAGE_ID_COL].isin(img_lookup.keys())].copy().reset_index(drop=True)
    df = df.merge(img_feat_df, on=IMAGE_ID_COL, how="left")

    aligned_images = np.stack([
        img_lookup[str(img_id)] for img_id in df[IMAGE_ID_COL].values
    ])

    return df, aligned_images


# =========================================================
# TABULAR PREPROCESSING (fit on the training fold only)
# =========================================================
def build_tabular_preprocessor(df_train):
    feature_df = df_train.drop(columns=[TARGET_COL], errors="ignore").copy()
    feature_df = feature_df.drop(columns=NON_FEATURE_COLS, errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])

    return preprocessor, numeric_cols, categorical_cols


def transform_tabular(preprocessor, df_train, df_val):
    x_train = df_train.drop(columns=[TARGET_COL], errors="ignore").drop(columns=NON_FEATURE_COLS, errors="ignore")
    x_val = df_val.drop(columns=[TARGET_COL], errors="ignore").drop(columns=NON_FEATURE_COLS, errors="ignore")

    x_train_tab = preprocessor.fit_transform(x_train)
    x_val_tab = preprocessor.transform(x_val)

    if hasattr(x_train_tab, "toarray"):
        x_train_tab = x_train_tab.toarray()
    if hasattr(x_val_tab, "toarray"):
        x_val_tab = x_val_tab.toarray()

    return x_train_tab.astype(np.float32), x_val_tab.astype(np.float32)


# =========================================================
# METRICS
# =========================================================
def compute_group_aurocs_binary(y_true_int, y_prob_pos, age_groups):
    """
    AUROC within each age group, plus the minimum across groups.
    The minimum is the model-selection metric: it rewards models that
    perform well for every age group, not just the dominant one.
    Groups containing only one class are skipped.
    """
    age_groups = np.asarray(age_groups)
    unique_groups = sorted(pd.Series(age_groups).astype(str).unique().tolist())
    group_auc_dict = {}

    for g in unique_groups:
        mask = (age_groups == g)
        y_g = np.asarray(y_true_int)[mask]
        p_g = np.asarray(y_prob_pos)[mask]

        if len(np.unique(y_g)) < 2:
            continue

        try:
            group_auc_dict[g] = float(roc_auc_score(y_g, p_g))
        except ValueError:
            continue

    min_group_auc = min(group_auc_dict.values()) if group_auc_dict else float("nan")
    return group_auc_dict, min_group_auc


# =========================================================
# TRAIN / VALIDATE
# =========================================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for images, tabular, labels in loader:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images, tabular)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_n += labels.size(0)

    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def validate(model, loader, criterion, device, age_groups_val):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    all_probs = []
    all_labels = []

    for images, tabular, labels in loader:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, tabular)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total_n += labels.size(0)

        # Positive-class probability only
        all_probs.append(probs[:, 1].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    overall_auc = roc_auc_score(all_labels, all_probs)
    group_auc_dict, min_group_auc = compute_group_aurocs_binary(
        all_labels, all_probs, age_groups_val
    )

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "overall_auc": overall_auc,
        "group_auc_dict": group_auc_dict,
        "min_group_auc": min_group_auc,
    }


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampler-power", type=float, default=0.15)
    parser.add_argument("--run-name", type=str, default="default_run")
    parser.add_argument("--verbose", action="store_true",
                        help="Print data, feature, and sampler diagnostics.")
    return parser.parse_args()


# =========================================================
# MAIN CV LOOP
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    args = parse_args()

    run_name = args.run_name
    metrics_log_file = f"cv_epoch_metrics_{run_name}.csv"
    group_log_file = f"cv_results_log_{run_name}.csv"
    checkpoint_file = f"best_fusion_model_{run_name}.pt"
    summary_file = "hyperparam_search_summary.csv"

    print(f"Using device: {DEVICE}")
    print(f"Run: {run_name} | lr={args.head_lr} wd={args.head_weight_decay} "
          f"dropout={args.dropout} batch={args.batch_size} power={args.sampler_power}")

    # ---------------------------
    # Load data
    # ---------------------------
    df, images = load_and_merge_data()

    print("\nClass x age-group counts:")
    print(pd.crosstab(df[TARGET_COL], df["age_group"]))

    if args.verbose:
        print("Merged dataframe shape:", df.shape)
        print("Image array shape:", images.shape)
        preview_cols = ["diameter_mean", "diameter_gt_6mm", "img_color_entropy",
                        "img_var_r", "img_var_g", "img_var_b", "img_edge_density"]
        print(df[[c for c in preview_cols if c in df.columns]].head())

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    y_all = df[TARGET_COL].astype(str).values
    groups_all = df[GROUP_COL].astype(str).values

    # ---------------------------
    # Log files
    # ---------------------------
    with open(metrics_log_file, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "run_name", "head_lr", "head_weight_decay", "dropout", "batch_size",
            "sampler_power", "fold", "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "overall_auc", "min_group_auc",
        ])

    with open(group_log_file, mode="w", newline="") as f:
        csv.writer(f).writerow(["fold", "epoch", "age_group", "age_group_auc"])

    fold_results = []
    best_fold_metric_global = -np.inf
    best_fold_checkpoint = None

    # ---------------------------
    # Folds
    # ---------------------------
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(df, y_all, groups_all), start=1):
        print("\n" + "=" * 70)
        print(f"FOLD {fold_idx}/{N_SPLITS}")
        print("=" * 70)

        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_val = df.iloc[val_idx].reset_index(drop=True)
        img_train = images[train_idx]
        img_val = images[val_idx]

        # Labels (encoder fit on the training fold)
        label_encoder = LabelEncoder()
        y_train_int = label_encoder.fit_transform(df_train[TARGET_COL].astype(str).values)
        y_val_int = label_encoder.transform(df_val[TARGET_COL].astype(str).values)
        num_classes = len(label_encoder.classes_)

        train_label_names = df_train[TARGET_COL].astype(str).values
        train_age_groups = df_train["age_group"].astype(str).values
        val_age_groups = df_val["age_group"].astype(str).values

        # Tabular preprocessing (fit on the training fold only, to avoid leakage)
        preprocessor, numeric_cols, categorical_cols = build_tabular_preprocessor(df_train)
        x_train_tab, x_val_tab = transform_tabular(preprocessor, df_train, df_val)

        # Weighted sampling over (class, age group) cells, with replacement
        sample_weights = make_class_age_sampler(
            label_names=train_label_names,
            age_groups=train_age_groups,
            power=args.sampler_power,
            combo_multipliers=SAMPLER_MULTIPLIERS,
        )
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        if args.verbose:
            print("Train target counts:\n", df_train[TARGET_COL].value_counts())
            print("Val target counts:\n", df_val[TARGET_COL].value_counts())
            print("Probability column 1 corresponds to:", label_encoder.classes_[1])
            print(f"Tabular dim after preprocessing: {x_train_tab.shape[1]}")
            print(f"Numeric cols: {numeric_cols}")
            print(f"Categorical cols: {categorical_cols}")
            combo_debug = pd.DataFrame({
                "label": train_label_names,
                "age_group": train_age_groups,
                "weight": sample_weights.numpy(),
            })
            print("Sampler weight per (class, age group):")
            print(combo_debug.groupby(["label", "age_group"])["weight"].first())

        # Datasets and loaders
        train_dataset = SkinFusionDataset(
            image_array=img_train,
            tabular_array=x_train_tab,
            labels_int=y_train_int,
            transform=build_train_transform(),
        )
        val_dataset = SkinFusionDataset(
            image_array=img_val,
            tabular_array=x_val_tab,
            labels_int=y_val_int,
            transform=build_eval_transform(),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Model and optimizer (only the non-frozen layers are optimized)
        model = EfficientNetTabularFusion(
            num_tabular_features=x_train_tab.shape[1],
            num_classes=num_classes,
            image_embed_dim=IMAGE_EMBED_DIM,
            tab_hidden_dims=TAB_HIDDEN_DIMS,
            fusion_hidden_dims=FUSION_HIDDEN_DIMS,
            dropout=args.dropout,
            freeze_backbone=FREEZE_BACKBONE,
        ).to(DEVICE)

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.head_lr,
            weight_decay=args.head_weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # ---------------------------
        # Epochs, with early stopping on minimum age-group AUROC
        # ---------------------------
        best_fold_state = None
        best_fold_metric = -np.inf
        epochs_without_improvement = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
            val_metrics = validate(model, val_loader, criterion, DEVICE, val_age_groups)

            with open(metrics_log_file, mode="a", newline="") as f:
                csv.writer(f).writerow([
                    run_name, args.head_lr, args.head_weight_decay, args.dropout,
                    args.batch_size, args.sampler_power, fold_idx, epoch,
                    train_loss, train_acc, val_metrics["loss"], val_metrics["acc"],
                    val_metrics["overall_auc"], val_metrics["min_group_auc"],
                ])

            with open(group_log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                for g, auc in val_metrics["group_auc_dict"].items():
                    writer.writerow([fold_idx, epoch, g, auc])

            print(f"\nFold {fold_idx} | Epoch {epoch}/{NUM_EPOCHS}")
            print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
            print(f"  Val loss  : {val_metrics['loss']:.4f} | Val acc: {val_metrics['acc']:.4f}")
            print(f"  Val overall AUROC       : {val_metrics['overall_auc']:.4f}")
            print(f"  Val min age-group AUROC : {val_metrics['min_group_auc']:.4f}")
            for g, auc in sorted(val_metrics["group_auc_dict"].items()):
                print(f"    Age group {g}: AUROC = {auc:.4f}")

            # Fall back to overall AUROC if no age group had both classes
            selection_metric = val_metrics["min_group_auc"]
            if not np.isfinite(selection_metric):
                selection_metric = val_metrics["overall_auc"]

            if np.isfinite(selection_metric) and selection_metric > best_fold_metric + MIN_DELTA:
                best_fold_metric = selection_metric
                epochs_without_improvement = 0
                best_fold_state = {
                    "fold": fold_idx,
                    "epoch": epoch,
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                    "label_classes": label_encoder.classes_.tolist(),
                    "num_tabular_features": x_train_tab.shape[1],
                    "best_min_group_auc": val_metrics["min_group_auc"],
                    "best_overall_auc": val_metrics["overall_auc"],
                    "selection_metric": selection_metric,
                }
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

        if best_fold_state is None:
            print(f"No valid checkpoint was found for fold {fold_idx}. Skipping this fold.")
            continue

        # Fold result = metrics at the epoch selected by minimum age-group
        # AUROC (so "best_overall_auc" is the overall AUROC at that epoch,
        # not the maximum overall AUROC across epochs).
        fold_summary = {
            "fold": fold_idx,
            "best_min_group_auc": best_fold_state["best_min_group_auc"],
            "best_overall_auc": best_fold_state["best_overall_auc"],
        }
        fold_results.append(fold_summary)

        print(f"\nFold {fold_idx} selected epoch {best_fold_state['epoch']}: "
              f"min age-group AUROC {fold_summary['best_min_group_auc']:.4f}, "
              f"overall AUROC {fold_summary['best_overall_auc']:.4f}")

        if np.isfinite(fold_summary["best_min_group_auc"]) and fold_summary["best_min_group_auc"] > best_fold_metric_global:
            best_fold_metric_global = fold_summary["best_min_group_auc"]
            best_fold_checkpoint = best_fold_state

    # ---------------------------
    # Cross-fold summary
    # ---------------------------
    if len(fold_results) == 0:
        print("No valid folds produced a checkpoint.")
        return

    results_df = pd.DataFrame(fold_results)
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(results_df)

    # .std() uses the sample standard deviation (n - 1)
    print(f"\nMean min age-group AUROC: {results_df['best_min_group_auc'].mean():.4f} "
          f"(std {results_df['best_min_group_auc'].std():.4f})")
    print(f"Mean overall AUROC      : {results_df['best_overall_auc'].mean():.4f} "
          f"(std {results_df['best_overall_auc'].std():.4f})")

    if best_fold_checkpoint is not None:
        torch.save(best_fold_checkpoint, checkpoint_file)
        print(f"\nSaved best fold checkpoint to {checkpoint_file}")

    # Append this run to the hyperparameter search summary
    file_exists = os.path.exists(summary_file)
    with open(summary_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "head_lr", "head_weight_decay", "dropout", "batch_size",
                "sampler_power", "mean_min_group_auc", "std_min_group_auc",
                "mean_overall_auc", "std_overall_auc",
            ])
        writer.writerow([
            args.head_lr, args.head_weight_decay, args.dropout, args.batch_size,
            args.sampler_power,
            results_df["best_min_group_auc"].mean(),
            results_df["best_min_group_auc"].std(),
            results_df["best_overall_auc"].mean(),
            results_df["best_overall_auc"].std(),
        ])


if __name__ == "__main__":
    main()
