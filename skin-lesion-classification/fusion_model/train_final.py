"""
Final training run and test-set predictions for the EfficientNet-B0 +
tabular MLP fusion model.

Trains once on the 80% training portion of the first patient-grouped
StratifiedGroupKFold split, keeps the epoch with the best validation
minimum age-group AUROC (on the remaining 20%), then predicts the test set.

Produced:
  - yproba1_test.txt     positive-class probabilities for the test set
                         (test AUROC 0.863, minimum age-group AUROC 0.797)
  - confidence_grid.png  most confident correct / incorrect validation
                         predictions (report Figure 6)

Final configuration: lr=1e-3, weight decay=1e-4, dropout=0.3, batch
size=16, sampler power=0.2. The hyperparameters are the defaults that
were kept after cross-validation; sampler power 0.2 outperformed 0.1 in
the fractional-factorial search. (The default CV run in report Table 5
used power 0.15.)

Usage (from data/):
    python ../fusion_model/train_final.py

Note: this code was cleaned for readability after the results were
produced. Changes are limited to comments, naming, logging, and removal
of unused code; model training behavior is unchanged.
"""

import argparse
import copy
import csv
import random

import matplotlib.pyplot as plt
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

X_TEST_CSV = "x_test.csv"
X_TEST_IMG_NPZ = "x_test_img.npz"

IMAGE_KEY = "images"
IMAGE_ID_KEY = "image_ids"

TARGET_COL = "coarse_label"   # binary: needs treatment ("1") or not ("0")
GROUP_COL = "patient_id"      # split groups, so no patient spans train and val
IMAGE_ID_COL = "img_id"

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

# Identifier and label-detail columns: never used as features.
# fine_label is loaded only to annotate the confidence-grid figure.
ID_COLS = ["patient_id", "img_id", "fine_label"]

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

    if age_value < 30:
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
def normalize_img_id(x):
    """Normalize image IDs that may arrive as bytes strings or floats ('12.0')."""
    s = str(x).strip()

    if s.startswith("b'") and s.endswith("'"):
        s = s[2:-1]

    if s.endswith(".0"):
        s = s[:-2]

    try:
        f = float(s)
        s = str(int(f)) if f.is_integer() else str(f)
    except ValueError:
        pass

    return s


def merge_images(df, images, image_ids, id_fn):
    """
    Attach image arrays and engineered image features to `df` by image ID.
    Returns the merged dataframe and the images aligned to its rows.
    """
    img_df = pd.DataFrame([
        {IMAGE_ID_COL: id_fn(img_id), "image_array": img}
        for img_id, img in zip(image_ids, images)
    ])

    img_feat_df = build_image_feature_df(images, image_ids)
    img_feat_df[IMAGE_ID_COL] = img_feat_df[IMAGE_ID_COL].apply(id_fn)

    df = df.merge(img_df, on=IMAGE_ID_COL, how="inner", validate="one_to_one")
    df = df.merge(img_feat_df, on=IMAGE_ID_COL, how="left", validate="one_to_one")

    aligned_images = np.stack(df["image_array"].values)
    df = df.drop(columns=["image_array"])
    return df, aligned_images


def load_train_data():
    x_df = pd.read_csv(X_TRAIN_CSV)
    y_df = pd.read_csv(Y_TRAIN_CSV)

    df = x_df.merge(
        y_df[[GROUP_COL, IMAGE_ID_COL, TARGET_COL, "fine_label"]],
        on=[GROUP_COL, IMAGE_ID_COL],
        how="inner",
    )

    if "age" not in df.columns:
        raise ValueError("Expected 'age' column in x_train.csv")

    df["age_group"] = df["age"].apply(make_age_group)
    df = add_diameter_engineered_features(df)
    df[IMAGE_ID_COL] = df[IMAGE_ID_COL].apply(normalize_img_id)

    npz = np.load(X_TRAIN_IMG_NPZ, allow_pickle=True)
    return merge_images(df, npz[IMAGE_KEY], npz[IMAGE_ID_KEY], normalize_img_id)


def load_test_data():
    x_df = pd.read_csv(X_TEST_CSV)

    if "age" not in x_df.columns:
        raise ValueError("Expected 'age' column in x_test.csv")

    x_df["age_group"] = x_df["age"].apply(make_age_group)
    x_df = add_diameter_engineered_features(x_df)

    # Note: test IDs are matched with str(), while training IDs use
    # normalize_img_id(). Left as originally run; the one-to-one merge
    # validation and the row-count prints confirm all test rows matched.
    x_df[IMAGE_ID_COL] = x_df[IMAGE_ID_COL].astype(str)

    npz = np.load(X_TEST_IMG_NPZ, allow_pickle=True)
    images = npz[IMAGE_KEY]

    print("Original x_test rows:", len(x_df))
    print("Images in npz:", len(images))

    x_df, aligned_images = merge_images(x_df, images, npz[IMAGE_ID_KEY], str)

    print("Rows after merging images:", len(x_df))
    print("Final test df shape:", x_df.shape)
    return x_df, aligned_images


# =========================================================
# TABULAR PREPROCESSING (fit on the training split only)
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


def _to_dense_float32(x):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return x.astype(np.float32)


def transform_tabular_train_val(preprocessor, df_train, df_val):
    x_train = df_train.drop(columns=[TARGET_COL], errors="ignore").drop(columns=NON_FEATURE_COLS, errors="ignore")
    x_val = df_val.drop(columns=[TARGET_COL], errors="ignore").drop(columns=NON_FEATURE_COLS, errors="ignore")

    x_train_tab = preprocessor.fit_transform(x_train)
    x_val_tab = preprocessor.transform(x_val)
    return _to_dense_float32(x_train_tab), _to_dense_float32(x_val_tab)


def transform_tabular_test(preprocessor, df_test):
    x_test = df_test.drop(columns=NON_FEATURE_COLS, errors="ignore")
    return _to_dense_float32(preprocessor.transform(x_test))


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
# TRAIN / VALIDATE / PREDICT
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

    all_probs_pos = []
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

        all_probs_pos.append(probs[:, 1].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs_pos = np.concatenate(all_probs_pos, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    overall_auc = roc_auc_score(all_labels, all_probs_pos)
    group_auc_dict, min_group_auc = compute_group_aurocs_binary(
        all_labels, all_probs_pos, age_groups_val
    )

    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "overall_auc": overall_auc,
        "group_auc_dict": group_auc_dict,
        "min_group_auc": min_group_auc,
    }


@torch.no_grad()
def predict_probabilities(model, loader, device):
    """Positive-class probabilities for an unlabeled loader."""
    model.eval()
    all_probs_pos = []

    for images, tabular in loader:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)

        logits = model(images, tabular)
        all_probs_pos.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())

    return np.concatenate(all_probs_pos, axis=0)


# =========================================================
# CONFIDENCE GRID (report Figure 6)
# =========================================================
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


@torch.no_grad()
def collect_predictions_with_images(model, loader, device):
    """Validation probabilities, labels, and de-normalized images for plotting."""
    model.eval()

    all_probs_pos = []
    all_labels = []
    all_images = []

    for images, tabular, labels in loader:
        images = images.to(device, non_blocking=True)
        tabular = tabular.to(device, non_blocking=True)

        probs = torch.softmax(model(images, tabular), dim=1)
        all_probs_pos.append(probs[:, 1].cpu().numpy())
        all_labels.append(labels.numpy())

        # Undo ImageNet normalization for display
        imgs_np = images.cpu().numpy().transpose(0, 2, 3, 1)
        imgs_np = np.clip(imgs_np * IMAGENET_STD + IMAGENET_MEAN, 0, 1)
        all_images.append(imgs_np)

    return (
        np.concatenate(all_probs_pos),
        np.concatenate(all_labels),
        np.concatenate(all_images),
    )


def save_confidence_grid(model, val_loader, device, label_classes,
                         age_groups_val, fine_labels_val,
                         output_path="confidence_grid.png", n=3):
    """
    Plot the n most confident correct predictions (top row) and the n most
    confident incorrect predictions (bottom row) on the validation split.
    """
    probs_pos, labels, images = collect_predictions_with_images(model, val_loader, device)
    age_groups = np.asarray(age_groups_val)
    fine_labels = np.asarray(fine_labels_val)

    preds = (probs_pos >= 0.5).astype(int)
    correct = (preds == labels)
    confidence = np.where(preds == 1, probs_pos, 1 - probs_pos)

    correct_idx = np.where(correct)[0]
    incorrect_idx = np.where(~correct)[0]

    top_correct = correct_idx[np.argsort(confidence[correct_idx])[::-1][:n]]
    top_incorrect = incorrect_idx[np.argsort(confidence[incorrect_idx])[::-1][:n]]

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 9))
    fig.suptitle("Most Confident Predictions on Held-Out Data", fontsize=14, fontweight="bold")

    row_meta = [
        ("Correct", top_correct, "#2ecc71"),
        ("Incorrect", top_incorrect, "#e74c3c"),
    ]

    for row, (row_label, indices, color) in enumerate(row_meta):
        for col, idx in enumerate(indices):
            ax = axes[row, col]
            ax.imshow(images[idx])
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(6)
                spine.set_visible(True)

            true_class = label_classes[int(labels[idx])]
            pred_class = label_classes[int(preds[idx])]

            ax.set_title(
                f"True: {true_class} ({fine_labels[idx]})  Pred: {pred_class}\n"
                f"Age group: {age_groups[idx]}\n"
                f"Confidence: {confidence[idx]:.3f}",
                fontsize=9,
            )

        axes[row, 0].set_ylabel(row_label, fontsize=12, fontweight="bold", color=color, labelpad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confidence grid to {output_path}")


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sampler-power", type=float, default=0.2)
    parser.add_argument("--run-name", type=str, default="final_problem2")
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def build_model(num_tabular_features, num_classes, dropout):
    return EfficientNetTabularFusion(
        num_tabular_features=num_tabular_features,
        num_classes=num_classes,
        image_embed_dim=IMAGE_EMBED_DIM,
        tab_hidden_dims=TAB_HIDDEN_DIMS,
        fusion_hidden_dims=FUSION_HIDDEN_DIMS,
        dropout=dropout,
        freeze_backbone=FREEZE_BACKBONE,
    ).to(DEVICE)


# =========================================================
# MAIN
# =========================================================
def main():
    set_seed(RANDOM_SEED)
    args = parse_args()

    run_name = args.run_name
    metrics_log_file = f"final_train_metrics_{run_name}.csv"
    group_log_file = f"final_train_group_aurocs_{run_name}.csv"
    checkpoint_file = f"best_fusion_model_{run_name}.pt"
    output_file = "yproba1_test.txt"

    print(f"Using device: {DEVICE}")
    print(f"Run: {run_name} | lr={args.head_lr} wd={args.head_weight_decay} "
          f"dropout={args.dropout} batch={args.batch_size} power={args.sampler_power}")

    # ---------------------------
    # Load training data
    # ---------------------------
    df, images = load_train_data()

    print("\nClass x age-group counts:")
    print(pd.crosstab(df[TARGET_COL], df["age_group"]))

    # ---------------------------
    # One patient-grouped 80/20 split (first fold of StratifiedGroupKFold).
    # The 20% is used for checkpoint selection and the confidence grid.
    # ---------------------------
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.split_seed)
    y_all = df[TARGET_COL].astype(str).values
    groups_all = df[GROUP_COL].astype(str).values
    train_idx, val_idx = next(iter(sgkf.split(df, y_all, groups_all)))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    img_train = images[train_idx]
    img_val = images[val_idx]
    val_fine_labels = df_val["fine_label"].astype(str).values   # figure titles only

    print("\nTrain target counts:\n", df_train[TARGET_COL].value_counts())
    print("Val target counts:\n", df_val[TARGET_COL].value_counts())

    label_encoder = LabelEncoder()
    y_train_int = label_encoder.fit_transform(df_train[TARGET_COL].astype(str).values)
    y_val_int = label_encoder.transform(df_val[TARGET_COL].astype(str).values)
    num_classes = len(label_encoder.classes_)

    if num_classes != 2:
        raise ValueError(f"Expected binary coarse_label task, got classes: {label_encoder.classes_}")
    print("Probability column 1 corresponds to:", label_encoder.classes_[1])

    train_label_names = df_train[TARGET_COL].astype(str).values
    train_age_groups = df_train["age_group"].astype(str).values
    val_age_groups = df_val["age_group"].astype(str).values

    # Tabular preprocessing (fit on the training split only)
    preprocessor, numeric_cols, categorical_cols = build_tabular_preprocessor(df_train)
    x_train_tab, x_val_tab = transform_tabular_train_val(preprocessor, df_train, df_val)

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
    model = build_model(x_train_tab.shape[1], num_classes, args.dropout)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.head_lr,
        weight_decay=args.head_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    with open(metrics_log_file, mode="w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc", "val_loss", "val_acc",
            "overall_auc", "min_group_auc",
        ])

    with open(group_log_file, mode="w", newline="") as f:
        csv.writer(f).writerow(["epoch", "age_group", "age_group_auc"])

    # ---------------------------
    # Training loop, with early stopping on minimum age-group AUROC
    # ---------------------------
    best_metric = -np.inf
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = validate(model, val_loader, criterion, DEVICE, val_age_groups)

        with open(metrics_log_file, mode="a", newline="") as f:
            csv.writer(f).writerow([
                epoch, train_loss, train_acc, val_metrics["loss"], val_metrics["acc"],
                val_metrics["overall_auc"], val_metrics["min_group_auc"],
            ])

        with open(group_log_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            for g, auc in val_metrics["group_auc_dict"].items():
                writer.writerow([epoch, g, auc])

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
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

        if np.isfinite(selection_metric) and selection_metric > best_metric + MIN_DELTA:
            best_metric = selection_metric
            epochs_without_improvement = 0

            best_state = {
                "epoch": epoch,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "label_classes": label_encoder.classes_.tolist(),
                "num_tabular_features": x_train_tab.shape[1],
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "best_min_group_auc": val_metrics["min_group_auc"],
                "best_overall_auc": val_metrics["overall_auc"],
                "selection_metric": selection_metric,
                "head_lr": args.head_lr,
                "head_weight_decay": args.head_weight_decay,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "sampler_power": args.sampler_power,
                "target_col": TARGET_COL,
            }
            torch.save(best_state, checkpoint_file)
            print(f"  Saved new best checkpoint to {checkpoint_file}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping triggered after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError("No valid checkpoint was found.")

    print("\nBest final-model summary:")
    print(f"  Selected epoch          : {best_state['epoch']}")
    print(f"  Best min age-group AUROC: {best_state['best_min_group_auc']:.4f}")
    print(f"  Overall AUROC (same epoch): {best_state['best_overall_auc']:.4f}")

    # ---------------------------
    # Test data (preprocessor fit on the training split is reused)
    # ---------------------------
    print("\nLoading test data...")
    df_test, img_test = load_test_data()
    x_test_tab = transform_tabular_test(preprocessor, df_test)

    test_dataset = SkinFusionDataset(
        image_array=img_test,
        tabular_array=x_test_tab,
        transform=build_eval_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ---------------------------
    # Reload the best checkpoint, plot, and predict
    # ---------------------------
    print(f"\nReloading best checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)

    best_model = build_model(checkpoint["num_tabular_features"], 2, args.dropout)
    best_model.load_state_dict(checkpoint["model_state_dict"])

    save_confidence_grid(
        model=best_model,
        val_loader=val_loader,
        device=DEVICE,
        label_classes=label_encoder.classes_.tolist(),
        age_groups_val=val_age_groups,
        fine_labels_val=val_fine_labels,
        output_path="confidence_grid.png",
        n=3,
    )

    yproba_test = predict_probabilities(best_model, test_loader, DEVICE)
    print(f"Test predictions shape: {yproba_test.shape}")
    np.savetxt(output_file, yproba_test, fmt="%.8f")
    print(f"Saved test predictions to {output_file}")


if __name__ == "__main__":
    main()
