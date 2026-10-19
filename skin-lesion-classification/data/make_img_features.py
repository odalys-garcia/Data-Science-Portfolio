"""
Step 0 (Problem 1): compute the engineered image features used by the
random forest scripts.

For every image in x_train_img.npz and x_test_img.npz, computes:
  img_mean_color      mean intensity over all pixels and channels
  img_color_variance  variance over all pixels and channels
  img_edge_density    fraction of pixels whose grayscale gradient
                      magnitude exceeds 20 (proxy for border complexity)

Writes x_train_img_features.csv and x_test_img_features.csv, keyed by
img_id. Uses the same feature function as train_rf.py, cv_search_rf.py,
and test_predictions_part1.py, which recompute these features for
augmented training images.

Usage (from data/):
    python make_img_features.py
"""

import numpy as np
import pandas as pd


def compute_edge_density(img_rgb: np.ndarray) -> float:
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


def load_npz(npz_path: str):
    """Find the image array (N, H, W, 3) and the 1-D id array in the npz."""
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

    return npz_obj[image_key], [str(x) for x in npz_obj[id_key]]


def build_feature_csv(npz_path: str, out_path: str):
    images, img_ids = load_npz(npz_path)

    rows = []
    for img_id, img in zip(img_ids, images):
        rows.append({"img_id": img_id, **extract_image_features_from_array(img)})

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(df)} images")


if __name__ == "__main__":
    build_feature_csv("x_train_img.npz", "x_train_img_features.csv")
    build_feature_csv("x_test_img.npz", "x_test_img_features.csv")
