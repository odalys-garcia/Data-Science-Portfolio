# Data

The data files are **not included** in this repository, because of GitHub's file-size limits and because they were distributed for the course. They are derived from the public **PAD-UFES-20** dataset (Pacheco et al., 2020), a collection of smartphone photos of skin lesions with patient clinical metadata, collected in Brazil.

> Pacheco, A. G. C., et al. (2020). PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones. *Data in Brief*, 32, 106221. https://doi.org/10.1016/j.dib.2020.106221

## Expected files

Place these files in this folder before running any scripts:

| File | Contents |
|---|---|
| `x_train.csv` | Clinical metadata for training lesions, one row per image |
| `y_train.csv` | Training labels: `fine_label` and `coarse_label` |
| `x_test.csv` | Clinical metadata for test lesions (no labels) |
| `x_train_img.npz` | Training images, as arrays `images` (N × H × W × 3, RGB) and `image_ids` |
| `x_test_img.npz` | Test images, same format |

Rows are matched across files by `patient_id` and `img_id`. The scripts also use the `age`, `diameter_1`, and `diameter_2` columns.

Test labels are not part of the data; test predictions were scored by the course autograder.

## Labels

| `fine_label` | Diagnosis | `coarse_label` |
|---|---|---|
| ACK | Actinic keratosis | 1 (needs treatment) |
| BCC | Basal cell carcinoma | 1 |
| MEL | Melanoma | 1 |
| SCC | Squamous cell carcinoma | 1 |
| NEV | Nevus | 0 (does not) |
| SEK | Seborrheic keratosis | 0 |

The random forest is trained on `fine_label` and sums the four positive classes' probabilities into one score. The fusion model is trained directly on `coarse_label`.

## Generated files

The random forest scripts need engineered image features. Generate them from this folder with:

```bash
python make_img_features.py
```

This writes:

- `x_train_img_features.csv`
- `x_test_img_features.csv`

Each contains `img_id`, `img_mean_color`, `img_color_variance`, and `img_edge_density`. The fusion model computes its own image features internally and doesn't need these files.

All other scripts also read from and write their outputs (logs, checkpoints, predictions, figures) to this folder. See the main [README](../README.md) for the run order.
