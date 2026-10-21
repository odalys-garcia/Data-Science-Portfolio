# Odalys Garcia-Lopez — Data Science Portfolio

MS in Mathematics (Tufts, 2026). I work at the intersection of rigorous quantitative methods and applied modeling, with a focus on healthcare and life-science problems. Each project below has a one-page summary and, where shareable, the code.

---

## 1. Skin Lesion Classification
**Problem:** Flag skin lesions that need treatment from images and tabular features, while performing fairly across patient age groups.

**What I did:** Built a RF model and fusion model (pretrained EfficientNet-B0 image embedding + MLP on tabular features), re-engineered image features using the dermatology ABCDE framework, and used weighted sampling to oversample underrepresented age × class groups.

**Result:** Highest test AUROC of 0.901 achieved by the RF model, but fusion model achieved a higher balance accuracy of 0.793. Lowered the melanoma decision threshold to reduce false negatives, where a missed case is far costlier than a false alarm.

📄 [Summary](skin-lesion-classification/project_B_summary.pdf) · 💻 [Code](skin-lesion-classification/)

---

## 2. Energy Demand Forecasting
**Problem:** Predict electricity demand 24 hours ahead for each ISO New England load zone using weather, historical load, and calendar data.

**What I did:** On a three-person team building hybrid CNN-Transformer models, I led the error and architectural analysis of our best models. I found that accuracy dropped during extreme weather (3.29% → 3.74% MAPE) because spatially averaged weather features were diluting the rare-weather signal, which guided changes to the final model.

**Result:** The final model achieved 3.307% MAPE, a 49.6% improvement over baseline, and the changes from my analysis lowered extreme-weather MAPE from 3.74% to 3.60%.

📄 [Summary](energy-demand-forecasting/energy_demand_prediction_project_summary.pdf) · 💻 [Code](energy-demand-forecasting/)

---

## 3. Minimally Sufficient Experimental Design
**Problem:** What is the fewest number of measurements needed to trust a drug model's parameter estimates, and on which days should they be collected?

**What I did:** Applied the minimally sufficient design framework of [Gevertz & Kareva (2024)](https://www.nature.com/articles/s41540-023-00325-1) to a PK/PD drug–receptor binding model. Co-developed the multistart optimization and profile likelihood code in MATLAB and ran an exhaustive protocol search on a computing cluster.

**Result:** 7 of 20 collection days is the minimum, but timing matters: only 52 of 77,520 possible 7-day schedules (0.0067%) allow both key parameters to be estimated reliably.

📄 [Summary](experimental-design/Minimally_Sufficient_Experimental_Design_Using_Mathematical_Modeling.pdf) · 💻 [Code](experimental-design/)

---

## Publications & Research
- [Fronts and patterns with a dynamic ramp], *Physica D: Nonlinear Phenomena*, 2025. [https://doi.org/10.1016/j.physd.2025.134972]
- [Identifiability of Directed-Cycle and Catenary Linear Compartmental Models], * SIAM Journal of Applied Dynamical Systems*, 2026. [https://doi.org/10.1137/25M1728636]
- [Identifiability catalog](https://github.com/[odalys-garcia]/[Identifiability-of-Directed-Cycle-and-Catenary-Linear-Compartmental-Models]) — [PDF and TXT files that store the identifiability of parameters in various models]

## Tools
Python (PyTorch, scikit-learn, pandas) · MATLAB · SQL · Git

## Contact
[LinkedIn](linkedin.com/in/odalys-garcia-lopez-844072420/) · [email](odalysgl1202@gmail.com)
