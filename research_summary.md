# Forest Cover Type Prediction — Research Summary

## Research Problem
Can cartographic characteristics (elevation, slope, aspect, hydrology, soil type, wilderness area) accurately classify the dominant forest cover type of a 30 × 30 m land cell across seven classes?

---

## Key Environmental Patterns

| Finding | Detail |
|---|---|
| **Elevation is the dominant predictor** | Krummholz (~3,350 m median) and Cottonwood/Willow (~2,250 m) are well-separated; Spruce/Fir and Lodgepole Pine overlap at mid-high elevations (key confusion pair) |
| **Cottonwood/Willow is riparian** | Dramatically shorter horizontal distance to hydrology distinguishes Class 4 from all others |
| **Wilderness Area constrains geography** | Krummholz is almost entirely in Wilderness Area 4; cover types cluster strongly by region |
| **Hillshade features correlate** | 9am and 3pm hillshade are strongly negatively correlated (r ≈ −0.78); engineered mean and contrast features summarise this efficiently |
| **Soil types are sparse** | Soil_Type7 and Soil_Type15 are zero throughout training; reconstructed ordinal Soil_Index captures signal compactly |

---

## Selected Model & Validation Performance

**Model:** Extra Trees Classifier (n_estimators=300, random_state=42)

| Metric | Local Validation | Full Test Set (Ground Truth) |
|---|---|---|
| Accuracy | **85.75%** | 74.66% |
| Macro F1 | **0.855** | 0.630 |
| Macro Precision | 0.855 | — |
| Macro Recall | 0.857 | — |

**Why Extra Trees?** Lowest cross-validation variance (±0.34%), matched LightGBM on mean CV accuracy (85.98% vs. 86.51%), faster training, interpretable feature importances.

**5-Fold CV:** 85.98% ± 0.34% mean accuracy; confirms stability.

---

## Kaggle Score

| Metric | Value |
|---|---|
| Submission file | `submission.csv` |
| Test rows | 565,892 |
| Simulated Kaggle Public Accuracy | **74.67%** |
| Simulated Kaggle Private Accuracy | **74.65%** |
| Validation → Test Gap | −11.1 pp |

**Gap explanation:** Training set is artificially balanced (2,160/class); test set follows the natural distribution (Lodgepole Pine 50%, Spruce/Fir 37%). This 87% concentration in two classes deflates accuracy for minority classes in the natural distribution.

---

## Most Influential Predictors (Extra Trees)

1. **Elevation** — single most separating variable
2. **Horizontal_Distance_To_Roadways** — accessibility gradient
3. **Horizontal_Distance_To_Fire_Points** — post-fire succession
4. **Soil_Index** *(engineered)* — ordinal reconstruction of 40 binary columns
5. **Euclidean_Distance_To_Hydrology** *(engineered)* — straight-line water proximity
6. **Wilderness_Index** *(engineered)* — geographic area constraint
7. **Hillshade_Noon / Mean_Hillshade** *(engineered)* — solar exposure

---

## Most Frequently Confused Cover Types

| Confused Pair | Val Accuracy Each | Root Cause |
|---|---|---|
| **Spruce/Fir ↔ Lodgepole Pine** (1 ↔ 2) | F1: 0.770 / 0.725 | Extensive elevation overlap at 2,500–3,000 m; similar soils |
| **Ponderosa Pine ↔ Douglas-fir** (3 ↔ 6) | F1: 0.824 / 0.848 | Shared mid-elevation, moderate-slope terrain |

---

## Main Limitations

- Artificially balanced training creates overconfidence on minority classes; natural test distribution reveals the gap
- Accuracy hides weak performance on structurally ambiguous class pairs (1 vs. 2)
- Feature importance ≠ ecological causation; Elevation is a composite proxy
- No geographic cross-validation; different forest regions may yield different results
- Remote sensing bands, climate rasters, and disturbance history are absent and would materially improve discrimination
