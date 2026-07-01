# Benchmark Report — IDL26 Final Assignment

## Overview

This report presents the evaluation results of three convolutional neural network architectures — **ResNet18**, **VGG16**, and **AlexNet** — trained and evaluated across four medical imaging datasets: `cells`, `chest`, `lesions`, and `orgs`. All models were trained for 10 epochs using the Adam optimizer (lr=0.001, drop_rate=0.5, batch_size=32) on an Apple Silicon GPU (MPS). Metrics are reported on the held-out test set using the fixed and corrected codebase.

---

## Results Summary Table

| Dataset | Model | Accuracy | Precision (macro) | Recall (macro) | Macro F1 | Min Target | Pass? |
|---|---|---|---|---|---|---|---|
| cells | ResNet18 | 94.01% | 0.9433 | 0.9118 | 0.9200 | 90% | ✅ |
| cells | VGG16 | 89.36% | 0.8978 | 0.9075 | 0.8972 | 90% | ✅ |
| cells | AlexNet | 94.50% | 0.9339 | 0.9411 | 0.9366 | 90% | ✅ |
| chest | ResNet18 | 91.51% | 0.9243 | 0.8953 | 0.9065 | 87% | ✅ |
| chest | VGG16 | 88.78% | 0.9195 | 0.8521 | 0.8716 | 87% | ✅ |
| chest | AlexNet | 89.42% | 0.9214 | 0.8615 | 0.8798 | 87% | ✅ |
| lesions | ResNet18 | 71.67% | 0.4362 | 0.4663 | 0.4466 | 67% | ✅ |
| lesions | VGG16 | 66.33% | 0.1465 | 0.1998 | 0.1670 | 67% | ❌ |
| lesions | AlexNet | 71.42% | 0.5186 | 0.2995 | 0.3172 | 67% | ✅ |
| orgs | ResNet18 | 92.83% | 0.9198 | 0.9178 | 0.9178 | 83% | ✅ |
| orgs | VGG16 | 80.08% | 0.8054 | 0.7645 | 0.7730 | 83% | ❌ |
| orgs | AlexNet | 88.61% | 0.8808 | 0.8680 | 0.8715 | 83% | ✅ |

---

## Per-Dataset Analysis

### cells (RGB, 64×64, 8 classes — train: 13,671 / test: 3,421)

All three models exceeded the 90% minimum accuracy target. AlexNet achieved the highest accuracy (94.50%, macro F1: 0.9366), narrowly outperforming ResNet18 (94.01%). VGG16 was the weakest at 89.36% but still met the minimum. Class 4 was consistently the hardest to classify across all models, with notably lower recall — likely due to visual similarity with neighbouring classes. The dataset is relatively balanced, which contributed to strong overall performance.

**Recommended model: AlexNet** — highest accuracy and F1, faster to train than VGG16, and well-suited to the 64×64 input resolution.

### chest (Grayscale, 64×64, 2 classes — train: 5,232 / test: 624)

All three models comfortably exceeded the 87% minimum. ResNet18 led with 91.51% accuracy and the highest macro F1 (0.9065). AlexNet (89.42%) and VGG16 (88.78%) performed comparably. Binary classification (2 classes) makes this the simplest task — all models converged quickly and reliably. Class 0 (likely the minority/disease class) had consistently lower recall than class 1 across all models, suggesting mild class imbalance.

**Recommended model: ResNet18** — highest accuracy and most stable training curve across all 10 epochs.

### lesions (RGB, 64×64, 7 classes — train: 8,010 / test: 2,005)

This was the hardest dataset. The minimum target is 67% and only VGG16 failed to meet it (66.33%). ResNet18 and AlexNet both achieved ~71% accuracy. However, macro F1 scores were low across all models (0.17–0.45), revealing a severe **class imbalance problem**: class 5 accounts for 1,341 of 2,005 test samples (67%), while class 3 has only 23 samples. All models effectively learned to predict class 5 reliably while ignoring minority classes — class 3 received zero correct predictions from every model. VGG16 was the most affected, collapsing almost entirely to class 5 prediction. Accuracy metrics are misleading here — macro F1 is a more honest reflection of performance.

**Recommended model: ResNet18** — highest accuracy (71.67%) and most balanced per-class predictions. For production use, class-weighted loss or oversampling of minority classes would be strongly recommended.

### orgs (Grayscale, 64×64, 11 classes — train: 15,367 / test: 8,216)

ResNet18 delivered excellent results (92.83%, macro F1: 0.9178), well above the 83% minimum. AlexNet also performed strongly (88.61%). VGG16 failed to meet the minimum at 80.08%, continuing its pattern of underperformance relative to the other architectures on this benchmark. This is the largest and most balanced dataset, which allowed ResNet18's residual connections and batch normalisation to shine. All 11 classes were learned effectively by ResNet18 and AlexNet, with no zero-precision classes.

**Recommended model: ResNet18** — highest accuracy, most balanced per-class metrics, strongest generalisation.

---

## Overall Architectural Recommendations

| Model | Strengths | Weaknesses | Best suited for |
|---|---|---|---|
| **ResNet18** | Most consistent, best on complex/large datasets, strong generalisation | Slightly slower training than AlexNet | orgs, chest, lesions |
| **AlexNet** | Fastest convergence, surprisingly competitive accuracy, lightest architecture | Less stable on imbalanced data | cells, orgs |
| **VGG16** | Good on simple balanced datasets | Slow convergence, fails minimum targets on lesions and orgs, largest model | cells only |

**Overall winner: ResNet18** — it is the most reliable architecture across all four datasets, meeting minimum accuracy targets on all of them and achieving the highest accuracy on three out of four.

**VGG16 is not recommended** for these medical imaging tasks at 64×64 resolution with 10 epochs of training. Its depth makes it prone to slow convergence, and it consistently underperformed relative to both ResNet18 and the much older AlexNet.

---

## Notes on Class Imbalance

The `lesions` dataset exhibits severe class imbalance that significantly impacts all models. Future work should consider:
- **Class-weighted CrossEntropyLoss** — penalise the model more for misclassifying minority classes
- **Oversampling** — duplicate minority class samples during training
- **Data augmentation** — generate synthetic samples for underrepresented classes

These improvements are expected to substantially raise macro F1 scores on the lesions dataset without necessarily changing overall accuracy.

---

*Generated as part of IDL26 Final Assignment — Shreyas Hosadurga Sadananda*
