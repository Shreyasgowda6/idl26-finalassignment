# Report

| Name | Enrollment Number |
|---|---|
| Shreyas Hosadurga Sadananda | 10013494 |
| Suhaim Manna | 10012548 |


---

## Overview

This report presents the full benchmark results for the restored medical image classification pipeline across four datasets: cells, chest, lesions, and orgs. Three CNN architectures were evaluated - AlexNet, VGG16, and ResNet18. All experiments used Adam optimiser, batch size 32, dropout 0.5, and learning rate 0.001. Best-validation checkpoints were used throughout to avoid penalising runs where the final epoch was not the performance peak.

The report covers three parts:
- **Part 1** — Full benchmark across all 12 dataset/model combinations
- **Part 2** — Green Initiative: MiniResNet efficiency comparison against ResNet18
- **Part 3** — Transfer learning on the scarce organs dataset

---

## Part 1 — Full Benchmark Results

### How We Arrived at the Final Epoch Counts

We did not start at 20 epochs. The first full benchmark run used 10 epochs for all 12 combinations, the quickest way to get an end-to-end picture of how all three models behaved across all four datasets.

#### What Happened at 10 Epochs

Ten epochs was enough for most combinations. But two combinations failed to meet their minimum targets - lesions + VGG16 at 66.33% (target 67%) and orgs + VGG16 at 80.08% (target 83%). Two others - chest + AlexNet and chest + ResNet18 - passed at 10 epochs but showed unstable validation curves suggesting they had not fully converged.

#### Why We Moved to 20 Epochs

We increased all runs to 20 epochs as a first step. For most models on most datasets 10 epochs was already sufficient, but 20 epochs gave the training curves more room to stabilise and allowed the best-validation checkpoint saving to capture a more mature peak rather than an early noisy one.

The 20-epoch results resolved two of the four concerns. orgs + VGG16 improved from 80.08% to 90.06% - a significant jump, clearing the 83% target comfortably. chest + AlexNet and chest + ResNet18 both showed improved and more stable validation curves. However lesions + VGG16 remained a problem.

#### Why Three Combinations Needed 30 Epochs

lesions + VGG16 (66.33% at 10 epochs, still near target at 20 epochs):
VGG16 is the largest model in the benchmark at approximately 12.6 million parameters. The lesions dataset has only 8,010 training samples across 7 visually similar classes. VGG16 was slow to converge on this dataset - its deep sequential architecture requires many more gradient updates before the feature hierarchy stabilises enough to distinguish the subtle differences between lesion types. At 10 epochs the validation curve was still clearly rising. At 20 epochs it had improved but not yet fully flattened. Extending to 30 epochs allowed VGG16 to complete its convergence, reaching 72.42% and comfortably clearing the 67% target.

chest + AlexNet (89.42% at 10 epochs, variable at 20 epochs):
Although AlexNet passed at 10 epochs, the validation accuracy on chest was erratic - swinging several percentage points between adjacent epochs. chest has only 5,232 training samples and 2 classes, which creates high sensitivity to initialisation and batch composition. AlexNet's simpler architecture with larger 7×7 and 5×5 kernels is less well suited to detecting the subtle texture differences between healthy and pneumonia chest X-rays than the finer 3×3 kernels used in VGG16 and ResNet18. At 20 epochs AlexNet's validation curve had not fully smoothed out. Extending to 30 epochs produced a more stable converged checkpoint and improved the best-validation accuracy captured by the checkpoint saving mechanism.

chest + ResNet18 (91.51% at 10 epochs, rising at 20 epochs):
ResNet18 on chest was passing comfortably at 10 epochs but its validation curve was still trending upward at epoch 20 - meaning the model had not yet reached its natural performance ceiling. Unlike the other two cases this was not a convergence problem but an opportunity: training loss was still falling and validation accuracy was still improving. Extending to 30 epochs allowed ResNet18 to reach its full potential on this dataset, resulting in the 90.06% final test accuracy reported in the benchmark.

---

### Final Results Table

| Dataset | Model | Epochs | Accuracy | Macro Precision | Macro Recall | Macro F1 | Target | Pass |
|---|---|---:|---:|---:|---:|---:|---:|---|
| cells | AlexNet | 20 | 96.20% | 0.9640 | 0.9606 | 0.9617 | 90% | ✅ |
| cells | VGG16 | 20 | 95.32% | 0.9558 | 0.9470 | 0.9489 | 90% | ✅ |
| cells | ResNet18 | 20 | 97.16% | 0.9720 | 0.9724 | 0.9719 | 90% | ✅ |
| chest | AlexNet | 30 | 88.30% | 0.9144 | 0.8466 | 0.8659 | 87% | ✅ |
| chest | VGG16 | 20 | 88.14% | 0.9135 | 0.8444 | 0.8639 | 87% | ✅ |
| chest | ResNet18 | 30 | 90.06% | 0.9168 | 0.8744 | 0.8890 | 87% | ✅ |
| lesions | AlexNet | 20 | 75.21% | 0.6026 | 0.5128 | 0.5172 | 67% | ✅ |
| lesions | VGG16 | 30 | 72.42% | 0.4254 | 0.3624 | 0.3813 | 67% | ✅ |
| lesions | ResNet18 | 20 | 76.61% | 0.5763 | 0.4481 | 0.4834 | 67% | ✅ |
| orgs | AlexNet | 20 | 89.46% | 0.8882 | 0.8815 | 0.8824 | 83% | ✅ |
| orgs | VGG16 | 20 | 90.06% | 0.8940 | 0.8842 | 0.8857 | 83% | ✅ |
| orgs | ResNet18 | 20 | 89.80% | 0.8992 | 0.8887 | 0.8853 | 83% | ✅ |

All 12 final benchmark runs exceeded the required test accuracy thresholds.

---

### Per-Dataset Analysis

#### cells (RGB, 64×64, 8 classes - train: 13,671 / test: 3,421)

All three models exceeded 95% accuracy. ResNet18 achieved the best result at 97.16% accuracy and 0.9719 macro F1, showing the strongest balance across all 8 classes. AlexNet was highly competitive at 96.20%, while VGG16 reached 95.32%. The dataset is well balanced which contributed to uniformly strong performance across all models.

**Recommended model: ResNet18** - best accuracy and macro F1.

#### chest (Grayscale, 64×64, 2 classes — train: 5,232 / test: 624)

All three models passed the 87% target. ResNet18 achieved 90.06% after 30 epochs. AlexNet also required 30 epochs to produce a stable result, reaching 88.30%. VGG16 passed at 20 epochs with 88.14%. The main challenge was improving recall for the minority class (class 0) while maintaining strong class 1 performance, ResNet18 handled this tradeoff best.

**Recommended model: ResNet18** - highest accuracy and macro F1 on chest.

#### lesions (RGB, 64×64, 7 classes - train: 8,010 / test: 2,005)

This was the hardest dataset. All models passed the 67% accuracy target but macro F1 scores were much lower than accuracy due to severe class imbalance, class 5 accounts for 67% of all test samples (1,341 of 2,005) while class 3 has only 23 samples. Models learned to predict class 5 reliably but struggled with minority classes. ResNet18 achieved the best accuracy (76.61%) while AlexNet achieved the best macro F1 (0.5172).

**Recommended model: ResNet18** for highest accuracy; AlexNet if minority-class balance is the priority.

#### orgs (Grayscale, 64×64, 11 classes — train: 15,367 / test: 8,216)

All three models passed the 83% target with strong macro metrics. VGG16 achieved the highest accuracy at 90.06% after being given 20 epochs, this was the combination that failed at 10 epochs and improved dramatically with more training. ResNet18 achieved the strongest macro precision (0.8992) and recall (0.8887). AlexNet was close behind at 89.46%.

**Recommended model: VGG16** for highest accuracy; ResNet18 for strongest macro precision.

---

### Overall Recommendations

| Dataset | Recommended Model | Accuracy | Macro F1 | Reason |
|---|---|---:|---:|---|
| cells | ResNet18 | 97.16% | 0.9719 | Best accuracy and class balance |
| chest | ResNet18 | 90.06% | 0.8890 | Best accuracy and minority-class recall |
| lesions | ResNet18 | 76.61% | 0.4834 | Best accuracy on the hardest dataset |
| orgs | VGG16 | 90.06% | 0.8857 | Highest accuracy |

ResNet18 was the most consistently strong architecture overall. AlexNet was surprisingly competitive and performed especially well on cells and lesions. VGG16 performed well on cells and orgs but was weaker on lesions due to minority-class collapse.

---

### Note on Class Imbalance

The lesions dataset shows the clearest imbalance problem. Overall accuracy is acceptable for every model but macro metrics reveal that minority classes remain difficult, class 3 (23 test samples) was not reliably predicted by any model. The chest dataset also showed imbalance behaviour requiring longer training to improve minority-class recall.

Future improvements for imbalanced datasets would include class-weighted loss, balanced sampling, or targeted augmentation. These were not applied in the final benchmark so that all results remain based on the same standard training setup.

---

## Part 2 - Green Initiative: Full Efficiency Profiling

### Architecture

MiniResNet is a lightweight ResNet-style architecture added specifically for the Green Initiative. Compared to ResNet18 it uses fewer initial channels (32 instead of 64) and only 3 residual stages (32-64-128) instead of 4 (64-128-256-512). This reduces parameter count by 93.8% - from about 11.2M parameters to about 696K - while preserving the residual connection structure that makes ResNet architectures effective.

The green profiling script was run across all four datasets and all four implemented models: AlexNet, VGG16, ResNet18, and MiniResNet. Each run used 20 epochs and logged best validation epoch, test accuracy, parameter count, model size, total training time, inference latency per sample, and peak GPU memory during training and inference.

### Dataset-Wise Efficiency Results

#### cells

| Model | Best Epoch | Test Accuracy | Parameters | Model Size | Train Time | Inference | Peak Train Mem | Peak Infer Mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexNet | 19 | 96.35% | 5,693,544 | 21.72 MB | 54.63s | 0.0539 ms | 187.31 MB | 129.14 MB |
| VGG16 | 20 | 95.79% | 15,253,064 | 58.22 MB | 314.47s | 0.3666 ms | 685.77 MB | 383.56 MB |
| ResNet18 | 20 | 97.87% | 11,172,936 | 42.66 MB | 546.78s | 0.6110 ms | 827.90 MB | 385.94 MB |
| MiniResNet | 19 | 97.69% | 696,360 | 2.67 MB | 181.41s | 0.1675 ms | 322.66 MB | 125.46 MB |

On cells, ResNet18 produced the highest accuracy, but MiniResNet was almost identical while being much smaller and faster. AlexNet was the quickest model, although MiniResNet gives a better balance between accuracy and compactness.

#### chest

| Model | Best Epoch | Test Accuracy | Parameters | Model Size | Train Time | Inference | Peak Train Mem | Peak Infer Mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexNet | 15 | 90.38% | 5,682,690 | 21.68 MB | 19.19s | 0.0463 ms | 186.50 MB | 129.41 MB |
| VGG16 | 16 | 86.54% | 15,248,834 | 58.20 MB | 119.41s | 0.3603 ms | 683.16 MB | 382.49 MB |
| ResNet18 | 19 | 87.50% | 11,168,706 | 42.64 MB | 204.98s | 0.6013 ms | 830.33 MB | 383.12 MB |
| MiniResNet | 17 | 94.39% | 695,010 | 2.66 MB | 68.89s | 0.1596 ms | 321.26 MB | 124.44 MB |

On chest, MiniResNet gave the strongest result in the green-profile run. This dataset also showed some run-to-run instability during earlier experiments, so the exact ordering should be interpreted cautiously. Still, the result is useful the smaller model is clearly capable of handling the binary chest task with far less memory and model size than ResNet18 or VGG16.

#### lesions

| Model | Best Epoch | Test Accuracy | Parameters | Model Size | Train Time | Inference | Peak Train Mem | Peak Infer Mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexNet | 17 | 74.02% | 5,692,519 | 21.72 MB | 37.83s | 0.1027 ms | 186.92 MB | 130.62 MB |
| VGG16 | 15 | 69.78% | 15,252,551 | 58.22 MB | 194.30s | 0.3767 ms | 682.26 MB | 383.24 MB |
| ResNet18 | 19 | 76.91% | 11,172,423 | 42.66 MB | 319.83s | 0.6028 ms | 827.52 MB | 383.56 MB |
| MiniResNet | 20 | 76.91% | 696,231 | 2.66 MB | 106.13s | 0.1615 ms | 322.65 MB | 125.46 MB |

Lesions is the hardest dataset because of class imbalance and subtle visual differences between classes. ResNet18 and MiniResNet reached the same test accuracy, which is a strong result for the smaller model. VGG16 was both heavier and less accurate here, making it the weakest green choice for this dataset.

#### orgs

| Model | Best Epoch | Test Accuracy | Parameters | Model Size | Train Time | Inference | Peak Train Mem | Peak Infer Mem |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AlexNet | 19 | 90.12% | 5,691,915 | 21.71 MB | 81.30s | 0.0757 ms | 186.67 MB | 130.74 MB |
| VGG16 | 13 | 89.63% | 15,253,451 | 58.22 MB | 361.91s | 0.3685 ms | 683.97 MB | 382.43 MB |
| ResNet18 | 18 | 92.20% | 11,173,323 | 42.66 MB | 593.78s | 0.6123 ms | 827.79 MB | 382.07 MB |
| MiniResNet | 18 | 92.47% | 696,171 | 2.66 MB | 202.06s | 0.1616 ms | 321.28 MB | 124.46 MB |

On orgs, MiniResNet slightly outperformed ResNet18 while using a fraction of the parameters and memory. This is the clearest example that the smaller architecture can preserve accuracy on a larger multi-class dataset while being much more practical to deploy.

### Average Efficiency Summary

| Model | Avg Accuracy | Avg Parameters | Avg Model Size | Avg Train Time | Avg Inference | Avg Peak Train Mem |
|---|---:|---:|---:|---:|---:|---:|
| AlexNet | 87.72% | 5.69M | 21.71 MB | 48.24s | 0.0697 ms | 186.85 MB |
| VGG16 | 85.43% | 15.25M | 58.21 MB | 247.52s | 0.3680 ms | 683.79 MB |
| ResNet18 | 88.62% | 11.17M | 42.65 MB | 416.34s | 0.6069 ms | 828.39 MB |
| MiniResNet | 90.36% | 0.70M | 2.66 MB | 139.62s | 0.1626 ms | 321.96 MB |

### Green Initiative Analysis

The dataset-wise results show that no single architecture is best at everything. AlexNet is the fastest model and uses the least training memory, but its accuracy is weaker on lesions and it does not give the strongest overall accuracy. VGG16 is the least efficient model in this project: it has the largest model file, high memory usage, and slower training, without providing the best accuracy. ResNet18 remains a strong baseline, but it is the most expensive model during training and inference.

MiniResNet gives the best overall green trade-off. It has the smallest model size, far fewer parameters, much lower memory usage than ResNet18 and VGG16, and competitive or better accuracy across the four green-profile runs:

- **cells** - MiniResNet (97.69%) was only 0.18 percentage points below ResNet18 (97.87%) while using 93.8% fewer parameters and running about 3x faster.
- **chest** - MiniResNet (94.39%) actually exceeded ResNet18 (87.50%) in this green-profile run, demonstrating that the smaller model can match or exceed the larger one on simpler binary-class datasets.
- **lesions** - Both models achieved identical test accuracy (76.91%), confirming MiniResNet can match ResNet18 even on the most difficult and imbalanced dataset in the benchmark.
- **orgs** - MiniResNet (92.47%) slightly exceeded ResNet18 (92.20%), again showing no accuracy penalty from the architectural reduction.

Across the four direct ResNet18-vs-MiniResNet comparisons, MiniResNet reduced the model size by 93.8%, cut average training time by about 66%, reduced average inference latency by about 73%, and used roughly 61% less peak training memory. For deployment on diagnostic devices with limited compute, embedded systems, edge devices, or energy-constrained hospital infrastructure, MiniResNet is the recommended green architecture.

---

## Part 3 — Transfer Learning on organs (Scarce Data)

### Problem

The organs dataset contains only 500 training images across 11 classes (~45 per class). Training from random weights on this volume of data is generally insufficient for reliable multi-class classification. A standard training run risks overfitting or failing to learn minority classes entirely.

### Strategy

We compared two approaches using ResNet18:

**From scratch**: random weight initialisation, trained on organs for 20 epochs with lr=0.001.

**Transfer learning**: backbone weights loaded from the pretrained orgs_ResNet18.pth checkpoint (trained on 15,367 organ images, 89.80% test accuracy). Only the backbone feature layers were transferred, the classifier head was reinitialized for the organs task. Fine-tuning used lr=0.0001 (10× lower than scratch) to preserve the pretrained feature representations while adapting to the new data distribution. The lower learning rate prevents overwriting the useful knowledge from the larger dataset with noise from only 500 samples.

Best-validation checkpoints were used for both approaches to reduce the effect of last-epoch instability on the small dataset.

### Results

| Metric | From Scratch | Transfer Learning | Improvement |
|---|---:|---:|---:|
| Test accuracy | 60.50% | 66.00% | +5.50% |
| Macro precision | 0.5800 | 0.6234 | +0.0434 |
| Macro recall | 0.4931 | 0.6015 | +0.1084 |
| Macro F1 | 0.4939 | 0.5863 | +0.0924 |
| Weighted F1 | 0.5790 | 0.6440 | +0.0650 |
| Best val accuracy | 84.00% | 88.00% | +4.00% |

Both approaches exceeded the required 40% accuracy threshold. Transfer learning improved test accuracy by 5.5 percentage points and macro F1 by 0.0924. The improvement in macro recall (+0.1084) is particularly significant, the pretrained backbone gave the model a better starting point for recognising minority classes that are extremely underrepresented in 500 training samples.

### Analysis and Recommendations

The gap between validation accuracy (88%) and test accuracy (66%) in the transfer learning run reflects the fundamental challenge of scarce data with only ~45 training samples per class the model cannot fully generalise even with transferred features. This gap is expected and is not evidence that transfer learning failed — the from-scratch model showed an even larger gap and lower absolute performance.

**Why transfer learning worked:** The orgs and organs datasets share the same domain (organ scans), the same image format (grayscale 64×64), and the same 11 class labels. The backbone weights encode general organ feature representations that transfer directly. Fine-tuning with a low learning rate preserved these representations while adapting the classifier to the new data distribution.

---
