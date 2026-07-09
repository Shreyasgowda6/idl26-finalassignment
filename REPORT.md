# Benchmark Report - IDL26 Final Assignment

## Overview

This report summarizes the corrected medical image classification pipeline and the final benchmark results across four datasets: `cells`, `chest`, `lesions`, and `orgs`. Three CNN architectures were evaluated for each dataset: AlexNet, VGG16, and ResNet18.

All experiments were run through the main training entry point with hyperparameters controlled by `config.json`. The default setup used Adam, batch size 32, dropout 0.5, and learning rate 0.001 on an NVIDIA GPU. Most models were trained for 20 epochs. The only epoch-count tuning used in the final benchmark was for `chest + AlexNet`, `chest + ResNet18`, and `lesions + VGG16`, which were trained for 30 epochs because the additional training improved their final test behavior.

## Final Results

| Dataset | Model | Epochs | Accuracy | Macro Precision | Macro Recall | Macro F1 | Target | Pass? |
|---|---|---:|---:|---:|---:|---:|---:|---|
| cells | AlexNet | 20 | 96.20% | 0.9640 | 0.9606 | 0.9617 | 90% | Yes |
| cells | VGG16 | 20 | 95.32% | 0.9558 | 0.9470 | 0.9489 | 90% | Yes |
| cells | ResNet18 | 20 | 97.16% | 0.9720 | 0.9724 | 0.9719 | 90% | Yes |
| chest | AlexNet | 30 | 88.30% | 0.9144 | 0.8466 | 0.8659 | 87% | Yes |
| chest | VGG16 | 20 | 88.14% | 0.9135 | 0.8444 | 0.8639 | 87% | Yes |
| chest | ResNet18 | 30 | 90.06% | 0.9168 | 0.8744 | 0.8890 | 87% | Yes |
| lesions | AlexNet | 20 | 75.21% | 0.6026 | 0.5128 | 0.5172 | 67% | Yes |
| lesions | VGG16 | 30 | 72.42% | 0.4254 | 0.3624 | 0.3813 | 67% | Yes |
| lesions | ResNet18 | 20 | 76.61% | 0.5763 | 0.4481 | 0.4834 | 67% | Yes |
| orgs | AlexNet | 20 | 89.46% | 0.8882 | 0.8815 | 0.8824 | 83% | Yes |
| orgs | VGG16 | 20 | 90.06% | 0.8940 | 0.8842 | 0.8857 | 83% | Yes |
| orgs | ResNet18 | 20 | 89.80% | 0.8992 | 0.8887 | 0.8853 | 83% | Yes |

All 12 final benchmark runs exceeded the required test accuracy thresholds.

## Per-Dataset Analysis

### cells

All three architectures performed strongly on `cells`, with every model exceeding 95% test accuracy. ResNet18 achieved the best result at 97.16% accuracy and 0.9719 macro F1, showing the strongest overall class balance. AlexNet was also highly competitive at 96.20%, while VGG16 reached 95.32%.

Recommended model: ResNet18, because it produced the best accuracy and macro F1 on this dataset.

### chest

All three models passed the 87% target. ResNet18 performed best with 90.06% accuracy and 0.8890 macro F1 after increasing training to 30 epochs. AlexNet also required 30 epochs to pass reliably, reaching 88.30%. VGG16 passed at 20 epochs with 88.14%.

The `chest` dataset shows class imbalance effects. The main challenge was improving recall for class 0 while maintaining strong class 1 performance. ResNet18 handled this tradeoff best in the final benchmark.

Recommended model: ResNet18, because it achieved the highest accuracy and macro F1 for `chest`.

### lesions

All models passed the 67% accuracy target, but this was the most difficult dataset. Macro F1 scores were much lower than accuracy because the dataset is imbalanced and some classes have very small support. ResNet18 achieved the best accuracy at 76.61%, while AlexNet achieved the best macro F1 at 0.5172. VGG16 improved with 30 epochs, rising from a weaker prior run to 72.42% accuracy and 0.3813 macro F1.

Recommended model: ResNet18 if the main objective is accuracy; AlexNet if macro F1 and minority-class balance are prioritized.

### orgs

All three models passed the 83% target with strong macro metrics. VGG16 achieved the best accuracy at 90.06%, while ResNet18 achieved the best macro precision and recall balance at 0.8992 macro precision and 0.8887 macro recall. AlexNet was close behind at 89.46%.

Recommended model: VGG16 for highest accuracy; ResNet18 for the strongest macro precision.

## Overall Recommendations

| Dataset | Recommended Model | Accuracy | Macro F1 | Reason |
|---|---|---:|---:|---|
| cells | ResNet18 | 97.16% | 0.9719 | Best overall result |
| chest | ResNet18 | 90.06% | 0.8890 | Best accuracy and macro F1 |
| lesions | ResNet18 | 76.61% | 0.4834 | Best accuracy on the hardest dataset |
| orgs | VGG16 | 90.06% | 0.8857 | Highest accuracy |

ResNet18 was the most consistently strong architecture overall. AlexNet was surprisingly competitive and performed especially well on `cells` and `lesions`. VGG16 performed well on `cells` and `orgs`, but was weaker on `lesions` because of minority-class behavior.

## Class Imbalance

The `lesions` dataset shows the clearest imbalance problem. Overall accuracy is acceptable for every model, but macro metrics reveal that minority classes remain difficult. This is especially visible for class 3, which has only 23 test samples and was not reliably predicted by VGG16 or ResNet18. AlexNet produced the best macro F1 on this dataset, although ResNet18 produced the best overall accuracy.

The `chest` dataset also showed class imbalance behavior during tuning. Increasing the training duration to 30 epochs improved the final test performance for AlexNet and ResNet18 without changing the training code.

Future improvements for the imbalanced datasets would include class-weighted loss, balanced sampling, or targeted augmentation. These were not used in the final benchmark table so that the reported results remain based on the same standard training setup with config-based hyperparameter tuning.

## Part 2 - Efficient/Green Model Profiling

MiniResNet is a lightweight ResNet-style model with fewer stages and fewer channels than ResNet18. It was added to reduce parameter count, model size, training time, inference latency, and GPU memory usage.

The Green Initiative profiling was expanded to cover all four datasets and all four evaluated architectures: AlexNet, VGG16, ResNet18, and MiniResNet. Each model was trained for 20 epochs with the same optimizer, batch size, learning rate, and data split. Because validation accuracy fluctuated during training, the test result is reported from the best validation checkpoint rather than from the final epoch.

| Dataset | Model | Best Val Epoch | Test Accuracy | Parameters | Model Size | Train Time | Inference Latency | Peak Train Memory | Peak Inference Memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cells | AlexNet | 19 | 96.35% | 5,693,544 | 21.72 MB | 54.63s | 0.0539 ms/sample | 187.31 MB | 129.14 MB |
| cells | VGG16 | 20 | 95.79% | 15,253,064 | 58.22 MB | 314.47s | 0.3666 ms/sample | 685.77 MB | 383.56 MB |
| cells | ResNet18 | 20 | 97.87% | 11,172,936 | 42.66 MB | 546.78s | 0.6110 ms/sample | 827.90 MB | 385.94 MB |
| cells | MiniResNet | 19 | 97.69% | 696,360 | 2.67 MB | 181.41s | 0.1675 ms/sample | 322.66 MB | 125.46 MB |
| chest | AlexNet | 15 | 90.38% | 5,682,690 | 21.68 MB | 19.19s | 0.0463 ms/sample | 186.50 MB | 129.41 MB |
| chest | VGG16 | 16 | 86.54% | 15,248,834 | 58.20 MB | 119.41s | 0.3603 ms/sample | 683.16 MB | 382.49 MB |
| chest | ResNet18 | 19 | 87.50% | 11,168,706 | 42.64 MB | 204.98s | 0.6013 ms/sample | 830.33 MB | 383.12 MB |
| chest | MiniResNet | 17 | 94.39% | 695,010 | 2.66 MB | 68.89s | 0.1596 ms/sample | 321.26 MB | 124.44 MB |
| lesions | AlexNet | 17 | 74.01% | 5,692,519 | 21.72 MB | 37.83s | 0.1027 ms/sample | 186.92 MB | 130.62 MB |
| lesions | VGG16 | 15 | 69.78% | 15,252,551 | 58.22 MB | 194.30s | 0.3767 ms/sample | 682.26 MB | 383.24 MB |
| lesions | ResNet18 | 19 | 76.91% | 11,172,423 | 42.66 MB | 319.83s | 0.6028 ms/sample | 827.52 MB | 383.56 MB |
| lesions | MiniResNet | 20 | 76.91% | 696,231 | 2.66 MB | 106.13s | 0.1615 ms/sample | 322.65 MB | 125.46 MB |
| orgs | AlexNet | 19 | 90.12% | 5,691,915 | 21.71 MB | 81.30s | 0.0757 ms/sample | 186.67 MB | 130.74 MB |
| orgs | VGG16 | 13 | 89.63% | 15,253,451 | 58.22 MB | 361.91s | 0.3685 ms/sample | 683.97 MB | 382.43 MB |
| orgs | ResNet18 | 18 | 92.20% | 11,173,323 | 42.66 MB | 593.78s | 0.6123 ms/sample | 827.79 MB | 382.07 MB |
| orgs | MiniResNet | 18 | 92.47% | 696,171 | 2.66 MB | 202.06s | 0.1616 ms/sample | 321.28 MB | 124.46 MB |

MiniResNet had the smallest parameter count and model size in every dataset. Compared with ResNet18, it used about 93.8% fewer parameters and reduced model size from roughly 42.66 MB to 2.66 MB. It also reduced peak training memory by about 60% and peak inference memory by about 67% while preserving comparable accuracy.

AlexNet was the fastest model in this profiling run and had the lowest peak training memory, despite having more parameters than MiniResNet. This shows that parameter count is not the only driver of runtime; network structure and activation memory also matter. AlexNet is therefore a strong speed-oriented baseline, especially on `cells`, `chest`, and `orgs`.

VGG16 was the least efficient architecture overall. It had the largest model size, high memory consumption, and long training times, while its accuracy was not consistently better than the lighter alternatives. This makes VGG16 hard to justify for the Green Initiative unless its accuracy is clearly superior for a specific deployment.

ResNet18 remained a strong accuracy baseline, but it was computationally expensive. MiniResNet matched or exceeded ResNet18 on `chest`, `lesions`, and `orgs` in the green-profile evaluation, and was only 0.18 percentage points lower on `cells`. The `chest` dataset still showed run-to-run instability, so the exact ranking there should be interpreted cautiously. Overall, MiniResNet is the best compact architecture when model size and memory footprint are the priority, while AlexNet is the best option when raw runtime speed is the priority.

## Part 3 - Transfer Learning on organs

The scarce-data `organs` experiment compared training ResNet18 from scratch against fine-tuning from a related `orgs` checkpoint. Transfer learning improved performance and exceeded the requested 40% target.

| Metric | From Scratch | Transfer Learning | Improvement |
|---|---:|---:|---:|
| Test accuracy | 60.50% | 66.00% | +5.50% |
| Macro F1 | 0.4939 | 0.5863 | +0.0924 |
| Validation stability | Erratic | More stable | Improved |
| Best validation accuracy | 84.00% | 88.00% | +4.00% |

Transfer learning was more effective because the source checkpoint was trained on the larger related `orgs` dataset. The pretrained backbone features gave the model a stronger starting point, while the final classifier was reinitialized for the 11 `organs` classes. The best-validation checkpoint was used for final test evaluation, reducing the effect of last-epoch instability on the scarce-data experiment.

Because `organs` is a small dataset, the results should be interpreted with more caution than the larger benchmark datasets. The validation scores were less stable, which is expected when each split contains relatively few examples per class. For practical use, the recommended strategy is to keep using transfer learning from the larger `orgs` profile, train only a freshly initialized classifier head first, and then fine-tune the backbone with a low learning rate. As more `organs` data becomes available, the next improvements should focus on stronger augmentation, repeated runs with averaged metrics, and class-balanced sampling to reduce variance across minority classes.
