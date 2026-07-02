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

## Part 2 - Efficient/Green Model: MiniResNet vs ResNet18

MiniResNet is a lightweight ResNet-style model with fewer stages and fewer channels than ResNet18. It was added to reduce parameter count, model size, training time, and inference latency.

| Metric | ResNet18 | MiniResNet | Reduction |
|---|---:|---:|---:|
| Parameters | 11,172,936 | 696,360 | 93.8% fewer |
| Model size | 42.66 MB | 2.67 MB | 93.8% smaller |
| Avg epoch time | 94.98s | 28.66s | 69.8% faster |
| Inference speed | 2.076 ms/img | 0.594 ms/img | 71.4% faster |
| Val accuracy (5 epochs) | 90.64% | 77.91% | -12.73% |

MiniResNet substantially reduces compute and memory cost, but it trades away accuracy after short training. This makes it useful for resource-constrained settings where model size and latency matter more than maximum performance. For the final benchmark datasets, the full ResNet18 remains the stronger accuracy-oriented option.

## Part 3 - Transfer Learning on organs

The scarce-data `organs` experiment compared training ResNet18 from scratch against fine-tuning from a related `orgs` checkpoint. Transfer learning improved performance and exceeded the requested 40% target.

| Metric | From Scratch | Transfer Learning | Improvement |
|---|---:|---:|---:|
| Test accuracy | 55.00% | 67.00% | +12.00% |
| Macro F1 | 0.4490 | 0.5922 | +0.1432 |
| Validation stability | Erratic | More stable | Improved |
| Epoch 1 validation accuracy | 8.00% | 80.00% | Immediate boost |

Transfer learning was more effective because the source checkpoint was trained on a related organ-image domain. The pretrained features gave the model a stronger starting point, improved validation stability, and produced a higher final test score than training from random initialization.
