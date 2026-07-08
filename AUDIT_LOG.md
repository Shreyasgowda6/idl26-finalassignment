# Audit Log — IDL26 Final Assignment

Bug audit for the corrupted medical imaging codebase. All bugs were identified, fixed, and committed to the repository. Bugs are grouped by file and ordered by severity type: crash bugs first, then silent/numerical bugs.

---

## Code/data.py

### Bug 1 — Wrong filename pattern
| Field | Detail |
|---|---|
| **File** | `Code/data.py` |
| **Line** | 11 |
| **Type** | Crash bug |
| **Before** | `d_path = Path(data_path) / f"{data}_data.pt"` |
| **After** | `d_path = Path(data_path) / f"{data}.pt"` |
| **Why** | The code looked for files named e.g. `cells_data.pt`, but the actual dataset files on disk are named `cells.pt`, `chest.pt` etc. (no `_data` suffix). This caused an immediate `FileNotFoundError` on every run — the pipeline could not load any data at all. |

### Bug 2 — Data leakage in train/val split
| Field | Detail |
|---|---|
| **File** | `Code/data.py` |
| **Lines** | 18–19 |
| **Type** | Silent bug (data leakage) |
| **Before** | `train_data = data_dict['train_images']` / `train_labels = data_dict['train_labels']` |
| **After** | `train_data = data_dict['train_images'][:val_start]` / `train_labels = data_dict['train_labels'][:val_start]` |
| **Why** | `train_data` was assigned the full training set including the validation slice. The model trained on the exact same images it was later evaluated against for validation. No crash occurred — the code ran fine — but validation accuracy was artificially inflated. After the fix, train and val sets are properly non-overlapping (90%/10% split). |

---

## Code/models.py

### Bug 3 — `activation_str = "Identity"` (no nonlinearity in ResNet18)
| Field | Detail |
|---|---|
| **File** | `Code/models.py` |
| **Line** | 9 |
| **Type** | Silent bug |
| **Before** | `activation_str = "Identity"` |
| **After** | `activation_str = "ReLU"` |
| **Why** | `ResNet18` uses `getattr(nn, activation_str)` to pick its activation. `nn.Identity` passes input through unchanged — it does nothing. Every activation in the entire ResNet18 was a no-op, making the network mathematically equivalent to a single linear layer. It trained without crashing but could not learn nonlinear patterns. Note: an intermediate attempt with `"ReLu"` (lowercase u) was also a bug — Python attribute lookup is case-sensitive, so `getattr(nn, "ReLu")` would crash with `AttributeError`. |

### Bug 4 — `VGGBlock`: `current_in_channels` never updated in loop
| Field | Detail |
|---|---|
| **File** | `Code/models.py` |
| **Lines** | 22–27 |
| **Type** | Crash bug |
| **Before** | Loop appended conv layers without updating `current_in_channels` after each layer |
| **After** | Added `current_in_channels = out_channels` at end of each loop iteration |
| **Why** | `current_in_channels` was set before the loop but never updated inside it. In a block with 2+ conv layers, every conv after the first expected the original input channel count, but actually received `out_channels` from the prior layer. This crashed with a channel-mismatch `RuntimeError` the moment VGG16 was used. |

### Bug 5 — `ResBlock.forward`: in-place addition `+=`
| Field | Detail |
|---|---|
| **File** | `Code/models.py` |
| **Line** | ~61 |
| **Type** | Numerical/stability bug |
| **Before** | `out += identity` |
| **After** | `out = out + identity` |
| **Why** | In-place operations can corrupt tensor values PyTorch needs for computing gradients during backpropagation, causing `RuntimeError` about in-place modification. Using `out = out + identity` creates a new tensor safely. |

### Bug 6 — `ResBlock.forward`: missing residual skip connection
| Field | Detail |
|---|---|
| **File** | `Code/models.py` |
| **Line** | ~61 |
| **Type** | Silent architecture bug |
| **Before** | Residual branch output was returned without adding the original identity path |
| **After** | Added the residual connection between the transformed output and the identity path |
| **Commit** | `042865e5` |
| **Why** | A ResNet block is supposed to add the learned transformation back to the original input. Without that skip connection, the block behaves like a plain convolutional block instead of a residual block, which weakens gradient flow and makes `ResNet18` different from the intended architecture. |

### Bug 7 — `AlexNet.__init__`: hardcoded `in_channels=3` and `num_classes=11`
| Field | Detail |
|---|---|
| **File** | `Code/models.py` |
| **Lines** | 66, 75, 103 |
| **Type** | Crash bug + silent bug |
| **Before** | `def __init__(self, **kwargs)` / `nn.Conv2d(3, 48, ...)` / `nn.Linear(1024, 11)` |
| **After** | `def __init__(self, in_channels, num_classes, **kwargs)` / `nn.Conv2d(in_channels, 48, ...)` / `nn.Linear(1024, num_classes)` |
| **Why** | `in_channels` and `num_classes` were not parameters — they were hardcoded to 3 and 11. This crashed on grayscale datasets (`chest`, `orgs`, `organs` — 1 channel) and silently produced wrong-shaped output on datasets with different class counts (`cells`=8, `chest`=2, `lesions`=7). VGG16 and ResNet18 already accepted these as constructor arguments; AlexNet needed to match. |

### Bug 8 — `AlexNet` and `VGG16`: hardcoded `Linear(2048, ...)` input size
| Field | Detail |
|---|---|
| **File** | `Code/models.py` |
| **Lines** | 95 (AlexNet), 126 (VGG16) |
| **Type** | Crash bug |
| **Before** | `nn.Linear(2048, 1024)` in both AlexNet and VGG16 classifiers |
| **After** | `nn.Linear(3072, 1024)` for AlexNet / `nn.Linear(4608, 1024)` for VGG16 |
| **Why** | The flattened feature size after the convolutional layers depends on the final spatial dimensions × channel count. For 64×64 inputs: AlexNet produces `192×4×4 = 3072` features, VGG16 produces `512×3×3 = 4608`. The hardcoded `2048` was wrong for both, causing a matrix-multiplication shape mismatch crash on the first forward pass. |

### Bug 9 — `ResNet18.forward`: classifier output not assigned or returned
| Field | Detail |
|---|---|
| **File** | `Code/models.py` |
| **Lines** | ~185–186 |
| **Type** | Silent bug |
| **Before** | `self.classifier(out)` / `return out` |
| **After** | `out = self.classifier(out)` / `return out` |
| **Commit** | `1addfcd4` |
| **Why** | `self.classifier(out)` computed the final class predictions but discarded the result — it was never assigned to anything. `return out` then returned the pre-classifier flattened tensor instead of actual predictions. The model ran without crashing but its output was meaningless (wrong shape, wrong values). |

---

## Code/train.py

### Bug 10 — `drop_rate=0.99` (extreme dropout)
| Field | Detail |
|---|---|
| **File** | `Code/train.py` |
| **Line** | 32 |
| **Type** | Numerical/silent bug |
| **Before** | `model_class(..., drop_rate=0.99, ...)` |
| **After** | `model_class(..., drop_rate=config.get("DROP_RATE", 0.5), ...)` |
| **Why** | `nn.Dropout(p=0.99)` zeros out 99% of neurons during each forward pass. Normal values are 0.2–0.5. At 0.99, the network has almost nothing to learn from per batch — training was crippled without crashing. Fixed to be configurable via `config.json`, defaulting to a sane 0.5. |

### Bug 11 — `activation_str=None` (dead argument)
| Field | Detail |
|---|---|
| **File** | `Code/train.py` |
| **Line** | 32 |
| **Type** | Dead code / silent bug |
| **Before** | `model_class(..., activation_str=None)` |
| **After** | Argument removed entirely |
| **Why** | No model class in `models.py` accepts `activation_str` as a parameter. It was silently absorbed into `**kwargs` and ignored — misleading code that appeared to configure something but did nothing. |

### Bug 12 — Device selection missing MPS (Apple Silicon GPU)
| Field | Detail |
|---|---|
| **File** | `Code/train.py` |
| **Line** | 19 |
| **Type** | Performance bug |
| **Before** | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` |
| **After** | `device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")` |
| **Why** | The original code only checked for CUDA (Nvidia GPU) or CPU. On Apple Silicon Macs (M-series), PyTorch supports GPU acceleration via MPS (Metal Performance Shaders). Without this fix, all training ran on CPU despite a capable GPU being available. |

---

## Code/fit.py

### Bug 13 — Missing `optimizer.zero_grad()`
| Field | Detail |
|---|---|
| **File** | `Code/fit.py` |
| **Line** | Before `loss.backward()` in `train_one_epoch` |
| **Type** | Numerical bug (gradient accumulation) |
| **Before** | No `zero_grad()` call anywhere in the training loop |
| **After** | `self.optimizer.zero_grad()` added before each forward pass |
| **Why** | PyTorch accumulates gradients by default — they are not reset between batches automatically. Without `zero_grad()`, gradients from batch 1 stack on top of batch 2, then batch 3, growing increasingly wrong with every step. No crash occurs, but training becomes unstable and the model cannot converge properly. |

### Bug 14 — Labels shape `(N, 1)` instead of `(N,)`
| Field | Detail |
|---|---|
| **File** | `Code/fit.py` |
| **Lines** | `train_one_epoch` and `evaluate` |
| **Type** | Crash bug |
| **Before** | `images, labels = images.to(self.device), labels.to(self.device)` |
| **After** | `images, labels = images.to(self.device), labels.to(self.device).squeeze(1)` |
| **Why** | Dataset labels are stored with shape `(N, 1)` — confirmed via testing (`torch.Size([32, 1])`). `nn.CrossEntropyLoss` requires targets shaped `(N,)` (1D integer class indices). Feeding `(N, 1)` caused `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`. Fixed by calling `.squeeze(1)` to remove the extra dimension. Applied in both `train_one_epoch` and `evaluate`. |

### Bug 15 — Variable named `sum` shadows Python built-in
| Field | Detail |
|---|---|
| **File** | `Code/fit.py` |
| **Line** | `train_one_epoch` |
| **Type** | Code quality / latent bug |
| **Before** | `correct, sum = 0, 0` / `sum += labels.size(0)` / `return running_loss / sum, (correct / sum) * 100` |
| **After** | `correct, total = 0, 0` / `total += labels.size(0)` / `return running_loss / total, (correct / total) * 100` |
| **Why** | `sum` is a Python built-in function. Naming a variable `sum` shadows it for the rest of the function scope — any future code in the function calling `sum()` would fail with a confusing `TypeError`. Renamed to `total` to match the naming already used in `evaluate`. |

---

## Summary

| # | File | Bug | Type |
|---|---|---|---|
| 1 | data.py | Wrong filename pattern (`_data.pt`) | Crash |
| 2 | data.py | Data leakage in train/val split | Silent |
| 3 | models.py | `activation_str = "Identity"` — no nonlinearity | Silent |
| 4 | models.py | VGGBlock channel not updated in loop | Crash |
| 5 | models.py | ResBlock in-place `+=` on residual | Numerical |
| 6 | models.py | ResBlock missing residual skip connection | Silent |
| 7 | models.py | AlexNet hardcoded `in_channels=3`, `num_classes=11` | Crash + Silent |
| 8 | models.py | AlexNet/VGG16 wrong Linear input size (2048) | Crash |
| 9 | models.py | ResNet18 classifier output not assigned/returned | Silent |
| 10 | train.py | `drop_rate=0.99` cripples training | Numerical |
| 11 | train.py | `activation_str=None` dead argument | Dead code |
| 12 | train.py | MPS device not checked (Mac GPU unused) | Performance |
| 13 | fit.py | Missing `optimizer.zero_grad()` | Numerical |
| 14 | fit.py | Labels `(N,1)` not squeezed to `(N,)` | Crash |
| 15 | fit.py | Variable `sum` shadows Python built-in | Code quality |

**Total: 15 bugs across 4 files**

---

*Audit completed by Shreyas Hosadurga Sadananda — IDL26 Final Assignment*
