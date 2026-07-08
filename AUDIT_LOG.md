# Audit Log 

| Name | Enrollment Number |
|---|---|
| Shreyas Hosadurga Sadananda | 10013494 |
| Suhaim Manna | |

---

## How We Approached This

We began by reading the assignment PDF and exploring the dataset files to understand shapes, channel counts and label formats before touching any code. We then read each Python file once to understand its role, and proceeded to run the code and fix whatever crashed first, repeating until the pipeline ran cleanly end to end. After the crash bugs were gone we inspected the logic carefully for silent bugs that produce wrong results without any error. In total we found 13 bugs across 4 files. 

---

## Bug Table

| # | File | Category | How It Manifests |
|---|------|----------|-----------------|
| 1 | data.py | Runtime crash | Wrong filename pattern — FileNotFoundError on every run |
| 2 | data.py | Silent logical flaw | Validation data leaks into training set |
| 3 | models.py | Numerical failure | Identity activation — network collapses to single linear layer |
| 4 | models.py | Runtime crash | VGGBlock never updates channel count inside loop |
| 5 | models.py | Numerical/stability | ResBlock in-place += corrupts autograd graph |
| 6 | models.py | Runtime crash | AlexNet ignores in_channels and num_classes entirely |
| 7 | models.py | Runtime crash | AlexNet classifier wrong input size — 2048 vs actual 3072 |
| 8 | models.py | Silent bug | ResNet18 forward() discards classifier output, returns wrong tensor |
| 9 | fit.py | Gradient failure | Missing zero_grad — gradients explode, model never learns |
| 10 | fit.py | Runtime crash | Labels shape [N,1] crashes CrossEntropyLoss |
| 11 | fit.py | Anti-pattern | Variable `sum` shadows Python built-in sum() |
| 12 | train.py | Numerical failure | drop_rate hardcoded at 0.99 — disables 99% of neurons |
| 13 | train.py | Dead code | activation_str=None passed but silently ignored by all models |

---

## Detailed Bug Reports

---

**File:** `data.py`

### BUG 01 — Wrong Filename Pattern

The code constructed the data path as `f"{data}_data.pt"`, producing filenames like `cells_data.pt`. The actual files on disk were named `cells.pt`, `chest.pt`, `lesions.pt` — no `_data` suffix. Every run crashed with `FileNotFoundError` before the pipeline could do anything at all.

```python
# before
d_path = Path(data_path) / f"{data}_data.pt"

# after
d_path = Path(data_path) / f"{data}.pt"
```

---

### BUG 02 — Validation Data Leaks Into Training Set

The split index `val_start` was calculated correctly as 90% of total samples. But the slice `[:val_start]` was only applied when creating `val_data` — never applied to `train_data`. The training set contained all images including the ones in the validation set. The model trained on validation images and was then evaluated against those same images. No crash occurred but all validation results were artificially inflated — this is data leakage and completely invalidates training metrics.

```python
# before
train_data   = data_dict['train_images']
train_labels = data_dict['train_labels']

# after
train_data   = data_dict['train_images'][:val_start]
train_labels = data_dict['train_labels'][:val_start]
```

---

**File:** `models.py`

### BUG 03 — Identity Activation Kills Network Learning

At the top of the file `activation_str = "Identity"`. ResNet18 uses `getattr(nn, activation_str)` to pick its activation. `nn.Identity` passes input through unchanged — it is not an activation function. With Identity used everywhere, the entire ResNet18 collapses mathematically into a single linear transformation regardless of depth. The network trained without crashing but could not learn any nonlinear patterns — accuracy stayed near random chance for all datasets.

```python
# before
activation_str = "Identity"

# after
activation_str = "ReLU"
```

---

### BUG 04 — VGGBlock Never Updates Channel Count in Loop

Inside VGGBlock, `current_in_channels` was initialised before the loop to `in_channels` but was never updated after each conv layer. In a block with `num_convs=2`, after the first conv transforms the data from `in_channels` (e.g. 3) to `out_channels` (e.g. 64), the second conv was still built expecting 3 input channels but was receiving 64. Immediate crash with channel mismatch RuntimeError the moment VGG16 was used on any dataset.

```python
# before
for i in range(num_convs):
    layers.append(nn.Conv2d(current_in_channels, out_channels, ...))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))

# after
for i in range(num_convs):
    layers.append(nn.Conv2d(current_in_channels, out_channels, ...))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    current_in_channels = out_channels
```

---

### BUG 05 — ResBlock In-Place Addition Unsafe for Autograd

The residual addition in ResBlock used the in-place operator `+=`. PyTorch's autograd engine needs to access the original tensor values to compute gradients correctly during backpropagation. In-place operations overwrite those values directly, which can corrupt the autograd graph and cause a RuntimeError about in-place modification of a tensor needed for gradient computation, or silently produce incorrect gradients.

```python
# before
out += identity

# after
out = out + identity
```

---

### BUG 06 — AlexNet Ignores in_channels and num_classes

AlexNet's constructor only accepted `**kwargs` with no named parameters. The first conv layer had `3` hardcoded and the final linear layer had `11` hardcoded. This meant AlexNet always built a 3-channel RGB network with 11 output classes regardless of what was passed in. It crashed on any grayscale dataset (`chest`, `orgs`, `organs` — 1 channel) and produced silently wrong output shapes for `cells` (8 classes), `chest` (2 classes) and `lesions` (7 classes). VGG16 and ResNet18 both accepted these as proper constructor arguments — AlexNet needed to match.

```python
# before
def __init__(self, **kwargs):
    ...
    nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=3)
    ...
    nn.Linear(1024, 11)

# after
def __init__(self, in_channels, num_classes, **kwargs):
    ...
    nn.Conv2d(in_channels, 48, kernel_size=7, stride=2, padding=3)
    ...
    nn.Linear(1024, num_classes)
```

---

### BUG 07 — AlexNet Classifier Wrong Input Size

After fixing Bug 06, AlexNet still crashed with `RuntimeError: mat1 and mat2 shapes cannot be multiplied`. The classifier had `nn.Linear(2048, 1024)` but the actual flattened feature size for a 64×64 input through AlexNet's conv stack is `192 × 4 × 4 = 3072`. The hardcoded 2048 did not match, causing an immediate shape mismatch crash on the first forward pass.

```python
# before
nn.Linear(2048, 1024)

# after
nn.Linear(3072, 1024)
```

---

### BUG 08 — ResNet18 forward() Discards Classifier Output

The final two lines of ResNet18's `forward()` were:

```python
self.classifier(out)
return out
```

In Python, calling a function without assigning the result discards it immediately. The classifier computed correct class predictions but they were thrown away. `return out` then returned the pre-classifier flattened feature vector — the wrong tensor entirely. The model ran without crashing but its output had nothing to do with predictions. Loss was computed on meaningless values, gradients were wrong, and training produced garbage silently with no error message.

```python
# before
self.classifier(out)
return out

# after
out = self.classifier(out)
return out
```

---

**File:** `fit.py`

### BUG 09 — Missing optimizer.zero_grad() in Training Loop

`loss.backward()` and `optimizer.step()` were called each batch but `optimizer.zero_grad()` was never called anywhere. PyTorch accumulates gradients by default — without resetting them, gradients from batch 1 carry into batch 2, batch 2 into batch 3, growing larger and more wrong with every step until they explode. No crash occurred but training loss grew instead of decreasing and the model learned nothing. `zero_grad()` must come before `loss.backward()` — the correct order is always reset → forward → loss → backward → update.

```python
# before
outputs = self.model(images)
loss = self.criterion(outputs, labels)
loss.backward()
self.optimizer.step()

# after
self.optimizer.zero_grad()
outputs = self.model(images)
loss = self.criterion(outputs, labels)
loss.backward()
self.optimizer.step()
```

---

### BUG 10 — Labels Shape [N,1] Crashes CrossEntropyLoss

`nn.CrossEntropyLoss` requires labels as a flat 1D tensor of shape `[N]`. The datasets store labels with shape `[N, 1]` — each class index wrapped in an extra dimension. CrossEntropyLoss interprets this as multi-label classification and crashes immediately with `RuntimeError: 0D or 1D target tensor expected, multi-target not supported`. The fix was applied in both `train_one_epoch` and `evaluate` since both load labels from the same dataloaders.

```python
# before
images, labels = images.to(self.device), labels.to(self.device)

# after
images, labels = images.to(self.device), labels.to(self.device).squeeze(1)
```

---

### BUG 11 — Variable sum Shadows Python Built-in

The running total counter inside `train_one_epoch` was named `sum`. Python's built-in `sum()` function is used to add up sequences. Naming a local variable `sum` shadows the built-in for the entire function scope. It was not crashing here because `sum()` was not called as a function in this method. But it is a dangerous anti-pattern — any future code in this function calling `sum()` would silently receive an integer instead of a callable, producing wrong results with no error. Renamed to `total` to match the naming already used in `evaluate()` below.

```python
# before
correct, sum = 0, 0
sum += labels.size(0)
return running_loss / sum, (correct / sum) * 100

# after
correct, total = 0, 0
total += labels.size(0)
return running_loss / total, (correct / total) * 100
```

---

**File:** `train.py`

### BUG 12 — drop_rate Hardcoded at 0.99

The model was created with `drop_rate=0.99` hardcoded directly in the Python source. `nn.Dropout(p=0.99)` randomly disables 99% of neurons during each training step — only 1% of the network is ever active at once. The model had almost no capacity to process information during training and could not learn complex patterns across 7, 8 or 11 classes. Training ran without crashing but accuracy barely moved. The correct range for dropout is typically 0.2 to 0.5.

```python
# before
model = model_class(..., drop_rate=0.99, activation_str=None)

# after
model = model_class(..., drop_rate=config.get("DROP_RATE", 0.5))
```

---

### BUG 13 — activation_str=None Passed as Dead Argument

On the same model creation line, `activation_str=None` was passed as a keyword argument. None of the model classes — AlexNet, VGG16 or ResNet18 — declare `activation_str` as a named parameter. They all use `**kwargs` which silently absorbs unknown keyword arguments without raising any error. This value was never read by any model. It was dead code that gave the false impression of configuring the activation function while having absolutely zero effect. `activation_str` is a module-level variable in models.py, not a per-instance constructor argument.

```python
# before
model = model_class(..., drop_rate=0.99, activation_str=None)

# after
model = model_class(..., drop_rate=config.get("DROP_RATE", 0.5))
```

---

