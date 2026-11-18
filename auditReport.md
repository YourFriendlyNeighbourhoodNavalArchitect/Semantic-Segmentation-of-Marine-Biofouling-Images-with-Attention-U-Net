# Audit Report: Semantic Segmentation of Marine Biofouling

Status as of 18 November 2026: a review of `3c831f2` on `main`, with no code changes in this pass. Earlier models and checkpoints are out of scope: everything will be retrained.

## Summary

| Area | Third pass | This review | Reason for the change |
|---|---|---|---|
| Project architecture and packaging | 82 | 82 | Unchanged |
| Python code quality | 73 | 71 | Nine unchecked `zip()` calls (O41); the configuration still accepts invalid values (O21, O22) |
| Reproducibility and experiment management | 79 | 72 | `deterministic = true` cannot work as written (O15); augmentations depend on `numWorkers` (O34); the study fingerprint ignores the split and the code (O35) |
| Tests, CI, documentation | 35 | 35 | Unchanged |
| Dataset curation | 65 | 65 | Unchanged |
| Model design | 66 | 65 | Overconfident initial output layer (O40) |
| Training and evaluation | 77 | 74 | Non-square resolutions crash (O33); divergence goes undetected (O37); the GPU export check may fail (O38) |
| **Overall** | **70** | **68** | The code is unchanged; the scores are lower because of new findings |

Scores are my estimates.

| Status | Count | Section |
|---|---|---|
| Resolved, verified and removed from this report | R1–R45 | |
| Decisions needed from you | 15 | 1 |
| Open issues | 31 (10 new: O33–O42) | 2 |
| Implementation proposals: O15, O16, O17, O18, O19, O20, O31 | 7 | 3 |
| Actions for you | 6 | 4 |
| Rectification plan | 7 phases | 5 |

**Verification of the earlier resolutions.** I read every module and the Windows app, installed the project from `uv.lock` (135 packages, no torchviz or graphviz), and confirmed R35–R45 in the code:
- The attention gate matches Oktay et al.: W_x is a 2×2 convolution with stride 2 and no bias, W_g carries b_g, ψ carries b_ψ, the gate has no batch normalisation, and the coefficients are resampled bilinearly.
- 22 + 4 convolutions that feed batch normalisation have no bias.
- Imports are by module.
- The dataset browser no longer blocks in its constructor.
- The Labelbox entries are validated field by field.
- Logging and type hints are in place.
- The `--interval` option, the `Pixel Accuracy` metric and `None` for absent classes are in place.
- The resolution and the monitored metric are validated at load time.
- The metrics module lives in `evaluation`.
- Study settings are compared key by key.

There are two corrections:
- **R45's reasoning is wrong.** `numWorkers` changes more than the speed: it changes the augmentations (O34). Excluding it from the study fingerprint is defensible, because the effect is equivalent to seed noise, but its comment is not.
- **The static-check claim was wrong.** `ruff check --select F,E9,B` does not pass. F and E9 pass, but B905 flags 9 `zip()` calls without `strict=` (O41).

**Checks run in this review (CPU, synthetic data):**
- A 64×96 resolution passes validation, then crashes at the first batch (O33).
- One augmented epoch gives a different hash for each of `numWorkers` 0, 1, 2 and 4. Each count reproduces itself (O34).
- `torch.use_deterministic_algorithms` (PyTorch 2.15) lists bilinear `interpolate` backward, `AdaptiveAvgPool2d` backward and `NLLLoss` as raising on CUDA. The project uses all three (O15).
- A bilinear ×2 upsampling built from slices matches `interpolate` to 7e-16, and a one-hot cross-entropy equals `CrossEntropyLoss` exactly. These are the deterministic replacements proposed for O15.
- The initial cross-entropy is 2.40 (Attention U-Net) and 3.61 (U-Net), against ln 4 = 1.39 for an uninformed prediction (O40).
- The configuration accepts `maxEpochs = 0`, `batchSize = -3`, `patience = 0` and a misspelt `[trainng]` section (O21, O22).
- Optuna still emits an `ExperimentalWarning` for each study plot (O25).
- PyTorch 2.15 enables TF32 for cuDNN convolutions by default (O38).

---

## 1. Decisions needed from you

⚠ marks a change that would make results differ from the published paper. D9 (the Windows app) is merged into the O20 proposal (3.6). D10 (old models) was dropped, because everything will be retrained.

| ID | Decision | Recommendation |
|---|---|---|
| D1 | **Pretrained encoder** ⚠ | Add a pretrained-encoder variant (ResNet or ConvNeXt encoder with your decoder and gates). This is likely the largest single gain. |
| D2 | **Seeds and uncertainty** ⚠ | Top 2–3 grid settings × 3 seeds; select by the mean; report mean ± standard deviation. This needs O36 and is built into the O16 proposal (3.2). |
| D3 | **Weight decay** ⚠ | AdamW, with batch-norm and bias parameters excluded; learning rate {1e-4, 2e-4, 4e-4, 8e-4} × decay {0, 1e-2, 5e-2}. |
| D4 | **Training crop scale and colour augmentation** ⚠ | Align the crop scale with evaluation; add underwater colour cast, turbidity, noise and compression. |
| D5 | **Evaluation at native resolution** ⚠ | Upsample the logits bilinearly to the native size, then take the argmax; score against the full-resolution masks. Use the same rule in the app (3.6). |
| D6 | **Aspect ratio** ⚠ | Resize the longer side and pad, or tile. Settle together with O33. |
| D7 | **Reporting the split** ⚠ | State that the split is image-level; report the grouped-split score as well. |
| D8 | **Loss weighting** ⚠ | Only if a class lags. |
| D11 | **Tests and CI** | As in the plan (Phase 1). |
| D12 | **Single-run resume** | Save the optimizer, scheduler, scaler and epoch state every epoch. |
| D13 | **`mlflow-skinny`** | Switch: 28 fewer packages, about 200 MB. The web interface then runs through `uvx`. |
| D14 | **Where usage lives** | The `biofouling --help` epilog. |
| D15 | **Package named `src`** | Keep. |
| D17 | **Downscaling interpolation** ⚠ | Area interpolation, in Python and the app together. |
| D18 | **Reproducing the paper.** `9789a9f` is the last commit whose networks equal the published ones | Tag `9789a9f` (e.g. `paper-2026`) and reproduce the published table there (Phase 3). Then train `main` with its own learning rate grid. Keep only the current gate in the code. |

---

## 2. Open issues

**New** marks an issue found in this review. All the others were re-checked against the code and are still open.

### Engineering

| ID | Issue | Suggested fix |
|---|---|---|
| O4 | The grouped split silently ignores unknown similarity IDs (`similarIDs & parent.keys()`) | Warn, listing the IDs |
| O8 | Only `<ID>.jpg` is accepted. Python ignores EXIF orientation, while the app (OpenCV `ImRead`) and Labelbox apply it. A rotated non-square image is only caught by Albumentations' shape check during training; a rotated square image is never caught | Accept other extensions; apply `ImageOps.exif_transpose` in Python; compare image and mask sizes in `split` (PIL reads only the header) |
| O21 | The top-level configuration is not validated: a misspelt section (e.g. `[trainng]`) is ignored, and unknown `paths` keys raise a bare `TypeError` | Check the top-level keys; build `Paths` through `buildSection` |
| O22 | Counts are not range-checked: `maxEpochs = 0` crashes at `load_state_dict(None)`; a batch size larger than the training subset gives zero batches and a division by zero; `batchSize = -3` and `patience = 0` are accepted. `monitorMode` is not tied to the metric, so `Loss` with `max` is accepted | Range checks at load time; derive the mode from the metric (`Loss` → `min`, everything else → `max`) and drop `monitorMode` |
| O24 | MLflow stores absolute artifact paths, so moving `outputs/` breaks the links | Fixed absolute path, or an MLflow server with `mlflow-artifacts:/`; decide before A3 |
| O25 | Optuna emits an `ExperimentalWarning` for each study plot (confirmed) | Filter `optuna.exceptions.ExperimentalWarning` around `plotStudy` |
| O26 | `peakMemoryGiB` is not reset between tuning trials, so each trial reports the maximum so far | `torch.cuda.reset_peak_memory_stats()` at the start of each run |
| O28 | `trialLearningRate` reads Optuna internals (`system_attrs['fixed_params']`) | `user_attrs` on `enqueue_trial`; this becomes necessary for O16's (learning rate, seed) points |
| **O33** | **New.** Non-square resolutions pass validation but crash training. `RandomRotate90` swaps height and width for half the samples, so collation fails (verified with 64×96). `RandomSizedCrop` also crops squares (`w2h_ratio = 1`) and stretches them to H×W | Until D6 is decided, require a square resolution in `validateConfiguration`. With D6, use flips and transposes that keep the shape |
| **O34** | **New.** Augmentations depend on `numWorkers`. Each worker seeds its own Albumentations generator, and samples are dealt to workers in turn, so a different worker count gives different augmentations (verified: four counts, four hashes). A run is reproducible only with the same `numWorkers`, and the R45 comment ("changes the speed only") is false | Seed every sample: see 3.1, step 3 |
| **O35** | **New.** The study fingerprint ignores the split contents and the code. If `split` is run again after images are added, or the model code changes (as it did in `3c831f2`), a study resumes and mixes its trials with old ones, skipping learning rates it believes are done. **This affects the retraining now:** an `outputs/optuna.db` from before `3c831f2` holds studies with the same names | Add the `splitHash` and a hash of the source files that affect training (`src/models`, `src/data`, `src/training`, `src/evaluation/computeMetrics.py`) to the fingerprint. Start the retraining with an empty `outputs/` (A6) |
| **O36** | **New.** One `seed` drives both the split and training. Varying the training seed (D2) therefore either changes the split, if `split` is run again, or is refused by the study fingerprint | Separate `split.seed` from `training.seed`. Seeds of a study become a dimension of its grid (3.2) |
| **O37** | **New.** Divergence is not detected. A run whose loss becomes NaN or infinite keeps training on constant predictions until `patience` runs out (50 epochs). If the monitored metric is never finite, `bestState` stays `None` and the run crashes. High grid points (1e-3) with mixed precision are the likely trigger | Accumulate a non-finite flag on the GPU and check it once per epoch. Stop the run as `FAILED`, or prune the trial, with a clear message |
| **O38** | **New, not verified (no GPU here).** The ONNX export check compares the model on the training device with ONNX Runtime on the CPU, using rtol = atol = 1e-3. On an Ampere or newer GPU, PyTorch's convolutions run in TF32 by default (`cudnn.allow_tf32 = True`, confirmed in 2.15), with about 1e-3 relative precision per layer across 23 layers, so the check may fail spuriously | Always export on the CPU: move the model to `cpu` in `exportRun`. This also makes the export independent of the device |
| **O39** | **New.** `gitCommit` marks a run as dirty with `git diff --quiet HEAD`, which ignores untracked files. A run that uses a new, uncommitted module is recorded as clean | `git status --porcelain --untracked-files=normal` |
| **O41** | **New.** Nine `zip()` calls have no `strict=` (ruff B905), e.g. class names against per-class values in `metricsToKeys`, where a length mismatch would silently drop classes | `strict = True`; enable B905 in the ruff configuration (Phase 1.3) |

### Dataset

| ID | Issue | Suggested fix |
|---|---|---|
| O9 | No data card | Generate it from `metadata.json` and the masks, including the per-image class coverage (see 3.7) |
| O10 | Overlapping annotations are resolved by export order | Fixed class priority |
| O11 | The grouped split balances group counts, not image counts | Size-aware assignment |
| O29 | Frame names (`<video> frame<n>`) cannot be referenced by similarity tags; leading zeros are dropped | Numeric frame IDs, or group frames by video |
| O31 | "Class Correlation Matrix" is a row-normalised co-occurrence, not a correlation | See 3.7 |
| O32 | Frame extraction seeks before every frame, which snaps to keyframes with many codecs | Sequential reading |

### Model

| ID | Issue | Suggested fix |
|---|---|---|
| O12 | No strong external baseline | DeepLabV3+, SegFormer or nnU-Net |
| **O40** | **New.** ⚠ Every convolution is initialised with Kaiming normal (fan-out, ReLU gain), including the output layer, which no ReLU follows. Measured at random initialisation: logit standard deviation 1.9 (Attention U-Net) and 3.3 (U-Net), and an initial cross-entropy of 2.4 and 3.6 against ln 4 = 1.39. The network starts confidently wrong, so early gradients first undo that confidence | Initialise the output layer with small weights (normal, std 0.01) and a zero bias. Low priority; do it with Phase 5.3 |

### Training

| ID | Issue | Suggested fix |
|---|---|---|
| O15 | `deterministic = true` does not deliver determinism, and the configuration comment claims it does. **Extended:** enabling `torch.use_deterministic_algorithms` as written would raise on three operations used by every architecture | See 3.1 |
| O16 | The learning rate tuned for the Attention U-Net is reused for the ablation, and the default 4.1e-4 was tuned for the old gate | See 3.2 |
| O17 | No dropout or gradient clipping | See 3.3 |
| **O42** | **New, not measured.** Every training sample decodes the full-resolution JPEG and resizes it to twice the model resolution, every epoch. With large photographs and 4 workers, data loading may limit GPU throughput | Measure GPU utilisation first (Phase 3.1). If it is the bottleneck, cache the training images once at 2H×2W, stored losslessly. That is the first step of the training pipeline, so the training inputs are unchanged |

### Evaluation

| ID | Issue | Suggested fix |
|---|---|---|
| O18 | No confidence intervals (a held-out vessel is out of scope) | See 3.4 |
| O19 | The best/worst ranking is distorted by stray-pixel classes | See 3.5 |

### Windows application

| ID | Issue | Suggested fix |
|---|---|---|
| O20 | No project file and **no `Program.cs`** (no entry point), so the app cannot be built from the repository. All images are loaded up front; the loops are per pixel, with one `float[]` allocated per pixel in the argmax; masks are saved at model resolution. **New:** `a.jpg` and `a.png` both write `a_mask.png`; every error shows a modal dialog through a synchronous `Invoke`, which stalls processing until it is dismissed; the "Operation cancelled" status is overwritten at once by `finally`; no class coverage is reported | See 3.6 |

---

## 3. Implementation proposals

### 3.1 O15: determinism that works as documented

**What breaks today.** `setSeed` only sets `cudnn.deterministic`, which covers convolutions. Three other operations remain nondeterministic on CUDA, and `torch.use_deterministic_algorithms(True)` would raise on each of them:

| Operation | Where | Deterministic replacement | Effect on results |
|---|---|---|---|
| Bilinear `interpolate` (backward) | Decoder upsampling (4×), attention gates (4×), Simple CNN output (×8) | `upsampleBilinear(x, factor)` built from slices, `cat`, weighted sums and an interleaving `reshape`. For an integer factor s, output phase r blends two neighbouring rows with weight \|(r + 0.5)/s − 0.5\|, clamped at the borders | None: matches `interpolate` (verified for ×2 to 7e-16) |
| `AdaptiveAvgPool2d(1)` (backward) | Squeeze-and-excitation | `x.mean((2, 3), keepdim = True)` | None |
| `NLLLoss` (inside `CrossEntropyLoss`) | Loss | `-(log_softmax(prediction.float(), 1) * oneHot).sum(1).mean()`, sharing the one-hot tensor already built for the Dice loss | None: verified identical |

**Steps**
1. Add `src/models/upsampling.py` with `upsampleBilinear`, and use it in `UpSample`, `AttentionGate` and `SimpleCNN`. Replace the pooling and the loss as in the table. The ONNX graph gains only standard operators (Slice, Concat, Mul, Add, Reshape), so the app needs no change.
2. In `setSeed`, call `torch.use_deterministic_algorithms(deterministic)` and keep the cuDNN flags. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` with `environ.setdefault` in `src/__init__.py`, next to the MLflow setting, because it must precede CUDA initialisation. It is harmless when determinism is off.
3. Make augmentations independent of `numWorkers` (O34). A small sampler in the main process wraps the shuffled indices and yields `(index, sampleSeed)` pairs, drawing the seeds from the seeded loader generator. `__getitem__` calls `transformCompose.set_random_seed(sampleSeed)` before transforming. Every sample then gets the same augmentation whichever worker loads it, and whatever the worker count. `seedWorker` keeps only the Python and NumPy seeds.
4. Correct the configuration comment: "Bitwise reproducible on the same GPU, driver and package versions, at the cost of speed." Add the cuDNN and CUDA versions to the run tags (`torch.backends.cudnn.version()`, `torch.version.cuda`).
5. Tests:
   - On CPU, in CI: each replacement equals the original operation; two 2-epoch synthetic runs give identical weight hashes; `numWorkers` 0 and 2 give identical batches.
   - On GPU, once, in Phase 3.2: two runs with `deterministic = true` give identical weights, and the slowdown is measured.

**Decision.** Keep `deterministic = false` for tuning, and use `true` for the final seeded runs if the measured slowdown is acceptable. Steps 1–3 are exact or seed-only changes, so they can land in Phase 1, before any training.

### 3.2 O16: a learning rate grid per architecture, with seeds (D2)

The mechanism already exists: the study name contains the architecture, so `biofouling tune --config <architecture file>` runs a separate study for each architecture. What is missing is the protocol, the configuration files and the seeds.

**Steps**
1. Add `configs/architectures/uNet.toml`, `spatialAttentionUNet.toml` and `simpleCNN.toml`. Each sets `model.architecture` and that architecture's tuned `training.learningRate`. `default.toml` holds the Attention U-Net. `train` then uses the right rate for each architecture, and the comment cites the study that produced it.
2. Use a denser grid where the optimum lies. The previous best, 4.1e-4, falls between the grid points 2e-4 and 5e-4. Proposal: {5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3}. `tune` warns if the best point is at an edge of the grid. Extending the grid only trains the new points, as now.
3. Seeds become a grid dimension (requires O36 and O28):
   - `tuning.seeds = [1]` gives the single-seed stage 1.
   - Stage 2 lists the top 3 learning rates with `tuning.seeds = [1, 2, 3]`. Completed (learning rate, seed) points are kept, so only the new seeds train.
   - The best learning rate is the one with the best mean over seeds, counting only learning rates with every seed complete.
   - Pruning stays on in stage 1 and is off in stage 2, where every seed must finish.
   - `training.seed` leaves the fingerprint; `split.seed` stays in it.
4. Cost: 4 architectures × 8 points + 4 × 3 learning rates × 2 extra seeds = 56 runs, against 28 for the current grid. Pruning cuts some of stage 1.

**Decision.** Should stage 2 run for all four architectures (a fair ablation, 56 runs), or only for the Attention U-Net and the U-Net (44 runs)? I recommend all four.

Weight decay (D3) adds a dimension later, in Phase 5.3. Do not add it now.

### 3.3 O17: gradient clipping and dropout ⚠

**Gradient-norm logging (no change to results; do it now).**
- In `trainOneEpoch`, call `scaler.unscale_(optimizer)`, then `norm = clip_grad_norm_(model.parameters(), maxNorm)` before `scaler.step`.
- With `maxNorm = inf`, the gradients are not modified and the norm is still returned.
- Accumulate the mean and maximum norm on the GPU, so there is no synchronisation per batch. Log `training/gradientNorm` and `training/gradientNormMax` each epoch.

**Clipping.**
- Add `training.gradientClipNorm`, with 0 meaning off.
- Choose the threshold from a reference run's logged norms (about their 99th percentile), not by guessing. Also log the fraction of clipped steps.
- Steps that mixed precision skips (infinite gradients) are unaffected.
- Expected effect: small at the tuned learning rate, but steadier high grid points and first epochs. The grid optimum may move up, so tune the learning rate again after enabling it.

**Dropout.**
- Add `model.dropout`, default 0.
- Use `Dropout2d(p)` on the bottleneck output, where Ronneberger et al. placed dropout ("at the end of the contracting path"). For the Simple CNN, place it before the last 1×1 convolution.
- Do not place it inside the convolution blocks: dropout before batch normalisation shifts the variance between training and inference (Li et al., 2019).
- Export uses `eval()`, so the ONNX model carries no dropout.

**Order.** Do this in Phase 5.3, together with D3: decay {0, 1e-2, 5e-2} × dropout {0, 0.2} at the tuned learning rate, then re-check the learning rate one grid step either side. I expect smaller gains than from D1 or D4. The data set is small, so measure rather than assume.

### 3.4 O18: confidence intervals (no held-out vessel)

**Data.**
- `ModelTester` already computes one confusion matrix per image. Keep them all as an (N, C, C) array.
- Save them with their IDs as `imageConfusionMatrices.npz`, attached to the run. They are a few kilobytes.

**Bootstrap.** Add `src/evaluation/bootstrap.py`:
1. **Resample clusters, not images.** Under the default stratified split, frames of one video share the testing subset and are correlated, so resampling images would understate the width. Use the similarity groups (`groupBySimilarity`, restricted to the test IDs), and sum each group's matrices into (G, C, C).
2. **Vectorise.** `counts = rng.multinomial(G, [1/G] * G, size = B)` gives a (B, G) array, and `totals = counts @ groupMatrices.reshape(G, -1)`. Metrics come from `computeMetrics`, generalised to a leading batch dimension (`(..., C, C)`), so a single code path serves the tester, training and the bootstrap. 10,000 resamples take well under a second.
3. **Fix the class set.** Average over the classes present in the full test ground truth, so that a resample that happens to lack a class does not change what is averaged. This is consistent with 3.5.
4. **Use percentile intervals** (95 %). Switch to BCa only if the distributions are visibly skewed.

**Output.**
- Console: `IoU 0.7123 (95 % CI 0.6810–0.7402)`.
- `testResults.json` gains `intervals`. MLflow gains `test/<metric>/low` and `test/<metric>/high`.
- The LaTeX row takes either the interval or ± half its width (decision below).

**Paired comparison.** `biofouling compare --run A --run B` refuses unless both runs share the `splitHash`. It resamples the same clusters for both models and reports the interval of each difference, with the fraction of resamples in which A is better. This is what the ablation table needs: two overlapping single-model intervals do not show that two models are equal.

**Seeds (D2) are a separate source of variance.** Report the mean ± standard deviation over 3 training seeds, and the bootstrap interval of the seed-averaged test metric. The interval covers resampling of test images from the same vessels and conditions. It does not cover new vessels, and the report should say so.

**Tests.** Identical images give a zero-width interval; the batched `computeMetrics` equals the single-matrix version; a fixed seed reproduces the interval.

**Decision.** For the LaTeX table, use [low, high] or ±. I recommend ±, with the half-width of the 95 % interval, because it fits the published column layout.

### 3.5 O19: ranking of best and worst predictions

**The flaw.** The per-image score is the macro Dice over classes present in the prediction or the ground truth. A class absent from the ground truth that receives a few stray pixels scores 0 and drags the image down by 1/k. A near-perfect clean-hull image with 20 stray "Hard Fouling" pixels scores 0.5 and ranks among the worst.

**Fix.** Rank by the macro Dice over the classes present in that image's ground truth (TP + FN > 0). The stray pixels still count, as false negatives of the true class, in proportion to their area, instead of as a whole class scored 0. The image above scores about 0.9999.

**Steps**
1. Add an `averageOver` argument (`'union'` or `'groundTruth'`) to `computeMetrics`. The subset metrics keep `'union'`, the published definition; the ranking uses `'groundTruth'`.
2. Save `imageMetrics.csv` (ID, Dice, IoU, pixel accuracy, and the ground-truth area of each class), so that any ranking can be checked.
3. Caption the figure: "Ranked by Dice over the ground-truth classes of each image".

An image that misses a tiny ground-truth class (e.g. 0.05 % barnacles) still ranks low. That is arguably right, but if it dominates the "worst" row, add a minimum class area (e.g. 0.1 %) for the ranking only.

### 3.6 O20: Windows application (formerly D9)

**1. Buildable.**
- Add `Application/BiofoulingSegmentation.csproj` (`net10.0-windows`, `UseWindowsForms`, `Nullable`, packages `Microsoft.ML.OnnxRuntime`, `OpenCvSharp4` and `OpenCvSharp4.runtime.win`).
- Add `Program.cs` (`[STAThread] Main`, `ApplicationConfiguration.Initialize()`, `Application.Run(new GUI())`).
- Add `Application/bin/`, `Application/obj/` and `.vs/` to `.gitignore`.
- A GitHub Actions job on `windows-latest` runs `dotnet build`, so the app cannot silently break.
- Release with `dotnet publish -c Release -r win-x64 --self-contained -p:PublishSingleFile=true`.

**2. Streaming.**
- Enumerate the paths, then load, infer, save and dispose one image at a time, so memory no longer grows with the folder.
- Progress is the file count. Cancellation is checked between images.

**3. Vectorised.**
- **Input:** `CvDnn.BlobFromImage(image, 1 / 255.0, size, swapRB: true)` resizes (bilinear, as now), converts BGR to RGB, scales and lays the data out as NCHW in one call. The pixel values are identical to the current path, because both resize the 8-bit image before scaling. Wrap the blob's memory in an `OrtValue`, without copying.
- **Argmax:** C passes over contiguous spans, keeping a running maximum and index, with no allocation per pixel.
- **Colouring:** `Cv2.LUT` with a 256-entry palette.

**4. Masks at native resolution, with the same rule as D5.** Resize each class's logit plane bilinearly to the original size, then take the argmax. That is about 190 MB for four classes at 12 MP. If that is too much, cap the output at, say, 4096 px on the longer side.

**5. Outputs that serve the inspection.**
- `<name>.<ext>.mask.png`: the extension is kept, which fixes the `a.jpg`/`a.png` collision.
- Optionally `<name>.<ext>.overlay.jpg`, with the mask blended over the photograph.
- `coverage.csv`: file, width, height, and the percentage of each class. That is the quantity an assessment of the hull actually uses.

**6. Status and errors.**
- Use `IProgress<T>` (`Progress<T>`) instead of hand-written `Invoke`.
- Collect per-image errors into a list and an `errors.log` in the output folder. Show one summary at the end, with no modal dialogs from the worker.
- Show "Cancelled" or "Completed with N errors" as the final state, instead of overwriting it in `finally`.
- Disable the folder and model buttons while processing.
- After a model is selected, show its architecture, classes, resolution and run ID, read from its metadata.

**7. Input.**
- Extensions `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`, with an optional subfolder search.
- EXIF orientation is already applied by `ImRead`; Python must follow (O8).

**8. Parity with Python.**
- Add `biofouling predict --model <onnx> --input <folder> --output <folder>`, sharing the preprocessing and post-processing functions with `ModelTester`.
- Run both on a few images and require identical masks. This test anchors D5, D6, D17 and O8 across the two code bases, and also gives inference on Linux.

**9. Optional: a GPU on Windows.** The `Microsoft.ML.OnnxRuntime.DirectML` package runs on any DirectX 12 GPU, with the CPU as fallback. The model is unchanged.

**Order.** Steps 1, 2, 3 and 6 are pure engineering and need no decisions. Steps 4, 5 and 8 depend on D5, D6 and D17, so they come after Phase 5.1.

### 3.7 O31: the "Class Correlation Matrix"

**What it is.** Cell (i, j) is the number of images that contain both i and j, divided by the number that contain i: the probability that j is present given that i is. It is asymmetric, its diagonal is 1 by construction, and the diverging `coolwarm` colormap suggests a meaningful midpoint at 0.5, which it does not have.

**Options**

| | Conditional co-occurrence (kept, corrected) | Correlation of presence (phi coefficient) |
|---|---|---|
| Question answered | "When soft fouling is present, how often is hard fouling present too?" | "Do two classes appear together more or less often than chance?" |
| Range and colormap | [0, 1]; sequential (`Blues`, as the confusion matrix) | [−1, 1]; diverging, centred on 0 |
| Symmetric | No: rows are the given class | Yes |
| Caveat | Mainly reflects how common each class is | Undefined for a class present in every image (zero variance), e.g. possibly background; show as n/a |

**Recommendation.** Keep the conditional co-occurrence, which is the easier figure to read in a paper, with these corrections:
- Title: "Class Co-occurrence: P(column class present | row class present)".
- Axis labels: "Given class" and "Co-occurring class".
- The sequential colormap.
- The diagonal shows each class's image count (or is blanked) instead of 1.00.

Check the paper's caption and text against this definition; if they say "correlation", an erratum may be needed. Add the phi matrix to the data card (O9) only if a correlation question arises.

**Further.** Co-occurrence uses presence only. For the data card, the distribution of each class's area per image (box or violin plot from the masks) is more informative: it shows the class imbalance that drives D8.

---

## 4. Actions for you

| ID | Action |
|---|---|
| A1 | **Pull `main` and run `uv sync`.** The package layout changed (`9789a9f`), and torchviz and graphviz were removed (`3c831f2`). |
| A2 | **Tag the paper commit (D18):** `git tag paper-2026 9789a9f` and `git push origin paper-2026`, if you agree. |
| A3 | **Move the data to the new layout:** images to `data/images/`, masks to `data/masks/`, `Metadata.json` to `data/metadata.json`, `Labels.ndjson` to `data/labels.ndjson`; then `uv run biofouling split`. |
| A4 | **First GPU run:** NVIDIA driver 580 or newer; decide O24 first if training elsewhere. |
| A5 | **Decide D13, D14, D17 and D18**, and the decisions in 3.1, 3.2 and 3.4. All are small. |
| A6 | **New: start the retraining with an empty `outputs/`.** Move aside `optuna.db`, `mlflow.db` and `mlruns/`. Studies and experiments keep their names across code changes (O35), so a resumed study would skip learning rates trained with the old gate. |

---

## 5. Rectification plan

The principles are unchanged:
- Engineering fixes come before changes to results.
- The measurement protocol is fixed before the model changes.
- Changes that affect results are made one at a time, against a recorded baseline.

### Phase 0: Housekeeping (no change to results)

| Step | Items |
|---|---|
| 0.1 | A1, A6, D13, D15 |
| 0.2 | O21, O22, O33 (require a square resolution), O37 |
| 0.3 | O28, O26, O35, O36, O39 |
| 0.4 | O4, O8 (extensions, size check), O25, O41 |
| 0.5 | O38 (export on CPU), D14 |

### Phase 1: Safety net (D11) and exact changes

| Step | Items | Outcome |
|---|---|---|
| 1.1 | Unit tests | Metrics (batched, absent classes, `averageOver`), loss, class remapping, configuration merging and every validation error, split, dataset, attention gate shapes, upsampling and cross-entropy equivalence |
| 1.2 | Smoke test | The synthetic run as one test, on CPU in about a minute, including determinism and independence from `numWorkers` |
| 1.3 | ruff and CI | ruff configured (F, E9, B including B905, naming rules that allow camelCase); GitHub Actions on CPU from `uv.lock`; a Windows job that builds the app once 3.6 step 1 exists |
| 1.4 | O15 steps 1–4, O34 | Deterministic operations and per-sample seeds, exact by construction; gradient-norm logging (3.3) |

### Phase 2: Data

| Step | Items | Outcome |
|---|---|---|
| 2.1 | A3 | Data in the new layout |
| 2.2 | O29, O32 | Evenly spaced frames, named so that they can be grouped |
| 2.3 | O10 ⚠, O8 (EXIF) ⚠ | Fixed class priority, oriented images; masks regenerated and checked |
| 2.4 | O11, O4 | Size-aware grouped split; warnings for unknown IDs |
| 2.5 | O9, O31 | Data card; corrected co-occurrence figure |

### Phase 3: Reference results (GPU)

| Step | Items | Outcome |
|---|---|---|
| 3.1 | A4, O24, D12, O42 (measure) | First GPU run, resumable, with a movable artifact store; data-loading throughput measured |
| 3.2 | O15 step 5 | Determinism verified on the GPU, and its cost measured |
| 3.3 | D18 | **Paper reproduction at tag `paper-2026`:** the four published architectures with the published protocol. They should match the paper within run-to-run noise; if they do not, the cause is found first |
| 3.4 | `main` | The same protocol on `main`. The U-Net and Simple CNN should equal 3.3 within noise (only biases differ). The difference for the attention variants measures the original gate alone |

### Phase 4: Measurement protocol ⚠

| Step | Items | Outcome |
|---|---|---|
| 4.1 | D5 | Native-resolution scoring |
| 4.2 | O18, D2 | Cluster-bootstrap intervals, paired comparisons, seeds |
| 4.3 | D7, O19 | Both split scores; corrected ranking |
| 4.4 | O16 | Grid and seeds per architecture, giving the new baseline table and the per-architecture learning rates |

### Phase 5: Model improvements ⚠, one at a time against Phase 4

| Step | Items |
|---|---|
| 5.1 | D6, D17, O33 (rectangular inputs): input geometry |
| 5.2 | D4: augmentation |
| 5.3 | D3, O17, O40: regularisation and initialisation |
| 5.4 | D8: loss weighting, if needed |
| 5.5 | D1: pretrained encoder, with the original gates |
| 5.6 | O12: external baseline |

### Phase 6: Deployment

| Step | Items |
|---|---|
| 6.1 | O20 steps 1, 2, 3, 6, 7, 9: buildable, streaming, vectorised app |
| 6.2 | O20 steps 4, 5, 8, with D5, D6, D17 and O8: native-resolution masks, coverage report, verified parity with `biofouling predict` |

---

## 6. Reference

**Settled decisions**
- **Structure:** package `src` (`data`, `models`, `training`, `evaluation`, `visualization`), command `biofouling`, configuration in `configs/`, Windows app in `Application/`, camelCase.
- **Dependencies:** 16 direct dependencies, each imported by the code except `onnxscript` (needed by the ONNX exporter); 135 packages in `uv.lock`; no optional extras. Albumentations stays at 2.0.8, the last MIT release.
- **Attention gate:** Oktay et al. (2018), as published. Channel attention: squeeze-and-excitation, unchanged.
- **Logging:** the `src` logger only; libraries keep their own settings.
- **Classes, split, tuning, tracking, evaluation:** as before.

**GPUs:** unchanged (driver 580+, compute capability 7.5–12.0, about 20 GB at batch 16, 512×512, mixed precision).
