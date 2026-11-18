# Audit Report: Semantic Segmentation of Marine Biofouling

Status as of 18 November 2026, for branch `claude/hopeful-lamport-jjxmku` after `559815d`.

**Rewritten history.** The commits of `main` were re-authored, with unchanged contents: `9789a9f` → `43aac81` and `3c831f2` → `d6206f0`. `main` itself still points to the old commits until you approve the rewrite (A7).

**Principle of this report.** It lists only problems that change results, make reported numbers wrong, or stop the pipeline from running. It proposes no checks of values that only the owner sets, and no guards against misuse. Section 3 lists existing defensive code that can go.

## Summary

| Area | Previous review | Now | Reason |
|---|---|---|---|
| Project architecture and packaging | 82 | 85 | The Windows app builds; the export runs on the CPU |
| Python code quality | 71 | 71 | Unchanged code; defensive code to remove (section 3) |
| Reproducibility and experiment management | 72 | 74 | A changed configuration starts a new study instead of raising (O35, proposed) |
| Tests, CI, documentation | 35 | 38 | Build workflow and README for the app |
| Dataset curation | 65 | 62 | Video frames cannot be grouped, so the grouped split leaks (O29) |
| Model design | 65 | 62 | Most attention coefficients start saturated (O40) |
| Training and evaluation | 74 | 76 | GPU export fixed (O38) |
| **Overall** | **68** | **70** | |

**Resolved since the previous review, and removed:**
- O38: the export always runs on the CPU (`a3dde03`).
- O20, build part: the project file, the `Program.cs` entry point, streaming, vectorised processing, error log, final status, distinct output names, the build workflow and `Application/README.md` (`559815d`). The app built with no warnings or errors (.NET 10 SDK on Linux, targeting Windows). On test images, its preprocessed input and its masks were bit-identical to the Python pipeline. The window itself has not been run on Windows.

**Dropped as unnecessary:**

| Items | Reason |
|---|---|
| O33 (non-square resolution), D6 as a separate decision | Images are always resized to a square. The aspect question moves into the scale decision (D5) |
| O36 (one seed for split and training) | Premise wrong: `seed` affects the split only when `split` runs; training reads the saved manifest |
| O37 (divergence detection) | A defence, not a fix |
| O9 (data card), O10 (overlap order), O11 (size-aware grouping), O32 (frame seeking) | No evidence that they change the reported numbers |
| O21, O22's range checks, O39, O41 | Checks of values only the owner sets, or style |
| O24 (artifact paths), O25 (Optuna warning), O28 (Optuna internals) | Cosmetic, or works as is |
| O26 (peak memory) | Delete the metric instead (S9) |
| O42 (decode cache) | Replaced by the scale decision (D5) |
| D9, D10, D12 (single-run resume), D13 (`mlflow-skinny`) | Merged, out of scope, or no gain |
| Gradient clipping and gradient-norm logging (from O17) | No observed problem to solve |

---

## 1. Decisions needed from you

⚠ marks a change that makes results differ from the published paper.

| ID | Decision | Recommendation |
|---|---|---|
| D1 | **Pretrained encoder** ⚠ | Add a ResNet or ConvNeXt encoder variant with your decoder and gates. This is likely the largest single gain. |
| D2 | **Several seeds** ⚠ | See below. Tune with one seed, then run `train` 3 times at each architecture's best learning rate and report mean ± standard deviation. |
| D3 | **Regularisation: weight decay and dropout (formerly O17)** ⚠ | AdamW, with batch-norm and bias parameters excluded; decay {0, 1e-2, 5e-2} × bottleneck dropout {0, 0.2} at the tuned learning rate, then re-check the learning rate one grid step either side. Dropout goes only on the bottleneck output (`Dropout2d`), where Ronneberger et al. placed it, and never inside a block before batch normalisation. |
| D4 | **Colour augmentation** ⚠ | Underwater colour cast, turbidity, noise and compression. The current jitter is ±5 %. |
| D5 | **Working scale and resolution (merges the former D4 crop scale, D5, D6, D17 and O42)** ⚠ | See below. |
| D7 | **Reporting the split** ⚠ | State that the split is image-level; also report the grouped-split score, after O29 is fixed. |
| D8 | **Loss weighting** ⚠ | Only if a class lags. |
| D11 | **Tests** | Unit tests for the metrics, class remapping and split, plus one synthetic CPU smoke test (split, train, export, test). |
| D14 | **Usage** | The command sequence, a few lines, in the README. |
| D15 | **Package named `src`** | Keep. |
| D18 | **Paper commit** | Tag `43aac81` (formerly `9789a9f`) as `paper-2026`. Reproduce the published table only if a reviewer or an erratum needs it. |
| D19 | **`deterministic` option (O15)** | Remove the option and always use `cudnn.benchmark = True`. See below. |

**D2: why several seeds.**
- A training run's score depends on chance: the random initial weights, the batch order and the augmentation draws. Two runs of the same model with the same settings, differing only in the seed, end at different scores.
- On a small dataset with a test subset of tens of images, that spread can be as large as the differences the ablation reports between architectures (Attention U-Net against U-Net, for example). With one run per architecture, an architecture effect cannot be told apart from a lucky seed.
- Picking the best of 8 learning rates from single runs also favours the luckiest run, so its validation score is optimistic.
- With 3 seeds, the table shows mean ± standard deviation, and a difference larger than the spread is real.
- **A cheap way to decide:** train the final Attention U-Net 3 times. If the spread is small next to the gaps between architectures, one seed per architecture is enough.
- **No code is needed:** change `seed` after running `split`. The split is read from its saved manifest, so only training changes.

**D5: working scale.**
- **What happens now.** Training squashes every photograph to 1024×1024, twice the model resolution, whatever its size and aspect ratio. Each epoch then takes one random square crop of side 512–1024 and resizes it to 512. Testing and the app squash the whole photograph to 512×512.
- **Cost of that.** A 12-megapixel photograph (4000×3000) keeps about 9 % of its pixels in training and 2 % at test. The fine texture that tells slime from calcareous deposits or barnacles is largely discarded, and the aspect ratio is distorted.
- **The alternative.** Resize each photograph so that its longer side is L, train on random 512×512 crops at that scale, and predict by tiling at the same scale. Tiles overlap by a margin and their logits are averaged. Upsample the result to the native resolution and score there.
  - At L = 2048, a 4:3 photograph holds about 12 distinct tiles, so each photograph yields many different crops across epochs.
  - The scale must be the same in training and inference. Training on fine crops while testing on squashed whole photographs would be worse than either alone.
  - Frames smaller than L (e.g. 1920×1080 video) are not upscaled.
- **Trade-off.** A larger L keeps more detail but gives each tile less context: a tile shows less of the hull around a patch of fouling. Inference costs grow with the number of tiles, which is still negligible on a GPU.
- **Recommendation.** Compare L = 1024 (about the current context, without the squashing) with L = 2048 on the validation subset, and keep the better. This settles crop scale, aspect ratio, native-resolution scoring and interpolation together, in Python and the app.

**D19: determinism.**
- `deterministic = true` makes only the convolutions deterministic. Bilinear upsampling, the squeeze-and-excitation pooling and the cross-entropy loss remain nondeterministic on the GPU. PyTorch's strict mode raises on all three, so the configuration comment's claim of exact reproducibility is false.
- Seeds already make the split, the initial weights, the batch order and the augmentations repeatable (for a fixed `numWorkers`, see O34). GPU kernels add noise far smaller than seed variance (D2).
- **Recommendation:** remove the option and its comment, and say in the README that GPU runs are repeatable up to that noise.
- If bitwise reproducibility is ever required, there are exact replacements, verified on the CPU: bilinear ×2 built from slices, `mean((2, 3))` for the pooling, and a one-hot cross-entropy. With them, `torch.use_deterministic_algorithms(True)` works.

---

## 2. Open issues

### Results and reported numbers

| ID | Issue | Fix |
|---|---|---|
| O29 | **Raised in priority.** Video frames are named `<video> frame<n>`, but similarity tags can only reference numeric IDs. If the dataset holds frames named that way, the grouped split does not keep frames of one video together: that is the leakage it exists to prevent, and it inflates the D7 score | In `groupBySimilarity`, also group IDs that share the part before ` frame` |
| O40 | ⚠ **Rewritten, measured.** See below | Two lines in `initializeModel`; make the change before the first training on `main` |
| O8 | ⚠ EXIF orientation. Labelbox displays photographs rotated by their EXIF tag, so masks are drawn in that orientation, while PIL (training) and now the app (to match training) ignore it. A rotated photograph is paired with a mask in another orientation | `ImageOps.exif_transpose` in `SegmentationDataset` and `ModelTester`; drop `ImreadModes.IgnoreOrientation` in the app (`GUI.cs`). Only the `.jpg` extension remains, as now |
| O43 | **New.** A mask that fails to download after three attempts makes `labels` skip the image with only a warning (`Labelbox.py:103-106`). A flaky network silently yields a smaller dataset, and so a different split | Raise instead. This also removes the skip code |
| O4 | A misspelt similarity ID is silently ignored, so near-duplicates end up in different subsets | One warning listing unknown IDs |
| O18 | No confidence intervals | See 4.2 |
| O19 | The best/worst ranking is distorted by stray-pixel classes | See 4.3 |
| O16 | The learning rate tuned for the Attention U-Net is reused for the ablation, and the default 4.1e-4 was tuned for the old gate | See 4.1 |
| O31 | The "Class Correlation Matrix" is a co-occurrence matrix | See 4.4 |
| O12 | No strong external baseline | DeepLabV3+, SegFormer or nnU-Net; optional, last |

**O40: what it means, why it happens, how to fix it.**
- **Cause.** `initializeWeights` (`modelFactory.py`) gives every convolution Kaiming-normal weights in fan-out mode with ReLU gain: std = √(2 / fan_out), where fan_out = output channels × kernel area. That suits the convolutions followed by batch normalisation and ReLU, which have many output channels. Two convolutions have few output channels and no ReLU after them:
  - ψ of each attention gate (1 output channel): std √2 ≈ 1.41.
  - The output layer (4 channels): std 0.71.
  - Both are 10–20 times PyTorch's default initialisation (about 0.07).
- **Effect on the gates** (measured at random initialisation, 256×256 input): 73–91 % of the coefficients of each gate are below 0.05 or above 0.95. Level means range from 0.06 to 0.84 depending on the seed, so some skip connections start almost closed. A saturated sigmoid passes its gradient scaled by c(1 − c) < 0.05, so those coefficients learn slowly.
  - This arrived with the published gate (`d6206f0`). The old gate normalised its logit with batch normalisation, which hid the scale.
- **Effect on the output.** The network starts confidently wrong: the initial cross-entropy is 2.4 (Attention U-Net) and 4.5 (U-Net), against ln 4 = 1.39 for an uninformed prediction. Early updates first undo that confidence.
- **Fix.** After `model.apply(initializeWeights)`, restore PyTorch's default initialisation for those layers with `reset_parameters()`: ψ in each gate, `UNet.output` and `SimpleCNN.fullyConnectedTwo`. Measured with that change: 0 % saturated, level means 0.42–0.65, and an initial cross-entropy of 1.40–1.44.
- Whether the final accuracy changes is unknown, because the gates may recover during training. The fix is two lines, removes a seed-dependent start, and costs nothing, since everything is retrained.

### Correctness of the code and its comments (no change to results)

| ID | Issue | Fix |
|---|---|---|
| O15 | The `deterministic` comment claims exact reproducibility | D19 |
| O34 | Augmentations depend on `numWorkers`: each worker has its own generator. The comment in `tuneModel.py` says the setting "changes the speed only" | Correct the comment, and keep `numWorkers` fixed for runs that are compared |
| O35 | A changed configuration makes `tune` raise instead of starting a new study, and a regenerated split goes unnoticed | Name the study `experimentName` plus a short hash of the study settings and the `splitHash`. Delete `checkStudyConfiguration` and its comparison (S7). New settings or a new split then start a new study by themselves |
| O22 | `monitorMode` can contradict the metric (e.g. `Loss` with `max`) | Derive the mode from the metric (`Loss` → `min`, everything else → `max`) and delete `monitorMode` and its check |
| O44 | **New.** The Dice loss comment ("smoothing yields a perfect score for absent classes") is false. Softmax probabilities are never exactly 0, so a class absent from a batch scores Dice ≈ 0. It adds a constant (1 − α)/C = 0.125 to the loss with almost no gradient. Training is hardly affected, but the validation loss depends on the batch composition | Correct the comment. Averaging the Dice term over the classes present in the batch would change results (⚠), so only do it with D8 |

### Windows application (remaining part of O20)

| ID | Issue | Fix |
|---|---|---|
| O20 | Masks are at model resolution, and there is no per-image summary | After D5: predict at the chosen scale, write masks at native resolution, and write `coverage.csv` (file and percentage of each class), the quantity a hull assessment uses |

---

## 3. Existing code to remove (no change to results)

Earlier passes added checks that re-raise what Python, NumPy, PyTorch or the dataclass constructors already raise, or that guard situations a single user does not meet.

| ID | Where | Change | Keep |
|---|---|---|---|
| S1 | `configuration.py:144-151` `buildSection` | Keep only the list → tuple conversion; the dataclass constructor already names unknown and missing keys | |
| S2 | `configuration.py:157-169`; `models/architectures.py` | Delete the ratio, architecture, resolution, monitor and mode checks, the file `architectures.py` and its import. Bad values crash at the first batch anyway, and `modelFactory` already rejects unknown architectures | Lines 155-156: `formGroups` would silently treat an unknown strategy as stratified |
| S3 | `annotationClasses.py:13-15, 20-22` | Delete; `.index` and NumPy raise anyway | Lines 17-18: a class absorbed twice would silently corrupt labels |
| S4 | `Labelbox.py`: `requireField` and its uses, 45-46, 80-86, 98-100, 111-112; the retry-and-skip path (with O43) | Plain indexing; raise on a failed download | 31-32 (missing API key), 121-123 (unannotated pixels) |
| S5 | `SegmentationDataset.py:18-20` | Delete the missing-file check | |
| S6 | `SubsetSplit.py:61-65, 104-110` | Delete the re-raise wrapper and the total and overlap checks, which are impossible by construction | The distribution table |
| S7 | `tuneModel.py:19-42, 103-104` | Replace with the hashed study name (O35); `best_trial` already raises | Recovery of interrupted trials |
| S8 | `commandLine.py:73-77` | Delete the "already exported" branch; exporting again takes seconds | |
| S9 | `trainModel.py:53-54` | Delete `peakMemoryGiB` (it was also wrong across trials) | |
| S10 | `ModelTester.py:34-35`; `experimentTracking.py:99-103` | Delete the missing-metadata check and the run-existence warning | `ModelTester.py:50-54`: the split-hash check protects the test numbers |
| S11 | `trainingInitialization.py`, `default.toml` | Remove `deterministic` (D19) | |

---

## 4. Implementation notes

### 4.1 O16: learning rate per architecture
1. Add `configs/architectures/uNet.toml`, `spatialAttentionUNet.toml` and `simpleCNN.toml`. Each sets `model.architecture` and its tuned `training.learningRate`; `default.toml` keeps the Attention U-Net. `tune --config <file>` already runs a separate study per architecture.
2. Use a denser grid where the optimum lies (the previous best, 4.1e-4, fell between 2e-4 and 5e-4): {5e-5, 1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 2e-3}. That is 32 runs for four architectures, fewer with pruning. If the best point lies at an edge, add a point; only new points are trained.
3. Then D2: 3 seeds at each architecture's best learning rate.

### 4.2 O18: confidence intervals
1. `ModelTester` already computes one confusion matrix per image. Save them in `testResults.json`.
2. Add one function in `src/evaluation/bootstrap.py`:
   - It resamples groups of similar images (similarity groups and video frames, O29), not single images, because frames of one video are correlated.
   - It draws 10,000 resamples of the summed matrices and reports the 2.5th and 97.5th percentiles of each metric.
   - It averages over the classes present in the full test ground truth, so a resample that lacks a class does not change what is averaged.
   - With two models, it resamples the same groups for both and reports the interval of each difference. That is what the ablation needs: two overlapping intervals do not show that two models are equal.
3. `test` prints each metric with its interval. The LaTeX row shows ± half the interval.
4. The intervals cover resampling of test images from the same vessels, not new vessels. Report seed spread (D2) separately.

### 4.3 O19: best/worst ranking
- **The flaw.** The per-image score averages Dice over the classes present in the prediction or the ground truth. A few stray pixels of a class absent from the ground truth score 0 for that class, so a near-perfect clean-hull image can score 0.5 and rank among the worst.
- **Fix.** Rank by the Dice averaged over the classes present in that image's ground truth. Stray pixels still count, as errors of the true class, in proportion to their area.
- The subset metrics keep their published definition.
- Caption the figure "Ranked by Dice over the ground-truth classes of each image".

### 4.4 O31: co-occurrence figure
- Cell (i, j) is the fraction of images containing class i that also contain class j. It is not symmetric, its diagonal is 1 by construction, and the diverging `coolwarm` colormap suggests a midpoint that does not exist.
- Retitle it "Class Co-occurrence: P(column class present | row class present)" and label the axes "Given class" and "Co-occurring class".
- Use the sequential `Blues` colormap, as in the confusion matrix, and show each class's image count on the diagonal.
- Check the paper's caption and text against this definition; if they call it a correlation, an erratum may be needed.

---

## 5. Actions for you

| ID | Action |
|---|---|
| A1 | Pull the branch and run `uv sync`. |
| A2 | Tag the paper commit (D18): `43aac81`, once `main` carries the rewritten history (A7). |
| A3 | Move the data to the new layout: images to `data/images/`, masks to `data/masks/`, `Metadata.json` to `data/metadata.json`, `Labels.ndjson` to `data/labels.ndjson`; then `uv run biofouling split`. |
| A4 | First GPU run: NVIDIA driver 580 or newer. |
| A5 | Decide D2, D3, D5, D18 and D19. |
| A6 | Start the retraining with an empty `outputs/`. |
| A7 | Approve replacing `main` with the re-authored history (a force-push; the file contents are identical). |

---

## 6. Plan

Changes to results are made one at a time, against a recorded baseline.

| Phase | Items | Changes results |
|---|---|---|
| 0. Clean-up | S1–S11, O35, O22, O34, O44 (comment), O43, O4 | No |
| 1. Tests | D11 | No |
| 2. Data | A3, O29, O8 ⚠, O31, then `split` | Labels and split |
| 3. Baseline | O40 ⚠, A4, A6, O16 with one seed per architecture, D2 seeds at the best rate, O18, O19, D7 | New baseline table |
| 4. Improvements | D5 scale, D4 colour, D3 regularisation, D8 if needed, D1 pretrained encoder, O12 | ⚠ each one |
| 5. Application | O20 remainder, with D5 and O8 | No |

**Settled:** package `src`, command `biofouling`, configuration in `configs/`, camelCase; 16 direct dependencies, 135 packages in `uv.lock`; attention gate of Oktay et al. (2018); squeeze-and-excitation channel attention; export on the CPU; Windows app on .NET 10 with ONNX Runtime 1.32 and OpenCvSharp 4.13.
