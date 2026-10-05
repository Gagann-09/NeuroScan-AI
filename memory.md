# NeuroScan AI --- Decision Memory

**Purpose:** Durable decision ledger for NeuroScan AI only.

This file records decisions, not general explanations.

## 1. Baseline Identity

-   Project: NeuroScan AI.
-   Core model: ARMT-GAN.
-   Repository: `Gagann-09/NeuroScan-AI`.
-   Baseline commit: `a0bbe53dd63834cea7349cd390c8ede8b915f52d`.
-   Baseline date: 2026-10-05.
-   Scope: research prototype.
-   Clinical status: not clinically validated.

## 2. Baseline Architecture Decisions

### D-001 --- ARMT-GAN remains canonical

**Decision:** Preserve ARMT-GAN as the central model.

**Reason:** It is the project's intended research contribution.

**Status:** Locked.

### D-002 --- Baseline is 2D

**Decision:** Use 2D axial segmentation for the validated baseline.

**Reason:** Current implementation is 2D and reproducible.

**Status:** Locked.

### D-003 --- Four MRI channels

**Decision:** Use T1, T1ce, T2, and FLAIR.

**Reason:** The current training contract already uses four modalities.

**Status:** Locked.

### D-004 --- Patient-level splitting

**Decision:** Split patients before slice extraction.

**Reason:** Slice-level splitting risks leakage.

**Status:** Locked.

### D-005 --- Real BraTS masks

**Decision:** Real tumor labels are mandatory.

**Reason:** Synthetic masks invalidate scientific evaluation.

**Status:** Locked.

## 3. Current Audit Decisions

### D-006 --- Correct API modality contract

**Finding:** Frontend sends four modalities.

**Finding:** Backend currently accepts one file.

**Decision:** Future API contract uses four modality fields.

**Status:** Required correction.

### D-007 --- Correct inference channel contract

**Finding:** Training expects four channels.

**Finding:** Current inference constructs three RGB channels.

**Decision:** Inference must construct the same four-channel tensor.

**Status:** Required correction.

### D-008 --- Remove confidence manipulation

**Finding:** Current code adds `0.40` below 50%.

**Decision:** Remove heuristic confidence modification.

**Status:** Required correction.

### D-009 --- Remove heuristic WHO grading

**Finding:** Current code maps area thresholds to WHO grades.

**Decision:** Do not present WHO grade from segmentation area.

**Status:** Required correction.

### D-010 --- Rename current XAI method

**Finding:** Current function is named `generate_gradcam`.

**Finding:** Implementation computes input gradients.

**Decision:** Describe it as gradient-based saliency until true Grad-CAM
exists.

**Status:** Required correction.

### D-011 --- Remove destructive startup reset

**Finding:** Application startup calls `drop_all()`.

**Decision:** Startup must never delete database tables.

**Status:** Required correction.

### D-012 --- Eliminate hard-coded credentials

**Finding:** Repository contains default credential values.

**Decision:** Credentials must come from environment or secret
management.

**Status:** Required correction.

### D-015 --- Shared preprocessing module for training/inference parity

**Date:** 2026-10-05

**Context:** Training used inline preprocessing; inference used completely different preprocessing (RGB conversion, 256x256 resize, single file).

**Decision:** Create `ai_pipeline/preprocessing/brats_preprocessing.py` as the single source of truth for BraTS preprocessing. Both training and inference must use this module.

**Reason:** Scientific reproducibility requires identical preprocessing for training and inference. The previous implementation had training expecting 4-channel [4, 224, 224] tensors but inference producing 3-channel [3, 256, 256] tensors.

**Affected files:**
- `backend/ai_pipeline/preprocessing/__init__.py` (new)
- `backend/ai_pipeline/preprocessing/brats_preprocessing.py` (new)
- `backend/ai_pipeline/training/train_armt_gan.py` (updated)
- `backend/ai_pipeline/evaluation/evaluate_armt_gan_baseline.py` (updated)
- `backend/app/services/ai_tasks.py` (rewritten)
- `backend/tests/test_preprocessing_parity.py` (new)

**Status:** Complete.

**Verification:** All 9 preprocessing parity tests pass, including critical `test_training_vs_inference_preprocessing`.

### D-016 --- Four-modality API contract

**Date:** 2026-10-05

**Context:** Frontend sends 4 modalities (t1, t1ce, t2, flair); backend accepted single file.

**Decision:** Update API to accept 4 explicit modality files. Add `ModalityFile` database model. Update `UploadRequest` schema.

**Reason:** Contract integrity (PRD FR-01, FR-09) requires explicit four-modality representation end-to-end.

**Affected files:**
- `backend/app/schemas/scan_schema.py` (added UploadRequest)
- `backend/app/api/routers/scans.py` (4-file upload endpoint)
- `backend/app/db/models.py` (added ModalityFile)
- `frontend/src/components/upload-pill.tsx` (4-modality UI)

**Status:** Complete.

### D-017 --- Removed heuristic confidence boost

**Date:** 2026-10-05

**Context:** Inference code added +0.40 to confidence scores below 0.50.

**Decision:** Remove heuristic confidence modification. Report raw model output probability.

**Reason:** Confidence Rules (rules.md #11): "No heuristic confidence boosting. No arbitrary score offsets."

**Affected files:**
- `backend/app/services/ai_tasks.py` (removed confidence boost)

**Status:** Complete.

### D-018 --- Removed heuristic WHO grading

**Date:** 2026-10-05

**Context:** Inference code mapped tumor area to WHO grades.

**Decision:** Remove WHO grade prediction. Model does not predict WHO grade.

**Reason:** Clinical Language Rules (rules.md #12): "WHO grade prediction" forbidden without validation. Product explicitly out of scope for WHO grading.

**Affected files:**
- `backend/app/services/ai_tasks.py` (removed WHO grade logic)

**Status:** Complete.

### D-019 --- Renamed XAI method to gradient-based saliency

**Date:** 2026-10-05

**Context:** Function named `generate_gradcam` but implemented input-gradient saliency.

**Decision:** Rename to `generate_gradient_saliency`. Keep deprecated alias with warning.

**Reason:** XAI Rules (rules.md #13): "Never label input-gradient saliency as Grad-CAM."

**Affected files:**
- `backend/app/services/xai.py` (renamed, added deprecation warning)
- `backend/app/services/ai_tasks.py` (updated import)

**Status:** Complete.

### D-020 --- Frozen TrainingConfig for ARMT-GAN reproducibility

**Date:** 2026-10-05

**Context:** Training hyperparameters were passed as function arguments with defaults but not centrally documented as a frozen configuration.

**Decision:** Create immutable `TrainingConfig` dataclass in `train_armt_gan.py` with all hyperparameters explicitly documented. Baseline config (`BASELINE_CONFIG`) serves as the single source of truth.

**Reason:** Training Rules (rules.md #8): "Every training run records: seed, dataset version, split, image size, batch size, learning rate, epochs, loss configuration, checkpoint version." A frozen config ensures this.

**Affected files:**
- `backend/ai_pipeline/training/train_armt_gan.py` (added TrainingConfig, BASELINE_CONFIG)
- `backend/ai_pipeline/evaluation/evaluate_armt_gan_baseline.py` (added EvaluationConfig, BASELINE_EVAL_CONFIG)

**Status:** Complete.

**Verification:** 
- Model initialization determinism verified
- P1 preprocessing parity tests: 9/9 pass
- Synthetic data smoke training: completes successfully
- Checkpoint save/load: verified
- Evaluation roundtrip: verified
- Two-run reproducibility: IDENTICAL weights and metrics (Dice=0.980397)

### D-021 --- Explicit loss configuration

**Date:** 2026-10-05

**Context:** Loss weights (L1=100, adversarial=1, D_real=0.5, D_fake=0.5) were hardcoded in training loop.

**Decision:** Move all loss weights into `TrainingConfig` as explicit fields.

**Reason:** Reproducibility requires all hyperparameters to be recorded and versioned.

**Affected files:**
- `backend/ai_pipeline/training/train_armt_gan.py` (loss weights in TrainingConfig)

**Status:** Complete.

## 4. Research Direction

### D-013 --- FGSM and PGD are future robustness phases

**Decision:** Evaluate adversarial robustness separately.

**Reason:** Robustness must be measured without corrupting baseline
comparisons.

**Status:** Locked.

### D-014 --- GAN component is not automatically augmentation

**Decision:** ARMT-GAN's discriminator loss is not described as
synthetic-data augmentation.

**Reason:** Current implementation trains a conditional segmentation
GAN.

**Status:** Locked.

## 5. Decision Format

Every future decision must record:

``` text
ID
Date
Context
Decision
Reason
Affected files
Status
Verification
```

## 6. Prohibited Memory

Do not store:

-   Secrets.
-   Passwords.
-   API keys.
-   Patient-identifying medical data.
-   Unverified metrics.
-   Temporary implementation guesses.

## 7. Update Rule

Update this file only for durable project decisions.

Do not turn it into a task checklist.
