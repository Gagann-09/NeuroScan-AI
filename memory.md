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

### D-022 --- Frozen EvaluationProtocol for ARMT-GAN scientific evaluation

**Date:** 2026-10-05

**Context:** Evaluation protocol parameters (split fractions, threshold, min_tumor_pixels, device) were scattered across CLI args and constants without a single frozen configuration.

**Decision:** Create immutable `EvaluationProtocol` dataclass with all protocol parameters. Three-way split (train/val/test) with default 0.70/0.15/0.15. Binary threshold fixed at 0.5. Protocol serialized in JSON experiment records.

**Reason:** Scientific evaluation requires a fully documented, versioned protocol. Evaluation Design (design.md #4): "Metrics must be computed on patient-disjoint evaluation data."

**Affected files:**
- `backend/ai_pipeline/evaluation/evaluate_armt_gan_baseline.py` (EvaluationProtocol, split_patients_three_way)
- `backend/tests/test_evaluation_protocol.py` (new)

**Status:** Complete.

**Verification:** 
- 26 evaluation protocol tests pass (metrics, split, protocol)
- 9 preprocessing parity tests pass
- Two evaluation runs produce identical metrics (Dice=0.980835, IoU=0.962391)
- Raw prediction artifacts saved as .npy files (12 files for test patient)

### D-023 --- Three-way patient-disjoint split

**Date:** 2026-10-05

**Context:** Previous evaluation used only train/validation split. Scientific evaluation requires a held-out test set that is never seen during training or validation.

**Decision:** Extend split logic to three-way (train/validation/test) with deterministic seed-based shuffling. Split sorts patient IDs before shuffling so filesystem order cannot affect split. Validates no patient leakage across any pair of splits.

**Reason:** Data Rules (rules.md #6): "Patient-level splitting is mandatory. No patient may appear in multiple evaluation partitions." PRD FR-03.

**Affected files:**
- `backend/ai_pipeline/evaluation/evaluate_armt_gan_baseline.py` (split_patients_three_way)

**Status:** Complete.

**Verification:** 
- Unit tests verify no leakage (train∩val=∅, train∩test=∅, val∩test=∅)
- Deterministic: same seed → identical splits
- All patients allocated exactly once

### D-024 --- Raw prediction artifact preservation

**Date:** 2026-10-05

**Context:** Evaluation computed metrics but did not save raw probability masks, preventing audit and re-analysis.

**Decision:** Evaluation saves raw probability masks as .npy files per slice in `output_dir/predictions/`. Each artifact includes metadata (patient_id, slice_index, shape, min/max, per-slice metrics). Artifact list serialized in JSON experiment record.

**Reason:** Storage Rules (rules.md #16): "Every artifact belongs to a study. Every derived artifact references its source study. Never overwrite artifacts without versioning."

**Affected files:**
- `backend/ai_pipeline/evaluation/evaluate_armt_gan_baseline.py` (evaluate() output_dir parameter, raw_artifacts)

**Status:** Complete.

**Verification:** 12 .npy files saved for test patient (BraTS20_Training_000), each with shape [224, 224], dtype float32, containing probability values in [0, 1].

### D-025 --- Frozen AttackConfig for FGSM/PGD robustness evaluation

**Date:** 2026-10-05

**Context:** Adversarial attack parameters (epsilon, step size, iterations, norm, clipping) were not centrally documented.

**Decision:** Create immutable `AttackConfig` dataclass with all attack parameters. Default epsilon=0.03 (L-inf), step_size=0.0075, 10 iterations, random start with fixed seed offset. Clipping bounds [-3.0, 3.0] for Z-score normalized input space.

**Reason:** Robustness Rules (rules.md #9): "Attack parameters must be explicit." Scientific evaluation requires documented attack budgets.

**Affected files:**
- `backend/ai_pipeline/robustness/attacks.py` (AttackConfig, BASELINE_ATTACK_CONFIG)
- `backend/ai_pipeline/robustness/config.py` (RobustnessConfig)
- `backend/tests/test_robustness_attacks.py` (new)

**Status:** Complete.

**Verification:** 23 attack tests pass (FGSM/PGD L∞ bound, projection, deterministic behavior, gradient existence, no weight modification).

### D-026 --- FGSM and PGD attack implementations for segmentation

**Date:** 2026-10-05

**Context:** No adversarial attack implementations existed for the ARMT-GAN segmentation model.

**Decision:** Implement FGSM (single-step) and PGD (iterative) attacks using soft Dice loss (1 - soft Dice) as differentiable attack objective. Attacks operate on model output probabilities (continuous), target ground-truth segmentation mask. Perturbation applied in Z-score normalized input space with single L∞ budget across all 4 modalities. Clean input clamped to [-3, 3] before attack to prevent clamping artifacts.

**Reason:** PRD FR-07 requires "FGSM and PGD use documented budgets." Design.md #5 specifies separate evaluation experiments.

**Affected files:**
- `backend/ai_pipeline/robustness/attacks.py` (new: soft_dice_loss, fgsm_attack, pgd_attack)
- `backend/ai_pipeline/robustness/evaluate_robustness.py` (new: run_robustness_evaluation)
- `backend/ai_pipeline/robustness/__init__.py` (exports)
- `backend/ai_pipeline/robustness/config.py` (AttackConfig, RobustnessConfig)

**Status:** Complete.

**Verification:** 23 attack unit tests pass. Clean/FGSM/PGD evaluation pipeline verified on synthetic data. Attack constraints verified (L∞ bound, clipping, determinism, no weight modification).

### D-027 --- Robustness evaluation pipeline integration with P3

**Date:** 2026-10-05

**Context:** P3 established clean evaluation protocol. P4 extends it with adversarial evaluation.

**Decision:** Create `run_robustness_evaluation` that reuses P3 evaluation machinery (metrics, patient splitting, artifact preservation). Runs clean, FGSM, and PGD on same test split. Produces `RobustnessResult` with clean/adversarial metrics and deltas. Artifacts saved per attack type (clean, FGSM, PGD). Outputs JSON experiment record.

**Reason:** P3 gate requires "Metrics are reproducible from stored predictions." P4 extends this to adversarial metrics.

**Affected files:**
- `backend/ai_pipeline/robustness/evaluate_robustness.py` (run_robustness_evaluation, print_robustness_result, save_robustness_result)
- `backend/ai_pipeline/robustness/__init__.py` (exports)

**Status:** Complete.

**Verification:** 58 total backend tests pass (9 preprocessing + 26 evaluation + 23 robustness). Robustness evaluation runs on synthetic data producing clean/FGSM/PGD metrics with deltas.

### D-028 --- Gradient-based input saliency as P5 XAI method

**Date:** 2026-10-06

**Context:** XAI method was named `generate_gradcam` but implemented input-gradient saliency. P5 requires method name to match implementation and provenance tracking.

**Decision:** Keep gradient-based input saliency (sensitivity analysis) as the P5 method. Rename function to `generate_gradient_saliency`, add `XAIProvenance` dataclass, remove semantic fallback, add raw attribution artifact preservation, add segmentation alignment metric.

**Reason:** PRD FR-08 requires "Attribution method is explicitly named." Design.md #6: "The current implementation is gradient-based input saliency. It must not be labelled Grad-CAM." Rules.md #13: "Never label input-gradient saliency as Grad-CAM. Store attribution provenance."

**Affected files:**
- `backend/app/services/xai.py` (rewritten: generate_gradient_saliency, XAIProvenance, compute_segmentation_alignment)
- `backend/app/services/ai_tasks.py` (updated to use new XAI function, provenance, alignment, raw artifact)
- `backend/tests/test_xai.py` (new: 28 tests)

**Status:** Complete.

**Verification:** 
- All 28 XAI tests pass (output shape, normalization range, flat-gradient behavior, deterministic attribution, gradient availability, no model-weight modification, provenance fields, alignment metric, empty-mask behavior)
- All 9 P1 preprocessing parity tests pass
- All 26 P3 evaluation protocol tests pass
- All 23 P4 robustness attack tests pass
- 86 total backend tests pass

### D-029 --- Remove hard-coded credential defaults from application configuration

**Date:** 2026-10-06

**Context:** Application configuration (`backend/app/core/config.py`) contained hard-coded default credentials for DATABASE_URL, MINIO_ACCESS_KEY, and MINIO_SECRET_KEY. These were development defaults but committed in plain text, violating security rules (rules.md #14) and memory.md D-012.

**Decision:** Remove all hard-coded credential defaults from `Settings` class. Make `DATABASE_URL`, `MINIO_ACCESS_KEY`, and `MINIO_SECRET_KEY` required fields with no default. Add validation that fails clearly when required credentials are missing. Preserve non-sensitive defaults (endpoints, timeouts, app metadata). Update Alembic to read DATABASE_URL from application settings. Docker Compose retains development credentials for local workflow.

**Reason:** Security Rules (rules.md #14): "No hard-coded production credentials. No default administrator passwords in committed code." Memory.md D-012: "Credentials must come from environment or secret management." PRD NFR: "Free of hard-coded credentials."

**Affected files:**
- `backend/app/core/config.py` (removed credential defaults, added validators, clear error messages)
- `backend/alembic.ini` (removed default sqlalchemy.url, now reads from app config)
- `backend/alembic/env.py` (updated to load DATABASE_URL from application Settings)
- `backend/tests/test_config_security.py` (new: 10 tests for configuration security)

**Status:** Complete.

**Verification:** 
- All 10 new configuration security tests pass
- All 96 backend tests pass (86 original + 10 new)
- Credential defaults `secure_password_123` and `minioadmin` no longer present in Python application code
- Development workflow preserved via Docker Compose environment variables
- `.env` file support retained for local development

### D-030 --- Establish persistence and provenance schema

**Date:** 2026-10-06

**Context:** The database schema was missing tables and columns required by design.md §9 for persistent provenance: `ModelVersion`, `Artifact`, `Prediction` model-version linkage, Dice/IoU fields, proper FK from `Prediction` to `Scan`, and raw XAI saliency path persistence.

**Decision:** Add `ModelVersion` and `Artifact` tables per design.md §9 ER diagram. Extend `Prediction` with `model_version_id` FK, `dice`, `iou`, `created_at` columns. Add proper FK from `Prediction.scan_id` to `Scan.id`. Add `xai_raw_path` to `Scan` for raw saliency artifact provenance. All changes via Alembic migration `002`.

**Reason:** design.md §9 requires `MODEL_VERSION`, `ARTIFACT`, `PREDICTION` with `model_version`, `dice`, `iou`. rules.md #16: "Every artifact belongs to a study. Every derived artifact references its source study." PRD FR-10: "Storage: source and derived artifacts have traceable IDs." rules.md #17: "Schema evolution must be migration-based."

**Affected files:**
- `backend/app/db/models.py` (added ModelVersion, Artifact; extended Prediction, Scan)
- `backend/alembic/versions/002_add_model_version_artifact_tables.py` (new migration)
- `backend/tests/test_db_models.py` (new: 26 model tests)

**Status:** Complete.

**Verification:** 
- All 26 new database model tests pass
- All 122 backend tests pass (96 previous + 26 new)
- Migration recognized by Alembic (001 -> 002)
- Schema matches design.md §9 ER diagram
- No breaking changes to existing data (nullable columns for backward compat)

### D-031 --- Wire application services into persistent provenance

**Date:** 2026-10-06

**Context:** Batch 3A established the persistence schema (ModelVersion, Artifact, Prediction.model_version_id, dice, iou, Scan.xai_raw_path). Batch 3B wires the application services to populate these fields during inference.

**Decision:** Wire the inference pipeline (ai_tasks.py) to:
- Create/reuse ModelVersion with deterministic SHA256 hash of checkpoint
- Link Prediction to ModelVersion via model_version_id FK
- Persist Scan.xai_raw_path with permanent MinIO object path
- Create Artifact records (mask, xai, xai_raw, report) linked to Prediction
- Set Prediction.dice/iou to None (populated by evaluation pipeline)
- Use permanent MinIO object paths for all provenance (no presigned URLs)
- Reuse existing XAIProvenance from P5, store config hash in ModelVersion

**Reason:** design.md §9 requires PREDICTION → MODEL_VERSION linkage and ARTIFACT records. rules.md #16: "Presigned URLs are delivery mechanisms. They are not permanent provenance identifiers." PRD FR-10: "Storage: source and derived artifacts have traceable IDs."

**Affected files:**
- `backend/app/services/ai_tasks.py` (ModelVersion get-or-create, Artifact creation, xai_raw_path, model_version_id)
- `backend/tests/test_config_security.py` (fixed .env file precedence test)
- `backend/tests/test_provenance_wiring.py` (new: 11 provenance wiring tests)

**Status:** Complete.

**Verification:** 
- All 11 new provenance wiring tests pass
- All 133 backend tests pass (122 original + 11 new)
- ModelVersion reuse verified (same checkpoint → same version_id)
- Artifact records use permanent object paths (not presigned URLs)
- Prediction.model_version_id FK populated correctly
- Scan.xai_raw_path populated with permanent MinIO path

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