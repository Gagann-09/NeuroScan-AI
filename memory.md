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

### D-032 --- Remove unsupported clinical claims from backend (P6 Batch 5A)

**Date:** 2026-10-06

**Context:** The P6 audit identified several unsupported clinical claims in the backend:
- `who_grade` field in Prediction model with heuristic area-threshold logic (previously removed in D-018 but field remained in DB)
- `anomaly_area_cm2` field computed using uncalibrated heuristic `tumor_area_px * 0.11`
- `confidence_score` field - renamed to `max_tumor_probability` to accurately reflect raw sigmoid output
- Report title "Clinical Diagnosis Report" - forbidden by rules.md #12
- Hardcoded "ARMT-GAN v1.0" model version in reports - not traceable to ModelVersion

**Decision:** 
1. Remove `who_grade`, `anomaly_area_cm2`, `confidence_score` columns from Prediction table via Alembic migration 003
2. Add `max_tumor_probability` column to store raw model output probability
3. Rename report function to `generate_segmentation_report` with research-prototype title
4. Add explicit research disclaimer to report
5. Use actual ModelVersion.id from provenance in report instead of hardcoded string
6. Update API schema to return `max_tumor_probability` and `model_version` instead of removed fields
7. Update all tests to verify unsupported fields are removed

**Reason:** 
- rules.md #12: "WHO grade prediction" forbidden without validation; "Clinical Diagnosis Report" forbidden
- PRD §2: Clinical diagnosis, automated WHO grading explicitly out of scope
- design.md §11: Reports must not call predictions diagnoses; must not assign WHO grade
- rules.md #11: No fabricated confidence - raw probability is scientifically accurate
- rules.md #16: Every artifact references its source study - ModelVersion provenance must be used

**Affected files:**
- `backend/app/db/models.py` (removed fields, added max_tumor_probability)
- `backend/app/services/ai_tasks.py` (removed computations, updated report call)
- `backend/app/api/routers/scans.py` (updated response)
- `backend/app/schemas/scan_schema.py` (updated ResultsResponse)
- `backend/app/services/reporting.py` (renamed function, updated title, added disclaimer, use ModelVersion)
- `backend/alembic/versions/003_remove_unsupported_clinical_fields.py` (new migration)
- `backend/tests/test_db_models.py` (verify removed fields)
- `backend/tests/test_provenance_wiring.py` (verify new fields, removed fields)

**Status:** Complete.

**Verification:** 
- All 135 backend tests pass (133 original + 2 new)
- No references to unsupported fields remain in backend app code
- Report title is "Research Prototype Segmentation Report"
- Report uses actual ModelVersion.id from provenance
- Migration 003 created for schema changes

### D-033 --- Separate preprocessing provenance from model configuration (P6 Batch 5B)

**Date:** 2026-10-06

**Context:** The P6 audit identified that preprocessing provenance was conflated with model configuration in `ModelVersion.config_hash`. The string `"preprocessing:v1|image_size:224|modality_order:t1,t1ce,t2,flair|normalize:nonzero_zscore"` mixed two independent provenance concepts:
- Model identity/configuration (what model/checkpoint produced the result)
- Preprocessing identity (what preprocessing pipeline transformed the input)

These must be independently auditable per PRD FR-04 ("preprocessing configuration are recorded") and design.md §8 ("Results must include: Preprocessing version").

**Decision:** 
1. Add `preprocessing_version` column to `ModelVersion` table via Alembic migration 004
2. Compute deterministic preprocessing version from `PreprocessingConfig` using SHA256 of canonical config representation
3. Separate `config_hash` to only contain model configuration (architecture, generator, discriminator, loss)
4. Update `_get_or_create_model_version` to accept and persist `preprocessing_version`
5. Use `DEFAULT_PREPROCESSING_CONFIG` and `PREPROCESSING_VERSION` constants for the inference pipeline

**Reason:** 
- PRD FR-04: "Seed and preprocessing configuration are recorded"
- design.md §8: "Results must include: Model version... Preprocessing version"
- rules.md #16: "Every artifact belongs to a study. Every derived artifact references its source study."
- The preprocessing pipeline is a separate versioned component from the model checkpoint

**Affected files:**
- `backend/app/db/models.py` (added preprocessing_version column)
- `backend/app/services/ai_tasks.py` (added _compute_preprocessing_version, PREPROCESSING_VERSION, updated config hash, updated ModelVersion creation)
- `backend/alembic/versions/004_add_preprocessing_version_to_model_version.py` (new migration)
- `backend/tests/test_db_models.py` (verify preprocessing_version column)
- `backend/tests/test_provenance_wiring.py` (verify new field, old conflated string removed)

**Status:** Complete.

**Verification:** 
- All 137 backend tests pass (135 original + 2 new)
- `preprocessing_version` column exists in ModelVersion model
- `config_hash` no longer contains preprocessing metadata
- Preprocessing version is deterministic (same config → same hash)
- Migration 004 created for schema changes

### D-034 --- P6 Batch 5C: Test stability fixes for config isolation and provenance test path

**Date:** 2026-10-07

**Context:** Five backend tests were failing:
1. Four config security tests failed because `database.py` and `storage.py` instantiated `get_settings()` at module import time, preventing `monkeypatch` from setting environment variables before validation.
2. One provenance wiring test failed because it used a hardcoded path `backend/app/services/ai_tasks.py` relative to the working directory, but tests run from the repository root.

**Decision:**
1. Make database engine and session factory initialization lazy in `database.py` using module-level globals with accessor functions (`_get_engine`, `_get_session_local`). Keep `SessionLocal` as a callable proxy for backward compatibility.
2. Make MinIO client initialization lazy in `storage.py` using a module-level global with accessor function (`_get_minio_client`). Keep `minio_client` as a proxy object for backward compatibility.
3. Fix provenance test to locate `ai_tasks.py` relative to the test file using `Path(__file__).parent.parent / "app" / "services" / "ai_tasks.py"`.

**Reason:**
- Rules.md #18: "A failing test blocks the phase."
- Tests must be able to monkeypatch environment variables before Settings instantiation.
- Test paths must be resilient to working directory changes.
- No production behavior changes; credential validation unchanged; no defaults introduced.

**Affected files:**
- `backend/app/db/database.py` (lazy engine/SessionLocal)
- `backend/app/core/storage.py` (lazy minio_client)
- `backend/tests/test_provenance_wiring.py` (fixed AI_TASKS_PATH)

**Status:** Complete.

**Verification:**
- All 137 backend tests pass (10/10 config security, 1/1 provenance path, 126 other tests).
- No new failures.
- No reduction in coverage or behavior.

### D-035 --- Integrate claim_scan_for_processing into processing entrypoint (P6 Batch 5D-3)

**Date:** 2026-10-09

**Context:** The `claim_scan_for_processing()` function existed with comprehensive tests but was not wired into the actual processing flow. The upload endpoint created Scan records with status `PROCESSING` immediately, bypassing the atomic claim mechanism that prevents duplicate processing.

**Decision:**
1. Change upload endpoint to create Scan records with status `PENDING` instead of `PROCESSING`.
2. Call `claim_scan_for_processing(db, scan_id)` at the start of `process_scan_task()`, before any expensive work (downloads, preprocessing, inference).
3. If claim fails (ALREADY_PROCESSING, ALREADY_COMPLETE, NOT_FOUND), log the reason and return early without processing.
4. If claim succeeds, continue with existing processing flow.
5. Preserve existing error handling: on processing failure, rollback and set scan status to `FAILED` (enabling retry via FAILED→PROCESSING claim).

**Reason:**
- Prevents duplicate processing via PostgreSQL row-level locking (SELECT FOR UPDATE NOWAIT).
- Enables safe retry of failed scans (FAILED→PROCESSING transition).
- Sets `processing_started_at` timestamp for observability.
- Minimal change: 2 files, ~5 lines changed; no schema, API contract, or frontend changes.

**Affected files:**
- `backend/app/api/routers/scans.py` (scan initial status: PROCESSING → PENDING)
- `backend/app/services/ai_tasks.py` (claim call at start of process_scan_task)
- `backend/tests/test_scan_claim_integration.py` (new: 5 integration tests with mocks)

**Status:** Complete.

**Verification:**
- test_scan_claim.py: 7/7 passed (scan claim/locking mechanism).
- test_artifact_lifecycle.py: 9/9 passed (artifact state transitions).
- test_scan_claim_integration.py: 5/5 passed (claim integration with process_scan_task).
- test_provenance_wiring.py: 14/14 passed (model definitions, provenance constants, artifact types).
- test_db_models.py: 27/27 passed (database schema and relationships).
- Production `model_version` ordering defect fixed in commit `3cb451f` (UnboundLocalError resolved by creating model_version before first use).
- All 21 database-dependent tests pass against live PostgreSQL.
- All 41 non-database provenance tests pass.
- No changes to inference, preprocessing, XAI, reports, database schema, or API contracts.

### D-036 --- SG3-F3-G: NIfTI validation verification, regression tests, and restoration of accidentally deleted worker helpers

**Date:** 2026-10-15

**Context:** Verification batch for the SG3-F3 series (MRI upload validation, NIfTI integrity validation, resource bounds, worker-side safety). The previous implementation attempt timed out, leaving five modified files uncommitted. Investigation found that HEAD commit `9aa22c7` (SG3-F3-E, "NIfTI Processing Safety Fix") had accidentally deleted ~434 lines of helper functions from `backend/app/services/ai_tasks.py`. As a result `process_scan_task` referenced undefined names (`claim_scan_for_processing`, `transition_artifact_state`, `create_reference_image`, `generate_clinical_overlays`, `_get_global_model`, `_get_weights_path`, `_get_or_create_model_version`, `_cleanup_derived_objects`, ...) and three test modules (`test_scan_claim.py`, `test_scan_claim_integration.py`, `test_artifact_lifecycle.py`) failed at collection.

**Decision:**
1. Restore the deleted helper functions verbatim from `HEAD~1` (the last known-good revision), preserving the working-tree NIfTI-validation changes. No ARMT-GAN architecture, weights, modality order, labels, normalization, resizing, or evaluation behavior was changed.
2. Keep the working-tree fix that changed `SUPPORTED_DTYPES` from scalar types (`{np.float32, ...}`) to `np.dtype` instances. The old set never matched `image.get_data_dtype()` because `np.dtype(np.float32) == np.float32` is `True` but their hashes differ, so set membership returned `False` and every valid file was rejected.
3. Keep the consolidation of NIfTI validation into the module-level `_validate_nifti_image`, reused by the training, evaluation, volumetric-3D, and worker paths.
4. Add `backend/tests/test_nifti_safety_regression.py` (Tiny synthetic fixtures and mocks only; 22 tests).
5. Fix genuinely-stale test setup in `backend/tests/test_auth.py` by supplying the now-required `CORS_ORIGINS` (SG3-F1). `TestConstants` was NOT stale: it failed only because of the deleted helpers and passes again after restoration.
6. Align the NIfTI-2 contract with the project's actual requirement: NIfTI-2 is not required (BraTS is NIfTI-1). Document `_validate_nifti_image` as NIfTI-1-only (docstring and error message) and replace the strict `xfail` with a test asserting that a valid NIfTI-2 image is rejected.
7. Record two facts found during the final diff audit: (a) the working tree drops a shadowed duplicate `class BraTSPreprocessor` that `HEAD~1` (SG3-F3-C) introduced and never fixed -- the first definition was fully shadowed by the second, so removal is behavior-preserving; (b) `ai_tasks.py` builds the derived-artifact paths from the outer `tmpdir` while `source_img_path` uses the inner `tmpdirname`.

**Reason:** rules.md #18 ("A failing test blocks the phase") and #20; PRD G7 ("fail closed when required inputs are invalid"); design.md §7 (validate modality completeness/file types before inference). The restoration recovers committed work that was lost by accident; it is not a new feature.

**Verified protections:**
- Header validation rejects unsupported datatypes, malformed dimensions (fewer than 3 dims; dims over the volume cap), invalid magic, and non-NIfTI objects.
- Validation uses header metadata only: `get_fdata()` is never called by validation ("validate before full allocation").
- Training and evaluation segmentation are validated before the payload is materialised.
- Invalid and truncated reference modalities are rejected before `_get_global_model()` (inference) is invoked.
- A valid NIfTI-1 file still loads correctly through the training and evaluation helpers.

**Known gaps (unresolved, deliberately not changed in this batch):**
- Header-only validation does not detect truncated payloads; truncation surfaces only when the data is read (`.get_fdata()` raises). There is no true decompressed-byte limit, only header-based size estimation.
- The module docstring lists supported dtypes without `float64`, which `SUPPORTED_DTYPES` includes.
- Reference modalities are loaded again (and re-validated) after `preprocess_brats_study` already loaded them: redundant IO, not a correctness bug.
- `ai_tasks.py` retains defects carried in from commit `9aa22c7` (not introduced or fixed by this batch): a triplicated `generate_segmentation_report` call with a doubled object-name block, a `db.close()` that runs only on the exception path (session leak on success), and derived-artifact paths built from the outer `tmpdir` while `source_img_path` uses the inner `tmpdirname` (inconsistent, cosmetic).
- `backend/app/api/routers/scans.py` (the separate upload-header validator) still names `NIFTI2_MAGIC = b"n+1"` (actually the NIfTI-1 single-file magic) and its docstrings say "NIfTI-1 or NIfTI-2", even though it rejects NIfTI-2 in practice. Left unchanged to keep this batch minimal.

**Affected files:**
- `backend/app/services/ai_tasks.py` (restored helpers; worker-side validation reuse)
- `backend/ai_pipeline/preprocessing/brats_preprocessing.py` (dtype set; standalone `_validate_nifti_image`)
- `backend/ai_pipeline/preprocessing/__init__.py` (exports)
- `backend/ai_pipeline/preprocessing/volumetric_3d/brats_prep.py` (validation before load)
- `backend/ai_pipeline/training/train_armt_gan.py` (validation before load)
- `backend/tests/test_nifti_safety_regression.py` (new: 22 tests)
- `backend/tests/test_auth.py` (required `CORS_ORIGINS` env)

**Status:** Complete (verification). NIfTI-1-only support is now explicit; true decompressed-byte limits and early truncated-payload rejection remain open risks.

**Verification:**
- Non-database suite: 196 passed, 0 xfailed.
- `test_nifti_safety_regression.py`: 22 passed, 0 xfailed.
- `test_provenance_wiring.py`: 14/14 passed (including `TestConstants`).
- `test_auth.py`, `test_config_security.py`, `test_db_models.py`: 74 passed.
- The three database-backed modules (`test_scan_claim.py`, `test_artifact_lifecycle.py`, `test_scan_claim_integration.py`; 31 tests) now collect (227 tests collected in `tests/` overall) but require a live PostgreSQL, which is unavailable in this environment; they were excluded from the non-database run. A separate root-level `backend/test_pipeline_direct.py` still fails to import (`MRIScan`/`process_mri_scan` no longer exist) and is outside `tests/`.

### D-037 --- SG3-F3-I: Bounded NIfTI-1 upload and preprocessing validation

**Date:** 2026-10-15

**Context:** Implementation batch for the SG3-F3-H audit findings. The upload validator buffered the whole gzip payload, checked the header magic at the wrong offset (bytes 0-2 instead of 344), relied on a shallow nibabel load, and had no true decompressed-byte limit. The preprocessing validator bounded only `shape[:3]` and estimated allocation from the stored itemsize. This allowed decompression bombs, truncated/CRC-corrupt/concatenated/trailing-garbage streams, extra-axis inflation, and it rejected valid uncompressed BraTS `.nii` (BraTS2020 ships uncompressed single-file NIfTI-1 with `vox_offset` != 352).

**Decision:**
1. `backend/app/api/routers/scans.py`: replace whole-payload buffering with bounded streaming gzip validation (`zlib.decompressobj(31)`), reading through EOF so the CRC32/ISIZE trailer is verified, under a true decompressed-byte cap `MAX_DECOMPRESSED_BYTES = MAX_DECOMPRESSED_SIZE_MB * 1024 * 1024` (200 MiB, documented). Malformed, truncated, concatenated, and trailing data are rejected explicitly via `eof`/`unused_data`. No temporary files are written.
2. Parse the NIfTI-1 header with `struct` (endianness detected from `sizeof_hdr`), read the magic at its real offset (344), require the single-file magic `n+1`, and validate dims, datatype code, bitpix, `vox_offset`, voxel count, and declared-vs-actual payload size using `vox_offset` (not a hard-coded 352). The container (gzip vs raw) must match the declared extension.
3. `backend/ai_pipeline/preprocessing/brats_preprocessing.py`: extract `_validate_nifti_shape_and_dtype(shape, dtype)` shared by the on-disk and upload validators; bound **every** declared dimension, use `prod(shape)` for the voxel count, and enforce a **float32** allocation estimate (`voxels * 4 bytes`) before `get_fdata(dtype=np.float32)`. Derive `SUPPORTED_DTYPES` from the new `NIFTI1_DTYPE_CODES` map (single source of truth).
4. `backend/app/services/ai_tasks.py`: the worker preserves the uploaded container (`.nii` or `.nii.gz`) when naming study files, so uncompressed BraTS inputs are loadable end to end.
5. Extend `backend/tests/test_nifti_safety_regression.py` (22 -> 47 tests) with upload-validator, resource-bound, worker, endpoint-cleanup, and uncompressed-modality tests. Real BraTS2020 `.nii` compatibility is guarded by a dataset-gated (skipped-if-absent), read-only test.

**Reason:** rules.md #18/#20; PRD G7 ("fail closed when required inputs are invalid"); design.md §7 (validate modality completeness/file types before inference). The SG3-F3-H audit confirmed no true decompressed-byte bound and a valid-uncompressed-`.nii` rejection.

**Verified protections / evidence:**
- `_validate_nifti_header` returns `True` for real BraTS2020 `.nii` (t1 `vox_offset=2880`, seg `vox_offset=2976`) and for the gzipped form of the same volume; `False` for truncated payloads/streams, CRC failure, malformed gzip, trailing garbage, concatenated members, over-limit decompression (patched cap), invalid magic (`n+2`), unsupported datatype code, extension/container mismatch, and inconsistent payload metadata.
- Preprocessing rejects oversized 4D/5D headers before any `get_fdata` (spied) and still accepts moderate 4D within bounds.
- The endpoint dispatches a valid uncompressed study and, on an invalid modality, returns 400 while cleaning up previously uploaded objects with no leftover temp files.

**Known limitations (not claimed as solved):**
- No real compressed BraTS `.nii.gz` sample and no live MinIO are available here; the exact `vox_offset + prod(shape)*itemsize` equality rule was validated against real uncompressed BraTS data and nibabel-generated gzip fixtures, not a real compressed BraTS file.
- The upload decompressed cap is a documented module constant, not env-configurable.
- `MAX_DECOMPRESSED_SIZE_MB` remains 200 for the float32 estimate (BraTS ~36 MB float32 is well within it).
- Decompression is streaming and bounded in bytes, but a single <=200 MiB decompressed stream is still accepted; the voxel/estimate caps are what bound the resulting allocation.

**Affected files:**
- `backend/app/api/routers/scans.py`
- `backend/ai_pipeline/preprocessing/brats_preprocessing.py`
- `backend/ai_pipeline/preprocessing/__init__.py`
- `backend/app/services/ai_tasks.py`
- `backend/tests/test_nifti_safety_regression.py`

**Status:** Complete (implementation). Supersedes the D-036 "known gaps" for the upload path: a true decompressed-byte limit and early truncated-payload rejection are now enforced.

**Verification:**
- `test_nifti_safety_regression.py`: 47 passed (including the dataset-gated real-BraTS test).
- `test_auth.py`, `test_config_security.py`, `test_db_models.py`, `test_provenance_wiring.py`, `test_preprocessing_parity.py`, `test_nifti_safety_regression.py`: 144 passed.
- Full non-database suite: 221 passed; the 31 database-backed tests still require a live PostgreSQL (unavailable here).
- Read-only real-data probe: BraTS2020 t1/seg `.nii` and gzipped t1 accepted.

### D-038 --- SG3-F4-A: PostgreSQL integration verification, worker session-leak fix, and report dedup

**Date:** 2026-10-15

**Context:** Verification batch for the three database-backed suites (`test_scan_claim.py`, `test_scan_claim_integration.py`, `test_artifact_lifecycle.py`). No PostgreSQL was running locally (no service, no listener on 5432, no server binaries; the Docker daemon was stopped). Docker Desktop was started and an isolated, throwaway PostgreSQL 16 container was used (dedicated `neuroscan_test` database, no shared volume), so no dev/production data was touched. Migrations 001 -> 006 were applied with `alembic upgrade head` before running the suites.

Audit findings:
1. `process_scan_task` opened a session and closed it only on the exception path (`db.close()` was inside the `except`). Every other exit -- success and the early `return` when the claim failed -- leaked the session. The early-return leak is a correctness bug, not just hygiene: `claim_scan_for_processing` runs `SELECT ... FOR UPDATE NOWAIT`, so the leaked session kept the row lock open and deadlocked the suite's fixture teardown. Observed: the run hung on `test_already_processing_scan_exits_early` before the fix.
2. `generate_segmentation_report` was called three times with identical arguments, with the object-name block duplicated twice. Side-effect trace: the function only writes one PDF to `report_path` (overwrite), so there were no duplicate artifacts or DB rows -- only redundant image IO. The triplication is a copy-paste artifact.
3. `TestUploadCompensation` returned HTTP 401 because the S3 ownership batch (`823e84f`, 2026-10-10) added `get_current_user` to `/upload` but never updated the test client helper to authenticate.
4. The same tests' NIfTI fixture (all-zero 10x10x10) gzips below the endpoint's 1 KB minimum-size guard added in `9f11037` (SG3-F3, 2026-10-10), so uploads were rejected as "too small".

**Decision:**
1. `backend/app/services/ai_tasks.py`: always release the primary session in a `finally` block (removed the exception-only `db.close()`), covering success, failure, and every early-return claim outcome and releasing the row lock.
2. `backend/app/services/ai_tasks.py`: call `generate_segmentation_report` exactly once and keep a single object-name block.
3. `backend/tests/test_scan_claim_integration.py`: override `get_current_user` with a persisted test identity in `_get_test_client_with_db`, and enlarge the upload fixture to incompressible 32x32x32 data so it clears the size guard. Test-harness only; no production or API-contract change.
4. `backend/tests/test_scan_claim_integration.py`: add `TestSessionCleanup` (2 tests) asserting the primary session is closed on success and that all sessions are closed on failure.

**Reason:** rules.md #17/#18/#20; PRD G7; design.md §7. The session leak contradicted the row-lock claim design (D-035) and blocked the DB-backed suite; a failing test blocks the phase. Transaction boundaries and artifact provenance are preserved: the `finally` only closes the session after `commit()`/error handling completes, and the fresh FAILED-marking transaction is unchanged.

**Affected files:**
- `backend/app/services/ai_tasks.py` (`finally: db.close()`; single report call)
- `backend/tests/test_scan_claim_integration.py` (`TestSessionCleanup`; upload helper auth + fixture size)

**Status:** Complete.

**Verification:**
- Isolated PostgreSQL 16 (ephemeral Docker container, dedicated `neuroscan_test` DB, migrations 001->006) with test-only env values for DATABASE_URL/MINIO/FIREBASE_PROJECT_ID/CORS_ORIGINS.
- Focused suites: `test_scan_claim.py` 7/7, `test_artifact_lifecycle.py` 9/9, `test_scan_claim_integration.py` 17/17 (the file was hanging before the fix; 5 upload tests were returning 401).
- Full backend suite `tests/`: **254 passed**, 0 failed.
- No production credentials read or exposed; `.env` was not read; the dev compose volume was untouched.

**Known limitations:**
- Verification used a throwaway container, not the compose `postgres_data` volume; CI must provision an equivalent disposable PostgreSQL.
- `TestUploadCompensation` uses an overridden `get_current_user` (no real Firebase token), consistent with the other integration tests.
- Pre-existing and unchanged: derived-artifact temp paths use the outer `tmpdir` while `source_img_path` uses the inner `tmpdirname` (cosmetic); `backend/test_pipeline_direct.py` (outside `tests/`) still fails to import (`MRIScan`/`process_mri_scan` removed).

### D-039 --- SG3-F4-B: PostgreSQL-backed backend CI workflow

**Date:** 2026-10-15

**Context:** D-038 left the explicit limitation "CI must provision an equivalent disposable PostgreSQL." Auditing the repository showed **no CI provider was configured** (no `.github/`, `.gitlab-ci.yml`, `Jenkinsfile`, Azure Pipelines, CircleCI, or Travis config). The only remote is GitHub (`Gagan-09/NeuroScan-AI`), so GitHub Actions is the smallest compatible provider. Three further blockers to a runnable CI were found:

1. `backend/requirements.txt` was **unsatisfiable**: `opencv-python-headless==5.0.0.93` declares `numpy>=2` for Python >=3.9, but the file pins `numpy==1.26.3` (needed by the frozen `torch==2.2.0` scientific stack, which predates NumPy 2). `pip install` fails with `ResolutionImpossible`. The OpenCV pin was introduced (commit `7c947e4`) alongside the NumPy pin and was never verified to install.
2. `alembic` was **not declared** anywhere, yet running migrations is required to build the schema.
3. `pytest` (and `pytest-asyncio`, required by `tests/test_auth.py`) were **not declared**; the suite is not runnable from a clean environment.

**Decision:**
1. Add `.github/workflows/backend-tests.yml`: a single job that provisions an isolated, disposable `postgres:16-alpine` service container (fresh per run, no volume) with **CI-only credentials** and a **dedicated test database** (`neuroscan_ci`), waits for readiness via the service health check plus an explicit `psycopg2` poll, recreates the `public` schema, applies `alembic upgrade head`, and runs the complete backend suite `python -m pytest tests/ -v`. No step reads `.env`; every required setting is supplied by the workflow `env` block. No production database or credentials are referenced.
2. Add `backend/requirements-dev.txt` pinning `pytest` and `pytest-asyncio` (test-only deps).
3. Correct `opencv-python-headless` to `4.11.0.86` (supports `numpy>=1.23.5`, same `cv2` surface used by the overlay renderer) and add `alembic==1.13.2` to `requirements.txt`.

**Reason:** rules.md #17 ("Schema evolution must be migration-based"), #18 ("No task is complete until required tests pass"), #20; D-038 ("CI must provision an equivalent disposable PostgreSQL"). The three database suites depend on PostgreSQL row-level locking (`SELECT ... FOR UPDATE NOWAIT`) and must never be substituted with SQLite. The OpenCV pin correction is a rendering-library version change only; ARMT-GAN architecture, modalities, labels, normalization, resizing, evaluation, database schema, and API contracts are untouched.

**Affected files:**
- `.github/workflows/backend-tests.yml` (new: PostgreSQL CI job)
- `backend/requirements-dev.txt` (new: pytest, pytest-asyncio)
- `backend/requirements.txt` (opencv pin corrected; alembic added)

**Status:** Complete.

**Verification:**
- Dependency resolution for the CI target (CPython 3.11 / Linux) succeeds: `uv pip compile backend/requirements.txt --python-version 3.11 --python-platform x86_64-unknown-linux-gnu` resolves `numpy==1.26.3`, `opencv-python-headless==4.11.0.86`, `torch==2.2.0`, `torchvision==0.17.0`, `alembic==1.13.2`. The pre-fix pair was demonstrated to fail with `ResolutionImpossible`.
- Equivalent local disposable-PostgreSQL run (ephemeral Docker `postgres:16-alpine`, dedicated `neuroscan_ci` DB, no volume; the dev `neuroscan_db` container/volume was never touched): migrations `001 -> 006` applied via `alembic upgrade head`; the workflow's readiness and schema-recreate scripts were executed verbatim; the complete backend suite `tests/`: **254 passed, 0 failed**; the three required modules (`test_scan_claim.py` 7, `test_artifact_lifecycle.py` 9, `test_scan_claim_integration.py` 17) re-verified at **33 passed** after a schema drop/recreate.
- Workflow YAML parsed successfully (single `backend-tests` job, 7 steps).
- No `.env` was read; no production credentials were used.

**Known limitations:**
- The workflow was not executed on a GitHub-hosted runner (none is available in this environment); the equivalent local run used the machine's system interpreter (CPython 3.13) rather than the workflow's pinned CPython 3.11 stack, so the exact pinned-stack execution remains unproven here.
- The workflow installs the full `requirements.txt` (including the large CUDA `torch` wheel); it is correct but heavyweight.
- The system interpreter's NumPy/OpenCV versions (2.x / 5.x) differ from the corrected pins, which is why the pin conflict was not caught earlier.

### D-040 --- SG3-F4-C: Hosted CI verification and closed backend test blockers

**Date:** 2026-10-15

**Context:** D-039 left the explicit limitation "the workflow has not run on a GitHub-hosted runner." The repository is public, so the hosted result is readable through the unauthenticated GitHub REST API (no `gh` CLI/token is available here, and raw job logs require sign-in). A push to `main` had already triggered run `38769341345` (run_number 1, head_sha `f39f070095704d3144caefd8a0ee7d11867a8972`), which **failed** on the final step `python -m pytest tests/ -v` with exit code 2 (a collection error). Steps 1-8 (checkout, Python 3.11 setup, dependency install, PostgreSQL readiness, schema recreate, `alembic upgrade head`) all succeeded. Reproducing the workflow verbatim against an isolated disposable PostgreSQL 16 container and the exact pinned dependencies (Python 3.11.17) surfaced two distinct blockers behind exit code 2:

1. `httpx` was not declared in either requirements file. `fastapi.testclient.TestClient` (Starlette) imports `httpx`; its absence makes `tests/test_auth.py` fail at import, which aborts collection of the whole file and yields pytest exit code 2 -- exactly the hosted symptom.
2. After adding `httpx`, the five `TestUploadValidation` tests failed: they send `Authorization: Bearer valid-token` and assert `401` with the comment "Auth fails first (mocked)", but Firebase is never mocked. With no Application Default Credentials (a clean CI runner), the real `auth.verify_id_token` raises `google.auth.exceptions.DefaultCredentialsError`, which is not a `FirebaseError`, so `app/core/security.py::verify_id_token` does not translate it to 401 and the request returns 500.

**Decision:**
1. `backend/requirements-dev.txt`: add `httpx==0.26.0`. Starlette 0.36.3 constructs its transport as `httpx.Client(app=self.app, ...)`; httpx removed the `app` argument in 0.27.0, so 0.26.0 is the newest compatible release. Test-only dependency.
2. `backend/tests/test_auth.py`: add a class-scoped autouse fixture to `TestUploadValidation` that patches `app.core.security.auth.verify_id_token` with `side_effect=InvalidIdTokenError("Invalid token")`, so the endpoint rejects authentication with 401 before upload validation runs, without requiring ADC. The tests' existing `== 401` assertions and intent are unchanged; the fixture mirrors the invalid/revoked-token pattern already used in the same file.

**Reason:** rules.md #3 (step 9, "Update affected source-of-truth records"), #18 ("No task is complete until required tests pass"; "a failing test blocks the phase"), #20; D-038/D-039. Both changes are test-harness/dependency only: no application logic, API contract, database schema, ARMT-GAN architecture, preprocessing, evaluation, or provenance behavior is altered. The upload endpoint's real authentication path is untouched (it is mocked, exactly as the other auth tests already do).

**Affected files:**
- `backend/requirements-dev.txt` (`httpx==0.26.0` added)
- `backend/tests/test_auth.py` (`TestUploadValidation` autouse auth-rejection fixture)

**Status:** Complete.

**Verification:**
- Reproduced the baseline hosted failure locally: exact pinned stack (Python 3.11.17, `requirements.txt` + `requirements-dev.txt`) with no `httpx` -> `ERROR collecting tests/test_auth.py` (`ModuleNotFoundError: No module named 'httpx'`) -> **exit code 2**, matching the hosted annotation.
- Isolated disposable `postgres:16-alpine` container (dedicated `neuroscan_ci` DB, CI-only credentials, no volume); migrations `001 -> 006` applied with `alembic upgrade head`; the workflow's readiness and schema-recreate steps executed verbatim.
- Pre-fix, with only `httpx` added: **5 failed, 249 passed** (254 collected) -- the five `TestUploadValidation` tests, all `DefaultCredentialsError`.
- Post-fix: `tests/test_auth.py` **28 passed**; complete backend suite `tests/`: **254 passed, 0 failed** (2 documented warnings: a Starlette `anyio.abc.BlockingPortal` deprecation and a pre-existing `AsyncMock` coroutine warning in `test_upload_ownership_derived_from_authenticated_identity`).
- `httpx` compatibility confirmed in-container: `inspect.signature(httpx.Client.__init__)` includes `app`, and `starlette.testclient.TestClient.__init__` passes `app=self.app`; Starlette declares `httpx>=0.22.0`.
- Dependency resolution for the CI target (CPython 3.11 / `x86_64-unknown-linux-gnu`) was verified in D-039 and is unchanged here.
- No `.env` was read; no production credentials were used.

**Known limitations:**
- Raw hosted job logs require GitHub sign-in (unavailable here); the hosted conclusion, step list, and the failing step's exit code are read from the public REST API.
- The workflow installs the full `requirements.txt` including the large CUDA `torch` wheel. The job's tests are CPU-only and do not require CUDA (`torch.cuda.is_available()` is never asserted true, and `torchvision` is used only by a non-tested preprocessing helper), so a CPU-only wheel would be lighter -- but no switch was made, as it is outside this task's scope and would require a PyTorch CPU index URL plus a re-verification of the scientific stack.

### D-041 --- Remove hardcoded administrator credential (SG3-F5-B)

**Date:** 2026-10-15

**Context:** `backend/init_system.py` bootstrapped a password-based `Admin` user by hashing a hardcoded default administrator password and creating `User(email, password_hash, role="Admin")`. The current `User` model (migration 006) keys identity on `firebase_uid` and stores no password or auth credentials; the columns `password_hash` and `role` no longer exist, so the script targets a nonexistent schema. Auth is Firebase UID-based (`app/core/security.py::get_current_user` gets/creates `User` from a verified Firebase ID token). The only reference to `init_system.py` was `backend/ingest_datasets.py` (unreferenced dead code that itself imports nonexistent `Patient`/`MRIScan`); nothing in CI, Docker, or scripts runs either script. No supported workflow depends on it.

**Decision:** Delete `backend/init_system.py` (removing the hardcoded-default-administrator-bootstrap path). Do not introduce password-based authentication or bypass the Firebase UID-based identity model. Leave `ingest_datasets.py` unmodified (unsupported dead code, contains no credential). Add a focused regression test in `backend/tests/test_config_security.py::TestAdminCredentialRemoval` asserting (a) `backend/init_system.py` no longer exists and (b) no tracked Python source contains the default administrator password (scanning tracked `.py` via `git ls-files`, excluding the test file itself which names the prohibited literal; never reading `.env`).

**Reason:** rules.md #14 ("No default administrator passwords in committed code"), #9/#17; PRD NFR "Free of hard-coded credentials"; memory.md D-012 ("Credentials must come from environment or secret management"), D-029.

**Affected files:**
- `backend/init_system.py` (deleted)
- `backend/tests/test_config_security.py` (added `TestAdminCredentialRemoval`, 2 tests)

**Status:** Complete.

**Verification:**
- `python -m pytest tests/test_config_security.py -v` from `backend/`: **21 passed** (incl. both new tests). Note: 3 pre-existing `TestConfigSecurity` tests require `CORS_ORIGINS` (provided by CI's workflow env block); they fail in a shell without that var but pass with it, unrelated to this change.
- Full backend suite `tests/` against an isolated disposable `postgres:16-alpine` (dedicated `neuroscan_ci` DB, CI-only credentials, migrations `001 -> 006`, `alembic upgrade head`): **256 passed, 0 failed**, matching the GitHub Actions procedure (`.github/workflows/backend-tests.yml`).
- `git diff --check`: clean. No `.env` read; no credentials exposed; no password literal recorded here.

### D-042 --- Reject wildcards anywhere in CORS origins (SG3-F5-C)

**Date:** 2026-10-15

**Context:** `backend/app/core/config.py::validate_cors_origins` rejected CORS wildcard only when the parsed list was exactly `["*"]` (singleton `"*"`). A mixed list such as `"http://localhost:3000,*,https://app.example.com"` → `["http://localhost:3000", "*", "https://app.example.com"]` passed validation and would be handed to `CORSMiddleware` with `allow_credentials=True` in `backend/app/main.py`, effectively allowing arbitrary origins alongside trusted ones. rules.md #14: "No wildcard CORS for production."

**Decision:** Update `validate_cors_origins` to reject any origin list containing `"*"`, including mixed wildcard-plus-explicit-origin lists. Preserve valid explicit-origin configurations and existing comma-separated parsing semantics. Validation runs in the pydantic field validator at `Settings()` instantiation (`get_settings()`), which is called before `CORSMiddleware` is configured in `main.py`, so rejection occurs before middleware setup. Add regression test `test_cors_origins_rejects_mixed_wildcard_and_explicit_origins` alongside the existing singleton-wildcard and valid-explicit-origins tests in `TestCORSConfiguration`. No changes to authentication, frontend behavior, or unrelated security findings.

**Reason:** rules.md #14 ("No wildcard CORS for production"); PRD G7 ("fail closed when required inputs are invalid").

**Affected files:**
- `backend/app/core/config.py` (`validate_cors_origins`: `if "*" in origins`)
- `backend/tests/test_config_security.py` (added `test_cors_origins_rejects_mixed_wildcard_and_explicit_origins`)

**Status:** Complete.

**Verification:**
- `python -m pytest tests/test_config_security.py -v` from `backend/`: **22 passed** (incl. new mixed-wildcard test). Note: 3 pre-existing `TestConfigSecurity` tests require `CORS_ORIGINS` (provided by CI's workflow env block); they fail in a shell without that var but pass with it, unrelated to this change.
- Full backend suite `tests/` against an isolated disposable `postgres:16-alpine` (dedicated `neuroscan_ci` DB, CI-only credentials, migrations `001 -> 006`, `alembic upgrade head`): **257 passed, 0 failed**, matching the GitHub Actions procedure.
- `git diff --check`: clean. No `.env` read; no credentials exposed.

### D-043 --- Firebase infrastructure error handling (SG3-F5-D)

**Date:** 2026-10-15

**Context:** The Firebase authentication path in `backend/app/core/security.py` had four infrastructure error handling defects:
1. Missing Application Default Credentials (ADC) raised `google.auth.exceptions.DefaultCredentialsError`, which is not a `FirebaseError`, causing unhandled 500 with exception details leaked.
2. Firebase Admin SDK initialization failures (project ID mismatch, config errors) raised generic exceptions, causing unhandled 500.
3. Google Auth infrastructure errors during token verification (`TransportError`, `TimeoutError`, etc.) are `GoogleAuthError` subclasses, not `FirebaseError`, causing unhandled 500.
4. Database errors during user lookup/creation in `get_current_user` raised `SQLAlchemyError`, causing unhandled 500 with connection details leaked.

All four cases represented server-side infrastructure failures that should fail closed with a safe 503 response, not be misrepresented as invalid user credentials (401) or leak internal details.

**Decision:** Update `backend/app/core/security.py` to:
1. In `_get_firebase_app()`: Catch `GoogleAuthError` (covers missing ADC, invalid credentials) and generic `Exception` (covers project ID mismatch, config errors), log with `exc_info=True` for operators, raise `HTTPException 503` with generic message.
2. In `verify_id_token()`: Catch `GoogleAuthError` during `auth.verify_id_token()` separately from `FirebaseError`. Token validation failures (`FirebaseError`) → 401 generic. Infrastructure failures (`GoogleAuthError`) → 503 generic. Log both with appropriate levels.
3. In `get_current_user()`: Wrap database operations in `try/except SQLAlchemyError`, log with `exc_info=True`, rollback, raise `HTTPException 503` generic.
4. Add structured logging for operator diagnosis without exposing credentials, tokens, or medical data.
5. Add regression tests in `backend/tests/test_auth.py::TestInfrastructureErrorHandling` (5 tests) covering: missing ADC, init failure, GoogleAuthError during verify, timeout during verify, database error in get_current_user. All verify 503 status, generic message, and no detail leakage.

Preserve existing behavior: invalid/expired/revoked tokens → 401 generic; missing/malformed bearer → 401 generic; valid tokens → decoded claims; cross-user ownership enforcement unchanged.

**Reason:** rules.md #14 (fail closed, no sensitive data in logs), #12 (no auth bypass), PRD G7 (fail closed when inputs invalid), memory.md D-012 (credentials from environment). Firebase infrastructure/configuration failures must not be misrepresented as invalid user credentials. Never return exception strings, tokens, credential contents, project secrets, or stack traces to clients.

**Affected files:**
- `backend/app/core/security.py` (added GoogleAuthError/SQLAlchemyError handling, logging, 503 responses)
- `backend/tests/test_auth.py` (added `TestInfrastructureErrorHandling` with 5 regression tests)

**Status:** Complete.

**Verification:**
- `python -m pytest tests/test_auth.py tests/test_config_security.py -v` from `backend/`: **55 passed** (incl. 5 new infrastructure tests).
- All non-database backend tests pass (224 tests across test_auth, test_config_security, test_db_models, test_provenance_wiring, test_preprocessing_parity, test_nifti_safety_regression, test_xai, test_robustness_attacks, test_evaluation_protocol).
- `git diff --check`: clean (trailing whitespace fixed).
- Defect verification: manual tests confirm missing ADC, init failure, GoogleAuthError, and database errors all return 503 with "Authentication service temporarily unavailable" and no internal details leaked.

### D-044 --- Rate limiting and resource-exhaustion controls (SG3-F5-E)

**Date:** 2026-10-15

**Context:** P8 Security + Reliability requires rate limiting for expensive API operations. The application currently runs background work via FastAPI `BackgroundTasks` (in-process, single worker), so an in-memory limiter would NOT be shared across uvicorn workers or service replicas. Redis is already part of the deployment topology (`docker-compose.yml` `redis:7-alpine`, `REDIS_URL` setting in `app/core/config.py`, `redis` package in `requirements.txt`) but is not wired into application code, so using it introduces no new mandatory infrastructure dependency.

**Decision:**
1. Add `backend/app/core/rate_limiting.py`: a Redis-backed fixed-window rate limiter using an atomic Lua script (`INCR` + `EXPIRE` + `TTL`), keyed per authenticated user (`{prefix}upload:{firebase_uid}`).
2. Enforce it on `POST /api/v1/scans/upload` only — the expensive endpoint that triggers scan processing (inference dispatched as a background task). Status/results polling are cheap DB reads and are deliberately not rate limited so normal scan monitoring is not broken.
3. Per-authenticated-user limits only; no IP-based controls (proxy handling not established as trustworthy; client-supplied forwarding headers never trusted).
4. Limits configurable via `Settings` (`app/core/config.py`) with validated defaults: `RATE_LIMIT_ENABLED=True`, `RATE_LIMIT_STORAGE_BACKEND=redis`, `RATE_LIMIT_KEY_PREFIX=neuroscan:ratelimit:`, `RATE_LIMIT_UPLOAD_MAX_REQUESTS=10`, `RATE_LIMIT_UPLOAD_WINDOW_SECONDS=60`.
5. Limiter backend unavailable: fail closed — `POST /upload` returns generic `503 Service Unavailable` with no internal details (matches the existing `security.py` infra-failure pattern). Exceeded limit returns `429` with generic body and `Retry-After` header (window TTL).
6. Wire `Depends(enforce_upload_rate_limit)` into `upload_scan` in `backend/app/api/routers/scans.py`; authentication (`get_current_user`) runs first, so unauthenticated requests get 401 before any limit check.

**Reason:** rules.md #14/#18/#20; PRD G7 (fail closed); requirement for a shared, production-suitable limiter in multi-worker deployments; fail-closed on backend unavailability; do not introduce undocumented infrastructure dependencies.

**Affected files:**
- `backend/app/core/rate_limiting.py` (new: RedisRateLimiter, get_rate_limiter, enforce_upload_rate_limit)
- `backend/app/core/config.py` (rate-limit settings + validators)
- `backend/app/api/routers/scans.py` (wire rate-limit dependency into upload)
- `backend/tests/test_rate_limiting.py` (new: 15 focused tests)
- `tasks.md` (SG3-F5-E marked verified)
- `memory.md` (this record)

**Status:** Verified.

**Verification:**
- New focused suite `tests/test_rate_limiting.py`: 15 passed, 0 failed (within-limit succeeds; exceeded → 429 + safe body; `Retry-After` correct; Redis failure → 503 with no detail leakage; per-user isolation; config validation; auth still runs first).
- Full non-database backend suite `tests/`: **244 passed, 0 failed** (baseline 229 + 15 new; no regressions in auth, ownership, upload validation, or error redaction).
- `git diff --check`: clean.
- DB-backed modules (`test_scan_claim.py`, `test_artifact_lifecycle.py`, `test_scan_claim_integration.py`) require a live PostgreSQL and run in CI via `.github/workflows/backend-tests.yml` (not executable in this environment); this change touches no database schema or those modules.

### D-045 --- Redact upload exception-detail leak (SG3-F6-A)

**Date:** 2026-10-16

**Context:** `backend/app/api/routers/scans.py::upload_scan` returned
`HTTPException(status_code=500, detail=str(e))` for unexpected upload/database
failures, leaking raw exception text (potentially credentials, object-storage
details, database details, file paths) to clients. Identified as finding F-01
in the final security acceptance audit.

**Decision:** Replace `detail=str(e)` with a generic `detail="Upload failed"`.
Log diagnostics server-side via `logger.exception("Unexpected error during scan
upload")` (traceback in controlled application logs; the log message itself
contains no tokens, secrets, uploaded content, or medical data). Preserve the
existing `db.rollback()` and best-effort MinIO source-object cleanup. Keep
`HTTPException` validation/auth responses (400/401) unchanged. Do not touch
other handlers or refactor the upload pipeline.

**Reason:** rules.md #14 (no sensitive data exposed), #18 (tests gate), #20;
PRD G7 (fail closed); design.md §12 (failures explicit).

**Affected files:**
- `backend/app/api/routers/scans.py` (generic 500 handler redacted)
- `backend/tests/test_scan_claim_integration.py` (new `TestUploadErrorRedaction`,
  3 hermetic tests: storage failure, database failure, validation unchanged)

**Status:** Verified.

**Verification:**
- New focused suite `tests/test_scan_claim_integration.py::TestUploadErrorRedaction`: 3 passed, 0 failed (storage failure → generic 500, no exception text, cleanup count 2; database failure → generic 500, no exception text, cleanup count 4; invalid extension → 400 unchanged).
- Regression suites: `tests/test_rate_limiting.py` 15 passed, `tests/test_config_security.py` 22 passed, `tests/test_auth.py` 33 passed; no regressions.
- `git diff --check`: clean. Diff limited to `scans.py` (1 line changed) and the new test class.
- Limitation: full backend suite `tests/` against disposable PostgreSQL 16 (CI procedure `.github/workflows/backend-tests.yml`) could not be re-run locally because no PostgreSQL instance or running Docker Desktop is available in this environment (D-036/D-038). Previously verified 254 passed, 0 failed (D-040). Changes are confined to the upload 500 path and do not touch DB-backed claim/artifact logic.

### D-046 --- SG3-F6-B: Stale-PENDING scan recovery loop with atomic retry accounting (remediates F-02)

**Date:** 2026-10-16

**Context:** Security finding F-02 (confirmed design gap): scans left in `PENDING` are stranded on worker restart because FastAPI `BackgroundTasks` are in-process and lost on restart. `claim_scan_for_processing` accepts `PENDING`/`FAILED` → `PROCESSING`, but nothing re-claims a stranded `PENDING` scan. `processing_started_at` (migration 005) existed but was never queried for recovery. No retry accounting or attempt cap existed.

**Decision:**
1. `backend/alembic/versions/007_add_scan_retry_count.py` (new migration 007): add `scans.attempt_count` (Integer, default 0) to durably track recovery re-dispatches per scan.
2. `backend/app/services/ai_tasks.py`: add `recover_stale_pending_scans()` which selects stranded scans (`status='PENDING'`, `processing_started_at IS NULL`, `created_at` older than the staleness threshold), locks each with `FOR UPDATE NOWAIT`, and in one transaction increments `attempt_count` and claims `PENDING → PROCESSING` (setting `processing_started_at`), committed durably *before* dispatch. Attempts are capped at `MAX_RECOVERY_ATTEMPTS = 3`; beyond the cap the scan is marked `FAILED` (terminal) and not retried again by the loop. Concurrent cycles cannot consume duplicate retry counts: an asyncio lock serializes cycles within a process, and the row lock + status change to `PROCESSING` serializes across processes.
3. `backend/app/services/ai_tasks.py`: `process_scan_task(scan_id, modality_objects, skip_claim=False)` gains a backward-compatible `skip_claim` parameter (default `False`, preserving the existing upload claim path). The recovery loop passes `skip_claim=True` because it already claimed atomically before dispatch; inference behavior is unchanged.
4. `backend/app/main.py`: FastAPI managed `lifespan` starts `_recovery_loop` on startup and stops it cleanly on shutdown. Each cycle runs within the asyncio lock (no overlapping cycles), scopes one `SessionLocal` to the operation, dispatches blocking `process_scan_task` via `asyncio.to_thread` (never blocks the asyncio event loop with model inference), logs and continues after cycle-level failures, and re-runs on a fixed interval.
5. `backend/tests/test_scan_recovery.py` (new): PostgreSQL-backed tests for attempt accounting, recent/processing/segmented/failed scan skipping, max-3-attempts cap → FAILED, concurrency (no duplicate retry consumption), upload claim-path preservation, and cycle dispatch/failure-continuation.
6. Condition 4 honored: only stale `PENDING` scans are recovered; `PROCESSING` scans are never reclaimed; no ownership, API, UI, or scientific-pipeline changes. Condition 3 honored: `process_scan_task` is synchronous/blocking and is never called on the event loop.

**Reason:** rules.md #17/#18/#20; PRD G7 (fail closed); design.md §12 (failures explicit, preserve scan state); memory.md D-038 (session hygiene). Resolves security finding F-02.

**Affected files:**
- `backend/alembic/versions/007_add_scan_retry_count.py` (new)
- `backend/app/services/ai_tasks.py` (recovery loop, attempt accounting, `skip_claim`)
- `backend/app/main.py` (lifespan-managed recovery loop)
- `backend/tests/test_scan_recovery.py` (new)
- `backend/memory.md` (this record), `backend/tasks.md` (P8 entry)

**Status:** Implemented; focused non-DB suite passes, new tests collect. PostgreSQL 16 DB-backed verification runs on the GitHub Actions workflow via the PR (CI success not yet claimed — condition 6).

**Verification:**
- `ai_tasks.py`, `main.py`, migration 007, `test_scan_recovery.py` all parse cleanly.
- App imports successfully with test settings; scan routes register.
- Non-DB backend suite: 123 passed (config security, rate limiting, XAI, preprocessing parity, evaluation protocol, robustness).
- New `test_scan_recovery.py`: 10 tests collected (require live PostgreSQL; run in CI).
- `git diff --check`: clean.

### D-047 --- SG3-F6-C: Redis security verification and real-Redis integration tests (remediates F-03)

**Date:** 2026-10-16

**Context:** Security finding F-03 (operational dependency): Redis in `docker-compose.yml` runs without authentication and is published on host port 6379; the fail-closed 503 path for the upload rate limiter is exercised only with in-memory fakes, never against a live Redis. CI provisions PostgreSQL but no Redis, so the real fail-closed `503` path is unproven in CI.

**Decision:**
1. `.github/workflows/backend-tests.yml` (updated): add a `redis:7-alpine` service container alongside PostgreSQL, with a `redis-cli ping` health check. Add `REDIS_URL: redis://localhost:6379/0` to the workflow `env` block. Add an explicit "Wait for Redis readiness" step that polls the Redis service before tests run. Preserve the existing PostgreSQL service, migration sequence, and test command unchanged.
2. `backend/tests/test_rate_limiting.py` (updated): add `TestRedisIntegration` class with mandatory real-Redis integration tests:
   - `test_429_when_limit_exceeded_real_redis`: exhausts the per-user counter against real Redis, then asserts 429 with `Retry-After` header on the upload endpoint.
   - `test_503_when_redis_unavailable`: points the limiter at a dead port, asserts generic 503 with no Redis failure details in the response body.
   - `test_auth_before_rate_limit_real_redis`: asserts unauthenticated requests return 401 before rate limiting is checked.
   - `test_no_credential_leakage_on_503`: asserts no Redis connection strings, credentials, or internal details appear in the 503 response body.
   - Tests clear the cached limiter and settings between runs. Local developers without Redis can opt out via `NEUROSCAN_SKIP_REDIS_INTEGRATION=1`; CI never sets this variable, so the tests are mandatory and never silently skipped.
3. `RedisRateLimiter.check()` exception handling preserved as-is: specific `RedisError` catch with fail-closed 503. No blanket `except Exception` added, as no reproducible failure demonstrated the need. The existing `test_backend_failure_raises_503_without_leak` unit test covers the `RedisError` → 503 path with `FakeRedis.should_raise`.
4. Documentation updated: `memory.md` (this record) and `tasks.md` (P8 entry) record production Redis requirements: private/internal network, authentication or ACLs (`requirepass`), no unnecessary host port exposure, and TLS where required by the deployment environment.

**Reason:** rules.md #17/#18/#20; PRD G7 (fail closed); design.md §12 (failures explicit); SECURITY_AUDIT_REPORT.md F-03 remediation.

**Affected files:**
- `.github/workflows/backend-tests.yml` (Redis service, readiness check, `REDIS_URL` env)
- `backend/tests/test_rate_limiting.py` (new `TestRedisIntegration` class)
- `backend/memory.md` (this record), `backend/tasks.md` (P8 entry)

**Status:** Implemented; existing mocked unit tests pass (18 passed locally without Redis). Real-Redis integration tests run in CI against the provisioned `redis:7-alpine` service. CI success not yet claimed — pending workflow run on the feature branch.

**Verification:**
- `test_rate_limiting.py` parses cleanly; YAML workflow parses cleanly.
- Existing mocked rate-limiting tests: 18 passed (locally, without Redis).
- New `TestRedisIntegration`: 3 tests pass without available Redis (503, auth-before, no-leakage); `test_429_when_limit_exceeded_real_redis` requires live Redis and runs in CI.
- `git diff --check`: clean.

### D-048 --- SG3-F6-D: Redis authentication enforcement (remediates F-03)

**Date:** 2026-10-16

**Context:** Security finding F-03 (confirmed): Redis is published on host port 6379 without authentication in `docker-compose.yml:24`; `REDIS_URL` defaults to `redis://localhost:6379/0` with no credentials in `config.py:34`; no validation exists for Redis authentication (unlike `DATABASE_URL`, `MINIO_ACCESS_KEY`, `FIREBASE_PROJECT_ID` which are required fields). Production deployments that don't override `REDIS_URL` inherit an unauthenticated connection.

**Decision:**
1. `backend/app/core/config.py` (updated): add `RATE_LIMIT_REDIS_REQUIRE_AUTH: bool = False` setting. When `True`, a `@model_validator(mode="after")` (`validate_redis_auth_configuration`) requires `REDIS_URL` to use `redis://` or `rediss://` scheme with a non-empty password component. Validation errors never include credentials or connection details. Default `False` preserves local development and CI workflows that use the unauthenticated `redis://localhost:6379/0` default.
2. `backend/app/core/rate_limiting.py`: no changes needed. `redis.Redis.from_url()` already supports authenticated URLs (`redis://user:pass@host:port/db`). `AuthenticationError` is a subclass of `RedisError`, already caught by the existing `except RedisError` block → generic 503, no detail leaked.
3. `.github/workflows/backend-tests.yml` (updated): provision `redis:7-alpine` with `--requirepass redis_ci_test_password_99`; health check uses `redis-cli -a <password> --no-auth-warning ping` (password never printed in output); set `REDIS_URL: redis://:redis_ci_test_password_99@localhost:6379/0` and `RATE_LIMIT_REDIS_REQUIRE_AUTH: "true"`.
4. `backend/tests/test_rate_limiting.py` (updated): add authenticated Redis tests to `TestRedisIntegration`:
   - `test_429_when_limit_exceeded_authenticated_redis`: correct credentials → requests succeed; exceeding limit → 429 + `Retry-After`.
   - `test_503_when_redis_wrong_password`: wrong credentials → generic 503, no credential leakage in body.
   - `test_auth_before_rate_limit_authenticated_redis`: unauthenticated → 401 before rate limiting.
   - `test_no_credential_leakage_on_wrong_password_503`: no Redis credentials or connection details in 503 body.
   - Update `test_429_when_limit_exceeded_real_redis` and `test_cleanup_preserves_unrelated_keys` to use authenticated URL when enforcement is enabled.
   - Add `TestRateLimitConfig` unit tests: default off (unauthenticated URL accepted), required URL credentials when on, `rediss://` scheme accepted, empty credentials rejected.
   - `TestRedisIntegration` is mandatory in CI (never silently skipped); local opt-out via `NEUROSCAN_SKIP_REDIS_INTEGRATION=1`.
5. `backend/tasks.md` and `backend/memory.md` (updated): document production Redis requirements — enforce authentication server-side, private/internal network, no unnecessary host port exposure, TLS where required by deployment environment.

**Reason:** rules.md #14 (no hard-coded production credentials), #17/#18/#20; PRD G7 (fail closed); PRD NFR "Free of hard-coded credentials"; SECURITY_AUDIT_REPORT.md F-03 remediation.

**Affected files:**
- `backend/app/core/config.py` (`RATE_LIMIT_REDIS_REQUIRE_AUTH`, `validate_redis_auth_configuration`)
- `.github/workflows/backend-tests.yml` (authenticated Redis service, `REDIS_URL`, `RATE_LIMIT_REDIS_REQUIRE_AUTH`)
- `backend/tests/test_rate_limiting.py` (new authenticated tests and config validation tests)
- `backend/memory.md` (this record), `backend/tasks.md` (P8 entry)

**Status:** Implemented; focused non-DB suite passes. Real-Redis authenticated integration tests run in CI against the provisioned `redis:7-alpine` service with `--requirepass`. CI success not yet claimed — pending workflow run on the feature branch.

**Verification:**
- `config.py`, `rate_limiting.py`, `test_rate_limiting.py`, workflow YAML all parse cleanly.
- Non-DB backend suite: tests pass locally (without Redis).
- New config validation tests: 5 passed (default off, requires password when on, accepts authenticated URL, accepts rediss://, rejects empty credentials).
- `git diff --check`: clean.

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