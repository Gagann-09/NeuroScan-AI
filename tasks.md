# NeuroScan AI --- Master Task Plan

**Status:** Authoritative execution plan\
**Project:** NeuroScan AI / ARMT-GAN only\
**Baseline:** 2026-10-05 repository audit

## 0. Phase Control

Only one phase may be active at a time.

Every phase has a gate.

A failed gate blocks dependent phases.

No phase may silently change scientific scope.

``` mermaid
flowchart LR
    P0["P0 Baseline Lock"] --> P1["P1 Data Contract"]
    P1 --> P2["P2 ARMT-GAN Reproducibility"]
    P2 --> P3["P3 Scientific Evaluation"]
    P3 --> P4["P4 FGSM/PGD Robustness"]
    P4 --> P5["P5 XAI Integrity"]
    P5 --> P6["P6 API + Storage"]
    P6 --> P7["P7 Frontend Integration"]
    P7 --> P8["P8 Security + Reliability"]
    P8 --> P9["P9 End-to-End Validation"]
    P9 --> P10["P10 Release Freeze"]
```

## P0 --- Baseline Lock

### Goal

Freeze the current repository as the starting reference.

### Tasks

-   [ ] Record baseline commit.
-   [ ] Verify repository tree.
-   [ ] Record current model architecture.
-   [ ] Record current API behavior.
-   [ ] Record known defects.
-   [ ] Add six source-of-truth files.
-   [ ] Confirm no other project rules apply.

### Gate

Six documents exist and agree.

------------------------------------------------------------------------

## P1 --- Data Contract

### Goal

Make the four-modality dataset contract explicit and executable.

### Tasks

-   [x] Validate BraTS directory discovery.
-   [x] Validate T1/T1ce/T2/FLAIR ordering.
-   [x] Validate segmentation labels.
-   [x] Implement patient-level split persistence.
-   [x] Record dataset provenance.
-   [x] Remove ambiguous preprocessing paths.
-   [x] Add preprocessing parity tests.

### Gate

The same preprocessing contract works for training and inference. **VERIFIED: All 9 parity tests pass.**

------------------------------------------------------------------------

## P2 --- ARMT-GAN Reproducibility

### Goal

Produce a trustworthy baseline checkpoint.

### Tasks

-   [x] Freeze generator architecture.
-   [x] Freeze discriminator architecture.
-   [x] Freeze loss configuration.
-   [x] Freeze seed policy.
-   [x] Record training configuration.
-   [x] Train baseline (synthetic data smoke test).
-   [x] Save checkpoint metadata.
-   [x] Re-run evaluation from checkpoint.
-   [x] Verify deterministic behavior where practical.

### Gate

Two controlled runs produce traceable, reproducible results. **VERIFIED: Synthetic data runs produce identical weights and metrics.**

> **Note:** Full BraTS reproducibility run pending until BraTS data is available. Mechanism is verified.

------------------------------------------------------------------------

## P3 --- Scientific Evaluation

### Goal

Measure segmentation performance correctly.

### Tasks

-   [x] Build patient-disjoint test split.
-   [x] Implement Dice.
-   [x] Implement IoU.
-   [x] Implement precision.
-   [x] Implement sensitivity.
-   [x] Add per-patient metrics.
-   [x] Add aggregate statistics.
-   [x] Preserve raw prediction artifacts.
-   [x] Generate baseline evaluation report.
-   [x] Freeze evaluation protocol.
-   [x] Add metric unit tests.
-   [x] Add split/leakage tests.
-   [x] Verify evaluation reproducibility.

### Gate

Metrics are reproducible from stored predictions. **VERIFIED: 26 tests pass; two evaluation runs produce identical metrics; raw prediction artifacts saved as .npy files.**

> **Note:** Full BraTS scientific evaluation pending real BraTS data. Mechanism is verified on synthetic data.

------------------------------------------------------------------------

## P4 --- Adversarial Robustness

### Goal

Evaluate ARMT-GAN under FGSM and PGD.

### Tasks

-   [x] Define threat model.
-   [x] Define norm.
-   [x] Define epsilon values.
-   [x] Define PGD step size.
-   [x] Define iteration counts.
-   [x] Implement FGSM.
-   [x] Implement PGD.
-   [x] Evaluate clean baseline.
-   [x] Evaluate adversarial predictions.
-   [x] Report robustness degradation.
-   [x] Prevent attack data leakage.

### Gate

Clean and adversarial metrics are independently reproducible. **VERIFIED: 23 attack tests pass; clean/FGSM/PGD evaluation pipeline runs on synthetic data; metrics and deltas computed.**

> **Note:** Full BraTS robustness evaluation pending real BraTS data. Mechanism is verified on synthetic data.

------------------------------------------------------------------------

## P5 --- XAI Integrity

### Goal

Make explanations scientifically honest.

### Tasks

-   [x] Rename current gradient-saliency implementation.
-   [x] Decide whether true Grad-CAM is required. (Decision: Not required; gradient saliency is acceptable when correctly named)
-   [x] Implement selected attribution method. (Kept gradient-based input saliency with explicit documentation)
-   [x] Record attribution provenance. (Added XAIProvenance dataclass with method, target, normalization, aggregation, model checkpoint, version, shapes)
-   [x] Test attribution shape. (test_output_shape, test_batch_size_greater_than_one, test_different_spatial_sizes)
-   [x] Test attribution normalization. (test_output_range, test_normalization_non_flat, test_constant_output_model, test_constant_nonzero_output_model)
-   [x] Compare attribution with segmentation region. (Added compute_segmentation_alignment with mean inside/outside tumor region)
-   [x] Remove unsupported "clinical explanation" language. (Not in XAI code; reporting.py changes deferred to P6/P7)

### Gate

XAI method name matches implementation. **VERIFIED: All 28 XAI tests pass; method is "gradient-based input saliency"; provenance recorded; raw artifact preserved; alignment metric computed.**

------------------------------------------------------------------------

## P6 --- API and Storage Integration

### Goal

Align backend with the four-modality model.

### Tasks

-   [ ] Replace single-file upload contract.
-   [ ] Accept T1/T1ce/T2/FLAIR explicitly.
-   [ ] Store each source modality.
-   [ ] Create study-level identity.
-   [x] Store model version. (Batch 3A: ModelVersion table + Prediction FK; Batch 3B: wired)
-   [x] Store preprocessing version. (Batch 5B: added preprocessing_version column to ModelVersion; separated from config_hash; deterministic hash from PreprocessingConfig)
-   [x] Store result provenance. (Batch 3A: Artifact table + Prediction dice/iou + Scan xai_raw_path; Batch 3B: wired)
-   [x] Remove `drop_all()` startup behavior. (Batch 1)
-   [x] Add migration strategy. (Batch 1)
-   [x] Remove hard-coded secrets. (Batch 2)
-   [x] Replace fake clinical fields. (Batch 5A: removed who_grade, anomaly_area_cm2, confidence_score; renamed to max_tumor_probability; updated report title and model version provenance)
-   [ ] Make failures transactional.

### Gate

A real four-modality study can complete end-to-end.

------------------------------------------------------------------------

## P7 --- Frontend Integration

### Goal

Make the UI display live backend evidence.

### Tasks

-   [ ] Remove silent mock-result merging.
-   [ ] Match upload contract.
-   [ ] Match result schema.
-   [ ] Remove unsupported telemetry.
-   [ ] Remove unsupported WHO grading.
-   [ ] Remove fake ensemble confidence.
-   [ ] Render real mask artifact.
-   [ ] Render real XAI artifact.
-   [ ] Render real report link.
-   [ ] Show research-prototype disclaimer.

### Gate

Every scientific UI value maps to a backend field.

------------------------------------------------------------------------

## P8 --- Security and Reliability

### Goal

Make the prototype safe for controlled demonstration.

### Tasks

-   [x] Remove default credentials. (SG3-F5-B: deleted obsolete `backend/init_system.py`, which hardcoded the default administrator password and bootstrapped a password-based `Admin` user against nonexistent `password_hash`/`role` columns; identity is Firebase UID-based and no supported workflow depends on the script. Added `TestAdminCredentialRemoval` regression tests in `backend/tests/test_config_security.py`. Full backend suite on disposable PostgreSQL 16: 256 passed, 0 failed.)
-   [ ] Validate upload types.
-   [ ] Validate file size.
-   [ ] Validate modality completeness.
-   [ ] Restrict CORS.
-   [ ] Add authentication enforcement.
-   [ ] Add authorization checks.
-   [ ] Sanitize logs.
-   [ ] Add failure-state tests.
-   [ ] Add storage access tests.

### Gate

No critical security or destructive-data defects remain.

> **SG3-F3 verification note (2026-10-15):** NIfTI upload validation (extension,
> magic, size), integrity validation, resource bounds, and worker-side
> validation before inference are implemented and covered by focused regression
> tests (`backend/tests/test_nifti_safety_regression.py`, plus
> `tests/test_auth.py::TestUploadValidation`). A pre-existing regression was
> found and repaired: commit `9aa22c7` had deleted ~434 lines of worker helper
> functions from `backend/app/services/ai_tasks.py`. Known gaps: truncated
> payloads are only detected at read time (no true decompressed-byte limit, only
> header-based estimation). NIfTI-2 is intentionally not supported: validation is
> documented and tested as NIfTI-1-only (BraTS is NIfTI-1). Database-backed
> security tests could not be executed because no PostgreSQL instance is
> available. See `memory.md` D-036.
>
> **SG3-F3-I implementation note (2026-10-15):** The SG3-F3-H audit findings were
> implemented. Upload validation now streams gzip in bounded chunks through EOF
> (verifying the CRC32/ISIZE trailer) under a true 200 MiB decompressed-byte cap,
> rejects malformed/truncated/concatenated/trailing data, reads the NIfTI-1 magic
> at its real header offset (344), and checks dims/datatype/voxel-count/
> `vox_offset`/payload-size consistency using `vox_offset` (so real BraTS
> `vox_offset != 352` files and uncompressed `.nii` are accepted). The
> preprocessing validator now bounds every declared dimension and enforces a
> float32-aware allocation estimate before `get_fdata`, sharing bounds via
> `_validate_nifti_shape_and_dtype`. The worker preserves the uploaded container
> so `.nii` and `.nii.gz` both load. Regression tests grew from 22 to 47, incl. a
> dataset-gated real-BraTS check. Remaining limitation: no real compressed BraTS
> `.nii.gz` sample or live MinIO here to prove compressed real-data compatibility.
> See `memory.md` D-037.
>
> **SG3-F4-A verification note (2026-10-15):** The three database-backed suites
> (`test_scan_claim.py`, `test_artifact_lifecycle.py`,
> `test_scan_claim_integration.py`) now run and pass against an isolated
> PostgreSQL 16 instance (dedicated `neuroscan_test` database, migrations
> 001 -> 006 via `alembic upgrade head`; no dev/prod data touched). Confirmed and
> fixed: (a) `process_scan_task` leaked its primary DB session on every
> non-exception exit -- `db.close()` ran only in the `except` block, so the early
> "claim failed" return kept a `FOR UPDATE` row lock open and deadlocked the
> suite; the session is now closed in a `finally`. (b) The triplicated
> `generate_segmentation_report` call (with a doubled object-name block) was
> reduced to one; tracing showed the only side effect was overwriting the same
> PDF, so no duplicate artifacts/rows were produced. (c) Two pre-existing
> test-harness regressions were repaired: `TestUploadCompensation` now
> authenticates (the S3 batch `823e84f` added `get_current_user` to `/upload`
> without updating the helper) and its NIfTI fixture is large enough to clear
> the 1 KB minimum-size guard added in `9f11037`. Added `TestSessionCleanup`
> (success + failure cleanup). Full backend suite `tests/`: 254 passed.
> See `memory.md` D-038.
>
> **SG3-F4-B CI integration note (2026-10-15):** No CI provider was configured
> (no `.github/`, GitLab, Jenkins, Azure, CircleCI, or Travis). Because the only
> remote is GitHub, the smallest compatible workflow is GitHub Actions:
> `.github/workflows/backend-tests.yml` provisions an isolated, disposable
> `postgres:16-alpine` service container (fresh per run) with CI-only credentials
> and a dedicated `neuroscan_ci` database, waits for readiness, recreates the
> `public` schema, applies `alembic upgrade head`, and runs the complete backend
> suite `python -m pytest tests/ -v` (no SQLite substitution, no database tests
> skipped, no `.env` read). Three CI blockers were fixed: `requirements.txt` was
> unsatisfiable (`opencv-python-headless==5.0.0.93` requires `numpy>=2` while the
> stack pins `numpy==1.26.3`/`torch==2.2.0`), `alembic` was undeclared, and the
> test runner (`pytest`/`pytest-asyncio`) was undeclared. OpenCV is corrected to
> `4.11.0.86` and `alembic==1.13.2` added; test deps live in
> `backend/requirements-dev.txt`. Verify: local disposable-PostgreSQL run
> (migrations 001->006; full `tests/` 254 passed; the three DB modules 33 passed;
> resolution succeeds for CPython 3.11/Linux). See `memory.md` D-039. Unproven:
> the workflow itself has not run on a GitHub-hosted runner.
>
> **SG3-F4-C hosted CI verification note (2026-10-15):** The workflow was
> exercised on the GitHub-hosted runner. The repository is public, so the run
> result is readable through the unauthenticated GitHub REST API: push run
> `38769341345` (head `f39f070`) reached the final step and **failed** --
> `python -m pytest tests/ -v` returned exit code 2 (a collection error), while
> every earlier step (checkout, Python 3.11, dependency install, PostgreSQL
> readiness, schema recreate, `alembic upgrade head`) succeeded. Reproducing the
> workflow verbatim against the exact pinned stack (Python 3.11.17) and a
> disposable PostgreSQL 16 container exposed two blockers: (a) `httpx` was
> undeclared, so `fastapi.testclient` could not be imported and `test_auth.py`
> aborted collection (exit 2); (b) five `TestUploadValidation` tests asserted 401
> but never mocked Firebase, so the real `auth.verify_id_token` raised
> `DefaultCredentialsError` (no ADC on a clean runner), which is not a
> `FirebaseError` and escaped as a 500. Fix: pin `httpx==0.26.0` in
> `backend/requirements-dev.txt` (newest release compatible with Starlette
> 0.36.3's `app=` constructor kwarg) and add a class-scoped autouse fixture to
> `TestUploadValidation` that makes authentication reject with 401 without ADC
> (assertions unchanged). Post-fix: `tests/test_auth.py` 28 passed; complete
> `tests/` 254 passed, 0 failed. See `memory.md` D-040. This resolves the D-039
> "has not run on a GitHub-hosted runner" limitation.

------------------------------------------------------------------------

## P9 --- End-to-End Validation

### Goal

Verify the complete NeuroScan pipeline.

### Tasks

-   [ ] Upload study.
-   [ ] Persist source artifacts.
-   [ ] Run preprocessing.
-   [ ] Run ARMT-GAN.
-   [ ] Generate mask.
-   [ ] Generate XAI.
-   [ ] Store artifacts.
-   [ ] Persist prediction metadata.
-   [ ] Render frontend.
-   [ ] Generate report.
-   [ ] Verify report values.
-   [ ] Verify failure handling.
-   [ ] Run complete automated test suite.

### Gate

A clean environment reproduces the documented workflow.

------------------------------------------------------------------------

## P10 --- Release Freeze

### Goal

Freeze a defensible research prototype.

### Tasks

-   [ ] Freeze model version.
-   [ ] Freeze dataset version.
-   [ ] Freeze evaluation scripts.
-   [ ] Freeze API schema.
-   [ ] Freeze frontend contract.
-   [ ] Freeze documentation.
-   [ ] Generate final limitations statement.
-   [ ] Archive experiment configuration.
-   [ ] Record final decision log.

### Gate

No undocumented behavior remains.

## Priority Order

1.  Scientific correctness.
2.  Data contract.
3.  Reproducibility.
4.  Evaluation.
5.  Robustness.
6.  XAI.
7.  API integration.
8.  Frontend.
9.  Security.
10. Presentation polish.

## Task Status Vocabulary

-   `LOCKED` --- requirement cannot change casually.
-   `READY` --- dependencies satisfied.
-   `ACTIVE` --- currently being implemented.
-   `BLOCKED` --- dependency or gate failed.
-   `VERIFY` --- implementation exists, testing pending.
-   `DONE` --- gate passed.
-   `DEFERRED` --- intentionally outside current phase.

## Definition of Done

A task is `DONE` only when code, tests, documentation, and provenance
agree.
