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

-   [ ] Define threat model.
-   [ ] Define norm.
-   [ ] Define epsilon values.
-   [ ] Define PGD step size.
-   [ ] Define iteration counts.
-   [ ] Implement FGSM.
-   [ ] Implement PGD.
-   [ ] Evaluate clean baseline.
-   [ ] Evaluate adversarial predictions.
-   [ ] Report robustness degradation.
-   [ ] Prevent attack data leakage.

### Gate

Clean and adversarial metrics are independently reproducible.

------------------------------------------------------------------------

## P5 --- XAI Integrity

### Goal

Make explanations scientifically honest.

### Tasks

-   [ ] Rename current gradient-saliency implementation.
-   [ ] Decide whether true Grad-CAM is required.
-   [ ] Implement selected attribution method.
-   [ ] Record attribution provenance.
-   [ ] Test attribution shape.
-   [ ] Test attribution normalization.
-   [ ] Compare attribution with segmentation region.
-   [ ] Remove unsupported "clinical explanation" language.

### Gate

XAI method name matches implementation.

------------------------------------------------------------------------

## P6 --- API and Storage Integration

### Goal

Align backend with the four-modality model.

### Tasks

-   [ ] Replace single-file upload contract.
-   [ ] Accept T1/T1ce/T2/FLAIR explicitly.
-   [ ] Store each source modality.
-   [ ] Create study-level identity.
-   [ ] Store model version.
-   [ ] Store preprocessing version.
-   [ ] Store result provenance.
-   [ ] Remove `drop_all()` startup behavior.
-   [ ] Add migration strategy.
-   [ ] Remove hard-coded secrets.
-   [ ] Replace fake clinical fields.
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

-   [ ] Remove default credentials.
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
