# NeuroScan AI --- Product Requirements Document

**Status:** Authoritative source of truth\
**Project:** NeuroScan AI only\
**Core model:** ARMT-GAN\
**Baseline audited:** 2026-10-05\
**Repository:** `Gagann-09/NeuroScan-AI`\
**Default branch audited:** `main`\
**Baseline commit:** `a0bbe53dd63834cea7349cd390c8ede8b915f52d`

> This document is project-specific. Do not import requirements from
> JARVIS, PQ-TLS, or any other project.

## 1. Product Definition

NeuroScan AI is a research prototype for brain-tumor MRI segmentation.

Its central technical artifact is ARMT-GAN.

The system must produce reproducible tumor segmentation.

The system may provide visual explanations and reports.

The system is not a clinical diagnostic device.

The system must never claim clinical validation without evidence.

## 2. Locked Prototype Scope

### In scope

-   BraTS multimodal MRI data.
-   T1, T1ce, T2, and FLAIR modalities.
-   Patient-level train/validation/test separation.
-   2D axial segmentation.
-   Real BraTS tumor annotations.
-   Lightweight U-Net-based ARMT-GAN generator.
-   Conditional PatchGAN discriminator.
-   Segmentation plus adversarial training.
-   Dice, IoU, precision, and sensitivity evaluation.
-   Reproducible checkpoints.
-   Controlled FGSM and PGD robustness experiments.
-   Explainability evaluation.
-   FastAPI inference integration.
-   PostgreSQL metadata persistence.
-   MinIO artifact storage.
-   Next.js frontend integration.
-   PDF report generation.
-   Prototype-level authentication and access controls.

### Explicitly out of scope for the baseline

-   Clinical diagnosis.
-   Automated WHO tumor grading.
-   Treatment recommendation.
-   Clinical deployment claims.
-   Clinical validation.
-   Regulatory certification.
-   Autonomous radiologist replacement.
-   3D segmentation as the baseline model.
-   Synthetic labels.
-   Fabricated confidence scores.
-   Fabricated performance metrics.

## 3. Current Baseline Reality

The repository currently contains a lightweight 2D ARMT-GAN.

The generator accepts four MRI channels.

The generator returns one sigmoid tumor-probability mask.

The discriminator consumes MRI plus mask channels.

The training script performs adversarial segmentation training.

The current backend inference path is not contract-compatible.

The current backend accepts one uploaded file.

The current frontend prepares four BraTS modality files.

The current inference code reduces NIfTI input to one middle slice.

The current inference tensor uses RGB-like three channels.

The trained model expects four channels.

The current XAI implementation is input-gradient saliency.

It is not true Grad-CAM despite its current function name.

The current confidence score contains a heuristic boost.

The current WHO grade is area-threshold logic.

Those outputs are not scientifically valid clinical classifications.

## 4. Product Goals

### G1 --- Scientific correctness

Every model result must originate from observed computation.

### G2 --- Reproducibility

Training and evaluation must be rerunnable from recorded configuration.

### G3 --- Model integrity

ARMT-GAN must remain the named project model.

### G4 --- Contract integrity

Frontend, backend, preprocessing, model, and storage schemas must agree.

### G5 --- Robustness research

FGSM and PGD must be evaluated separately from ordinary segmentation.

### G6 --- Explainability integrity

XAI must describe model behavior without implying clinical proof.

### G7 --- Safe prototype behavior

The application must fail closed when required inputs are invalid.

## 5. Functional Requirements

  -----------------------------------------------------------------------
  ID                      Requirement             Acceptance condition
  ----------------------- ----------------------- -----------------------
  FR-01                   Four-modality input     T1/T1ce/T2/FLAIR are
                                                  explicitly represented

  FR-02                   Real labels             Segmentation uses
                                                  verified BraTS labels

  FR-03                   Patient split           No patient crosses
                                                  evaluation partitions

  FR-04                   Determinism             Seed and preprocessing
                                                  configuration are
                                                  recorded

  FR-05                   Checkpoints             Model weights include
                                                  architecture metadata

  FR-06                   Segmentation metrics    Dice and IoU are
                                                  mandatory

  FR-07                   Robustness              FGSM and PGD use
                                                  documented budgets

  FR-08                   XAI                     Attribution method is
                                                  explicitly named

  FR-09                   API contract            Upload and result
                                                  schemas are versioned

  FR-10                   Storage                 Source and derived
                                                  artifacts have
                                                  traceable IDs

  FR-11                   Reports                 Reports contain
                                                  evidence and
                                                  disclaimers

  FR-12                   Clinical claims         Clinical language is
                                                  prohibited without
                                                  validation
  -----------------------------------------------------------------------

## 6. Non-Functional Requirements

-   Reproducible.
-   Testable.
-   Auditable.
-   Deterministic where practical.
-   Modular.
-   Explicitly typed.
-   Dataset-version aware.
-   Model-version aware.
-   Artifact-version aware.
-   Safe against destructive startup behavior.
-   Free of hard-coded credentials.

## 7. System Flow

``` mermaid
flowchart LR
    A["BraTS MRI<br/>T1/T1ce/T2/FLAIR"] --> B["Validated preprocessing"]
    B --> C["2D axial slice dataset"]
    C --> D["ARMT-GAN Generator"]
    C --> E["PatchGAN Discriminator"]
    D --> F["Tumor probability mask"]
    E --> D
    F --> G["Evaluation"]
    G --> H["FGSM / PGD robustness"]
    F --> I["XAI"]
    F --> J["Artifact storage"]
    I --> J
    J --> K["FastAPI"]
    K --> L["Next.js UI"]
    J --> M["PDF report"]
```

## 8. Acceptance Gates

A phase is complete only when its exit criteria pass.

No phase may silently redefine another phase.

No metric may be reported before its evaluation protocol exists.

No deployment work may bypass scientific validation.

## 9. Final Success Definition

NeuroScan AI is complete when:

1.  ARMT-GAN training is reproducible.
2.  Evaluation uses patient-disjoint data.
3.  Segmentation metrics are independently reproducible.
4.  FGSM and PGD experiments are reproducible.
5.  XAI outputs are correctly described.
6.  API and frontend contracts match.
7.  Storage is traceable and non-destructive.
8.  Reports contain only evidence-backed claims.
9.  Automated tests cover critical contracts.
10. The project remains explicitly a research prototype.
