# NeuroScan AI --- Technical Design

**Status:** Authoritative source of truth\
**Scope:** ARMT-GAN research prototype and its application integration

## 1. Design Objectives

The design prioritizes scientific traceability over feature count.

Every result must have a computational origin.

Every experiment must have a reproducible configuration.

Every UI value must map to an API field or explicit static metadata.

## 2. Model Contract

``` text
Input:
  Tensor shape = [B, 4, H, W]
  Channel order = T1, T1ce, T2, FLAIR

Output:
  Tensor shape = [B, 1, H, W]
  Range = [0, 1]
  Meaning = tumor probability
```

The current training default is 224×224.

The inference path must use the same preprocessing contract.

## 3. Training Design

### 3.1 Dataset

BraTS patient directories are discovered strictly.

A usable patient requires all four modalities.

A usable training patient requires a segmentation label.

Only tumor-containing slices enter the current dataset.

### 3.2 Split

Split patients before slice expansion.

Never split individual slices across partitions.

Record:

-   Dataset identifier.
-   Dataset version.
-   Patient IDs or hashed IDs.
-   Seed.
-   Validation fraction.
-   Image size.
-   Preprocessing configuration.

### 3.3 Loss

The current prototype uses:

``` text
L_D = 0.5 * (BCE(D(real)) + BCE(D(fake)))

L_G = 100 * L1(fake_mask, real_mask)
      + BCE(D(fake_mask), 1)
```

Future loss changes require an experiment record.

Do not replace the baseline silently.

## 4. Evaluation Design

Mandatory segmentation metrics:

-   Dice coefficient.
-   Intersection over Union.
-   Precision.
-   Sensitivity.

Secondary metrics may include specificity and Hausdorff distance.

Secondary metrics cannot replace mandatory metrics.

Metrics must be computed on patient-disjoint evaluation data.

## 5. Robustness Design

FGSM and PGD are separate evaluation experiments.

They must not alter the baseline model silently.

Each experiment records:

-   Attack method.
-   Epsilon.
-   Step size.
-   Iteration count.
-   Norm.
-   Seed.
-   Dataset split.
-   Clean metric.
-   Adversarial metric.

``` mermaid
flowchart LR
    A["Clean MRI"] --> B["ARMT-GAN"]
    B --> C["Clean segmentation"]
    A --> D["FGSM / PGD"]
    D --> E["Perturbed MRI"]
    E --> B
    B --> F["Adversarial segmentation"]
    C --> G["Robustness delta"]
    F --> G
```

## 6. XAI Design

The current implementation is gradient-based input saliency.

It must not be labelled Grad-CAM.

Target design:

-   Use an explicitly implemented attribution method.
-   Record method name.
-   Record target tensor.
-   Record normalization.
-   Record model checkpoint.
-   Preserve raw attribution before visualization.

XAI is explanatory evidence.

XAI is not proof of clinical correctness.

## 7. Inference Design

Inference must:

1.  Validate modality completeness.
2.  Validate file types.
3.  Apply training-equivalent preprocessing.
4.  Load a named checkpoint.
5.  Run deterministic inference where practical.
6.  Generate raw prediction.
7.  Generate derived visualization.
8.  Store traceable artifacts.
9.  Return evidence-backed metadata.

No heuristic confidence modification is allowed.

## 8. API Design

### Upload

Target request:

``` text
POST /api/v1/scans/upload
multipart:
  t1
  t1ce
  t2
  flair
```

### Status

``` text
GET /api/v1/scans/status/{scan_id}
```

### Results

``` text
GET /api/v1/scans/results/{scan_id}
```

Results must include:

-   Scan ID.
-   Processing status.
-   Model version.
-   Preprocessing version.
-   Mask artifact.
-   XAI artifact.
-   Metrics.
-   Provenance.
-   Disclaimer.

## 9. Database Design

Minimum scientific entities:

``` mermaid
erDiagram
    STUDY ||--o{ MODALITY_FILE : contains
    STUDY ||--o| PREDICTION : produces
    PREDICTION ||--o{ ARTIFACT : creates
    MODEL_VERSION ||--o{ PREDICTION : generated_by

    STUDY {
      string id PK
      string status
      datetime created_at
    }

    MODALITY_FILE {
      int id PK
      string study_id FK
      string modality
      string object_path
    }

    PREDICTION {
      int id PK
      string study_id FK
      string model_version
      float dice
      float iou
    }

    ARTIFACT {
      int id PK
      string prediction_id FK
      string type
      string object_path
    }

    MODEL_VERSION {
      string id PK
      string checkpoint_path
      string config_hash
    }
```

## 10. Frontend Design

The frontend consumes API truth.

Mock data is permitted only in isolated UI development.

Mock data must never merge silently into live results.

The results dashboard must remove unsupported claims.

The current UI contains telemetry, DICOM, WHO-grade, and confidence
concepts.

Those fields require backend evidence before live display.

## 11. Report Design

Reports must include:

-   Study identifier.
-   Model version.
-   Input modalities.
-   Segmentation visualization.
-   XAI visualization.
-   Computed metrics where applicable.
-   Timestamp.
-   Research-prototype disclaimer.

Reports must not call predictions diagnoses.

Reports must not assign WHO grade without validated classification
evidence.

## 12. Error Handling

Errors must preserve scan state.

Failures must be explicit.

Partial artifacts must be marked incomplete.

A failed inference must never return stale successful results.

## 13. Testing Strategy

``` mermaid
flowchart TB
    A["Unit tests"] --> B["Contract tests"]
    B --> C["Integration tests"]
    C --> D["Model regression tests"]
    D --> E["End-to-end acceptance"]
```

Critical tests cover:

-   modality ordering.
-   tensor shapes.
-   preprocessing parity.
-   checkpoint loading.
-   API schemas.
-   storage paths.
-   failure states.
-   no fabricated metrics.
-   patient-disjoint evaluation.
