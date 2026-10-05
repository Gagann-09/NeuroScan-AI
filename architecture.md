# NeuroScan AI --- Architecture

**Status:** Authoritative source of truth\
**Project:** NeuroScan AI / ARMT-GAN only\
**Baseline:** `main` @ `a0bbe53dd63834cea7349cd390c8ede8b915f52d`

## 1. Architectural Principle

The model pipeline is the scientific core.

Application services wrap the model.

The frontend never defines scientific truth.

The database never invents model outputs.

Storage preserves evidence, not conclusions.

## 2. Target Architecture

``` mermaid
flowchart TB
    subgraph DATA["Data Layer"]
      A["BraTS patients"]
      B["NIfTI modalities"]
      C["Verified tumor masks"]
      D["Patient-level split"]
    end

    subgraph ML["ARMT-GAN Layer"]
      E["Preprocessing"]
      F["ARMTGenerator2D<br/>4 → 64 → 128 → 256 → 512"]
      G["Decoder + skip connections"]
      H["1-channel sigmoid mask"]
      I["ARMTDiscriminator2D<br/>MRI + mask"]
      J["Segmentation loss"]
      K["Adversarial loss"]
    end

    subgraph ROB["Robustness Layer"]
      L["FGSM"]
      M["PGD"]
      N["Robustness evaluation"]
    end

    subgraph APP["Application Layer"]
      O["Inference service"]
      P["FastAPI"]
      Q["PostgreSQL"]
      R["MinIO"]
      S["XAI"]
      T["Report generator"]
      U["Next.js"]
    end

    A --> B --> E
    C --> E
    D --> E
    E --> F --> G --> H
    E --> I
    H --> K
    I --> K
    J --> K
    K --> F
    F --> L --> N
    F --> M --> N
    H --> O
    O --> P
    O --> S
    O --> R
    O --> Q
    S --> R
    T --> R
    P --> U
```

## 3. Layer Responsibilities

### 3.1 Data

Owns dataset discovery, validation, splits, and provenance.

### 3.2 Preprocessing

Owns normalization, resizing, slicing, and tensor construction.

### 3.3 Model

Owns ARMT-GAN architecture and forward computation.

### 3.4 Training

Owns optimization, losses, checkpoints, and experiment configuration.

### 3.5 Robustness

Owns FGSM, PGD, attack budgets, and robustness metrics.

### 3.6 Inference

Owns loading validated checkpoints and executing inference.

### 3.7 Application

Owns API, persistence, artifact storage, and orchestration.

### 3.8 Presentation

Owns UI rendering and user interaction.

The presentation layer must not calculate scientific metrics.

## 4. Current ARMT-GAN Architecture

``` mermaid
flowchart LR
    A["4 MRI channels"] --> B["DoubleConv 64"]
    B --> C["MaxPool"]
    C --> D["DoubleConv 128"]
    D --> E["MaxPool"]
    E --> F["DoubleConv 256"]
    F --> G["MaxPool"]
    G --> H["Bottleneck 512"]
    H --> I["Up 256 + skip"]
    I --> J["Up 128 + skip"]
    J --> K["Up 64 + skip"]
    K --> L["1×1 Conv"]
    L --> M["Sigmoid tumor mask"]
```

The current generator is a lightweight U-Net variant.

The current discriminator is a conditional PatchGAN.

The current model does not contain a Transformer.

Transformer fusion must never be claimed unless implemented.

## 5. Scientific Data Contract

Input channels are ordered:

1.  T1
2.  T1ce
3.  T2
4.  FLAIR

Target masks are binary:

-   `0` = background.
-   `1` = tumor.

BraTS non-zero labels may be collapsed to binary for segmentation.

Patient identity must remain attached during splitting.

## 6. Application Contract

``` mermaid
sequenceDiagram
    participant UI as Next.js
    participant API as FastAPI
    participant DB as PostgreSQL
    participant OBJ as MinIO
    participant ML as ARMT-GAN

    UI->>API: Upload validated four-modality study
    API->>OBJ: Store source artifacts
    API->>DB: Create scan record
    API->>ML: Run inference
    ML-->>API: Mask + metrics + metadata
    API->>OBJ: Store derived artifacts
    API->>DB: Store traceable result
    UI->>API: Poll status
    UI->>API: Request results
    API-->>UI: Evidence-backed result payload
```

## 7. Mandatory Contract Correction

Current repository state is inconsistent.

The frontend constructs four modality uploads.

The backend endpoint currently accepts one `UploadFile`.

The inference path also expects three RGB channels.

The trained generator expects four channels.

This must be resolved before production-like integration.

The target contract is four modality inputs end-to-end.

## 8. Storage Architecture

PostgreSQL stores metadata and relational state.

MinIO stores source and derived image artifacts.

Model weights remain versioned separately.

Reports reference stored evidence.

Raw medical data must not be embedded in database rows.

## 9. Architectural Anti-Patterns

Forbidden:

-   UI-generated medical metrics.
-   Hard-coded model confidence.
-   Fake WHO grading.
-   Silent modality substitution.
-   Mid-slice-only claims for 3D studies.
-   Destructive database initialization.
-   Hard-coded passwords.
-   Undocumented model architecture changes.
-   Hidden fallback data.
-   Metrics without dataset provenance.

## 10. Architecture Evolution Rule

Any change affecting model input, model output, preprocessing,
evaluation, API schemas, or storage requires updates to all affected
source-of-truth files before implementation.
