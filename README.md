# NeuroScan AI

**A research prototype for reproducible brain-tumour MRI segmentation
with ARMT-GAN.**

[Repository](https://github.com/Gagann-09/NeuroScan-AI) ·
[Issues](https://github.com/Gagann-09/NeuroScan-AI/issues) ·
[Source-of-truth documents](#project-documents) · [API
documentation](#api-and-interaction)

> \[!IMPORTANT\] NeuroScan AI is a **research prototype**, not a
> clinical diagnostic device. It has not been established as clinically
> validated. Model outputs, saliency visualisations, and reports must
> not be used for diagnosis or treatment decisions.

------------------------------------------------------------------------

## Contents

-   [What this project does](#what-this-project-does)
-   [How the system fits together](#how-the-system-fits-together)
-   [Current model contract](#current-model-contract)
-   [Application workflows](#application-workflows)
-   [Repository map](#repository-map)
-   [Getting started](#getting-started)
-   [Training and evaluation](#training-and-evaluation)
-   [API and interaction](#api-and-interaction)
-   [Security and data handling](#security-and-data-handling)
-   [Known limitations and project
    status](#known-limitations-and-project-status)
-   [Project documents](#project-documents)
-   [Contributing and reporting
    issues](#contributing-and-reporting-issues)

## What this project does

NeuroScan AI combines an experimental segmentation pipeline with a web
application.

-   **ARMT-GAN:** a lightweight 2D U-Net generator paired with a
    conditional PatchGAN discriminator.
-   **BraTS research pipeline:** uses four MRI modalities and real
    annotated tumour masks for segmentation experiments.
-   **Reproducible experiments:** records configuration, patient-level
    splits, checkpoint metadata, and evaluation outputs.
-   **Robustness research:** includes separate FGSM and PGD
    attack/evaluation code.
-   **Explainability:** provides gradient-based input saliency as an
    experimental attribution visualisation.
-   **Application layer:** includes a Next.js frontend, FastAPI backend,
    PostgreSQL metadata, MinIO artifact storage, Firebase
    authentication, and PDF research reports.

These are project components, not a claim that every end-to-end workflow
has passed acceptance testing. See [known limitations and project
status](#known-limitations-and-project-status).

## How the system fits together

``` mermaid
flowchart TD
    A["BraTS patient study"] --> B["T1 · T1ce · T2 · FLAIR"]
    B --> C["Shared preprocessing"]
    D["Real BraTS tumour annotation"] --> E["Binary target mask"]
    C --> F["2D axial slices"]
    E --> F
    F --> G["ARMT-GAN generator"]
    F --> H["Conditional PatchGAN discriminator"]
    G --> I["One-channel sigmoid tumour-probability mask"]
    H --> J["Adversarial training signal"]
    I --> K["Evaluation and raw prediction artifacts"]
    K --> L["Gradient-based input saliency"]
    K --> M["Research PDF report"]
    K --> N["Application API and UI"]
```

The model is the source of segmentation output. The UI must present
results and metadata; it must not invent scientific measurements.

## Current model contract

  Property                         Defined baseline
  -------------------------------- ---------------------------------------------
  Canonical model                  ARMT-GAN
  Generator                        Lightweight 2D U-Net variant
  Discriminator                    Conditional PatchGAN
  Input channel order              T1, T1ce, T2, FLAIR
  Input tensor                     `[batch, 4, height, width]`
  Default training image size      `224 × 224`
  Output                           One sigmoid tumour-probability mask
  Target labels                    Binary: background versus tumour
  Canonical segmentation dataset   BraTS
  Data split                       Patient-level; no patient across partitions
  XAI method                       Gradient-based input saliency, not Grad-CAM

The current training defaults are configured in
`backend/ai_pipeline/training/train_armt_gan.py`. The evaluation
protocol is defined separately in
`backend/ai_pipeline/evaluation/evaluate_armt_gan_baseline.py`. Always
record the exact configuration and checkpoint for a reported result.

## Application workflows

### BraTS / ARMT-GAN segmentation

The intended input is a complete four-modality study:

1.  T1
2.  T1ce
3.  T2
4.  FLAIR

The backend upload route is `POST /api/v1/scans/upload`; status and
results are exposed through the corresponding scan endpoints. The
application code includes NIfTI validation, scan ownership checks,
artifact handling, model inference tasks, saliency generation, and
research-report generation.

**Important:** the end-to-end model/API/frontend contract is still
subject to the project validation gate. Do not assume that the existence
of each component proves a complete successful live scan.

### Kaggle image classification

The repository also contains a separate single-image classification
workflow for JPG/PNG inputs. Its four output labels are glioma,
meningioma, no tumor, and pituitary, as defined by that workflow. This
classifier is distinct from ARMT-GAN segmentation and its metrics must
not be presented as segmentation performance.

### Research report and saliency

The report generator can assemble the supplied MRI slice, predicted
mask, gradient-based saliency visualisation, technical slice
measurements when available, model metadata, and a research-use
disclaimer.

-   A segmentation mask is a **model prediction**, not a verified
    clinical contour.
-   Peak sigmoid output is **not accuracy** and is not a calibrated
    patient-level diagnostic probability.
-   Segmentation alone does not establish tumour histology or WHO grade.
-   Saliency is an experimental attribution method; it does not prove
    that a prediction is correct.

## Repository map

``` text
.
├── PRD.md
├── architecture.md
├── design.md
├── tasks.md
├── memory.md
├── rules.md
├── backend/
│   ├── app/                 # FastAPI routes, services, schemas, configuration
│   ├── ai_pipeline/
│   │   ├── models/          # ARMT-GAN and classification models
│   │   ├── preprocessing/   # Shared BraTS preprocessing
│   │   ├── training/        # Reproducible training configuration and loop
│   │   ├── evaluation/      # Segmentation metrics and evaluation protocol
│   │   └── robustness/      # FGSM/PGD experiments
│   ├── tests/
│   └── requirements.txt
├── frontend/                # Next.js application
└── .github/workflows/       # CI workflows
```

Paths above describe the principal project areas; consult the repository
for the complete file tree.

## Getting started

### Prerequisites

-   Python version compatible with the backend dependencies.
-   Node.js and npm compatible with the frontend lockfile.
-   PostgreSQL configured for the application.
-   Redis configured for rate limiting and its integration tests.
-   MinIO configured for source and derived artifacts.
-   Firebase authentication configured for the intended environment.
-   BraTS data and a valid model checkpoint for real segmentation
    experiments.

Do not commit credentials, patient data, datasets, or model weights
unless the project explicitly permits them.

### 1. Clone the repository

``` bash
git clone https://github.com/Gagann-09/NeuroScan-AI.git
cd NeuroScan-AI
```

### 2. Install backend dependencies

``` bash
cd backend
python -m venv .venv
```

Activate the virtual environment:

**Windows PowerShell**

``` powershell
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

Then install dependencies:

``` bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Configure the required environment variables for the database, Firebase,
MinIO, Redis, CORS, and model/checkpoint paths using the current
settings implementation and deployment configuration. Never copy
production secrets into source control.

Start the API from the `backend` directory:

``` bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

When the backend is running, open [FastAPI Swagger
UI](http://127.0.0.1:8000/docs) or [ReDoc](http://127.0.0.1:8000/redoc).
These interfaces document the routes registered by the running
configuration; protected endpoints still require valid authentication
and configured services.

### 3. Install and start the frontend

In a second terminal, from the repository root:

``` bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:3000>. Configure the frontend API origin and
Firebase client settings according to the existing frontend
configuration. Do not assume the UI can process a scan until the
backend, database, object storage, authentication, Redis, and model
checkpoint are correctly configured.

### Useful frontend commands

Run these from `frontend/`:

  Command           Purpose
  ----------------- -------------------------------------------
  `npm run dev`     Start the Next.js development server
  `npm run build`   Build the production frontend
  `npm run start`   Start the production server after a build
  `npm run lint`    Run ESLint

## Training and evaluation

Run Python commands from the `backend/` directory so project imports
resolve as expected. The following scripts are the relevant entry
points:

  --------------------------------------------------------------------------------------------
  Purpose                             Script
  ----------------------------------- --------------------------------------------------------
  ARMT-GAN training                   `ai_pipeline/training/train_armt_gan.py`

  Baseline segmentation evaluation    `ai_pipeline/evaluation/evaluate_armt_gan_baseline.py`

  ARMT-GAN architecture               `ai_pipeline/models/armt_gan.py`

  Shared BraTS preprocessing          `ai_pipeline/preprocessing/brats_preprocessing.py`

  Robustness attack implementation    `ai_pipeline/robustness/attacks.py`
  --------------------------------------------------------------------------------------------

Inspect each script's CLI help before launching an experiment:

``` bash
python ai_pipeline/training/train_armt_gan.py --help
python ai_pipeline/evaluation/evaluate_armt_gan_baseline.py --help
```

Use a real, correctly structured BraTS dataset and a compatible
checkpoint. Preserve the generated experiment configuration and outputs.
Evaluation must use patient-disjoint partitions; do not substitute
synthetic data for real scientific results.

### Reporting metrics responsibly

Any published metric must identify:

-   Dataset and dataset version.
-   Patient split and evaluation cohort.
-   Model/checkpoint version.
-   Evaluation script and configuration.
-   Experiment date or identifier.
-   Whether the value comes from training, validation, or held-out
    testing.

Required segmentation metrics include Dice, IoU, precision, and
sensitivity. Do not report numbers that were not produced by a traceable
evaluation run. Robustness metrics must separate clean, FGSM, and PGD
results.

## API and interaction

The FastAPI application exposes interactive API documentation at `/docs`
when the backend is running.

Primary scan routes include:

  ------------------------------------------------------------------------------------------
  Method                  Route                                      Purpose
  ----------------------- ------------------------------------------ -----------------------
  `POST`                  `/api/v1/scans/upload`                     Upload the four
                                                                     required NIfTI
                                                                     modalities

  `GET`                   `/api/v1/scans/status/{scan_id}`           Check scan processing
                                                                     status

  `GET`                   `/api/v1/scans/results/{scan_id}`          Retrieve authorized
                                                                     scan results

  `POST`                  `/api/v1/scans/kaggle/upload`              Submit a single JPG/PNG
                                                                     classification image

  `GET`                   `/api/v1/scans/kaggle/status/{scan_id}`    Check classification
                                                                     status

  `GET`                   `/api/v1/scans/kaggle/results/{scan_id}`   Retrieve classification
                                                                     results
  ------------------------------------------------------------------------------------------

The running OpenAPI schema is the definitive reference for request
fields, response schemas, authentication, and error responses. Scan
access is user-scoped; do not disable authentication to make a demo
work.

## Security and data handling

-   Never commit API keys, Firebase service-account credentials,
    database passwords, Redis credentials, or production environment
    files.
-   Use authenticated Redis in production and keep it on a
    private/internal network; use TLS where required.
-   Keep CORS origins explicit; wildcard CORS is not permitted by the
    project rules.
-   Use only authorized datasets and protect MRI data as sensitive
    information.
-   Do not put patient identifiers, tokens, credentials, or sensitive
    medical details in logs.
-   Use migration-based database schema changes; application startup
    must not destructively reset tables.
-   Treat presigned object-storage URLs as temporary delivery links, not
    permanent provenance identifiers.

## Known limitations and project status

This section is intentionally explicit so the README does not overstate
project readiness.

-   **Research only:** the project is not clinically validated and is
    not a diagnostic device.
-   **2D baseline:** ARMT-GAN predicts one binary mask for a 2D axial
    slice; it is not a validated 3D segmentation model.
-   **No WHO grade prediction:** segmentation output does not determine
    tumour histology, grade, or treatment.
-   **XAI terminology:** the current attribution method is
    gradient-based input saliency, not Grad-CAM.
-   **Metric provenance:** no metric should be treated as a verified
    real-BraTS result without a traceable experiment record.
-   **Integration gate:** `tasks.md` identifies end-to-end validation as
    a gated phase. A component existing in the repository does not mean
    the complete user workflow has passed.
-   **Current branch CI:** check the latest GitHub Actions run before
    describing the branch as passing; a previous local test count is not
    proof that the current revision is green.

Do not remove these limitations merely to make the project appear more
complete. Update this section only when the relevant implementation,
tests, and source-of-truth records support the change.

## Project documents

These files govern project scope and decisions. Resolve conflicts in the
order specified by `rules.md`.

1.  [`rules.md`](rules.md) --- implementation boundaries and
    scientific-integrity rules.
2.  [`PRD.md`](PRD.md) --- product scope, requirements, and exclusions.
3.  [`architecture.md`](architecture.md) --- system architecture and
    data contracts.
4.  [`design.md`](design.md) --- model, evaluation, XAI, API, and report
    design.
5.  [`tasks.md`](tasks.md) --- phase gates, acceptance criteria, and
    task status.
6.  [`memory.md`](memory.md) --- decision ledger and verification
    evidence.

Useful implementation references:

-   [ARMT-GAN model](backend/ai_pipeline/models/armt_gan.py)
-   [Training pipeline](backend/ai_pipeline/training/train_armt_gan.py)
-   [Evaluation
    protocol](backend/ai_pipeline/evaluation/evaluate_armt_gan_baseline.py)
-   [Robustness attacks](backend/ai_pipeline/robustness/attacks.py)
-   [Scan API](backend/app/api/routers/scans.py)
-   [PDF report generator](backend/app/services/reporting.py)
-   [Backend tests](backend/tests/)
-   [Frontend](frontend/)

## Contributing and reporting issues

1.  Read all six project source-of-truth documents before proposing
    implementation changes.
2.  Check the active phase and its gate in `tasks.md`.
3.  Make the smallest evidence-driven change; avoid unrelated refactors.
4.  Preserve model architecture, data contracts, and scientific scope
    unless a documented decision authorizes a change.
5.  Run relevant tests and report their actual results.
6.  Update the affected documentation and record consequential
    decisions.

Use [GitHub Issues](https://github.com/Gagann-09/NeuroScan-AI/issues) to
report reproducible defects. Include the branch/commit, steps to
reproduce, expected versus actual behavior, and relevant redacted logs.
Never attach private medical data or secrets.

------------------------------------------------------------------------

**Project:** NeuroScan AI · **Core model:** ARMT-GAN · **Status:**
Research prototype; end-to-end validation remains gated.
