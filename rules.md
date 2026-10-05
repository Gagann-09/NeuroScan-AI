# NeuroScan AI --- Vibe Coding Rules

**Authority:** Highest implementation boundary for this project.

## 1. Project Isolation

These rules apply only to NeuroScan AI.

Never copy source-of-truth documents from another project.

Never import architecture decisions from JARVIS.

Never import research workflow rules from PQ-TLS.

Every new decision must be justified against NeuroScan AI.

## 2. Source-of-Truth Order

When conflicts exist:

1.  `rules.md`
2.  `PRD.md`
3.  `architecture.md`
4.  `design.md`
5.  `tasks.md`
6.  `memory.md`
7.  Existing code
8.  README claims

Code is not automatically authoritative.

Documentation must describe verified behavior.

## 3. Before Any Coding

An agent must:

1.  Read all six source-of-truth files.
2.  Inspect the relevant existing code.
3.  Identify current behavior.
4.  Identify affected contracts.
5.  Check task phase and dependencies.
6.  State the smallest permitted change.
7.  Implement only that change.
8.  Run required verification.
9.  Update affected source-of-truth records.

## 4. Scope Lock

Do not expand scope during implementation.

Do not add unrelated refactors.

Do not replace ARMT-GAN with another architecture.

Do not add Transformers merely because they appear in papers.

Do not add 3D segmentation to the baseline.

Do not add clinical features before scientific validation.

## 5. Scientific Integrity

Never fabricate:

-   Accuracy.
-   Dice.
-   IoU.
-   Sensitivity.
-   Precision.
-   Confidence.
-   Robustness.
-   Clinical grade.
-   Clinical diagnosis.

Every reported metric needs:

-   Dataset.
-   Split.
-   Model version.
-   Evaluation script.
-   Configuration.
-   Date or experiment ID.

## 6. Data Rules

BraTS is the canonical segmentation dataset.

Use four modalities:

-   T1.
-   T1ce.
-   T2.
-   FLAIR.

Patient-level splitting is mandatory.

No patient may appear in multiple evaluation partitions.

No synthetic masks may substitute for real labels.

No random fallback dataset is allowed.

## 7. Model Rules

ARMT-GAN is the canonical model.

The current generator is a lightweight 2D U-Net.

The current discriminator is conditional PatchGAN.

The current model input is four channels.

The current model output is one sigmoid mask.

Architecture changes require explicit versioning.

## 8. Training Rules

Every training run records:

-   Seed.
-   Dataset version.
-   Split.
-   Image size.
-   Batch size.
-   Learning rate.
-   Epochs.
-   Loss configuration.
-   Checkpoint version.

Do not overwrite the best checkpoint without provenance.

## 9. Robustness Rules

FGSM and PGD are evaluation features.

They are not silently merged into baseline training.

Attack parameters must be explicit.

Clean and adversarial metrics must remain separate.

## 10. Inference Rules

Training preprocessing and inference preprocessing must match.

No RGB conversion may be used for the four-channel ARMT-GAN path.

No single middle slice may represent an entire 3D study.

3D support remains future scope until implemented and validated.

## 11. Confidence Rules

No heuristic confidence boosting.

No arbitrary score offsets.

No fabricated ensemble agreement.

No UI confidence unless the backend computes it legitimately.

## 12. Clinical Language Rules

Forbidden unless validated:

-   "clinical-grade."
-   "diagnosis."
-   "clinically proven."
-   "WHO grade prediction."
-   "radiologist replacement."
-   "treatment recommendation."

Preferred:

-   "research prototype."
-   "segmentation prediction."
-   "model output."
-   "experimental attribution."

## 13. XAI Rules

Never label input-gradient saliency as Grad-CAM.

Never imply XAI proves correctness.

Store attribution provenance.

## 14. Security Rules

No hard-coded production credentials.

No default administrator passwords in committed code.

No wildcard CORS for production.

No destructive startup database reset.

No sensitive medical data in logs.

No authentication bypass for live inference.

## 15. API Rules

Frontend and backend schemas must match exactly.

A contract change requires updates to:

-   Backend schema.
-   Backend route.
-   Frontend API client.
-   Frontend state types.
-   Tests.
-   Relevant documentation.

## 16. Storage Rules

Every artifact belongs to a study.

Every derived artifact references its source study.

Never overwrite artifacts without versioning.

Presigned URLs are delivery mechanisms.

They are not permanent provenance identifiers.

## 17. Database Rules

Never call `drop_all()` during application startup.

Schema evolution must be migration-based.

Missing tables must fail clearly.

Seed data must be explicit.

## 18. Testing Gate

No task is complete until required tests pass.

A failing test blocks the phase.

Warnings may be accepted only when documented.

## 19. Vibe Coding Boundary

AI coding agents may implement bounded tasks.

They may not invent requirements.

They may not choose scientific metrics without approval.

They may not change architecture silently.

They may not delete evidence to make tests pass.

They may not mark a task complete without verification.

## 20. Completion Rule

A phase closes only after:

-   Implementation passes.
-   Tests pass.
-   Documentation matches code.
-   No known contract contradiction remains.
-   `memory.md` records consequential decisions.
-   `tasks.md` marks the phase verified.
