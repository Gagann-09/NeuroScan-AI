# NeuroScan AI — Final Security Acceptance Audit (Read-Only)

**Repository:** Gagann-09/NeuroScan-AI  
**Branch:** `main` (working tree clean, up to date with `origin/main`)  
**Audited commit (HEAD):** `2c27e96c83cbb39db576795c728e069bc66c55ef` — "SG3-F5-E: bypass rate limiter in upload compensation tests"  
**Remote:** `https://github.com/Gagann-09/NeuroScan-AI.git`  
**Auditor role:** Senior application-security auditor / backend architect  
**Mode:** Read-only audit. No files edited, no `.env` read, no secrets printed, no migrations run, no containers altered, no commits/pushes.  
**Source-of-truth order applied:** rules.md → PRD.md → architecture.md → design.md → tasks.md → memory.md → code.

---

## Preflight

| Check | Result |
|---|---|
| Branch / HEAD | `main` @ `2c27e96c83cbb39db576795c728e069bc66c55ef` |
| Working tree | Clean (`git status`: nothing to commit) |
| Remote | `origin` = https://github.com/Gagann-09/NeuroScan-AI.git (fetch + push) |
| Commit `2c27e96` present on `main` | Yes — `git cat-file -t 2c27e96` = `commit` |
| HEAD == requested commit | Yes (`2c27e96...`) |
| GitHub Actions PostgreSQL backend test check for exact commit | **Could not be independently confirmed.** `api.github.com` returns `403` (unauthenticated/rate-limited) for both `/commits/{sha}/check-runs` and `/actions/workflows/.../runs?head_sha=...`. `memory.md` (D-040/D-041/D-042/D-044) documents that the workflow passed on the GitHub-hosted runner for the SG3-F5 series (254–257 passed, 0 failed), and local equivalent disposable-PostgreSQL runs pass (254/256/257). The exact-commit hosted result for `2c27e96` remains unverified by this audit. |
| Discrepancies | (1) PRD.md/architecture.md document baseline commit `a0bbe53...`; HEAD is `2c27e96` (the repo has advanced well past the documented baseline — all SG3-F3 through SG3-F5 security work is now in tree). (2) CI check for exact commit not externally confirmable (see above). |

No repository dirtiness or commit mismatch was found; proceeding with the audit.

---

## Executive Summary and Overall Acceptance Status

**Overall status: ACCEPTABLE FOR CONTROLLED RESEARCH-PROTOTYPE DEMONSTRATION, WITH CONDITIONS.**

No confirmed Critical-severity, remotely exploitable vulnerability was found. The security work tracked in `tasks.md` P8 (SG3-F5-B through SG3-F5-E) and SG3-F3/SG3-F4 is implemented and tested:

- Firebase ID-token verification with `check_revoked=True`; invalid/expired/revoked → generic 401; infrastructure failures (missing ADC, init errors, network/timeout, DB errors) → generic 503 with no detail leakage. **Verified in code and tests.**
- CORS: no wildcard anywhere (including mixed lists), `allow_credentials=True`, origins required, validated at `Settings()` instantiation before middleware setup. **Verified.**
- Configuration: no hard-coded credential defaults in Python application code; `DATABASE_URL`, `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `FIREBASE_PROJECT_ID`, `CORS_ORIGINS` are required fields. **Verified.**
- Rate limiting: Redis-backed fixed-window, atomic Lua script (`INCR`+`EXPIRE`+`TTL`), keyed per authenticated Firebase UID, `429` + `Retry-After`, fail-closed `503` when Redis unavailable, wired on `POST /upload` only, auth runs first. **Verified in code and focused tests.**
- NIfTI upload validation: header magic at real offset 344, NIfTI-1 only, bounded streaming gzip decompression with CRC/ISIZE trailer verification, decompressed-byte cap (200 MiB), dimension/datatype/voxel-count/`vox_offset`/payload-size consistency, truncated/trailing/concatenated data rejected. **Verified.**
- Database: Alembic migrations 001→006 (no `drop_all()`), session closure in `finally`, ownership FK `scans.user_id → users.firebase_uid`, artifact lifecycle check constraint, claim-locking with `SELECT FOR UPDATE NOWAIT`. **Verified against live PostgreSQL** (33 DB-backed tests pass).
- Scientific integrity: report is titled "Research Prototype Segmentation Report" with an explicit disclaimer; `max_tumor_probability` is the raw sigmoid output; Dice/IoU are `NULL` at inference (populated only by the evaluation pipeline); XAI is explicitly "gradient-based input saliency", not Grad-CAM. **Verified.**

**Conditions / residual risks (do not block prototype acceptance, but must be remediated before any production-like exposure):**

1. **High — Information disclosure:** `POST /upload` returns `HTTPException(500, detail=str(e))` (`backend/app/api/routers/scans.py:430`), which can leak exception text (endpoints, paths, possibly credential fragments) to clients.
2. **Medium — Availability/reliability:** scans left in `PENDING` are never retried. `BackgroundTasks` are in-process; a worker restart before `claim_scan_for_processing()` strands the scan forever. No reaper/retry path exists for `PENDING` scans.
3. **Medium — Operational dependency:** uploads fail closed with `503` when Redis is unavailable. CI does not run Redis, so the fail-closed `503` path is exercised only with a fake client, not against a live Redis. Redis in `docker-compose.yml` runs without a password/ACL and is exposed on the host port.
4. **Medium/Deployment — Object storage transport:** `MINIO_SECURE` defaults to `False` (plain HTTP); tunable via env but production should enforce TLS.
5. **Low — Data governance:** `scans.user_id` is nullable; legacy ownerless scans are fail-closed (inaccessible to everyone) but have no backfill or access-recovery path.
6. **Informational — Documentation drift:** `PRD.md` and `architecture.md` baseline references (`a0bbe53...`) no longer match HEAD (`2c27e96`); `tasks.md`/`memory.md` are current.

**What was NOT verified:** hosted GitHub Actions result for the exact commit `2c27e96` (GitHub API 403); fail-closed `503` against a live Redis in production; compressed real-world BraTS `.nii.gz` against live MinIO (dataset-gated test skipped when data absent); frontend (out of scope per audit boundaries); ARMT-GAN model weights/training science (out of scope).

The application must **not** be described as production-secure, clinically validated, or a diagnostic device. It remains a research prototype (PRD §1, rules.md #12).

---

## Findings Ranked by Severity

### F-01 — High (Confirmed): Raw exception details leaked to clients on upload failure

- **Severity:** High
- **File/line:** `backend/app/api/routers/scans.py:425-430`, specifically line 430 `raise HTTPException(status_code=500, detail=str(e))`.
- **Evidence:** The `except Exception` block in `upload_scan` rolls back the DB, attempts source-object cleanup, then raises `HTTPException(500, detail=str(e))`. `str(e)` can embed MinIO/S3 error messages, SQLAlchemy connection details (host, database, possibly user), file paths, or underlying driver errors. Clients receive this verbatim in the response body.
- **Reproducible impact:** Force a MinIO failure (e.g., make `bucket_exists`/`fput_object` raise) after valid authentication; the 500 body returns `detail=str(e)`. Information disclosure to an authenticated attacker.
- **Remediation:** Replace `detail=str(e)` with a generic message (e.g., `"Upload failed"`), log the full exception server-side with `exc_info=True` (operator-only), and return status 500 without internals — consistent with the pattern already used in `app/core/security.py` (503) and `app/core/rate_limiting.py` (503).
- **Blocks acceptance?** No under the P8 gate ("no critical security or destructive-data defects"), but it is a confirmed information-disclosure defect and should be fixed before any production-like demo.

### F-02 — Medium (Confirmed design gap): `PENDING` scans are stranded on worker restart; no retry/reaper

- **Severity:** Medium
- **File/line:** `backend/app/api/routers/scans.py:412` (`background_tasks.add_task(process_scan_task, ...)`) and `backend/app/services/ai_tasks.py:487-497` (claim only transitions `PENDING`/`FAILED` → `PROCESSING`; nothing transitions stale `PENDING` back).
- **Evidence:** FastAPI `BackgroundTasks` are in-process and tied to the request/response worker. If the worker restarts after the upload endpoint enqueues `process_scan_task` but before the task runs (or claims the scan), the scan remains `PENDING` indefinitely. `claim_scan_for_processing` accepts `PENDING` → `PROCESSING`, but nothing ever re-claims a `PENDING` scan. `processing_started_at` (migration 005) exists to detect staleness but is never queried for recovery.
- **Reproducible impact:** Restart the API worker mid-upload; the scan record stays `PENDING` forever, never processed, never retried. Denial of service for that scan; user sees `PROCESSING`/`PENDING` status with no resolution.
- **Remediation:** Add a startup/periodic reaper (or a lightweight endpoint) that claims scans stuck in `PENDING` beyond a threshold (using `processing_started_at` is NULL or old) and transitions them back for processing, or exposes a manual retry. Document the single-worker `BackgroundTasks` limitation for the prototype.
- **Blocks acceptance?** No (prototype scope), but an operational reliability gap.

### F-03 — Medium (Operational): Uploads fail closed `503` when Redis unavailable; fail-closed path not tested against live Redis; Redis exposed without auth in compose

- **Severity:** Medium
- **File/line:** `backend/app/core/rate_limiting.py:119-128` (fail-closed 503 on `RedisError`); `backend/app/core/config.py:34` (`REDIS_URL` default `redis://localhost:6379/0`); `docker-compose.yml:20-30` (Redis with no password, host port `6379:6379`).
- **Evidence:** The limiter is required on `POST /upload` (`scans.py:320`). When Redis is down, `enforce_upload_rate_limit` raises 503 before any upload validation runs — correct fail-closed behavior. However CI runs PostgreSQL only (no Redis service); the upload-compensation tests bypass the limiter entirely (commit `2c27e96`, HEAD), so the real fail-closed `503` path is exercised only with an in-memory fake (`tests/test_rate_limiting.py`). `docker-compose.yml` runs Redis without `requirepass`/ACL and publishes it to the host.
- **Reproducible impact:** Stop Redis; authenticated uploads return 503 (safe but a denial of service for the expensive endpoint). In a host-exposed compose setup, Redis is reachable without authentication from the host network.
- **Remediation:** (a) Add an integration test against a real Redis container asserting `503` on upload when Redis is down and `429`+`Retry-After` when exceeded. (b) For production deployment, bind Redis to an internal network, add `requirepass`/ACL, and do not publish `6379` to the host. (c) Consider whether upload unavailability during a Redis outage is acceptable, or provide an ops escape hatch (admin override) — a product decision.
- **Blocks acceptance?** No for the prototype (fail-closed is the secure default); required before production-like exposure.

### F-04 — Medium/Deployment-dependent: MinIO object storage default plain HTTP (`MINIO_SECURE=False`)

- **Severity:** Medium (deployment-dependent)
- **File/line:** `backend/app/core/config.py:31` (`MINIO_SECURE: bool = False`); `docker-compose.yml:33-49` (MinIO ports `9000:9000`, `9001:9001` published to host).
- **Evidence:** The setting exists and can be overridden via env, but the default is `False` and MinIO is exposed on host ports in compose. Presigned URLs (15-minute TTL, `config.py:51`) and artifact traffic would travel plain-text unless the deployment sets `MINIO_SECURE=True` with TLS-terminated MinIO.
- **Remediation:** Document production requirement: set `MINIO_SECURE=True` and terminate TLS; do not publish MinIO ports to the public host network. Consider making `MINIO_SECURE` required (no default) for non-local contexts.
- **Blocks acceptance?** No (deployment-dependent).

### F-05 — Low (Confirmed): Legacy ownerless scans are fail-closed with no access-recovery path

- **Severity:** Low
- **File/line:** `backend/app/db/models.py:36` (`user_id` nullable, FK to `users.firebase_uid`); `migrations/006_add_user_table_and_scan_ownership.py:32` ("nullable for backward compatibility with existing scans"); `scans.py:443-445` and `463-465` (ownership comparison `scan.user_id != current_user.firebase_uid`).
- **Evidence:** `user_id` is nullable, so pre-migration scans have no owner. The ownership check in Python (`None != "<uid>"` is `True`) returns 403 for every authenticated user — fail-closed, which is safe. `tests/test_auth.py:260-272` (`mock_legacy_scan`) and `:436-456` confirm a legacy scan returns 403 to any user. No migration or backfill assigns ownership, and no admin workflow exists to reclaim orphan scans.
- **Remediation:** Either backfill legacy scans to a known service account or document that pre-migration scans are intentionally inaccessible. Do not relax the ownership check to "public if ownerless" (that would be an authorization bypass).
- **Blocks acceptance?** No.

### F-06 — Low: Presigned-URL generation silently swallows errors and returns `None`

- **Severity:** Low
- **File/line:** `backend/app/core/storage.py:53-58` (`get_presigned_url`: bare `except Exception: return None`).
- **Evidence:** `get_scan_results` (`scans.py:471-473`) emits `mask_url`/`xai_url`/`report_url` as `None` when generation fails, indistinguishable from "artifact not yet produced". No error is surfaced; callers cannot tell storage failure from absence.
- **Remediation:** Distinguish "not ready" (`None` path absent) from "URL generation failed" (server error or explicit field), or at least log the failure server-side. Keep presigned URLs issuance gated behind the ownership check (already done).
- **Blocks acceptance?** No.

### F-07 — Low (Hygiene): Partial temp file can remain on MinIO upload failure during source upload

- **Severity:** Low
- **File/line:** `backend/app/api/routers/scans.py:363-384`. `temp_path` is written at lines 366-367 (`_copyfileobj_with_limit`), uploaded at line 383 (`fput_object`), and removed at line 384 only on success. If `fput_object` raises, the exception is caught at line 425, but `temp_path` was never appended to `uploaded_objects` (append happens at line 387 after successful upload), so the temp file under `temp_uploads/` is orphaned.
- **Remediation:** Remove `temp_path` in the `except HTTPException`/size-limit path as well, or wrap write+upload in a `try/finally` that deletes the temp file unless the scan record is successfully created.
- **Blocks acceptance?** No.

### F-08 — Informational: Celery pinned in `requirements.txt` but unused; `BackgroundTasks` is the actual async mechanism

- **Severity:** Informational
- **File/line:** `backend/requirements.txt:13` (`celery==5.3.6`); `scans.py:412` (`BackgroundTasks`).
- **Evidence:** Celery is declared but nothing imports it. Background work runs in-process via FastAPI `BackgroundTasks`, which is not shared across uvicorn workers and is lost on restart (see F-02). This is not a vulnerability but adds an unmaintained dependency and could mislead operators about task durability.
- **Remediation:** Remove celery from `requirements.txt` or wire it in. Document that background processing is single-worker/in-process for the prototype.
- **Blocks acceptance?** No.

### F-09 — Informational: `test_pipeline_direct.py` and `ingest_datasets.py` are dead/broken code outside the test suite

- **Severity:** Informational
- **File/line:** `backend/test_pipeline_direct.py` (fails to import: references removed `MRIScan`/`process_mri_scan`, noted in `memory.md` D-036/D-038); `backend/ingest_datasets.py` (imports nonexistent `Patient`/`MRIScan`; unreferenced dead code, noted in D-041).
- **Evidence:** Neither is collected by `pytest tests/`; both reference removed symbols. No security impact, but they are noise in the repository.
- **Remediation:** Remove or fix the dead files.
- **Blocks acceptance?** No.

### F-10 — Informational: CORS permits all methods/headers with credentials (origin restriction is the control)

- **Severity:** Informational
- **File/line:** `backend/app/main.py:30-31` (`allow_methods=["*"]`, `allow_headers=["*"]`, `allow_credentials=True`).
- **Evidence:** Wildcard methods/headers combined with `allow_credentials=True` is broader than strictly necessary, though `allow_origins` is restricted to explicit origins (no `*`) by `validate_cors_origins` (`config.py:115-137`), which is the critical control for credentialed cross-origin requests. Tests in `test_config_security.py::TestCORSConfiguration` cover wildcard rejection.
- **Remediation:** Consider narrowing `allow_methods`/`allow_headers` to the set the frontend needs.
- **Blocks acceptance?** No.

### F-11 — Informational: Documentation baseline references stale vs. HEAD

- **Severity:** Informational
- **File/line:** `PRD.md:9` ("Baseline commit: `a0bbe53...`"), `architecture.md:5` ("Baseline: `main` @ `a0bbe53...`"); actual HEAD `2c27e96`.
- **Evidence:** The repository has advanced past the documented baseline (all SG3-F3/F4/F5 security work is in tree), but `PRD.md` and `architecture.md` still reference `a0bbe53...`. `tasks.md` and `memory.md` are current.
- **Remediation:** Update `PRD.md` and `architecture.md` baseline lines to `2c27e96` (or remove the now-stale baseline claim).
- **Blocks acceptance?** No.

---

## Control-by-Control Evidence Matrix

| # | Control | Source-of-truth requirement | Implementation reference | Test reference | Classification | Status |
|---|---|---|---|---|---|---|
| 1 | Firebase token verification + revocation | rules.md #14 (no auth bypass for live inference); PRD G7 (fail closed) | `app/core/security.py:63-121` — `verify_id_token(token, check_revoked=True)`; FirebaseError→401; GoogleAuthError→503 | `tests/test_auth.py` `TestVerifyIdToken` (valid/invalid/expired/revoked/cert-fetch) | Verified in code and tests | ✅ Verified |
| 2 | Auth infrastructure failure handling | rules.md #14 (no sensitive data in logs, fail closed); memory.md D-043 | `security.py:26-60` (missing ADC/GoogleAuthError→503); `:124-172` (`SQLAlchemyError`→503 + rollback); generic messages only | `tests/test_auth.py` `TestInfrastructureErrorHandling` (5 tests: ADC, init failure, TransportError, TimeoutError, DB error — all 503, no leak) | Verified | ✅ Verified |
| 3 | User-to-scan ownership | design.md §9; PRD FR-10; rules.md #16 | `models.py:36` (`user_id` FK); `scans.py:391-396` (set from `current_user.firebase_uid`); `:443-445`, `:463-465` (403 on mismatch) | `tests/test_auth.py` `TestCrossUserOwnership` (403 for other user; 403 for legacy ownerless scan; presigned URLs not generated on 403) | Partially verified (no backfill for legacy scans — F-05) | ⚠️ Partially |
| 4 | Cross-user access blocked | Same as #3 | Ownership check before any presigned URL generation (`scans.py:471-473` runs after 403) | `tests/test_auth.py:458-492` (`get_presigned_url.assert_not_called()` on unauthorized) | Verified | ✅ Verified |
| 5 | Legacy ownerless scans | memory.md D-035/D-036 context | `user_id` nullable (`006` migration); comparison returns 403 for ownerless | `tests/test_auth.py:260-272, 436-456` | Verified as fail-closed; recovery path missing (F-05) | ⚠️ Partially |
| 6 | Artifact URL issuance after auth | rules.md #16 (presigned = delivery, not provenance); design.md §8 | `get_presigned_url` called only post-ownership-check (`scans.py:471-473`); TTL 15 min default, max 1440 (`config.py:51`, validator `139-152`) | `tests/test_auth.py:388-400` (mocked presigned for owner); `test_config_security.py::TestPresignedURLTTLConfiguration` | Partially verified (silent `None` on error — F-06) | ⚠️ Partially |
| 7 | Upload size limits | PRD G7 (fail closed on invalid input) | `scans.py:36` (100 MB/modality); `_copyfileobj_with_limit` `:271-290` (HTTP 413 over limit); decompressed cap 200 MiB (`scans.py:60`, `187-190`) | `tests/test_nifti_safety_regression.py`; `tests/test_auth.py::TestUploadValidation` (auth-mocked; size/malformed cases assert 401 because auth runs first) | Verified | ✅ Verified |
| 8 | NIfTI validation (magic, header, bounds) | design.md §7 (validate before inference); memory.md D-036/D-037 | `scans.py:66-154` (magic at offset 344, `n+1` only, `vox_offset`, payload-size consistency); `:157-206` (bounded gzip stream, CRC/ISIZE via `eof`, reject concatenated/trailing); `brats_preprocessing.py:79-152` (shape/dtype bounds shared) | `tests/test_nifti_safety_regression.py` (47 tests, incl. dataset-gated real BraTS `.nii`) | Verified (NIfTI-1 only; compressed real-BraTS `.nii.gz` against live MinIO unproven) | ✅ Verified |
| 9 | Malformed/truncated input rejection | Same as #8 | `_validate_nifti1_single_file` raises on truncated/trailing (`scans.py:140-154`); gzip EOF trailer verified (`193-204`) | `tests/test_nifti_safety_regression.py` | Verified | ✅ Verified |
| 10 | Cleanup on upload failure | PRD G7; design.md §12 (failures explicit) | `_cleanup_source_objects` (`scans.py:293-308`); `uploaded_objects` tracking (`333`, `387`, `420-424`); temp-file edge (F-07) | `tests/test_scan_claim_integration.py::TestUploadCompensation` (5 tests: cleanup counts, NoSuchKey, cleanup-failure does not mask original error) | Verified | ✅ Verified |
| 11 | Concurrent upload / duplicate processing | design.md §12 (preserve scan state); rules.md #17/#18 | `claim_scan_for_processing` (`ai_tasks.py:187-242`, `SELECT FOR UPDATE NOWAIT`); wired into `process_scan_task` (`493-497`); PENDING initial status (`scans.py:394`) | `tests/test_scan_claim.py` (7 tests incl. concurrency); `test_scan_claim_integration.py` (17 tests incl. duplicate-dispatch, session cleanup) | Verified | ✅ Verified |
| 12 | Rate limiting: Redis shared counters, atomicity | tasks.md P8 SG3-F5-E; memory.md D-044 | `rate_limiting.py:67-80` (Lua `INCR`+`EXPIRE`+`TTL`); keyed per Firebase UID (`:110`) | `tests/test_rate_limiting.py` (15 tests; FakeRedis; per-user isolation; 429+Retry-After; 503 on backend failure) | Verified (fake Redis; live-Redis fail-closed unproven — F-03) | ✅ Verified |
| 13 | Rate limiting: per-user isolation, 429, Retry-After | SG3-F5-E | `enforce_upload_rate_limit` (`rate_limiting.py:180-200`); `429` + `Retry-After: <TTL>` | `tests/test_rate_limiting.py:196-205, 236-255` | Verified | ✅ Verified |
| 14 | Rate limiting: fail-closed, test-only overrides | SG3-F5-E | `RedisError`→503 generic (`rate_limiting.py:119-128`); `RATE_LIMIT_ENABLED` + `reset_rate_limiter_for_tests()` (`:158-177`) | `tests/test_rate_limiting.py:161-171, 207-222`; bypass in upload-compensation tests only (HEAD `2c27e96`) | Verified | ✅ Verified |
| 15 | Auth runs before rate limit | SG3-F5-E | `enforce_upload_rate_limit` has `Depends(get_current_user)` first (`rate_limiting.py:181`); upload Depends order (`scans.py:318-320`) | `tests/test_rate_limiting.py:258-265` (401 before rate limit); `tests/test_auth.py:199-212` (upload 401 without auth) | Verified | ✅ Verified |
| 16 | CORS: no wildcard, credentials, trusted origins | rules.md #14 ("No wildcard CORS for production") | `config.py:115-137` (rejects `"*"` anywhere, incl. mixed); `main.py:26-32` (`allow_credentials=True`, origins from settings) | `tests/test_config_security.py::TestCORSConfiguration` (singleton wildcard, mixed wildcard, explicit origins) | Verified | ✅ Verified |
| 17 | Configuration: no hard-coded creds | rules.md #14; PRD NFR; memory.md D-012/D-029 | `config.py:24,29-30,38,46` (required fields, no defaults); `get_settings` clear error (`:197-216`); `init_system.py` deleted (D-041) | `tests/test_config_security.py` (10 tests incl. required-field failures, wildcard CORS, AdminCredentialRemoval scanning tracked `.py`) | Verified in app code; docker-compose retains dev creds (deployment-dependent) | ✅ Verified |
| 18 | Error responses: no exception leakage | rules.md #14 | `security.py` 503/401 generic; BUT `scans.py:430` `detail=str(e)` (F-01); `storage.py:53-58` silent `None` (F-06) | `tests/test_auth.py` (no detail in 503/401) | Partially verified (F-01 leak confirmed) | ⚠️ Partially |
| 19 | Secret logging | rules.md #14 ("No sensitive medical data in logs") | `security.py` logs with `exc_info=True` for operators; detail not returned to client (D-043) | `tests/test_auth.py:712-713` etc. (assert strings not in detail) | Verified (client-side); server-side log content not asserted | ✅ Verified (client-side) |
| 20 | Redis access controls / availability | architecture.md (Redis in APP layer); docker-compose | `rate_limiting.py` (Redis only; `validate_rate_limit_backend` rejects non-redis `config.py:154-170`); no Redis password in compose | `tests/test_rate_limiting.py` (fake); no live-Redis integration test | Partially verified (deployment-dependent: compose Redis no auth, host-exposed — F-03) | ⚠️ Partially |
| 21 | BackgroundTasks lifecycle / worker restart | PRD G7; design.md §12 | `scans.py:412` (`BackgroundTasks`); `ai_tasks.py:487` (`SessionLocal()`); `finally: db.close()` (`:717-720`, fixed D-038); no PENDING reaper | `tests/test_scan_claim_integration.py::TestSessionCleanup` (sessions closed on success/failure) | Partially verified (PENDING-strand gap — F-02) | ⚠️ Partially |
| 22 | Duplicate work / failure recovery | design.md §12 | Claim-locking prevents duplicates (`test_scan_claim.py` concurrency); FAILED→PROCESSING retry allowed (`claim_scan_for_processing`:226-232) | `test_scan_claim_integration.py` (duplicate dispatch → 1 prediction; failed→claimed again) | Verified | ✅ Verified |
| 23 | Migrations / no destructive startup | rules.md #17 ("Never call drop_all()"); architecture.md anti-patterns | `main.py` docstring (startup no DDL); Alembic 001→006; `alembic/env.py:34` reads `get_settings()` | CI workflow (`alembic upgrade head`); `tests/` 254 pass against live PostgreSQL (D-038/D-040) | Verified | ✅ Verified |
| 24 | Session closure / transaction handling | rules.md #17; memory.md D-038 | `get_db` finally `db.close()` (`database.py:36-41`); `process_scan_task` finally `db.close()` (`ai_tasks.py:717-720`); rollback on failure (`:703`) | `test_scan_claim_integration.py::TestSessionCleanup` (2 tests); integration commit-failure rollback (`test_commit_failure_after_upload_rolls_back_artifacts`) | Verified | ✅ Verified |
| 25 | Ownership constraints / schema consistency | design.md §9 | `scans.user_id` FK (`006`); `Prediction.scan_id` / `model_version_id` FKs; `Artifact.status` check constraint (`005`) | `tests/test_db_models.py` (27 tests); `test_artifact_lifecycle.py` (9 tests incl. terminal states) | Verified against live PostgreSQL | ✅ Verified |
| 26 | Dependency pinning | tasks.md; CI | `requirements.txt` pinned (torch 2.2.0, numpy 1.26.3, opencv 4.11.0.86, alembic 1.13.2); `requirements-dev.txt` (pytest, pytest-asyncio, httpx 0.26.0) | CI installs both files; dependency resolution verified for CPython 3.11/Linux (D-039) | Verified (Celery unused — F-08) | ✅ Verified |
| 27 | Docker Compose / exposed services | Deployment-dependent | `docker-compose.yml`: db `5432`, redis `6379`, minio `9000/9001` published to host; dev credentials in compose | None (deployment) | Deployment-dependent | ⚠️ Deployment-dependent |
| 28 | CI coverage (PostgreSQL backend) | tasks.md P8; memory.md D-039/D-040 | `.github/workflows/backend-tests.yml` (postgres:16-alpine service, `alembic upgrade head`, `pytest tests/ -v`) | Documented hosted pass (D-040: 254 passed); exact commit `2c27e96` not confirmable (API 403) — preflight limitation | Partially verified (hosted result for exact commit unconfirmed) | ⚠️ Partially |
| 29 | Scientific-data integrity / no clinical claims | rules.md #5/#11/#12; PRD §1/§2; design.md §11 | `reporting.py:11` "Research Prototype Segmentation Report"; disclaimer `:25-26`; `max_tumor_probability` raw sigmoid (`ai_tasks.py:605`, `scan_schema.py:37-38`); Dice/IoU `NULL` at inference (`:668-669`); XAI named "gradient-based input saliency" (`xai.py:14`, `:43-55`) | `tests/test_xai.py` (method name, provenance); report disclaimer present in code | Verified | ✅ Verified |

Classification key:
- ✅ Verified in code and tests
- ⚠️ Partially verified (gap noted in findings)
- ⚠️ Deployment-dependent (cannot be verified from repo alone)

---

## Confirmed Gaps vs Deployment-Dependent Risks

**Confirmed gaps (in code, actionable now):**

| ID | Severity | Gap | Remediation |
|---|---|---|---|
| F-01 | High | `detail=str(e)` leaks exception text on upload 500 (`scans.py:430`) | Generic 500 message; log server-side only |
| F-02 | Medium | `PENDING` scans stranded on worker restart; no reaper | Add stale-`PENDING` reaper/retry or document single-worker limit |
| F-07 | Low | Temp file orphaned if `fput_object` fails during source upload (`scans.py:363-384`) | `try/finally` temp cleanup |
| F-06 | Low | `get_presigned_url` returns `None` silently on error (`storage.py:53-58`) | Distinguish "not ready" from "generation failed"; log server-side |
| F-05 | Low | Legacy ownerless scans (`user_id` NULL) inaccessible, no backfill | Document fail-closed; optional backfill to service account |

**Deployment-dependent risks (cannot be verified from repo; must be validated at deployment time):**

- Hosted GitHub Actions result for exact commit `2c27e96` (API returned 403; prior commits in the series passed — D-040/D-044).
- Fail-closed `503` against a live Redis outage in production (only fake-Redis tested locally).
- Redis without password/ACL and published on host port in `docker-compose.yml` (local-dev only; not safe for production exposure).
- `MINIO_SECURE=False` default and MinIO ports published to host (plain-text object traffic unless overridden).
- Compressed real-world BraTS `.nii.gz` end-to-end against live MinIO (dataset-gated test skips when data absent; D-037).

**Not a gap (secure by design, verified):**
- Ownerless legacy scans return 403 to everyone (fail-closed) — do not relax this.
- No `drop_all()` at startup; migration-based schema (rules.md #17).
- No hard-coded credential defaults in Python code.
- CORS rejects wildcard including mixed lists.
- Firebase revocation checking (`check_revoked=True`) with 401/503 separation.

---

## Prioritized Remediation Plan (Small Implementation Batches)

**Batch 1 — Critical defect fix (do first)**
- [ ] `backend/app/api/routers/scans.py:430`: replace `detail=str(e)` with generic `"Upload failed"`; log exception server-side with `exc_info`.
- [ ] Add unit test asserting upload 500 detail contains no endpoint/path/credential text.

**Batch 2 — Reliability**
- [ ] Add a reaper/startup job that claims `PENDING` scans with stale or NULL `processing_started_at` (migration 005 column exists) and re-dispatches them; document `BackgroundTasks` single-worker limitation.
- [ ] Add integration test against a real Redis container: upload returns `503` when Redis down; `429`+`Retry-After` when exceeded.

**Batch 3 — Hygiene (low effort)**
- [ ] `scans.py:363-384`: `try/finally` to remove `temp_path` on upload failure (F-07).
- [ ] `storage.py:53-58`: log presigned-url failures server-side; consider surfacing distinct failure vs not-ready (F-06).
- [ ] Remove unused `celery` from `requirements.txt` or wire it in (F-08).
- [ ] Remove/fix `backend/test_pipeline_direct.py` and `backend/ingest_datasets.py` dead code (F-09).

**Batch 4 — Documentation & deployment hardening**
- [ ] Update `PRD.md:9` and `architecture.md:5` baseline references from `a0bbe53...` to `2c27e96` (F-11).
- [ ] Document production requirements: `MINIO_SECURE=True` + TLS, Redis `requirepass`/ACL, do not publish `6379`/`9000`/`9001` to public host networks (F-03/F-04).
- [ ] Narrow CORS `allow_methods`/`allow_headers` to frontend needs (F-10, optional).

**Batch 5 — Verification (no code change required from this audit)**
- [ ] Confirm hosted GitHub Actions result for `2c27e96` with repository access (preflight limitation).
- [ ] Re-run full backend suite against disposable PostgreSQL after Batch 1-3 (expect no regressions; baseline was 254–257 passed, 0 failed).

---

## Statement of What Has and Has Not Been Independently Verified

**Independently verified from repository evidence:**

1. Git state: `main`, clean tree, HEAD `2c27e96` == requested commit, remote `Gagann-09/NeuroScan-AI`.
2. Firebase auth: token verification with `check_revoked=True`; invalid/expired/revoked/cert-fetch → generic 401; missing ADC, init failure, `TransportError`, `TimeoutError`, and DB `SQLAlchemyError` → generic 503 with no detail leakage (code + 13 auth tests).
3. Authorization: ownership enforced on status/results/upload; cross-user 403; legacy ownerless scan → 403; presigned URLs only generated after ownership check (code + `TestCrossUserOwnership`).
4. Upload validation: NIfTI-1 magic at offset 344; bounded gzip streaming with CRC/ISIZE trailer; decompressed 200 MiB cap; dimension/datatype/voxel-count/`vox_offset`/payload-size consistency; truncated/trailing/concatenated rejected; 100 MB/modality cap; 1 KB minimum; compensation cleanup on failure (code + 47 NIfTI safety tests + upload-compensation tests).
5. Rate limiting: Redis Lua fixed-window, atomic, per-Firebase-UID, `429`+`Retry-After`, fail-closed `503`, auth-first, test-only override/bypass (code + 15 rate-limiting tests; bypass applied only in upload-compensation tests at HEAD `2c27e96` because CI lacks Redis).
6. CORS: wildcard rejected anywhere (incl. mixed), credentials enabled, origins required and validated before middleware (code + `TestCORSConfiguration`).
7. Configuration: no hard-coded credential defaults in Python; required fields; `init_system.py` deleted with regression test scanning tracked `.py` (code + `TestConfigSecurity`, `TestAdminCredentialRemoval`).
8. Database: Alembic migrations 001→006, no `drop_all()`, sessions closed in `finally` (including the D-038 fix), ownership FK, artifact lifecycle check constraint, claim-locking concurrency — all 33 DB-backed tests pass against live PostgreSQL 16 (D-038/D-040).
9. Scientific integrity: research-prototype report title + disclaimer; raw sigmoid `max_tumor_probability`; Dice/IoU null at inference; XAI explicitly "gradient-based input saliency" with provenance, not Grad-CAM (code + XAI tests).

**Not independently verified (limitations):**

- Hosted GitHub Actions check result for the exact commit `2c27e96` — GitHub API returned 403 (no credentials available); prior commits in the series passed on the hosted runner per `memory.md` D-040/D-044, and local equivalent runs pass, but the exact-commit result is unconfirmed.
- Fail-closed `503` against a live Redis outage in production (only fake-Redis unit tests; CI bypasses the limiter because it runs no Redis service).
- Compressed real BraTS `.nii.gz` end-to-end against live MinIO (dataset-gated test skips when data absent; D-037).
- Frontend, ARMT-GAN model weights/training science, dataset, and scientific methodology — explicitly out of scope per audit boundaries.
- Any `.env` file contents or secret values — deliberately not read (boundary).

---

**Bottom line:** The codebase satisfies the P8 security gate's spirit — no critical or destructive-data defects, auth/infrastructure failure handled safely, CORS hardened, credentials de-hardcoded, rate limiting fail-closed, NIfTI uploads validated, DB migrations and session hygiene verified against PostgreSQL, and scientific claims limited to research-prototype language. The one confirmed defect that should be fixed before any exposure beyond a controlled demo is **F-01 (exception detail leakage on upload 500)**. The application is **not production-secure** and **not clinically validated**; it remains a research prototype per PRD §1 and rules.md #12.
