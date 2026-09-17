# Implementation Plan — Containerized Audience Segmentation Service

## Dataset Status

> [!WARNING]
> **Official Dataset Missing.** All dataset-dependent phases (2–13) are paused.
> Place the official hackathon dataset file (CSV) in `data/` before proceeding.
> No synthetic, generated, or benchmark data is used or accepted.

---

## Phase 1 — Project Audit (COMPLETE)

All scaffolding files have been reviewed and corrected:

| File | Change |
|------|--------|
| `docker-compose.yml` | `trainer` now uses `condition: service_completed_successfully`; `evaluator` mounts project root, uses env vars; added `start_period` to healthcheck |
| `.gitignore` | Added `secrets/`, `.env.*`, `*.swp`, `.DS_Store`, pinned model artifacts |
| `trainer/requirements.txt` | Pinned all dependency versions |
| `api/requirements.txt` | Pinned all dependency versions |
| `evaluator/requirements.txt` | Pinned version |
| `trainer/Dockerfile` | Added non-root user |
| `api/Dockerfile` | Added `curl` (required by healthcheck), added non-root user |
| `evaluator/Dockerfile` | Added non-root user |
| `trainer/main.py` | Validates dataset presence; exits 1 if missing; clear TODO markers |
| `api/main.py` | Lifespan-based model loading; 503 if model missing; no stack traces |
| `evaluator/main.py` | Health poll with timeout (120s); validates `model_loaded: true`; writes metrics.json |

---

## Phase 2 — Dataset Inspection (PENDING OFFICIAL DATASET)

Once the official dataset is placed in `data/`, inspect **without modifying**:
1. Filename, row count, column count
2. All column names and data types
3. Missing values per column
4. Duplicate rows
5. Invalid / negative numeric values
6. Unique categorical values (genres, user segments if any)
7. Numerical statistics (mean, std, min, max, percentiles)
8. Potential outliers (IQR method)
9. Determine: user-level records vs. activity events (aggregation needed?)

---

## Phase 3 — Feature Engineering (PENDING PHASE 2)

- Propose a compact behavioral feature set based on actual dataset columns only.
- Categories: engagement, session, content-preference, activity-pattern.
- **Do not use user_id as a feature.**
- **Do not create supervised labels.**
- Document preprocessing: imputation, encoding, clipping, scaling.
- Present feature proposal to user for approval before training.

---

## Phase 4 — Model Development (PENDING PHASE 3 APPROVAL)

- Use `KMeans` as baseline (`StandardScaler` → `KMeans`).
- Evaluate `K = 2` through `K = 8` (or adjusted range based on dataset size).
- For each K: compute silhouette score and inertia.
- Select final K using evidence (not arbitrarily).
- Fixed `random_state=42` for reproducibility.

---

## Phase 5 — Cluster Interpretation (PENDING PHASE 4)

- Calculate cluster sizes and feature summaries per cluster.
- Inspect centroids.
- **Derive human-readable segment names from observed characteristics only.**
- No pre-assigned segment names.

---

## Phase 6 — Trainer Service (PENDING PHASE 5)

Implement `trainer/main.py`:
- Load → Validate → Clean → Engineer → Preprocess → Evaluate K → Train → Profile → Persist.
- Save to `/app/models/`: `pipeline.pkl`, `metadata.json`, `clustering_metrics.json`.

---

## Phase 7 — API Service (PENDING PHASE 6)

Implement `api/main.py`:
- Finalize Pydantic schema from actual feature set.
- `POST /recommend`: validate input, preprocess, predict, compute distance, return response.
- `GET /health`: return `{status, model_loaded}`.
- Handle all edge cases gracefully (no stack traces exposed).

---

## Phase 8 — Personalization (PENDING PHASE 5)

- Implement transparent rule-based recommendation catalog.
- Rules derived from actual cluster characteristics.
- No LLMs, no paid APIs.

---

## Phase 9 — Evaluator Service (PENDING PHASE 7)

Implement complete test suite in `evaluator/main.py`:
- Health polling with 120s timeout.
- Valid profile tests, edge case tests, malformed input tests.
- Clustering metrics collection.
- Auto-generate `metrics.json` at `METRICS_PATH`.

---

## Phase 10 — metrics.json (PENDING PHASE 9)

Machine-readable output including:
`selected_k`, `candidate_k_results`, `silhouette_score`, `inertia`, `cluster_counts`,
`cluster_distribution`, API health result, test results, response times.

---

## Phase 11 — Docker (PENDING PHASE 10)

- All Dockerfiles: lean image, pinned versions, non-root user.
- `docker-compose.yml`: correct ordering, shared volume, healthcheck.

---

## Phase 12–16 — Verification, Reproducibility, Documentation, Security, Final Checklist

To be completed after Phase 11.
