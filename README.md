# OTT Audience Segmentation & Personalization Service

> **IMPORTANT NOTICE:** This repository uses a **synthetic data generator** because the official dataset was not provided. All model artifacts and evaluation metrics generated reflect patterns embedded in this synthetic dataset. Please refer to `REPORT.md` for architectural and design decisions.

## Project Structure
- `data_generator/` - Creates `data/synthetic_ott_users.csv`.
- `trainer/` - Validates data, extracts features, performs dynamic K-selection, and atomically saves `pipeline.pkl` and `metadata.json` to the shared volume.
- `api/` - FastAPI service that lazily loads the pipeline, providing strict validation and rule-based segment mapping for recommendations.
- `evaluator/` - Comprehensive end-to-end integration test suite covering health, clustering metrics validation, and edge cases (missing fields, outliers, unseen genres).
- `models/` (generated) - Shared volume for artifacts.
- `output/` (generated) - Mount point for `metrics.json`.

## Quickstart

Run the full end-to-end suite with a single command:
```bash
docker compose up --build
```
This will:
1. Spin up the `trainer` service, cluster the synthetic data, and output the model.
2. Spin up the `api` service (which waits for the trainer to complete).
3. Spin up the `evaluator` service (which waits for the API to become healthy).
4. Run all evaluation tests and write results to `output/metrics.json`.

## Development (Local)
To generate a new dataset:
```bash
python3 data_generator/generate_dataset.py
```

## Security & Reliability
- **Non-root**: All Docker containers drop privileges and run as `appuser`.
- **Atomic Writes**: The trainer avoids corrupting model artifacts via atomic file swapping.
- **Determinism**: Fixed seeds ensure consistent cluster boundaries across runs.
