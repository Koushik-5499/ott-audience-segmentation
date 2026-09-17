# System Architecture

## Overview
The system follows a three-service containerized architecture orchestrated by Docker Compose. The components are decoupled, allowing modular development and testing.

## Components

### 1. Trainer Service
- **Responsibility:** Data loading, preprocessing, model training, and persistence.
- **Workflow:** 
  1. Reads tabular dataset from `/app/data`. (Features to be finalized after the actual dataset is provided).
  2. Applies `StandardScaler`.
  3. Trains unsupervised clustering model (e.g., KMeans).
  4. Generates segment metadata (centroid names, sizes, characteristics).
  5. Saves the trained pipeline (scaler + model + metadata) to a shared Docker volume `/models`.
- **Exit Strategy:** Exits cleanly after model persistence.

### 2. API Service (FastAPI)
- **Responsibility:** Exposes REST endpoints for health monitoring and personalization inference.
- **Workflow:**
  1. Waits for model to be available in the shared `/models` volume at startup.
  2. Loads the persisted pipeline.
  3. Rejects invalid requests gracefully.
  4. Maps generated segments to predefined recommendation logic.
- **Endpoints:**
  - `GET /health`: Returns `{ "status": "ok", "model_loaded": true/false }`
  - `POST /recommend`: Accepts user behavioral profile, scales it, runs inference, and returns segment info alongside recommendations.

### 3. Evaluator Service
- **Responsibility:** Validates API correctness and ML clustering quality.
- **Workflow:**
  1. Polls `GET /health` until the API is fully ready.
  2. Submits representative edge-case and valid test profiles to `POST /recommend`.
  3. Analyzes cluster evidence (Silhouette score, cluster sizes).
  4. Writes final results to `metrics.json` locally.
- **Exit Strategy:** Exits cleanly after generating evidence.

## Shared Infrastructure
- **Volumes:** A shared volume mapped to `/models` allows the trainer to write artifacts that the API reads without restarting containers.
- **Network:** An isolated Docker bridge network allows components to communicate securely (e.g., Evaluator calling `http://api:8000`).
