# Product Requirements Document (PRD)

## 1. Objective
Build a lightweight, containerized machine-learning service that understands OTT viewer behavior, groups users into meaningful behavioral segments using unsupervised clustering, and exposes personalized recommendations through a REST API.

## 2. Business Need
OTT platforms serve diverse audiences with different interests, engagement levels, and session patterns. A single recommendation strategy is insufficient. This automated audience intelligence service discovers user groups with similar behaviors without relying on manual labels, enabling targeted downstream personalization.

## 3. Core Requirements
- **Unsupervised Clustering:** Segment audience using lightweight algorithms (KMeans or GMM) on behavioral features (features to be finalized after the actual dataset is provided).
- **Three-Service Architecture:**
  - **Trainer:** Scales data, trains clustering model, and persists the pipeline.
  - **API:** Loads the model and provides segments + recommendations for users.
  - **Evaluator:** Tests the API, calculates clustering/evaluation metrics, and generates a `metrics.json` report.
- **Dockerized Deployment:** Reproducible system orchestrated by Docker Compose with proper health checks and startup ordering.
- **Rule-Based Personalization:** Map machine-generated cluster IDs to transparent recommendation behaviors based on predefined logic.

## 4. Input & Output Constraints
- **Data:** Tabular dataset containing user activity records. *Note: Features to be finalized after the actual dataset is provided.*
- **API Contracts:**
  - `POST /recommend`: Accepts user profile (JSON), returns segment ID/name, recommendations, and distance to centroid.
  - `GET /health`: Returns status and model loading state.
- **Evaluation:** Must produce a `metrics.json` capturing clustering quality (silhouette score) and API correctness.

## 5. System Constraints
- Compute: CPU-friendly; no GPU required.
- External Dependencies: No paid APIs or heavy LLMs.
- Portability: Run completely offline after initial setup, reproducible with a single `docker compose up` command.
