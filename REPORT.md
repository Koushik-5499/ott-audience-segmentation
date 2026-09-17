# Containerized Audience Segmentation & Personalization Service
## Implementation Report

### 1. Problem Understanding and Assumptions
The objective of this project is to build an automated, unsupervised audience segmentation system for an OTT streaming platform, enabling tailored content recommendations. 

**Assumptions & Constraints:**
- The provided dataset is purely synthetic. Therefore, "true" viewer patterns do not inherently exist beyond what the generator was programmed to output. We assume the generated dataset approximates real user behavior to a sufficient degree for proving the architecture.
- A lightweight, CPU-friendly model (KMeans) is preferred over deep learning for speed and reproducibility.
- Features are uniformly standardized since they operate on vastly different scales (e.g., watch hours vs session counts).

### 2. Dataset Description and Preprocessing
The input dataset (`synthetic_ott_users.csv`) consists of 2,000 synthetic viewer profiles.
**Preprocessing Steps:**
- **Deduplication:** Dropped 15 exactly duplicated rows.
- **Negative Value Handling:** Converted physically impossible negative values in `total_watch_time_hours` and `avg_session_mins` to `NaN`.
- **Imputation:** Filled missing numerical values (`NaN`) using the global median of the respective columns to avoid dropping sparse user data. Missing `top_genres` were imputed as empty strings.

### 3. Feature-Selection Rationale
Raw features were engineered to emphasize behavioral patterns without allowing genre dominance.
- **Behavioral Variables:** 
  - `log_watch_time`: Log1p transformed `total_watch_time_hours` to handle extreme right-skew.
  - `avg_session_mins`, `sessions_per_week`, `weekend_ratio` (used directly).
- **Content Preferences (Compact Genres):** Using 15 binary multi-hot genre columns (e.g. Action, Romance) mathematically swamped the 4 behavioral features, leading to poor clustering (Silhouette ~0.09). To correct this, genres were grouped into 3 broad indicator families (`genre_action_thriller`, `genre_comedy_family`, `genre_drama_romance`).
- **Diversity:** `num_genres` was extracted as a measure of user exploration.

### 4. Model and Hyperparameter Choices
- **Algorithm:** `KMeans` with `n_init=10` and `max_iter=300`.
- **Scaling:** `StandardScaler` to ensure features (0 to 1 weekend ratio vs 0 to 1000 watch hours) contribute equally.
- **Why not GMM?** We implemented a GMM baseline as well, but KMeans achieved slightly higher Silhouette scores in this synthetic space and is computationally cheaper. Both are tracked in our `clustering_metrics.json`.

### 5. How the Number of Clusters was Selected
We sweep `K` from 2 to 8. K is selected programmatically using a strict rule-based pipeline rather than arbitrary manual choice:
1. **Balance Filter:** Reject K if any cluster is <5% (too niche) or >50% (too dominant).
2. **Uniqueness Filter:** Reject K if two segments generate identical semantic profiles.
3. **Primary Metric:** Highest Silhouette score among remaining candidates.
4. **Noise Margin Rule:** If a smaller K has a silhouette score within `0.01` of the highest K, the smaller K is preferred to reduce unnecessary complexity.

*Result:* K=5 was selected (Silhouette: 0.4647) over K=6 (Silhouette: 0.4727) due to the noise margin rule.

### 6. Cluster Profiles and Segment Naming Logic
Segment names are completely data-driven. We calculate each cluster's mean for every feature, compare it to the global mean (z-score), and assign semantic labels.

**Engagement Profile Logic:**
- `Heavy-Engagement`: watch_time relative > 0.4 & session relative > 1.5
- `Low-Activity`: watch_time relative < -0.4 & session relative < -1.5
- `Quick-Bite`: session relative < -1.5
- `Weekend-Binge`: weekend relative > 0.2
- `Frequent`: frequency relative > 0.5
- `Core`: Fallback.

**Content Profile Logic:**
- `Genre-Explorers`: diversity mean > 2.5
- `{top_genre}-Purists`: diversity mean < 1.5
- `{top_genre}-Fans`: Fallback.

*Discovered Segments:*
1. Heavy-Engagement action_thriller-Purists
2. Quick-Bite comedy_family-Fans
3. Weekend-Binge drama_romance-Fans
4. Quick-Bite Genre-Explorers
5. Low-Activity drama_romance-Fans

### 7. API Design
Built with **FastAPI** for high performance and automatic validation.
- **POST /recommend**: Validates incoming profiles (using Pydantic strict bounds, ensuring no negative time, bounded lists), runs the profile through the persisted `joblib` pipeline, and returns the segment assignment and rule-based recommendations.
- **GET /health**: Verifies if the API is alive and exposes whether the model volume has been successfully loaded yet.

### 8. Docker Architecture
A three-tier Docker Compose architecture leveraging a shared volume (`model_volume`):
- **trainer:** Reads data, builds and saves the model to `/app/models/pipeline.pkl`.
- **api:** Mounts `/app/models/` as Read-Only. Uses a background thread to lazily retry loading if the model isn't immediately present (preventing crash loops).
- **evaluator:** Waits for the API to report healthy (`model_loaded=true`), then fires tests and outputs `metrics.json`.

### 9. Evaluation Methodology and Metrics
Evaluation ensures both model quality and engineering robustness.
The Evaluator issues HTTP requests against the live API container. It verifies 25 unique test states covering:
- Happy paths (valid profiles hit the expected segments).
- Edge cases (missing fields, extreme values).
- Cold starts (ensuring graceful 503s before the model loads).
It automatically compiles latency percentiles and test results into `metrics.json`.

### 10. Results and Observations
- 25/25 Evaluator tests passed.
- P95 Latency for inference is ~3.5ms.
- Silhouette Score: 0.4647 (Excellent for behavioral clustering).
- Stability (ARI): 1.0000.

### 11. Failure Cases / Edge Cases Tested
1. Unknown/unseen genre values (Gracefully ignores and falls back).
2. Empty `top_genres` list (Defaults to generic popular content).
3. Zero watch time.
4. Absurdly large watch time (e.g. 5000 hours, correctly rejected by 422 Unprocessable Entity).
5. Missing required fields (Caught by Pydantic, returns 422).
6. Strings supplied where numeric value expected.
7. Negative values (Caught by Pydantic, returns 422).
8. Repeated requests (Returns strictly identical outputs).
9. Cold start (API responds with 503 instead of crashing).

### 12. Limitations and Future Improvements
- **Limitation:** The feature engineering manually groups genres. On a real platform, a learned embedding space (e.g., Word2Vec on viewing sequences) would be far more effective.
- **Limitation:** The personalization catalog is rule-based and static.
- **Improvement:** Connect the API to a live feature store and implement an A/B testing framework to measure real CTR (Click Through Rate) of the returned recommendations.

### 13. Reproducibility Instructions
To cleanly reproduce the entire environment from scratch:
```bash
docker compose down -v
docker compose up --build
```
Check `./output/metrics.json` after the Evaluator container exits with code 0.

### 14. What the team tried and changed during development
- **Attempt 1 (Multi-hot genres):** Initially used 15 multi-hot genre columns. The clustering grouped people purely by genre, ignoring engagement behaviors, resulting in a low Silhouette score of ~0.09.
- **Attempt 2 (Compact families):** We consolidated the 15 genres into 3 semantic families. Silhouette instantly jumped to ~0.46, and the clusters became significantly more behavior-oriented (e.g., distinguishing "Quick-Bite" vs "Weekend-Binge").
- **Pickling Bug:** We encountered an issue where `api` couldn't unpickle the trainer's model because the custom `OTTFeatureTransformer` was tied to the trainer's `__main__` namespace. We fixed this by gracefully injecting the transformer directly into the API's namespace.
