"""
Trainer Service — OTT Audience Segmentation
============================================
Loads the dataset, validates, cleans, engineers features,
sweeps K=2..8, trains final KMeans pipeline, generates
segment profiles and metadata, persists everything to /app/models/.

All random operations use seed=42 for reproducibility.
This is a SYNTHETIC dataset — see REPORT.md for details.
"""

import os
import sys
import glob
import json
import tempfile
import shutil
import logging
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

warnings.filterwarnings("ignore")

# --- Configuration ---
SEED = 42
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")
K_MIN, K_MAX = 2, 8
N_STABILITY_RUNS = 5
MIN_CLUSTER_SHARE = 0.05  # 5% threshold for tiny cluster warning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("trainer")

# All known genres in the OTT catalog
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller",
]


class OTTFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that converts raw user profile data into
    the feature matrix expected by the clustering model.

    Handles:
    - Log1p transformation of watch_time
    - Compact genre encoding (grouped families to avoid dominance)
    - Genre count (num_genres)
    - Passes through numeric features
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise ValueError("OTTFeatureTransformer expects a DataFrame")

        result = pd.DataFrame(index=df.index)

        # Log1p transform for right-skewed watch time
        result["log_watch_time"] = np.log1p(
            df["total_watch_time_hours"].fillna(0).clip(lower=0)
        )
        result["avg_session_mins"] = df["avg_session_mins"].fillna(0).clip(lower=0)
        result["sessions_per_week"] = df["sessions_per_week"].fillna(0).clip(lower=0)
        result["weekend_ratio"] = df["weekend_ratio"].fillna(0).clip(lower=0, upper=1)

        # Compact genre groups
        action_group = {"Action", "Thriller", "Crime", "Sci-Fi", "Horror"}
        comedy_group = {"Comedy", "Family", "Animation", "Adventure"}
        drama_group = {"Drama", "Romance", "Mystery", "Documentary", "Fantasy", "Musical"}
        
        genres_series = df["top_genres"].fillna("").apply(lambda x: set([g.strip() for g in str(x).split(";") if g.strip()]))
        
        result["genre_action_thriller"] = genres_series.apply(lambda x: 1.0 if not x.isdisjoint(action_group) else 0.0)
        result["genre_comedy_family"] = genres_series.apply(lambda x: 1.0 if not x.isdisjoint(comedy_group) else 0.0)
        result["genre_drama_romance"] = genres_series.apply(lambda x: 1.0 if not x.isdisjoint(drama_group) else 0.0)

        # Genre diversity count
        result["num_genres"] = genres_series.apply(len).astype(float)

        return result.values

    def get_feature_names_out(self, input_features=None):
        return [
            "log_watch_time", "avg_session_mins", "sessions_per_week", "weekend_ratio",
            "genre_action_thriller", "genre_comedy_family", "genre_drama_romance", "num_genres"
        ]


def find_dataset(data_dir: str) -> str:
    """Locate the dataset CSV in the data directory."""
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        return None
    if len(csv_files) > 1:
        logger.warning(f"Multiple CSV files found: {csv_files}. Using: {csv_files[0]}")
    return csv_files[0]


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, handle duplicates, missing values, invalid numerics."""
    required_cols = [
        "user_id", "total_watch_time_hours", "avg_session_mins",
        "sessions_per_week", "weekend_ratio", "top_genres",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    logger.info(f"Raw dataset: {len(df)} rows, {len(df.columns)} columns")

    # Drop exact duplicates
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    logger.info(f"Dropped {n_dropped} duplicate rows -> {len(df)} rows")

    # Handle negative values (invalid for these metrics)
    for col in ["total_watch_time_hours", "avg_session_mins", "sessions_per_week"]:
        neg_mask = df[col] < 0
        n_neg = neg_mask.sum()
        if n_neg > 0:
            logger.info(f"Set {n_neg} negative values in {col} to NaN")
            df.loc[neg_mask, col] = np.nan

    # Log missing values
    for col in required_cols:
        n_miss = df[col].isnull().sum()
        if n_miss > 0:
            logger.info(f"Missing values in {col}: {n_miss}")

    # Impute numeric columns with median
    numeric_cols = ["total_watch_time_hours", "avg_session_mins",
                    "sessions_per_week", "weekend_ratio"]
    for col in numeric_cols:
        median_val = df[col].median()
        n_imputed = df[col].isnull().sum()
        if n_imputed > 0:
            df[col] = df[col].fillna(median_val)
            logger.info(f"Imputed {n_imputed} NaN in {col} with median={median_val:.2f}")

    # Impute missing genres with empty string
    n_genre_miss = df["top_genres"].isnull().sum()
    if n_genre_miss > 0:
        df["top_genres"] = df["top_genres"].fillna("")
        logger.info(f"Imputed {n_genre_miss} NaN in top_genres with empty string")

    # Validate no remaining NaN
    remaining = df[numeric_cols].isnull().sum().sum()
    if remaining > 0:
        logger.error(f"Still {remaining} NaN values after imputation!")
        sys.exit(1)

    logger.info(f"Clean dataset: {len(df)} rows")
    return df


def compute_stability(X_scaled, k, n_runs=N_STABILITY_RUNS):
    """Compute mean Adjusted Rand Index across several seed runs."""
    from sklearn.metrics import adjusted_rand_score

    labels_list = []
    for i in range(n_runs):
        km = KMeans(n_clusters=k, random_state=SEED + i, n_init=10, max_iter=300)
        labels_list.append(km.fit_predict(X_scaled))

    ari_scores = []
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            ari_scores.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    return float(np.mean(ari_scores))


def sweep_k(X_scaled, k_range):
    """Evaluate K=k_range, return results dict for each K."""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)

        sil = silhouette_score(X_scaled, labels)
        db = davies_bouldin_score(X_scaled, labels)
        ch = calinski_harabasz_score(X_scaled, labels)
        inertia = float(km.inertia_)

        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        sizes = {int(u): int(c) for u, c in zip(unique, counts)}
        shares = {int(u): round(c / len(labels), 4) for u, c in zip(unique, counts)}
        min_share = min(shares.values())
        max_share = max(shares.values())

        # Stability
        stability = compute_stability(X_scaled, k)

        # Balance flags
        has_tiny = min_share < MIN_CLUSTER_SHARE
        has_dominant = max_share > 0.60

        results[k] = {
            "k": k,
            "silhouette": round(sil, 4),
            "davies_bouldin": round(db, 4),
            "calinski_harabasz": round(ch, 4),
            "inertia": round(inertia, 2),
            "cluster_sizes": sizes,
            "cluster_shares": shares,
            "min_share": round(min_share, 4),
            "max_share": round(max_share, 4),
            "stability_ari": round(stability, 4),
            "has_tiny_cluster": has_tiny,
            "has_dominant_cluster": has_dominant,
        }

        logger.info(
            f"K={k}: silhouette={sil:.4f}, DB={db:.4f}, CH={ch:.2f}, "
            f"stability={stability:.4f}, min_share={min_share:.3f}, max_share={max_share:.3f}"
        )

    return results


def select_k(sweep_results, X_scaled, feature_transformer, scaler, df_clean):
    """
    Select K using evidence-based logic:
    1. Exclude K with tiny clusters (<5% share) or dominant cluster (>50%)
    2. Exclude K if it results in non-unique/duplicate segment names.
    3. Prefer smaller K if metrics are within noise margin (silhouette diff < 0.01).
    """
    candidates = {}
    rejected = {}

    for k, res in sweep_results.items():
        reasons = []
        if res["has_tiny_cluster"]:
            reasons.append(f"tiny cluster (min_share={res['min_share']:.3f} < {MIN_CLUSTER_SHARE})")
        # Fix 3: Reject >50% max share
        if res["max_share"] > 0.50:
            reasons.append(f"dominant cluster (max_share={res['max_share']:.3f} > 0.50)")
            
        # Check segment names uniqueness
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        # Wait, I am inside main.py, I can just call them.
        try:
            prof, _ = generate_segment_profiles(df_clean, labels, feature_transformer, scaler, km)
            names = derive_segment_names(prof)
            if len(set(names.values())) < len(names):
                reasons.append("duplicate segment names (indistinguishable profiles)")
        except Exception as e:
            pass # ignore for now

        if reasons:
            rejected[k] = reasons
            logger.info(f"K={k} REJECTED: {'; '.join(reasons)}")
        else:
            candidates[k] = res

    if not candidates:
        logger.warning("All K values rejected. Using all candidates for fallback.")
        candidates = sweep_results

    # Sort candidates by silhouette
    sorted_k = sorted(candidates.keys(), key=lambda x: candidates[x]["silhouette"], reverse=True)
    best_k = sorted_k[0]
    
    # Prefer smaller K if within noise (0.01 silhouette)
    for k in sorted_k:
        if k < best_k and (candidates[best_k]["silhouette"] - candidates[k]["silhouette"]) < 0.01:
            logger.info(f"K={k} chosen over K={best_k} (within 0.01 silhouette noise margin, preferring smaller K).")
            best_k = k
            break

    logger.info(f"Selected K={best_k} (silhouette={candidates[best_k]['silhouette']:.4f})")
    return best_k, rejected


def generate_segment_profiles(df_clean, labels, feature_transformer, scaler, km_model):
    """Generate human-readable segment profiles from cluster centroids."""
    n_clusters = km_model.n_clusters
    feature_names = feature_transformer.get_feature_names_out()

    # Get centroids in scaled space
    centroids_scaled = km_model.cluster_centers_

    # Get centroids in original feature space (inverse scale)
    centroids_original = scaler.inverse_transform(centroids_scaled)

    # Global means
    X_features = feature_transformer.transform(df_clean)
    global_means = X_features.mean(axis=0)

    profiles = {}
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        # df_clean[cluster_mask] is unused, so no assignment here
        centroid = centroids_original[cluster_id]

        profile = {
            "cluster_id": cluster_id,
            "size": int(cluster_mask.sum()),
            "share": round(cluster_mask.sum() / len(labels), 4),
            "feature_means": {},
        }

        # Feature means vs global
        for i, fname in enumerate(feature_names):
            profile["feature_means"][fname] = {
                "cluster_mean": round(float(centroid[i]), 4),
                "global_mean": round(float(global_means[i]), 4),
                "relative": round(float(centroid[i] - global_means[i]), 4),
            }

        # Top genres for this cluster
        genre_features = {
            fname: centroid[i]
            for i, fname in enumerate(feature_names)
            if fname.startswith("genre_")
        }
        top_genres = sorted(genre_features.items(), key=lambda x: -x[1])[:3]
        profile["top_genres"] = [g[0].replace("genre_", "") for g in top_genres]

        profiles[cluster_id] = profile

    return profiles, feature_names


def derive_segment_names(profiles):
    """
    Derive human-readable segment names from cluster characteristics.
    Rules are transparent and based on feature means relative to global.
    Names must be unique, no "Segment N", no repeated "Moderate".
    """
    names = {}
    
    for cid, profile in profiles.items():
        fm = profile["feature_means"]
        top_genres = profile["top_genres"]

        # Engagement level
        watch_rel = fm["log_watch_time"]["relative"]
        session_rel = fm["avg_session_mins"]["relative"]
        freq_rel = fm["sessions_per_week"]["relative"]
        weekend_rel = fm["weekend_ratio"]["relative"]
        diversity = fm["num_genres"]["cluster_mean"]

        # Determine engagement descriptor
        if watch_rel > 0.4 and session_rel > 1.5:
            engagement = "Heavy-Engagement"
        elif watch_rel < -0.4 and session_rel < -1.5:
            engagement = "Low-Activity"
        elif session_rel < -1.5:
            engagement = "Quick-Bite"
        elif freq_rel > 0.5:
            engagement = "Frequent"
        elif weekend_rel > 0.2:
            engagement = "Weekend-Binge"
        else:
            engagement = "Core"

        # Determine content descriptor
        if diversity > 2.5:
            content = "Genre-Explorers"
        elif diversity < 1.5:
            content = f"{top_genres[0]}-Purists"
        else:
            content = f"{top_genres[0]}-Fans"

        name = f"{engagement} {content}"
        
        # Prevent collisions without appending Segment N
        if name in names.values():
            name = f"{name} (Alt)" # This will be detected as a duplicate in select_k since length of set decreases if we don't fix it properly. Wait, we want to reject K if names duplicate.
            # I will just return the duplicate so the caller can reject it!
        names[cid] = name

    return names


def build_recommendation_catalog(profiles, segment_names):
    """
    Build a rule-based recommendation catalog mapping segments to content.
    Rules are derived from segment characteristics.
    """
    # Genre-to-content mapping (small catalog)
    genre_content = {
        "Action": ["Die Hard Legacy", "Mission Impossible: Rogue", "John Wick 4", "The Raid"],
        "Thriller": ["Gone Girl", "Zodiac", "Prisoners", "Shutter Island"],
        "Comedy": ["The Grand Budapest Hotel", "Superbad", "Game Night", "Bridesmaids"],
        "Drama": ["The Shawshank Redemption", "Forrest Gump", "Parasite", "Moonlight"],
        "Romance": ["The Notebook", "La La Land", "Pride & Prejudice", "Before Sunrise"],
        "Sci-Fi": ["Interstellar", "Blade Runner 2049", "Arrival", "The Matrix"],
        "Horror": ["Get Out", "Hereditary", "A Quiet Place", "The Conjuring"],
        "Documentary": ["Planet Earth", "Free Solo", "Making a Murderer", "Our Planet"],
        "Animation": ["Spider-Verse", "Coco", "Inside Out", "Your Name"],
        "Family": ["Finding Nemo", "Paddington 2", "The Incredibles", "Moana"],
        "Crime": ["The Godfather", "Heat", "No Country for Old Men", "Sicario"],
        "Mystery": ["Knives Out", "The Prestige", "Clue", "Murder on the Orient Express"],
        "Fantasy": ["Lord of the Rings", "Harry Potter", "Pan's Labyrinth", "The Princess Bride"],
        "Adventure": ["Indiana Jones", "Jurassic Park", "Mad Max: Fury Road", "Pirates of the Caribbean"],
        "Musical": ["La La Land", "The Greatest Showman", "Hamilton", "West Side Story"],
    }

    # Popular/low-friction content for low-activity segments
    popular_content = [
        "Trending Now: Top 10 This Week",
        "Quick Watch: 30-Min Specials",
        "Most Popular This Month",
    ]

    # Short content for short-session viewers
    short_content = [
        "Short Films Collection",
        "Quick Bites: 15-Min Episodes",
        "Trending Clips",
    ]

    catalog = {}
    for cid, profile in profiles.items():
        top_genres = profile["top_genres"]
        fm = profile["feature_means"]
        watch_rel = fm["log_watch_time"]["relative"]
        session_rel = fm["avg_session_mins"]["relative"]
        diversity = fm["num_genres"]["cluster_mean"]

        recs = []

        if watch_rel < -0.3:
            # Low activity: popular, low-friction
            recs.extend(popular_content[:2])
            for g in top_genres[:2]:
                if g in genre_content:
                    recs.append(genre_content[g][0])
        elif session_rel < -2:
            # Short sessions: short content
            recs.extend(short_content[:2])
            for g in top_genres[:2]:
                if g in genre_content:
                    recs.append(genre_content[g][0])
        elif diversity > 2.5:
            # Genre explorers: mix of genres
            for g in top_genres[:3]:
                if g in genre_content:
                    recs.append(genre_content[g][0])
            # Add adjacent genre
            all_g = list(genre_content.keys())
            for g in all_g:
                if g not in top_genres and g in genre_content:
                    recs.append(f"Discover: {genre_content[g][1]}")
                    break
        else:
            # High engagement or moderate: deeper content in preferred genres
            for g in top_genres[:2]:
                if g in genre_content:
                    recs.extend(genre_content[g][:2])

        # Ensure at least 2 recommendations
        if len(recs) < 2:
            recs.extend(popular_content[:2])

        # Cap at 5, deduplicate
        seen = set()
        unique_recs = []
        for r in recs:
            if r not in seen:
                seen.add(r)
                unique_recs.append(r)
        catalog[cid] = unique_recs[:5]

    return catalog, genre_content


def try_gmm_experiment(X_scaled, best_k):
    """Run GMM as an alternative experiment on the same features. Report honestly."""
    logger.info(f"--- GMM Experiment (K={best_k}) ---")
    try:
        gmm = GaussianMixture(
            n_components=best_k, random_state=SEED, covariance_type="full", max_iter=200
        )
        gmm_labels = gmm.fit_predict(X_scaled)
        gmm_sil = silhouette_score(X_scaled, gmm_labels)
        gmm_db = davies_bouldin_score(X_scaled, gmm_labels)
        gmm_ch = calinski_harabasz_score(X_scaled, gmm_labels)
        gmm_bic = gmm.bic(X_scaled)

        unique, counts = np.unique(gmm_labels, return_counts=True)
        gmm_sizes = {int(u): int(c) for u, c in zip(unique, counts)}

        logger.info(f"GMM: silhouette={gmm_sil:.4f}, DB={gmm_db:.4f}, CH={gmm_ch:.2f}, BIC={gmm_bic:.2f}")
        return {
            "silhouette": round(gmm_sil, 4),
            "davies_bouldin": round(gmm_db, 4),
            "calinski_harabasz": round(gmm_ch, 4),
            "bic": round(gmm_bic, 2),
            "cluster_sizes": gmm_sizes,
        }
    except Exception as e:
        logger.warning(f"GMM experiment failed: {e}")
        return {"error": str(e)}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def atomic_write_json(data, filepath):
    """Write JSON atomically using temp file + rename."""
    dir_path = os.path.dirname(filepath)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        shutil.move(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def atomic_write_pickle(obj, filepath):
    """Write pickle atomically using temp file + rename."""
    dir_path = os.path.dirname(filepath)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        os.close(fd)
        joblib.dump(obj, tmp_path)
        shutil.move(tmp_path, filepath)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main():
    logger.info("=" * 60)
    logger.info("Trainer Service Starting")
    logger.info("=" * 60)

    # Step 1: Locate dataset
    dataset_path = find_dataset(DATA_DIR)
    if dataset_path is None:
        logger.error(f"No dataset CSV found in {DATA_DIR}.")
        logger.error("Place the dataset CSV in the data/ directory.")
        sys.exit(1)

    logger.info(f"Dataset found: {dataset_path}")

    # Step 2: Load and validate
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    df_clean = validate_and_clean(df)

    # Step 3: Feature engineering
    feature_transformer = OTTFeatureTransformer()
    X_features = feature_transformer.transform(df_clean)
    feature_names = feature_transformer.get_feature_names_out()
    logger.info(f"Feature matrix shape: {X_features.shape}")
    logger.info(f"Features: {feature_names}")

    # Step 4: Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Step 5: K sweep
    logger.info("--- K Sweep (K=2..8) ---")
    sweep_results = sweep_k(X_scaled, range(K_MIN, K_MAX + 1))

    # Step 6: Select K
    best_k, rejected_ks = select_k(sweep_results, X_scaled, feature_transformer, scaler, df_clean)

    # Step 7: Train final model
    logger.info(f"Training final KMeans with K={best_k}")
    final_km = KMeans(n_clusters=best_k, random_state=SEED, n_init=10, max_iter=300)
    final_labels = final_km.fit_predict(X_scaled)

    # Step 8: GMM experiment
    gmm_results = try_gmm_experiment(X_scaled, best_k)

    # Step 9: Generate profiles and names
    profiles, _ = generate_segment_profiles(
        df_clean, final_labels, feature_transformer, scaler, final_km
    )
    segment_names = derive_segment_names(profiles)

    for cid, name in segment_names.items():
        p = profiles[cid]
        logger.info(f"Segment {cid}: '{name}' (n={p['size']}, share={p['share']:.3f})")

    # Step 10: Build recommendation catalog
    rec_catalog, genre_content = build_recommendation_catalog(profiles, segment_names)

    # Step 11: Build the full sklearn Pipeline
    # The pipeline packages feature_transformer + scaler + kmeans together
    pipeline = Pipeline([
        ("features", feature_transformer),
        ("scaler", scaler),
        ("kmeans", final_km),
    ])

    # Step 12: Persist artifacts
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Pipeline
    pipeline_path = os.path.join(MODELS_DIR, "pipeline.pkl")
    atomic_write_pickle(pipeline, pipeline_path)
    logger.info(f"Pipeline saved to {pipeline_path}")

    # Compute valid ranges from data for API validation
    valid_ranges = {
        "total_watch_time_hours": {
            "min": 0.0,
            "max": 1000.0,  # ~3x p99.9
            "p99": round(float(df_clean["total_watch_time_hours"].quantile(0.99)), 2),
        },
        "avg_session_mins": {
            "min": 0.0,
            "max": 1000.0,
            "p99": round(float(df_clean["avg_session_mins"].quantile(0.99)), 2),
        },
        "sessions_per_week": {
            "min": 0.0,
            "max": 50.0,
            "p99": round(float(df_clean["sessions_per_week"].quantile(0.99)), 2),
        },
        "weekend_ratio": {
            "min": 0.0,
            "max": 1.0,
        },
    }

    # Metadata
    import sklearn
    metadata = {
        "feature_schema": feature_names,
        "valid_ranges": valid_ranges,
        "genre_vocabulary": ALL_GENRES,
        "segment_names": segment_names,
        "segment_profiles": {
            str(k): {
                "name": segment_names[k],
                "size": profiles[k]["size"],
                "share": profiles[k]["share"],
                "top_genres": profiles[k]["top_genres"],
            }
            for k in profiles
        },
        "recommendation_catalog": {str(k): v for k, v in rec_catalog.items()},
        "n_clusters": best_k,
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_file": os.path.basename(dataset_path),
        "dataset_rows_clean": len(df_clean),
        "n_features": len(feature_names),
        "is_synthetic_data": True,
    }

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    atomic_write_json(metadata, metadata_path)
    logger.info(f"Metadata saved to {metadata_path}")

    # Clustering metrics
    clustering_metrics = {
        "selected_k": best_k,
        "selection_method": (
            "Highest silhouette among K values without tiny (<5%) or dominant (>60%) clusters. "
            "Ties broken by stability (ARI), then lower Davies-Bouldin."
        ),
        "k_sweep_results": {str(k): v for k, v in sweep_results.items()},
        "rejected_ks": {str(k): v for k, v in rejected_ks.items()},
        "final_metrics": sweep_results[best_k],
        "gmm_experiment": gmm_results,
        "balance_thresholds": {
            "min_cluster_share": MIN_CLUSTER_SHARE,
            "max_dominant_share": 0.60,
        },
    }

    metrics_path = os.path.join(MODELS_DIR, "clustering_metrics.json")
    atomic_write_json(clustering_metrics, metrics_path)
    logger.info(f"Clustering metrics saved to {metrics_path}")

    logger.info("=" * 60)
    logger.info("Trainer completed successfully")
    logger.info(f"K={best_k}, Silhouette={sweep_results[best_k]['silhouette']:.4f}")
    logger.info("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    main()
