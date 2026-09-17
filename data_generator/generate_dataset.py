#!/usr/bin/env python3
"""
Synthetic OTT User Activity Dataset Generator
==============================================
Generates a SYNTHETIC dataset for development and testing.
This is NOT the official hackathon dataset.

The generator creates realistic but artificial OTT viewer behavior data
with overlapping behavioral distributions (not hard-coded groups).

Fixed seed: 42
Output: data/synthetic_ott_users.csv
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_USERS = 2000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_ott_users.csv")

# All available genres in the OTT catalog
ALL_GENRES = [
    "Action", "Thriller", "Comedy", "Drama", "Romance",
    "Sci-Fi", "Horror", "Documentary", "Animation", "Family",
    "Crime", "Mystery", "Fantasy", "Adventure", "Musical"
]


def generate_dataset(seed: int = SEED, n_users: int = N_USERS) -> pd.DataFrame:
    """Generate synthetic OTT user activity data with realistic quirks."""
    rng = np.random.RandomState(seed)

    # --- Base behavioral distributions (overlapping, not clustered) ---
    # Watch time: mixture of light, moderate, heavy viewers
    mix = rng.choice([0, 1, 2], size=n_users, p=[0.35, 0.40, 0.25])
    watch_time = np.where(
        mix == 0,
        rng.exponential(scale=8.0, size=n_users),        # light viewers
        np.where(
            mix == 1,
            rng.normal(loc=35.0, scale=15.0, size=n_users),  # moderate
            rng.normal(loc=80.0, scale=25.0, size=n_users),   # heavy
        )
    )
    # Clip negatives from normal dist tails (but leave a few for testing)
    watch_time = np.maximum(watch_time, 0.0)

    # Session duration: correlated with watch time but with noise
    avg_session = 15.0 + 0.6 * watch_time + rng.normal(0, 10, n_users)
    avg_session = np.maximum(avg_session, 1.0)

    # Sessions per week: inversely correlated with session length for some users
    sessions_per_week = rng.poisson(lam=3.5, size=n_users).astype(float)
    sessions_per_week += rng.normal(0, 0.5, n_users)
    sessions_per_week = np.maximum(sessions_per_week, 0.0)

    # Weekend ratio: 0-1, some users are weekend-heavy
    weekend_ratio = rng.beta(a=2.0, b=3.0, size=n_users)

    # Genre preferences: each user gets 1-4 top genres as semicolon-delimited string
    top_genres = []
    for i in range(n_users):
        n_genres = rng.choice([1, 2, 3, 4], p=[0.20, 0.40, 0.30, 0.10])
        chosen = rng.choice(ALL_GENRES, size=n_genres, replace=False)
        top_genres.append(";".join(chosen))

    # User IDs
    user_ids = [f"USR-{i:05d}" for i in range(n_users)]

    df = pd.DataFrame({
        "user_id": user_ids,
        "total_watch_time_hours": np.round(watch_time, 2),
        "avg_session_mins": np.round(avg_session, 1),
        "sessions_per_week": np.round(sessions_per_week, 1),
        "weekend_ratio": np.round(weekend_ratio, 3),
        "top_genres": top_genres,
    })

    # --- Introduce realistic data quality issues ---

    # ~2% missing values in numeric columns (scattered)
    numeric_cols = ["total_watch_time_hours", "avg_session_mins",
                    "sessions_per_week", "weekend_ratio"]
    for col in numeric_cols:
        mask = rng.random(n_users) < 0.02
        df.loc[mask, col] = np.nan

    # ~1% missing genres
    genre_mask = rng.random(n_users) < 0.01
    df.loc[genre_mask, "top_genres"] = np.nan

    # Add ~15 duplicate rows (exact copies of random rows)
    dup_indices = rng.choice(n_users, size=15, replace=False)
    duplicates = df.iloc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # Inject ~5 negative values in watch_time (invalid)
    neg_indices = rng.choice(len(df), size=5, replace=False)
    df.loc[neg_indices, "total_watch_time_hours"] = -rng.uniform(1, 10, size=5).round(2)

    # Inject ~3 negative session values (invalid)
    neg_sess = rng.choice(len(df), size=3, replace=False)
    df.loc[neg_sess, "avg_session_mins"] = -rng.uniform(1, 20, size=3).round(1)

    # Inject a few outliers (extremely high values)
    outlier_idx = rng.choice(len(df), size=4, replace=False)
    df.loc[outlier_idx, "total_watch_time_hours"] = rng.uniform(200, 500, size=4).round(2)

    outlier_sess = rng.choice(len(df), size=3, replace=False)
    df.loc[outlier_sess, "avg_session_mins"] = rng.uniform(300, 600, size=3).round(1)

    # Shuffle rows
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Synthetic dataset generated: {OUTPUT_FILE}")
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] This is SYNTHETIC data, NOT the official hackathon dataset.")


if __name__ == "__main__":
    main()
