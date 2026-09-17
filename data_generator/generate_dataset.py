#!/usr/bin/env python3
"""
Synthetic OTT User Activity Dataset Generator
==============================================
Generates a SYNTHETIC dataset for development and testing.
This is NOT the official hackathon dataset.

Archetypes (Hidden - NOT written to CSV):
1. Heavy-engagement, long-session, weekend-casual, Action/Thriller focused.
2. Frequent short-session, weekday-heavy, Comedy/Family focused.
3. Casual/low-activity, mixed-genre.
4. Genre-explorers, high-activity, highly diverse genres.
5. Weekend-binge, Drama/Romance focused.
"""

import os
import numpy as np
import pandas as pd

SEED = 42
N_USERS = 2000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "synthetic_ott_users.csv")

ALL_GENRES = [
    "Action", "Thriller", "Comedy", "Drama", "Romance",
    "Sci-Fi", "Horror", "Documentary", "Animation", "Family",
    "Crime", "Mystery", "Fantasy", "Adventure", "Musical"
]

def generate_dataset(seed: int = SEED, n_users: int = N_USERS) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    
    # 5 latent archetypes with realistic overlap
    # Proportions: 25%, 25%, 20%, 15%, 15%
    archetypes = rng.choice(5, size=n_users, p=[0.25, 0.25, 0.20, 0.15, 0.15])
    
    watch_time = np.zeros(n_users)
    avg_session = np.zeros(n_users)
    sessions_per_week = np.zeros(n_users)
    weekend_ratio = np.zeros(n_users)
    top_genres = []
    
    for i in range(n_users):
        arch = archetypes[i]
        if arch == 0:
            # Heavy engagement, long sessions
            wt = rng.normal(120, 30)
            sess = rng.normal(90, 20)
            spw = rng.normal(5, 1.5)
            wr = rng.beta(3, 3)
            # Action/Thriller focused
            genres = rng.choice(["Action", "Thriller", "Crime", "Sci-Fi"], size=rng.choice([1, 2]), replace=False)
        elif arch == 1:
            # Frequent short session
            wt = rng.normal(40, 15)
            sess = rng.normal(15, 5)
            spw = rng.normal(12, 3)
            wr = rng.beta(2, 5) # Weekday heavy
            genres = rng.choice(["Comedy", "Family", "Animation"], size=rng.choice([1, 2, 3]), replace=False)
        elif arch == 2:
            # Casual / low activity
            wt = rng.exponential(15)
            sess = rng.normal(30, 10)
            spw = rng.normal(1.5, 0.5)
            wr = rng.beta(1.5, 1.5)
            genres = rng.choice(ALL_GENRES, size=rng.choice([1, 2]), replace=False)
        elif arch == 3:
            # Genre explorers, high activity
            wt = rng.normal(90, 25)
            sess = rng.normal(45, 15)
            spw = rng.normal(7, 2)
            wr = rng.beta(2, 2)
            genres = rng.choice(ALL_GENRES, size=rng.choice([4, 5, 6]), replace=False)
        else:
            # Weekend binge
            wt = rng.normal(70, 20)
            sess = rng.normal(120, 30)
            spw = rng.normal(2, 0.5)
            wr = rng.beta(8, 2) # Weekend heavy
            genres = rng.choice(["Drama", "Romance", "Mystery"], size=rng.choice([1, 2]), replace=False)
            
        watch_time[i] = max(wt, 0.1)
        avg_session[i] = max(sess, 1.0)
        sessions_per_week[i] = max(spw, 0.1)
        weekend_ratio[i] = np.clip(wr, 0.0, 1.0)
        top_genres.append(";".join(genres))
        
    user_ids = [f"USR-{i:05d}" for i in range(n_users)]
    
    df = pd.DataFrame({
        "user_id": user_ids,
        "total_watch_time_hours": np.round(watch_time, 2),
        "avg_session_mins": np.round(avg_session, 1),
        "sessions_per_week": np.round(sessions_per_week, 1),
        "weekend_ratio": np.round(weekend_ratio, 3),
        "top_genres": top_genres,
    })
    
    # Introduce data quality issues
    for col in ["total_watch_time_hours", "avg_session_mins", "sessions_per_week", "weekend_ratio"]:
        mask = rng.random(n_users) < 0.02
        df.loc[mask, col] = np.nan
        
    genre_mask = rng.random(n_users) < 0.01
    df.loc[genre_mask, "top_genres"] = np.nan
    
    dup_indices = rng.choice(n_users, size=15, replace=False)
    df = pd.concat([df, df.iloc[dup_indices].copy()], ignore_index=True)
    
    neg_indices = rng.choice(len(df), size=5, replace=False)
    df.loc[neg_indices, "total_watch_time_hours"] = -rng.uniform(1, 10, size=5).round(2)
    
    neg_sess = rng.choice(len(df), size=3, replace=False)
    df.loc[neg_sess, "avg_session_mins"] = -rng.uniform(1, 20, size=3).round(1)
    
    outlier_idx = rng.choice(len(df), size=4, replace=False)
    df.loc[outlier_idx, "total_watch_time_hours"] = rng.uniform(500, 1000, size=4).round(2)
    
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = generate_dataset()
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[INFO] Synthetic dataset generated: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
