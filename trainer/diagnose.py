import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans

def diagnose():
    print("--- DIAGNOSE SCRIPT ---")
    
    # Load dataset
    df = pd.read_csv("../data/synthetic_ott_users.csv")
    
    # Same preprocessing as trainer
    df = df.drop_duplicates(subset=["user_id"]).copy()
    df.loc[df["total_watch_time_hours"] < 0, "total_watch_time_hours"] = np.nan
    df.loc[df["avg_session_mins"] < 0, "avg_session_mins"] = np.nan
    
    # Impute
    df["total_watch_time_hours"] = df["total_watch_time_hours"].fillna(df["total_watch_time_hours"].median())
    df["avg_session_mins"] = df["avg_session_mins"].fillna(df["avg_session_mins"].median())
    df["sessions_per_week"] = df["sessions_per_week"].fillna(df["sessions_per_week"].median())
    df["weekend_ratio"] = df["weekend_ratio"].fillna(df["weekend_ratio"].median())
    df["top_genres"] = df["top_genres"].fillna("")
    
    # Feature extraction (old way: 20 features)
    df["log_watch_time"] = np.log1p(df["total_watch_time_hours"])
    df["num_genres"] = df["top_genres"].apply(lambda x: len(x.split(";")) if x else 0)
    
    all_genres = [
        "Action", "Thriller", "Comedy", "Drama", "Romance",
        "Sci-Fi", "Horror", "Documentary", "Animation", "Family",
        "Crime", "Mystery", "Fantasy", "Adventure", "Musical"
    ]
    for g in all_genres:
        df[f"genre_{g}"] = df["top_genres"].apply(lambda x: 1.0 if g in x else 0.0)
        
    beh_cols = ["log_watch_time", "avg_session_mins", "sessions_per_week", "weekend_ratio", "num_genres"]
    genre_cols = [f"genre_{g}" for g in all_genres]
    all_cols = beh_cols + genre_cols
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_all = scaler.fit_transform(df[all_cols])
    X_beh = scaler.fit_transform(df[beh_cols])
    X_genre = scaler.fit_transform(df[genre_cols])
    
    # Run KMeans K=6
    labels_all = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(X_all)
    labels_beh = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(X_beh)
    labels_genre = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(X_genre)
    
    print(f"Silhouette All 20 features: {silhouette_score(X_all, labels_all):.4f}")
    print(f"Silhouette Behavioral only: {silhouette_score(X_beh, labels_beh):.4f}")
    print(f"Silhouette Genre only:      {silhouette_score(X_genre, labels_genre):.4f}")
    
    # ARI stability for K=2..8
    print("\n--- ARI Stability (10 seeds) ---")
    for k in range(2, 9):
        labels_list = []
        for seed in range(10):
            l = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(X_all)
            labels_list.append(l)
        aris = []
        for i in range(len(labels_list)):
            for j in range(i+1, len(labels_list)):
                aris.append(adjusted_rand_score(labels_list[i], labels_list[j]))
        print(f"K={k}: Mean ARI={np.mean(aris):.4f}")

if __name__ == "__main__":
    diagnose()
