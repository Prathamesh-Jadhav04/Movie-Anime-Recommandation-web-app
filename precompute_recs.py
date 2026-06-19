"""
Precompute NeuMF hybrid recommendations for the 6 demo users.
Run once locally (requires TF): python precompute_recs.py
Output: model/precomputed_recs.json
"""
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

USER_NAMES = {1: "Arjun", 2: "Priya", 3: "Raj", 5: "Kiran", 7: "Sneha", 8: "Amit"}
ALPHA      = 0.7   # default CF/CB blend
TOP_N      = 20

print("Loading model and maps...")
model    = load_model("model/hybrid_model.h5", compile=False)
user_map = pd.read_csv("model/user_map.csv")
item_map = pd.read_csv("model/item_map.csv")

print("Loading content data...")
try:
    anime_df = pd.read_csv("data/anime.csv")
except FileNotFoundError:
    anime_df = pd.DataFrame(columns=["title", "description", "type"])
try:
    movie_df = pd.read_csv("data/movies.csv")
except FileNotFoundError:
    movie_df = pd.DataFrame(columns=["title", "description", "type"])

for df, is_anime in [(anime_df, True), (movie_df, False)]:
    if "description" not in df.columns:
        for col in ["synopsis", "overview", "desc"]:
            if col in df.columns:
                df["description"] = df[col]
                break
        else:
            df["description"] = ""
    if "title" not in df.columns:
        for col in ["name", "anime_name", "movie_title"]:
            if col in df.columns:
                df["title"] = df[col]
                break
        else:
            df["title"] = ""
    df["description"] = df["description"].fillna("")
    df["title"]       = df["title"].fillna("")
    df["type"]        = "Anime" if is_anime else "Movie"

combined_df = pd.concat(
    [anime_df[["title", "description", "type"]],
     movie_df[["title", "description", "type"]]],
    ignore_index=True,
)

print("Building TF-IDF matrix...")
tfidf      = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_mat  = tfidf.fit_transform(combined_df["description"])

item_ids = item_map["item_idx"].values
results  = {}

for uid, uname in USER_NAMES.items():
    print(f"  Scoring user {uid} ({uname})...")

    urow = user_map[user_map["user_id"] == uid]["user_idx"]
    if urow.empty:
        print(f"    SKIP user {uid} not in user_map")
        results[str(uid)] = []
        continue

    uidx  = urow.values[0]
    u_arr = np.full(len(item_ids), uidx)

    cf_raw  = model.predict([u_arr, item_ids], batch_size=4096, verbose=0).flatten()
    cf_norm = (cf_raw - cf_raw.min()) / (cf_raw.max() - cf_raw.min() + 1e-8)

    # CB: average TF-IDF sim of top-3 CF seeds
    seeds   = item_map.iloc[cf_raw.argsort()[::-1][:3]]["title"].tolist()
    cb      = np.zeros(len(combined_df))
    matched = 0
    for t in seeds:
        hit = combined_df[combined_df["title"] == t]
        if hit.empty:
            continue
        cb     += cosine_similarity(tfidf_mat[hit.index[0]], tfidf_mat).flatten()
        matched += 1
    if matched:
        cb /= matched

    cb_map  = dict(zip(combined_df["title"], cb))
    cb_item = item_map["title"].map(cb_map).fillna(0.0).values
    cb_norm = (cb_item - cb_item.min()) / (cb_item.max() - cb_item.min() + 1e-8)

    scores    = ALPHA * cf_norm + (1.0 - ALPHA) * cb_norm
    top_items = item_map.iloc[scores.argsort()[::-1][:TOP_N]]

    recs = []
    for _, row in top_items.iterrows():
        recs.append({
            "title":  row["title"],
            "source": row.get("source", "Movie"),
            "genre":  row.get("genre", ""),
            "score":  round(float(scores[row.name]), 4),
        })

    results[str(uid)] = {"seeds": seeds, "recs": recs}
    print(f"    OK {len(recs)} recs, seeds: {seeds}")

out_path = "model/precomputed_recs.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved → {out_path}")
