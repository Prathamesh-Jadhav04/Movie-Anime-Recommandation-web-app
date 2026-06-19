import pandas as pd

# ------------ Load Anime Dataset ------------
anime_df = pd.read_csv("data/anime.csv")
anime_ratings = pd.read_csv("data/ratings.csv")

# ✅ Clean anime dataset
anime_df = anime_df.rename(columns={"genres": "genre"})  # fix genre column
anime_df = anime_df[["anime_id", "title", "genre", "type"]]
anime_ratings = anime_ratings[anime_ratings["rating"] > 0]  # remove -1 ratings

# ✅ Merge anime info + ratings
anime_merged = anime_ratings.merge(anime_df, on="anime_id")
anime_merged["item_id"] = anime_merged["anime_id"]
anime_merged["source"] = "Anime"

# ------------ Load MovieLens Dataset ------------
MOVIELENS_GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]

mov_ratings = pd.read_csv("data/u.data", sep="\t", names=["user_id", "movie_id", "rating", "timestamp"])

genre_cols = {i + 5: f"g_{g}" for i, g in enumerate(MOVIELENS_GENRES)}
all_cols = {0: "movie_id", 1: "title"}
all_cols.update(genre_cols)
movie_df = pd.read_csv(
    "data/u.item", sep="|", encoding="latin-1", header=None,
    usecols=list(all_cols.keys()), names=list(all_cols.values())
)

def extract_genres(row):
    return ", ".join(g for g in MOVIELENS_GENRES if row.get(f"g_{g}", 0) == 1)

movie_df["genre"] = movie_df.apply(extract_genres, axis=1)
movie_df = movie_df[["movie_id", "title", "genre"]]

# ✅ Merge movie info + ratings
mov_merged = mov_ratings.merge(movie_df, on="movie_id")
mov_merged["item_id"] = mov_merged["movie_id"]
mov_merged["type"] = "Movie"
mov_merged["source"] = "Movie"

# ------------ Combine Anime + Movie Ratings ------------
columns = ["user_id", "item_id", "title", "type", "genre", "rating", "source"]
combined_df = pd.concat([
    anime_merged[columns],
    mov_merged[columns]
], ignore_index=True)

# ✅ Save to CSV
combined_df.to_csv("data/combined_ratings.csv", index=False)
print("[OK] Preprocessing complete! Saved to data/combined_ratings.csv")

# ✅ Save precomputed popular items for fast loading in Streamlit
popular_pool = (
    combined_df.groupby(["title", "source"]).size()
    .reset_index(name="cnt")
    .sort_values("cnt", ascending=False)
    .drop_duplicates("title")
)
popular_pool.to_csv("data/popular_items.csv", index=False)
print("[OK] Precomputed popular items saved to data/popular_items.csv")

# ✅ Precompute user profiles (for Arjun, Priya, Raj, Kiran, Sneha) for Tab 2
user_ids = [1, 2, 3, 4, 5]
user_profiles = {}
import json
for uid in user_ids:
    user_ratings = combined_df[combined_df["user_id"] == uid]
    if user_ratings.empty:
        continue
    # Genres breakdown
    genres_list = (user_ratings["genre"].dropna()
                   .str.replace(r'[/|;]', ',', regex=True)
                   .str.split(",")
                   .explode()
                   .str.strip()
                   .tolist())
    genres_series = pd.Series([g.title() for g in genres_list if g])
    top_genres = genres_series.value_counts().head(5).to_dict()
    
    # Calculate percentages
    total_g = sum(top_genres.values())
    top_genres_pct = {k: round((v / total_g) * 100) for k, v in top_genres.items()} if total_g > 0 else {}
    
    # Total rated items
    total_rated = len(user_ratings)
    # Average rating
    avg_rating = float(user_ratings["rating"].mean())
    # History list of rated items
    history = user_ratings.sort_values("rating", ascending=False).head(10)[["title", "rating", "source"]].to_dict("records")
    
    user_profiles[uid] = {
        "top_genres": top_genres_pct,
        "total_rated": total_rated,
        "avg_rating": round(avg_rating, 2),
        "history": history
    }

with open("model/user_profiles.json", "w") as f:
    json.dump(user_profiles, f, indent=2)
print("[OK] User profiles precomputed and saved to model/user_profiles.json")



