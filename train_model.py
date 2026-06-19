"""
Train a NeuMF (Neural Collaborative Filtering) hybrid recommender.

Architecture follows He et al. (2017), "Neural Collaborative Filtering":
fuses a Generalized Matrix Factorization (GMF) branch with a Multi-Layer
Perceptron (MLP) branch, then concatenates both and feeds a final logistic
output layer. This is a true NeuMF — not a plain MF-concat.

Evaluation:
  * Regression  : RMSE / MAE on a held-out test split
  * Baseline    : item-popularity predictor (to prove the model adds value)
  * Ranking     : leave-one-out HR@10 + NDCG@10  (1 held-out positive vs
                  99 sampled negatives — the standard NCF protocol)
"""

import json
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Dense, Concatenate, Multiply, Dropout
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

# Load configuration with fallback defaults
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except Exception:
    config = {
        "model": {
            "mf_dim": 32,
            "mlp_dim": 32,
            "mlp_layers": [64, 32, 16],
            "reg": 1e-6,
            "epochs": 20,
            "batch_size": 512,
            "val_split": 0.1
        },
        "paths": {
            "combined_ratings": "data/combined_ratings.csv",
            "model_output": "model/hybrid_model.h5",
            "user_map": "model/user_map.csv",
            "item_map": "model/item_map.csv",
            "metrics": "model/metrics.json",
            "learning_curves": "model/learning_curves.png"
        }
    }


# -- 1. Load & normalize ratings -----------------------------------------------
df = pd.read_csv(config["paths"]["combined_ratings"])

df["user_idx"] = df["user_id"].astype("category").cat.codes
df["item_idx"] = df["item_id"].astype("category").cat.codes

# Anime ratings are 1-10; MovieLens are 1-5 — normalize both to [0, 1]
df["rating_norm"] = df.groupby("source")["rating"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)
)

n_users = df["user_idx"].nunique()
n_items = df["item_idx"].nunique()
print(f"Users: {n_users} | Items: {n_items} | Ratings: {len(df):,}")

# -- 2. Train/test split -------------------------------------------------------
train, test = train_test_split(df, test_size=0.2, random_state=42)

# -- 3. Popularity baseline ----------------------------------------------------
item_avg = train.groupby("item_idx")["rating_norm"].mean()
global_mean = train["rating_norm"].mean()

pop_preds = test["item_idx"].map(item_avg).fillna(global_mean)
pop_rmse = float(np.sqrt(((pop_preds - test["rating_norm"]) ** 2).mean()))
pop_mae  = float((pop_preds - test["rating_norm"]).abs().mean())
print(f"\nPopularity Baseline -> RMSE: {pop_rmse:.4f} | MAE: {pop_mae:.4f}")

# -- 4. Build NeuMF model (GMF + MLP fusion) -----------------------------------
MF_DIM     = config["model"]["mf_dim"]            # GMF embedding size
MLP_DIM    = config["model"]["mlp_dim"]           # MLP embedding size
MLP_LAYERS = config["model"]["mlp_layers"]        # MLP tower
REG        = float(config["model"]["reg"])

user_input = Input(shape=(1,), name="user")
item_input = Input(shape=(1,), name="item")

# --- GMF branch: element-wise product of user & item embeddings ---
gmf_u = Flatten()(Embedding(n_users, MF_DIM, embeddings_regularizer=l2(REG),
                            name="gmf_user_emb")(user_input))
gmf_i = Flatten()(Embedding(n_items, MF_DIM, embeddings_regularizer=l2(REG),
                            name="gmf_item_emb")(item_input))
gmf_vec = Multiply()([gmf_u, gmf_i])

# --- MLP branch: concat embeddings → dense tower ---
mlp_u = Flatten()(Embedding(n_users, MLP_DIM, embeddings_regularizer=l2(REG),
                            name="mlp_user_emb")(user_input))
mlp_i = Flatten()(Embedding(n_items, MLP_DIM, embeddings_regularizer=l2(REG),
                            name="mlp_item_emb")(item_input))
mlp_vec = Concatenate()([mlp_u, mlp_i])
for units in MLP_LAYERS:
    mlp_vec = Dense(units, activation="relu")(mlp_vec)
    mlp_vec = Dropout(0.2)(mlp_vec)

# --- NeuMF fusion: concat GMF + MLP → logistic output ---
neumf = Concatenate()([gmf_vec, mlp_vec])
output = Dense(1, activation="sigmoid", name="prediction")(neumf)

model = Model([user_input, item_input], output)
model.compile(optimizer="adam", loss="mse")
model.summary()

# -- 5. Train with early stopping ----------------------------------------------
early_stop = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
)

history = model.fit(
    [train.user_idx, train.item_idx],
    train.rating_norm,
    epochs=config["model"]["epochs"],
    batch_size=config["model"]["batch_size"],
    validation_split=config["model"]["val_split"],
    callbacks=[early_stop],
)

# -- 6. Save learning curves ---------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history.history["loss"], label="Train Loss", marker="o", markersize=3)
ax.plot(history.history["val_loss"], label="Val Loss", marker="s", markersize=3)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("NeuMF Training & Validation Loss")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(config["paths"]["learning_curves"], dpi=150)
plt.close()
print(f"Saved: {config['paths']['learning_curves']}")

# -- 7. Regression metrics on test set ----------------------------------------
ncf_preds = model.predict(
    [test.user_idx.values, test.item_idx.values], batch_size=4096, verbose=0
).flatten()

ncf_rmse = float(np.sqrt(((ncf_preds - test["rating_norm"].values) ** 2).mean()))
ncf_mae  = float(np.abs(ncf_preds - test["rating_norm"].values).mean())
print(f"NeuMF Model     -> RMSE: {ncf_rmse:.4f} | MAE: {ncf_mae:.4f}")
print(f"RMSE improvement over baseline: {(pop_rmse - ncf_rmse) / pop_rmse * 100:.1f}%")

# -- 8. Ranking metrics: leave-one-out HR@10 & NDCG@10 ------------------------
# Standard NCF protocol (He et al. 2017): for each evaluated user, hold out ONE
# relevant positive, pit it against 99 sampled negatives the user never saw,
# rank all 100, and measure whether the positive lands in the top-K.
#
#   HR@K   = fraction of users whose held-out item is in the top-K
#   NDCG@K = position-discounted version (rewards a higher rank)
#
# Using exactly one positive per user (not "all relevant in the pool") keeps
# the metric honest and comparable to published NeuMF numbers.

def ndcg_at_k_loo(rank: int, k: int = 10) -> float:
    """rank is 0-indexed position of the single positive item."""
    return 1.0 / np.log2(rank + 2) if rank < k else 0.0

K = 10
RELEVANCE_THRESHOLD = 0.6  # ≈ 6/10 anime or 3/5 movie after normalization
N_NEGATIVES = 99
EVAL_USERS = 500

all_item_idxs = np.arange(n_items)
rng = np.random.default_rng(42)

test_users = test["user_idx"].unique()
sample_users = rng.choice(test_users, size=min(EVAL_USERS, len(test_users)), replace=False)

hr_scores, ndcg_scores = [], []

for uid in sample_users:
    user_test = test[test["user_idx"] == uid]
    relevant = user_test[user_test["rating_norm"] >= RELEVANCE_THRESHOLD]["item_idx"].values
    if len(relevant) == 0:
        continue

    # Hold out exactly ONE positive
    pos_item = int(rng.choice(relevant))

    # 99 negatives the user has never interacted with (train or test)
    seen = set(df[df["user_idx"] == uid]["item_idx"].tolist())
    pool = np.setdiff1d(all_item_idxs, list(seen))
    if len(pool) < N_NEGATIVES:
        continue
    negatives = rng.choice(pool, size=N_NEGATIVES, replace=False).tolist()

    candidates = [pos_item] + negatives
    user_arr = np.full(len(candidates), uid)
    scores = model.predict(
        [np.array(user_arr), np.array(candidates)], batch_size=512, verbose=0
    ).flatten()

    # Rank candidates by score (desc); find the position of the positive (index 0)
    ranked = list(scores.argsort()[::-1])
    rank = ranked.index(0)  # 0-indexed rank of the held-out positive

    hr_scores.append(1.0 if rank < K else 0.0)
    ndcg_scores.append(ndcg_at_k_loo(rank, K))

hr_at_k   = float(np.mean(hr_scores))
ndcg_at_k = float(np.mean(ndcg_scores))
print(f"\nRanking Metrics — leave-one-out (K={K}, {len(hr_scores)} users, "
      f"1 positive vs {N_NEGATIVES} negatives):")
print(f"  HR@{K}:   {hr_at_k:.4f}")
print(f"  NDCG@{K}: {ndcg_at_k:.4f}")

# -- 9. Save metrics summary ---------------------------------------------------
metrics = {
    "model":             "NeuMF (GMF + MLP)",
    "popularity_rmse":   round(pop_rmse,   4),
    "popularity_mae":    round(pop_mae,    4),
    "ncf_rmse":          round(ncf_rmse,   4),
    "ncf_mae":           round(ncf_mae,    4),
    "rmse_gain_vs_baseline_pct": round((pop_rmse - ncf_rmse) / pop_rmse * 100, 1),
    "hr_at_10":          round(hr_at_k,    4),
    "ndcg_at_10":        round(ndcg_at_k,  4),
    "epochs_trained":    len(history.history["loss"]),
    "eval_users":        len(hr_scores),
}
with open(config["paths"]["metrics"], "w") as f:
    json.dump(metrics, f, indent=2)

# -- 10. Model comparison table (printed) -------------------------------------
print("\n--- Model Comparison --------------------------------------")
print(f"{'Model':<22}{'RMSE':>8}{'MAE':>8}")
print(f"{'Popularity baseline':<22}{pop_rmse:>8.4f}{pop_mae:>8.4f}")
print(f"{'NeuMF (GMF + MLP)':<22}{ncf_rmse:>8.4f}{ncf_mae:>8.4f}")
print("-----------------------------------------------------------")
print(f"  RMSE gain vs baseline : {(pop_rmse - ncf_rmse) / pop_rmse * 100:.1f}%")
print(f"  HR@10                 : {hr_at_k:.4f}")
print(f"  NDCG@10               : {ndcg_at_k:.4f}")
print("-----------------------------------------------------------")
print(f"Saved: {config['paths']['metrics']}")

# -- 11. Save model & mappings -------------------------------------------------
model.save(config["paths"]["model_output"])
df[["user_id", "user_idx"]].drop_duplicates().to_csv(config["paths"]["user_map"], index=False)
df[["item_id", "item_idx", "title", "type", "genre", "source"]].drop_duplicates().to_csv(
    config["paths"]["item_map"], index=False
)
print("\n[OK] NeuMF model trained and saved.")
