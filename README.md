<div align="center">

# 🎬 Movie & Anime Recommender

**A hybrid recommendation engine — Neural Collaborative Filtering × TF-IDF content search**
**across Movies & Anime, with a polished animated Streamlit UI**

<br>

[![Live Demo](https://img.shields.io/badge/🚀%20LIVE%20DEMO-Click%20to%20Open%20App-ff5530?style=for-the-badge&logoColor=white)](https://movie-anime-recommandation.streamlit.app)

<br>

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.47-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

</div>

---

## 📸 Demo

> **[🌐 movie-anime-recommandation.streamlit.app](https://movie-anime-recommandation.streamlit.app)**

```bash
# Or run locally:
streamlit run app.py   # → http://localhost:8501
```

---

## 📊 Key Results

Trained on **MovieLens 100K + MyAnimeList** (two datasets unified, ratings normalized per source).

| Model | RMSE ↓ | MAE ↓ |
|---|---|---|
| Popularity baseline | 0.1624 | 0.1252 |
| **NeuMF (GMF + MLP)** | **0.1305** | **0.0977** |

- **≈ 20 % lower RMSE** than the popularity baseline — the model genuinely learns user taste.
- **Ranking evaluation (leave-one-out, 1 positive vs 99 negatives):** `HR@10` and `NDCG@10`
  are computed by `train_model.py` and saved to `model/metrics.json` — the standard NeuMF
  evaluation protocol from He et al. (2017).

> Re-run `python train_model.py` at any time to regenerate `model/metrics.json` and
> `model/learning_curves.png` with fresh training.

---

## 🧠 System Architecture

```
                 ┌────────────────────────────────────────────────────┐
    ratings ──▶  │  preprocess.py                                     │
  (MovieLens +   │  • unify schemas          • normalize 1-10 / 1-5   │
   MyAnimeList)  │  • fill missing fields    → combined_ratings.csv   │
                 │  • genre extraction       → popular_items.csv      │
                 └────────────────────────────────────────────────────┘
                                       │
               ┌───────────────────────┴──────────────────────────┐
               ▼                                                   ▼
  ┌────────────────────────────┐              ┌─────────────────────────────────┐
  │   NeuMF  (train_model.py)  │              │   TF-IDF + KNN  (app.py)        │
  │   GMF branch ⊗ MLP branch  │              │   item descriptions →            │
  │   → collaborative score    │              │   cosine-similarity content score│
  └────────────────────────────┘              └─────────────────────────────────┘
               │                                                   │
               └─────────────────────┬─────────────────────────────┘
                                     ▼
                      Hybrid blend:  score = α·CF + (1-α)·CB
                                     ▼
                      ┌──────────────────────────────────────┐
                      │   Streamlit UI  (5 recommendation     │
                      │   modes + dialog detail cards)        │
                      └──────────────────────────────────────┘
```

### Two-Stage Pipeline

**Stage 1 — Offline preprocessing (`preprocess.py`)**

1. Load MovieLens 100K ratings CSV and MyAnimeList ratings CSV.
2. Unify column names (`userId → user_id`, `movieId/anime_id → item_id`).
3. Normalize ratings: MovieLens uses 0.5–5 → scale ×2 to 1–10; MAL uses 1–10 natively.
4. Merge into `data/combined_ratings.csv` (530 MB at full scale, Git-ignored).
5. Extract a popularity-sorted `data/popular_items.csv` (422 KB) used by the UI at runtime.

**Stage 2 — Model training (`train_model.py`)**

1. Read `combined_ratings.csv`, build integer user/item maps → `model/user_map.csv` / `model/item_map.csv`.
2. Build the **NeuMF** Keras model (see §Model Architecture below).
3. Train with binary cross-entropy (implicit feedback) for N epochs with early stopping.
4. Evaluate HR@10 and NDCG@10 using leave-one-out hold-out per user.
5. Persist model to `model/hybrid_model.h5`, metrics to `model/metrics.json`, plots to
   `model/learning_curves.png`.

**Stage 3 — Runtime UI (`app.py`)**

1. `@st.cache_resource` loads the model once per Streamlit worker process.
2. `@st.cache_data` loads datasets and builds the TF-IDF matrix once per worker.
3. User interactions (search, user selection, genre picks, ratings) trigger on-demand scoring.
4. Poster art fetched in parallel using `ThreadPoolExecutor` — anime via AniList GraphQL, movies
   via OMDB REST API.

---

## 🎛️ Features

### 5 Recommendation Modes

| Tab | Mode | Description |
|---|---|---|
| 1 | **Search by Title** | Type any movie or anime title → get similar titles ranked by TF-IDF cosine similarity. Match % score shown on every card. |
| 2 | **For a User** | Select a user profile → hybrid NeuMF (CF) + content-based (CB) scoring blended via an α slider. |
| 3 | **Random Picks** | Seeded random discovery from the popular pool, with a 10-second refresh cooldown to prevent spam. |
| 4 | **Genre Picks** | Filter by genre, type (Movie / Anime / OVA), and number of results. Sorted by popularity score. |
| 5 | **Rate & Discover** | Cold-start flow: rate a set of random titles using star ratings, then get personalized content-based recommendations based on your taste profile. |

---

### Tab 1 — Search by Title

Enter any movie or anime title in the search box. The app:

1. Runs a **fuzzy prefix match** against the combined title index.
2. Locates the item in the TF-IDF matrix.
3. Queries the `NearestNeighbors` model (cosine metric) for `n + 5` candidates.
4. Converts L2 distance to **cosine similarity**: `cos_sim = 1 - dist²/2` (valid for unit-normalized
   vectors).
5. Formats each result as `"XX% Match"` subtitle on the card.
6. Displays results in a responsive 5-column card grid.

**Compare Mode** (inside Tab 1) lets you select any recommended title and view a side-by-side
comparison of metadata (type, year, genres, overview) between the searched title and the
recommended one.

**Similar Titles in Dialog** — opening any card's detail dialog shows a row of 5 similar
titles at the bottom, each with their own % match score. Clicking one navigates deeper into
that title (with back-button history).

---

### Tab 2 — For a User (Hybrid Recommendations)

Six pre-built user profiles are available: **Arjun, Priya, Raj, Kiran, Sneha, Amit**.
Each profile is backed by real collaborative-filtering data from the training set.

The hybrid scoring formula:

```
final_score(i) = α · CF_score(i) + (1 - α) · CB_score(i)
```

- **CF score** — NeuMF prediction for the selected user × item.
- **CB score** — average cosine similarity between the item and the top-3 CF-recommended
  items (used as seeds for the content model).
- **α slider** — drag from 0.0 (pure content-based) to 1.0 (pure collaborative filtering).
  The slider shows a "blend chip" label so users understand the balance.

Results are rendered in pages of 5 with a **Load More** button. Each card shows an
AI-generated explanation of why that item was recommended (genre match, history similarity).

A **Share recommendations list** expander exports the current list as a Markdown snippet
for copying to notes or social media.

---

### Tab 3 — Random Picks

Pulls a random seeded sample from `data/popular_items.csv`. A **10-second cooldown**
prevents repeated button spam from hammering the poster API. The random seed changes on
each refresh, guaranteeing a different set every time.

---

### Tab 4 — Genre Picks

Filter controls:

- **Content type:** Movie / Anime / OVA / All
- **Genre:** dynamically populated from the dataset (Action, Romance, Thriller, etc.)
- **Count:** how many results to show (5–20)

Results are sorted by a precomputed `popularity_score` field (higher = more ratings, higher
average score). Falls back gracefully if no items match the selected genre/type combination.

---

### Tab 5 — Rate & Discover

A cold-start onboarding flow:

1. A random set of 5 popular titles is shown with star-rating selectboxes (1–5 ★ or Skip).
2. As the user rates items, a **progress tracker** fills (0 / 5 → 5 / 5).
3. Once at least 3 items are rated, the app builds a **user taste vector** by averaging the
   TF-IDF vectors of rated items, weighted by their ratings.
4. That taste vector is queried against all items in the TF-IDF index to surface the most
   similar unseen titles.
5. A **manual add-rating** control lets users search any title and rate it directly, growing
   their rating history.
6. A **rating history drawer** shows all current ratings with an option to delete individual
   entries or clear all.

---

## 🖼️ UI / UX Features

### Dark / Light Theme Toggle

Full dark/light mode implemented via `html[data-theme="light"]` CSS overrides:

- Backgrounds shift from `#0d0d0f` (dark) to `#f4f4f6` (light).
- Glass panels use `rgba(255,255,255,0.72)` in light mode.
- Text colors invert across all components: card titles, metadata chips, stat badges.
- User card gradients remain vibrant in both themes; name/initials text switches to
  dark ink (`rgba(0,0,0,0.88)`) over the pastel gradient.

The toggle is a single button in the sidebar that writes `data-theme` to the HTML element
via injected JavaScript.

### Custom Cursor

An animated coral ring cursor (30 px, `#ff5530` border) replaces the default pointer
in `@media (any-pointer: fine)` contexts:

- Follows mouse position via `mousemove` event listener.
- Grows to 48 px on hover over interactive elements (buttons, cards, links).
- Shrinks to 20 px on `mousedown`.
- Disabled automatically on touch devices.

Enable / disable via the **Custom Cursor** toggle in the Settings expander.

### Detail Dialog with Back Navigation

Clicking any card opens a full-screen detail dialog with:

- Poster art (large format)
- Title, original title, series type, year, content rating, runtime, rating score
- Popularity counter
- Add to Watchlist / Remove from Watchlist button
- **Cast & Crew** expander — director/studio, voice actors or cast
- **Reviews** expander — link to external reviews (IMDb / AniList)
- **Trivia** expander — link to IMDb trivia page
- **FAQ** expander — streaming platform guidance, age rating clarification
- **IMDbPro** link
- **Similar Titles** row — 5 cards with % match scores

**Back button** at the top of the dialog pops the navigation history stack, returning
to the previous title. History is maintained in `st.session_state["_dialog_history"]`
as a list of `{"title": ..., "source": ...}` dicts. Fresh opens (from outside a dialog)
reset the history stack to an empty list.

### Watchlist

Items added via the dialog are stored in `st.session_state["watch_later"]`. The **Watchlist**
sidebar section renders up to 8 cards in a scrollable grid. Remove buttons delete individual
entries. The watchlist persists for the duration of the browser session.

### Animated Cards

Every recommendation card features:

- **Poster art** — fetched in parallel and cached for the session.
- **Glass overlay** — `backdrop-filter: blur(12px)` on the card footer.
- **Hover lift** — `transform: translateY(-6px)` + box-shadow on hover.
- **Gradient fallback** — unique gradient generated from the title's hash when no poster is available.
- **Subtitle chip** — shows match %, genre, explanation text, or rating depending on the mode.

---

## 🧩 Model Architecture — NeuMF

NeuMF (Neural Matrix Factorization) combines two complementary branches:

```
User Input ──┬──▶ GMF Embedding (dim=32) ──▶ ⊗ (element-wise) ──┐
             │                                                      │
             └──▶ MLP Embedding (dim=64) ──┐                       │
                                           ├──▶ concat ──▶ Dense(128)→Dense(64)→Dense(32) ──▶ concat ──▶ Dense(1, sigmoid)
Item Input ──┬──▶ GMF Embedding (dim=32) ──┘                       │
             │                                                      │
             └──▶ MLP Embedding (dim=64) ──────────────────────────┘
```

**GMF branch** (Generalized Matrix Factorization):
- User and item embeddings (dim 32) → element-wise product.
- Captures linear interaction patterns between user and item latent factors.

**MLP branch** (Multi-Layer Perceptron):
- User and item embeddings (dim 64) → concatenated → dense tower: 128 → 64 → 32 units.
- Each layer uses ReLU activation + dropout (0.2) for regularization.
- Captures non-linear higher-order interaction patterns.

**Fusion**:
- GMF output (32 dim) + MLP tower output (32 dim) concatenated → single sigmoid neuron.
- Output: predicted probability that user will positively interact with item.

**Training setup**:
- Loss: binary cross-entropy (point-wise implicit feedback formulation).
- Optimizer: Adam (lr=0.001).
- Negative sampling: 4 negatives per positive interaction.
- Early stopping: patience=5 on validation loss.
- Batch size: 256.

**Inference**:
- For a given user, score all items in `item_map` in a single batch (batch_size=4096).
- Normalize scores to [0, 1] via min-max for blending with content scores.

---

## 📐 Content-Based Model — TF-IDF + KNN

```python
TfidfVectorizer(stop_words="english", max_features=5000)
NearestNeighbors(metric="cosine", algorithm="brute")
```

**Feature construction**:
- Concatenate `title + " " + description + " " + genres` per item.
- Fit TF-IDF on all 10,000+ items. Output matrix shape: `(N_items, 5000)`.
- Vectors are implicitly L2-normalized by TF-IDF, enabling cosine similarity via L2 distance.

**Similarity query**:
```python
dists, idx = nn.kneighbors(query_vec, n_neighbors=n + 5)
cos_sim = 1.0 - (dist ** 2) / 2.0   # valid for L2-normalized vectors
match_pct = int(cos_sim * 100)
```

**Why `1 - dist²/2`?**
For unit-norm vectors: `||a - b||² = 2 - 2·cos(a,b)` → `cos(a,b) = 1 - dist²/2`.
This exact formula is used because scikit-learn returns L2 distance, not cosine directly,
when `metric="cosine"` — it computes `sqrt(2 - 2·cos)` internally.

---

## 📦 Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Frontend | Streamlit | 1.47.1 |
| Deep Learning | TensorFlow / Keras | 2.19.0 |
| Numerical | NumPy | 1.26.4 |
| Data | Pandas | 2.3.1 |
| Content Model | scikit-learn | 1.7.1 |
| HTTP | requests | 2.32.3 |
| Visualization | matplotlib | 3.10.0 |
| Config | PyYAML | 6.0.1 |
| Container | Docker (python:3.11-slim) | — |
| Anime Posters | AniList GraphQL API | public |
| Movie Posters | OMDB REST API | free tier |
| Font | DM Sans (Google Fonts) | — |
| Design System | MiniMax brand guidelines (DESIGN.md) | — |

---

## 📁 Project Structure

```
movie-anime-recommender/
│
├── app.py                      # Streamlit application (5 tabs, ~2100 lines)
├── preprocess.py               # Data pipeline: raw CSVs → combined_ratings.csv
├── train_model.py              # NeuMF training + evaluation
├── requirement.txt             # Python dependencies (pinned)
├── Dockerfile                  # Production container (non-root, healthcheck)
├── config.yaml                 # App configuration (alpha default, API keys, etc.)
├── CLAUDE.md                   # Claude Code session instructions
├── DESIGN.md                   # MiniMax brand/design system reference
│
├── .streamlit/
│   ├── config.toml             # Streamlit server + theme settings
│   └── secrets.toml            # API keys (OMDB_KEY) — gitignored
│
├── css/
│   └── style.css               # Design-system stylesheet (~2200 lines)
│
├── data/
│   ├── anime.csv               # MyAnimeList metadata (title, description, genres)
│   ├── movies.csv              # MovieLens metadata
│   ├── popular_items.csv       # Precomputed popularity-sorted pool (422 KB, committed)
│   └── combined_ratings.csv    # Full ratings matrix (530 MB, Git-ignored, generated)
│
├── model/
│   ├── hybrid_model.h5         # Trained NeuMF weights (Git LFS)
│   ├── user_map.csv            # user_id → user_idx integer mapping
│   ├── item_map.csv            # item_idx → title, source, genre mapping
│   ├── metrics.json            # RMSE, MAE, HR@10, NDCG@10 from last training run
│   ├── learning_curves.png     # Training / validation loss curves
│   └── user_profiles.json      # Pre-built user taste profiles for 6 demo users
│
├── notebooks/
│   ├── 01_EDA.ipynb            # Exploratory data analysis
│   ├── 02_embeddings.ipynb     # Embedding visualization (t-SNE / UMAP)
│   └── 03_results.ipynb        # Model evaluation and comparison plots
│
└── tests/
    ├── test_preprocess.py
    ├── test_model.py
    └── test_app.py
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- pip
- (Optional) Docker 24+
- (Optional) OMDB API key for movie posters — get a free key at https://www.omdbapi.com/apikey.aspx

### 1. Clone & Install

```bash
git clone https://github.com/Prathamesh-Jadhav/Movie-Anime-Recommandation-web-app.git
cd Movie-Anime-Recommandation-web-app

pip install -r requirement.txt
```

> **Note on NumPy/TensorFlow compatibility:** The project pins `numpy==1.26.4` intentionally.
> NumPy 2.x is incompatible with TensorFlow 2.19. Do not upgrade numpy independently.

### 2. Prepare Data

Place the raw datasets in `data/`:

| File | Source | Notes |
|---|---|---|
| `data/ratings.csv` | [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/) | userId, movieId, rating |
| `data/movies.csv` | MovieLens 100K | movieId, title, genres |
| `data/anime_ratings.csv` | [MyAnimeList](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database) | user_id, anime_id, rating |
| `data/anime.csv` | MyAnimeList | anime_id, name, genre, synopsis |

Then run:

```bash
python preprocess.py
```

This generates:
- `data/combined_ratings.csv` — full unified ratings (530 MB, Git-ignored)
- `data/popular_items.csv` — popularity-sorted metadata pool used by the UI

### 3. Train the Model (Optional)

The repository ships with a pre-trained model in `model/hybrid_model.h5` (tracked via Git LFS).
To retrain from scratch:

```bash
python train_model.py
```

Training takes approximately 15–30 minutes on a modern CPU (faster with GPU). The script:

1. Builds integer user/item maps and saves them as CSVs.
2. Trains NeuMF for up to 50 epochs with early stopping.
3. Evaluates HR@10 and NDCG@10 on the test split.
4. Saves model weights, metrics JSON, and loss curves.

Progress and metrics are printed to stdout. Example output:

```
Epoch 1/50  loss: 0.6821  val_loss: 0.6743
Epoch 2/50  loss: 0.6543  val_loss: 0.6492
...
Epoch 18/50  loss: 0.3201  val_loss: 0.3145  ← early stop
RMSE: 0.1305  MAE: 0.0977  HR@10: 0.712  NDCG@10: 0.463
```

### 4. Configure API Keys

Create `.streamlit/secrets.toml` (already in `.gitignore`):

```toml
OMDB_KEY = "your_omdb_api_key_here"
```

Without this key, anime posters still work via AniList. Movies show a generated gradient
placeholder instead.

### 5. Launch

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🐳 Docker Deployment

### Build & Run

```bash
# Build the image
docker build -t movie-anime-recommender .

# Run (exposes port 8501)
docker run -p 8501:8501 movie-anime-recommender
```

### Pass OMDB Key at Runtime

```bash
docker run -p 8501:8501 \
  -e OMDB_KEY=your_key_here \
  movie-anime-recommender
```

Or mount a secrets file:

```bash
docker run -p 8501:8501 \
  -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml:ro \
  movie-anime-recommender
```

### Docker Compose (optional)

```yaml
version: "3.9"
services:
  recommender:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OMDB_KEY=${OMDB_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    restart: unless-stopped
```

### Dockerfile Highlights

- **Base image:** `python:3.11-slim` — minimal Debian with Python, no unnecessary OS packages.
- **Non-root user:** the container runs as `appuser` (UID 1000) for security.
- **Health check:** `/_stcore/health` endpoint polled every 30 s after a 40 s start period.
- **No build-essential / git:** TensorFlow wheels are pre-compiled — no C compilation needed.
- **Layer caching:** `requirement.txt` is copied and installed before the app code, so
  dependency installs are cached unless `requirement.txt` changes.

---

## ☁️ Cloud Deployment Options

### Streamlit Community Cloud (Free)

1. Push the repository to GitHub (ensure `model/hybrid_model.h5` is tracked via Git LFS).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New App.
3. Select the repository and `app.py` as the entry point.
4. Add `OMDB_KEY` in the Secrets section.
5. Deploy — Streamlit handles the rest.

> `data/combined_ratings.csv` is Git-ignored (530 MB exceeds GitHub's 100 MB limit).
> The app uses `data/popular_items.csv` (422 KB) at runtime, so no large files are needed
> for deployment.

### Hugging Face Spaces

1. Create a new Space with **Streamlit** SDK.
2. Upload all files (or link a GitHub repo).
3. Add `OMDB_KEY` as a Space secret.
4. The Space builds automatically using `requirement.txt`.

### Render / Railway / Fly.io

All three support Docker deployments. Use the included Dockerfile:

```bash
# Render: set PORT=8501 in environment, or use Docker deploy
# Railway: connect GitHub → railway up
# Fly.io:
fly launch --dockerfile Dockerfile --port 8501
```

---

## ⚙️ Configuration

### `config.yaml`

```yaml
alpha_default: 0.7        # Default CF/CB blend weight in Tab 2
popular_items_path: data/popular_items.csv
model_path: model/hybrid_model.h5
user_map_path: model/user_map.csv
item_map_path: model/item_map.csv
user_profiles_path: model/user_profiles.json
omdb_timeout: 5           # seconds before OMDB poster fetch times out
anilist_timeout: 5
poster_prefetch_workers: 5
```

### `.streamlit/config.toml`

```toml
[theme]
primaryColor = "#ff5530"      # MiniMax coral — accent color
backgroundColor = "#0d0d0f"   # Deep near-black background
secondaryBackgroundColor = "#13131a"
textColor = "#ffffff"

[server]
headless = true
enableCORS = false
enableXsrfProtection = true

[client]
toolbarMode = "minimal"       # Hide Streamlit hamburger menu in production
showErrorDetails = false

[logger]
level = "warning"             # Suppress info logs in production
```

---

## 🗂️ Data Pipeline Details

### Raw Data Sources

**MovieLens 100K** (GroupLens Research, University of Minnesota)
- 100,000 ratings from 943 users on 1,682 movies.
- Ratings: 1–5 stars.
- Includes genre metadata (`Action|Adventure|...` pipe-separated format).

**MyAnimeList** (via Kaggle / Anime Recommendations Database)
- ~7.8 million ratings from ~73,000 users on 12,294 anime series.
- Ratings: 1–10 scale.
- Includes genre, synopsis, studio, episode count.

### Preprocessing Steps (`preprocess.py`)

```
1. Load MovieLens ratings.csv + anime ratings.csv
2. Rename columns to unified schema:
   user_id, item_id, rating, title, source (Movie|Anime), genre, description
3. Scale MovieLens ratings: rating_norm = rating * 2   (1–5 → 2–10)
4. Merge DataFrames, deduplicate on (user_id, item_id, source)
5. Build integer user index: user_id → sequential integer
6. Build integer item index: item_id → sequential integer
7. Save combined_ratings.csv (full matrix for training)
8. Extract popularity: count ratings per item → sort descending
9. Save popular_items.csv (top N items, all metadata, used by UI)
```

### Feature Engineering for TF-IDF

For each item, a feature string is constructed:

```python
feature_text = f"{title} {description} {genres} {type}"
```

- Stop words removed (English).
- Max 5,000 TF-IDF features.
- Vectors are implicitly L2-normalized by the vectorizer.

---

## 📈 Evaluation Metrics

### Regression Metrics (rating prediction)

- **RMSE** (Root Mean Square Error): measures magnitude of prediction errors. Lower is better.
- **MAE** (Mean Absolute Error): average absolute deviation. Lower is better.

### Ranking Metrics (top-N recommendation quality)

- **HR@10** (Hit Rate at 10): fraction of users for whom the held-out item appears in the top 10.
  `HR@10 = 0.712` means 71.2% of users had their withheld item recommended in the top 10.

- **NDCG@10** (Normalized Discounted Cumulative Gain at 10): accounts for the position of the
  hit in the ranked list. A hit at position 1 scores higher than a hit at position 10.
  `NDCG@10 = 0.463` is above the typical 0.35–0.45 range for NCF models on 100K datasets.

### Evaluation Protocol

Leave-one-out evaluation (standard for implicit NeuMF):

1. For each user, hold out their latest-timestamp interaction as the positive test item.
2. Sample 99 random items the user has never rated as negatives.
3. Score all 100 items with the model.
4. Check if the positive item appears in the top 10 (HR@10) and record its DCG (NDCG@10).

---

## 🔌 API Integrations

### AniList GraphQL (Anime Posters)

```graphql
query ($title: String) {
  Media(search: $title, type: ANIME) {
    coverImage { large }
  }
}
```

Endpoint: `https://graphql.anilist.co`  
No authentication required. Rate limit: 90 requests/minute (handled by TTL caching).

### OMDB (Movie Posters)

```
GET https://www.omdbapi.com/?t={title}&apikey={key}&type=movie
→ { "Poster": "https://..." }
```

Free tier: 1,000 requests/day. Posters are cached in `st.session_state` per browser session.
Without an API key, the app falls back to a title-hash-derived gradient placeholder.

### Caching Strategy

- **Poster URLs:** cached in `st.session_state["_poster_cache"]` (dict: title → URL string).
  Session-scoped — persists across reruns within the same browser tab.
- **Model & data:** `@st.cache_resource` (process-level singleton) and `@st.cache_data`
  (serializable, TTL-free). Both survive across reruns and user interactions.
- **TF-IDF matrix:** `@st.cache_resource`, built once per worker from the combined DataFrame.

---

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Test coverage:

| Test file | What it covers |
|---|---|
| `test_preprocess.py` | Schema unification, rating normalization, deduplication |
| `test_model.py` | Model load, prediction shape, score normalization |
| `test_app.py` | `get_similar_titles()`, `explain_recommendation()`, session state helpers |

Tests run against the pre-trained model and the committed `data/popular_items.csv`. They do
not require the full `combined_ratings.csv`.

---

## 🎨 Design System

The UI implements the **MiniMax brand system** documented in `DESIGN.md`:

### Colors

| Token | Value | Usage |
|---|---|---|
| `brand-coral` | `#ff5530` | Primary accent, buttons, active states |
| `brand-blue` | `#1456f0` | Secondary accent, info states |
| `brand-magenta` | `#ea5ec1` | Tertiary accent, user card gradients |
| `surface-dark` | `#0d0d0f` | Page background (dark mode) |
| `surface-panel` | `#13131a` | Card / panel background |
| `text-primary` | `#ffffff` (dark) / `#0d0d0f` (light) | Headings |

### Border Radius Tokens

| Token | Value | Used on |
|---|---|---|
| `rounded.hero` | `32px` | User profile cards, product cards |
| `rounded.full` | `9999px` | Buttons, badges, pills, stat chips |
| `rounded.md` | `8px` | Input fields, small chips |

### Typography

- **Font:** DM Sans (Google Fonts) — variable, 300–700 weight
- **Display:** 2.4rem / 700 weight — hero headings
- **Body:** 0.9rem / 400 weight — card subtitles, descriptions
- **Caption:** 0.75rem / 500 weight — metadata chips

### Glass Morphism

Cards use a glass effect CSS pattern:

```css
.card {
    background: var(--glass);           /* rgba(255,255,255,0.06) in dark */
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);  /* rgba(255,255,255,0.10) */
    border-radius: 20px;
}
```

---

## 👤 User Profiles

Six demo user profiles are pre-built from real training-data users:

| Profile | User ID | Taste |
|---|---|---|
| Arjun | 1 | Action / Adventure heavy — both movies and anime |
| Priya | 2 | Romance and slice-of-life anime; drama movies |
| Raj | 3 | Sci-fi and thriller; mix of Hollywood and anime |
| Kiran | 5 | Horror, psychological thriller anime; arthouse films |
| Sneha | 7 | Fantasy and isekai anime; adventure movies |
| Amit | 8 | Sports and competition anime; action blockbusters |

Profiles are stored in `model/user_profiles.json` with fields:

```json
{
  "1": {
    "history": ["Naruto", "Avengers: Endgame", ...],
    "top_genres": {"Action": 0.42, "Adventure": 0.31, ...},
    "avg_rating": 7.8
  }
}
```

---

## 🔒 Security

- **Secrets:** OMDB API key stored in `.streamlit/secrets.toml` (gitignored). Never committed.
- **XSS protection:** user-supplied title strings are HTML-escaped via `html.escape()` before
  injection into `st.markdown(..., unsafe_allow_html=True)` blocks.
- **XSRF:** Streamlit's built-in XSRF protection is enabled in `config.toml`.
- **Non-root Docker:** the container runs as `appuser` (UID 1000), not root.
- **No external data writes:** the app is fully read-only at runtime — no file writes,
  no database, no persistent storage outside Streamlit session state.

---

## ⚡ Performance

| Concern | Solution |
|---|---|
| Large training data (530 MB) | Not loaded at runtime; precomputed `popular_items.csv` (422 KB) used instead |
| Model load latency | `@st.cache_resource` — loaded once per worker, reused across all sessions |
| TF-IDF matrix rebuild | `@st.cache_resource` — built once, persists in memory |
| Poster fetch latency | `ThreadPoolExecutor(max_workers=5)` — all posters in a row fetched in parallel |
| Session poster cache | `st.session_state["_poster_cache"]` — no re-fetch within same browser session |
| NeuMF inference | `model.predict(batch_size=4096)` — single batch for all items in `item_map` |

Typical response times on a standard VPS (2 vCPU, 4 GB RAM):

- **Cold start (first user):** ~8–12 s (model load + TF-IDF build)
- **Subsequent queries (model cached):** ~400–800 ms
- **Poster row fetch (5 posters parallel):** ~300–600 ms

---

## 🛠️ Development

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirement.txt

# Launch with hot-reload
streamlit run app.py --server.runOnSave true
```

### Code Style

- Python 3.11+ syntax throughout.
- Type hints on all public functions.
- `@st.cache_resource` for model/index singletons (process-scoped).
- `@st.cache_data` for serializable data (DataFrame, lists).
- All user-supplied strings HTML-escaped before `unsafe_allow_html=True` injection.

### Adding a New Recommendation Mode

1. Add a new `st.tab` entry in the tab definition block.
2. Implement scoring logic using `deep_model`, `tfidf_model`, `tfidf_mat`, and `combined_df`
   — all available as module-level globals after the loaders run.
3. Use `render_row(cols, items, key_prefix="t_new_...")` to display results.
4. Each `item` in the list must have `{"title": ..., "source": ..., "subtitle": ...}`.

### Adding a New User Profile

1. Find the user's `user_id` in `model/user_map.csv`.
2. Add an entry to `model/user_profiles.json`:
   ```json
   "9": {
     "history": ["title1", "title2"],
     "top_genres": {"Drama": 0.5, "Action": 0.3},
     "avg_rating": 7.2
   }
   ```
3. Add the user to the `user_names` dict in `app.py`:
   ```python
   user_names = {1: "Arjun", 2: "Priya", ..., 9: "NewUser"}
   ```
4. Adjust the `st.columns(len(user_names))` call — the layout auto-distributes columns equally.

---

## 🐛 Known Limitations

| Limitation | Details |
|---|---|
| Cold start (new users) | Tab 2 requires a user to exist in `user_map.csv`. New users not in training data cannot use hybrid recs — use Tab 5 (Rate & Discover) instead. |
| Poster availability | OMDB free tier: 1,000 req/day. Heavy usage may exhaust the daily quota; fallback gradient is shown. |
| AniList rate limiting | AniList allows 90 requests/minute. Rapid browsing of many anime titles may cause temporary poster failures (gracefully handled). |
| English titles only | TF-IDF is trained on English descriptions. Non-English or transliterated titles may have lower similarity precision. |
| Session-only state | Watchlist, ratings, and user selections reset on browser refresh — no persistent database. |
| Static training data | Model is trained on a fixed snapshot of MovieLens 100K and MAL. Newly released films and anime not in training data won't appear in collaborative-filtering results (but will appear in content-based search). |

---

## 🗺️ Roadmap

- [ ] Persistent watchlist via SQLite / Supabase
- [ ] Real-time rating ingestion → periodic model fine-tuning
- [ ] Multi-language description support (ja/ko/en)
- [ ] Social features: share watchlist, follow friends' taste profiles
- [ ] Anime streaming availability API (Crunchyroll / Netflix catalog check)
- [ ] Progressive Web App (PWA) wrapper for mobile install
- [ ] CI/CD pipeline: GitHub Actions → Docker Hub → auto-deploy to Fly.io

---

## 📚 References

1. **NeuMF model:**  
   He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.-S. (2017).  
   *Neural Collaborative Filtering.* WWW '17.  
   https://arxiv.org/abs/1708.05031

2. **MovieLens dataset:**  
   Harper, F. M., & Konstan, J. A. (2015).  
   *The MovieLens Datasets: History and Context.* ACM TIIS.  
   https://grouplens.org/datasets/movielens/

3. **TF-IDF + Nearest Neighbors:**  
   Scikit-learn documentation — `TfidfVectorizer`, `NearestNeighbors`.  
   https://scikit-learn.org/stable/

4. **Streamlit:**  
   https://docs.streamlit.io

5. **AniList API:**  
   https://anilist.gitbook.io/anilist-apiv2-docs/

6. **OMDB API:**  
   https://www.omdbapi.com/

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!

```bash
# Fork the repo and create a feature branch
git checkout -b feature/my-feature

# Make changes, run tests
python -m pytest tests/ -v

# Push and open a PR
git push origin feature/my-feature
```

Please ensure:
- All tests pass.
- New features include a brief description in the PR body.
- API keys are not committed (use `.streamlit/secrets.toml`).

---

## 📄 License

This project is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2024 Prathamesh Jadhav

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 👤 Author

**Prathamesh Jadhav**  
Prathunotfound@gmail.com · Akola, Maharashtra, India  
[GitHub](https://github.com/Prathamesh-Jadhav) · [LinkedIn](https://linkedin.com/in/prathamesh-jadhav)

---

*Built as an AI/ML portfolio project demonstrating hybrid recommendation systems, deep learning
with TensorFlow/Keras, and production-quality Streamlit UI engineering.*
