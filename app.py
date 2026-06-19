import os
import re
import time
import random
import requests
from html import escape as _esc
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import streamlit.components.v1 as _c1
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
import json
import urllib.parse
import textwrap

# ── Page config (must be first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Movie & Anime Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject fonts + CSS ────────────────────────────────────────────────────
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
try:
    with open("css/style.css", encoding="utf-8") as _f:
        st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# ── Custom cursor CSS (st.markdown works fine for styles) ────────────────
st.markdown("""<style>
@media (any-pointer: fine) {
  body.has-custom-cursor, body.has-custom-cursor *, body.has-custom-cursor *::before, body.has-custom-cursor *::after {
    cursor: none !important;
  }
}
#cx-ring{
  position:fixed;top:0;left:0;width:30px;height:30px;
  border:1.5px solid rgba(255,85,48,0.6);border-radius:50%;
  pointer-events:none;z-index:999999;will-change:transform;
  transition:width .2s ease,height .2s ease,border-color .2s ease,background .2s ease;
}
#cx-ring.hov{width:46px;height:46px;border-color:#ff5530;background:rgba(255,85,48,0.08);}
#cx-ring.clk{width:18px;height:18px;background:rgba(255,85,48,0.3);border-color:#ff5530;transition:none;}
#cx-dot{
  position:fixed;top:0;left:0;width:6px;height:6px;
  background:#ff5530;border-radius:50%;pointer-events:none;z-index:1000000;will-change:transform;
}
#cx-mute{
  position:fixed;bottom:22px;right:22px;width:44px;height:44px;
  border-radius:50%;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.12);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  display:flex;align-items:center;justify-content:center;z-index:99998;
  font-size:18px;transition:background .2s,transform .15s,opacity .2s;user-select:none;
}
#cx-mute:hover{background:rgba(255,255,255,0.14);transform:scale(1.1);}
#cx-mute.off{opacity:0.35;}
.cx-ripple{
  position:fixed;border-radius:50%;pointer-events:none;z-index:999997;
  background:rgba(255,85,48,0.25);animation:cx-rpl .5s ease-out forwards;
  transform:translate(-50%,-50%);
}
@keyframes cx-rpl{from{width:0;height:0;opacity:.7}to{width:66px;height:66px;opacity:0}}
/* Restore cursor inside dropdowns & popovers so native pointer shows */
[data-baseweb="popover"] *,[data-baseweb="popover"] *::before,[data-baseweb="popover"] *::after,
[data-baseweb="menu"] *,[role="listbox"] *,[role="option"],[data-baseweb="list-item"]{cursor:pointer !important}
input,textarea,[data-baseweb="input"] input,select,[contenteditable="true"]{cursor:text !important}
/* Slider: track area = pointer (click-to-jump), thumb = grab */
[data-testid="stSlider"],[data-testid="stSlider"] *,
[data-testid="stSelectSlider"],[data-testid="stSelectSlider"] *{cursor:pointer !important}
[data-testid="stSlider"] [role="slider"],
[data-testid="stSelectSlider"] [role="slider"]{cursor:grab !important}
[data-testid="stSlider"] [role="slider"]:active,
[data-testid="stSelectSlider"] [role="slider"]:active{cursor:grabbing !important}
</style>""", unsafe_allow_html=True)

# ── Cursor + sound JS via components.html (actually executes scripts) ─────
# st.markdown blocks <script> execution via React's dangerouslySetInnerHTML.
# components.html creates a same-origin iframe; window.parent reaches the
# real document so cursor elements and event listeners land on the main page.
_c1.html("""<script>
(function(){
  var W=window.parent, D=W.document;
  /* guard: skip if already loaded AND elements still in DOM */
  if(W.__cxLoaded&&D.getElementById('cx-ring'))return;
  W.__cxLoaded=true;

  var touch=!matchMedia('(any-pointer: fine)').matches;

  /* ── Audio (Web Audio API) ─────────────────────────────────────── */
  var AC=null,muted=false;
  function getAC(){
    if(!AC)AC=new(W.AudioContext||W.webkitAudioContext)();
    if(AC.state==='suspended')AC.resume();
    return AC;
  }
  function tone(freq,vol,dur,type,delay){
    if(muted)return;
    try{
      var a=getAC(),o=a.createOscillator(),g=a.createGain();
      o.connect(g);g.connect(a.destination);
      o.type=type||'sine';o.frequency.value=freq;
      var t=a.currentTime+(delay||0);
      g.gain.setValueAtTime(0,t);
      g.gain.linearRampToValueAtTime(vol,t+0.012);
      g.gain.exponentialRampToValueAtTime(0.0001,t+dur);
      o.start(t);o.stop(t+dur+0.015);
    }catch(e){}
  }
  function sndClick()  {tone(520,0.09,0.07);}
  function sndHover()  {tone(780,0.022,0.028);}
  function sndTab()    {tone(440,0.065,0.065);}
  function sndSuccess(){tone(523,0.07,0.18);tone(659,0.07,0.18,'sine',0.11);tone(784,0.07,0.22,'sine',0.22);}
  function sndMute()   {tone(300,0.06,0.09,'square');}

  /* ── Cursor elements in parent doc ─────────────────────────────── */
  var ring=D.createElement('div');ring.id='cx-ring';
  var dot =D.createElement('div');dot.id ='cx-dot';
  if(!touch){
    D.body.classList.add('has-custom-cursor');
    D.body.appendChild(ring);
    D.body.appendChild(dot);
  }

  var mx=W.innerWidth/2,my=W.innerHeight/2,rx=mx,ry=my;

  D.addEventListener('mousemove',function(e){
    mx=e.clientX;my=e.clientY;
    dot.style.transform='translate('+(mx-3)+'px,'+(my-3)+'px)';
  });

  /* rAF loop — runs in iframe context, still 60fps */
  (function loop(){
    rx=rx+(mx-rx)*0.13;ry=ry+(my-ry)*0.13;
    ring.style.transform='translate('+(rx-15)+'px,'+(ry-15)+'px)';
    requestAnimationFrame(loop);
  })();

  var SEL='button,a,label,[role="tab"],[data-testid="stSelectbox"],.rc,.rc-noposter,#cx-mute,[data-testid="stSlider"],[data-testid="stSelectSlider"]';
  D.addEventListener('mouseover',function(e){
    if(e.target.closest(SEL)){ring.classList.add('hov');sndHover();}
  });
  D.addEventListener('mouseout',function(e){
    if(e.target.closest(SEL))ring.classList.remove('hov');
  });
  D.addEventListener('mousedown',function(e){
    ring.classList.add('clk');
    var r=D.createElement('div');r.className='cx-ripple';
    r.style.left=e.clientX+'px';r.style.top=e.clientY+'px';
    D.body.appendChild(r);setTimeout(function(){r.remove();},520);
  });
  D.addEventListener('mouseup',function(){ring.classList.remove('clk');});
  D.addEventListener('click',function(e){
    if(e.target.closest('[role="tab"]')){sndTab();return;}
    if(e.target.closest('button'))sndClick();
  });

  /* ── Mute button ─────────────────────────────────────────────────── */
  var btn=D.createElement('div');
  btn.id='cx-mute';btn.textContent='🔊';btn.title='Toggle sound [M]';
  D.body.appendChild(btn);
  btn.addEventListener('click',function(){
    muted=!muted;
    btn.classList.toggle('off',muted);
    btn.textContent=muted?'🔇':'🔊';
    if(!muted)sndClick();else sndMute();
  });
  D.addEventListener('keydown',function(e){
    var ae=D.activeElement,tn=ae&&ae.tagName;
    if(tn==='INPUT'||tn==='TEXTAREA'||(ae&&ae.isContentEditable))return;
    if((e.key==='m'||e.key==='M')&&!e.ctrlKey&&!e.metaKey)btn.click();
  });

  /* ── Theme toggle (☀️ / 🌙) ──────────────────────────────────────── */
  var tgl=D.createElement('div');tgl.id='theme-toggle';
  function applyTheme(t){
    if(t==='light'){
      D.documentElement.setAttribute('data-theme','light');
      tgl.textContent='🌙';
      tgl.title='Switch to dark mode  [L]';
    } else {
      D.documentElement.removeAttribute('data-theme');
      tgl.textContent='☀️';
      tgl.title='Switch to light mode  [L]';
    }
    try{localStorage.setItem('cx-theme',t);}catch(e){}
  }
  var savedTheme='dark';
  try{savedTheme=localStorage.getItem('cx-theme')||'dark';}catch(e){}
  applyTheme(savedTheme);
  D.body.appendChild(tgl);
  tgl.addEventListener('click',function(){
    var isLight=D.documentElement.getAttribute('data-theme')==='light';
    applyTheme(isLight?'dark':'light');
    sndTab();
  });
  D.addEventListener('keydown',function(e){
    var ae=D.activeElement,tn=ae&&ae.tagName;
    if(tn==='INPUT'||tn==='TEXTAREA'||(ae&&ae.isContentEditable))return;
    if((e.key==='l'||e.key==='L')&&!e.ctrlKey&&!e.metaKey)tgl.click();
  });

  /* ── Success sound when rec cards render ────────────────────────── */
  var _played=false;
  new MutationObserver(function(muts){
    if(_played)return;
    for(var i=0;i<muts.length;i++){
      var nl=muts[i].addedNodes;
      for(var j=0;j<nl.length;j++){
        var n=nl[j];
        if(n.nodeType===1&&n.querySelector&&n.querySelector('.rc')){
          _played=true;sndSuccess();
          setTimeout(function(){_played=false;},2000);return;
        }
      }
    }
  }).observe(D.body,{childList:true,subtree:true});

})();
</script>""", height=0)

# ── OMDB key ──────────────────────────────────────────────────────────────
try:
    _OMDB_KEY: str = st.secrets.get("OMDB_KEY", os.environ.get("OMDB_KEY", ""))
except Exception:
    _OMDB_KEY = os.environ.get("OMDB_KEY", "")

_SENTINEL = object()

# ── DESIGN.md badge tokens ────────────────────────────────────────────────
_BADGE = {
    "Anime": ("background:rgba(255,85,48,0.18);color:#ff8a72", "ANIME"),
    "Movie": ("background:rgba(20,86,240,0.18);color:#6ea3ff", "MOVIE"),
}


# ═══════════════════════════════════════════════════════════════════════════
#  DATA / MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_deep_model():
    if not _TF_AVAILABLE:
        try:
            user_map = pd.read_csv("model/user_map.csv")
            item_map = pd.read_csv("model/item_map.csv")
        except Exception:
            user_map = pd.DataFrame()
            item_map = pd.DataFrame()
        return None, user_map, item_map
    try:
        model    = load_model("model/hybrid_model.h5", compile=False)
        user_map = pd.read_csv("model/user_map.csv")
        item_map = pd.read_csv("model/item_map.csv")
        return model, user_map, item_map
    except Exception as _e:
        return None, pd.DataFrame(), pd.DataFrame()


@st.cache_data
def load_data():
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
                    df["description"] = df[col]; break
            else:
                df["description"] = ""
        if "title" not in df.columns:
            for col in ["name", "anime_name", "movie_title"]:
                if col in df.columns:
                    df["title"] = df[col]; break
            else:
                df["title"] = ""
        df["description"] = df["description"].fillna("")
        df["title"]       = df["title"].fillna("")
        df["type"]        = "Anime" if is_anime else "Movie"
    return pd.concat(
        [anime_df[["title", "description", "type"]],
         movie_df[["title", "description", "type"]]],
        ignore_index=True,
    )


@st.cache_resource
def build_tfidf(descriptions):
    tfidf  = TfidfVectorizer(stop_words="english", max_features=5000)
    matrix = tfidf.fit_transform(descriptions)
    nn     = NearestNeighbors(metric="cosine", algorithm="brute")
    nn.fit(matrix)
    return nn, matrix


@st.cache_data
def _popular_pool() -> pd.DataFrame:
    # Read precalculated popular items instead of the 530MB combined_ratings.csv
    try:
        return pd.read_csv("data/popular_items.csv")
    except FileNotFoundError:
        # Fallback to computing from combined_ratings.csv if it exists (for local fallback)
        if os.path.exists("data/combined_ratings.csv"):
            return (
                pd.read_csv("data/combined_ratings.csv")
                .groupby(["title", "source"]).size()
                .reset_index(name="cnt")
                .sort_values("cnt", ascending=False)
                .drop_duplicates("title")
            )
        else:
            return pd.DataFrame(columns=["title", "source", "cnt"])



def get_popular_items(n_anime: int = 3, n_movie: int = 2, seed: int = 42) -> list[dict]:
    pool = _popular_pool()
    a = pool[pool["source"] == "Anime"].head(80).sample(min(n_anime, 80), random_state=seed)
    m = pool[pool["source"] == "Movie"].head(80).sample(min(n_movie, 80), random_state=seed)
    return (
        pd.concat([a, m]).sample(frac=1, random_state=seed)[["title", "source"]]
        .to_dict("records")
    )


# ═══════════════════════════════════════════════════════════════════════════
#  POSTER FETCHING
# ═══════════════════════════════════════════════════════════════════════════
def _clean_search_title(title: str) -> str:
    """Strip year, season tags, etc. for cleaner API search."""
    t = re.sub(r'\s*\(\d{4}\)\s*$', '', title)
    t = re.sub(r':\s*(Season|Part|Arc|Cour)\s+\d+.*$', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+\d+(?:st|nd|rd|th)\s+Season.*$', '', t, flags=re.IGNORECASE)
    return t.strip() or title


def extract_year(title: str) -> int | None:
    m = re.search(r'\((\d{4})\)', title)
    return int(m.group(1)) if m else None



# Gradient pool for no-poster cards (hashed per title → always same colour)
_CARD_GRADS = [
    ("rgba(255,85,48,0.50)",  "rgba(234,94,193,0.40)"),
    ("rgba(20,86,240,0.50)",  "rgba(61,174,255,0.40)"),
    ("rgba(168,85,247,0.50)", "rgba(20,86,240,0.40)"),
    ("rgba(255,85,48,0.45)",  "rgba(168,85,247,0.40)"),
    ("rgba(61,174,255,0.45)", "rgba(168,85,247,0.40)"),
    ("rgba(234,94,193,0.50)", "rgba(61,174,255,0.40)"),
]

def _card_gradient(title: str) -> tuple[str, str]:
    return _CARD_GRADS[sum(ord(c) for c in title) % len(_CARD_GRADS)]

def _initials(title: str) -> str:
    words = [w for w in re.split(r'\W+', title) if w.isalpha()][:2]
    return "".join(w[0] for w in words).upper() if words else title[:2].upper()


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_poster(title: str, source: str) -> str | None:
    try:
        if source == "Anime":
            search = _clean_search_title(title)
            queries = [search] if search == title else [search, title]
            # 1) AniList
            for q in queries:
                try:
                    r = requests.post(
                        "https://graphql.anilist.co",
                        json={"query": "query($s:String){Media(search:$s,type:ANIME){coverImage{large}}}",
                              "variables": {"s": q}},
                        timeout=7,
                    )
                    url = r.json().get("data", {}).get("Media", {}).get("coverImage", {}).get("large")
                    if url:
                        return url
                except Exception:
                    pass
            # 2) Jikan (MyAnimeList) fallback
            try:
                r2 = requests.get(
                    "https://api.jikan.moe/v4/anime",
                    params={"q": search, "limit": 1},
                    timeout=8,
                )
                data = r2.json().get("data", [])
                if data:
                    img = data[0].get("images", {}).get("jpg", {}).get("large_image_url")
                    if img and "questionmark" not in img:
                        return img
            except Exception:
                pass
            return None
        if not _OMDB_KEY:
            return None
        r = requests.get(
            "http://www.omdbapi.com/",
            params={"t": title, "apikey": _OMDB_KEY},
            timeout=7,
        )
        url = r.json().get("Poster", "N/A")
        return url if url and url != "N/A" else None
    except Exception:
        return None


@st.cache_data(ttl=86_400, show_spinner=False)
def resolve_card_links(title: str, source: str) -> dict[str, str]:
    """Resolves exact IMDb URL and Watch URL for a given title."""
    imdb_url = None
    
    # Try OMDB if API key is present
    if _OMDB_KEY:
        try:
            r = requests.get(
                "http://www.omdbapi.com/",
                params={"t": title, "apikey": _OMDB_KEY},
                timeout=5,
            )
            d = r.json()
            if d.get("Response") == "True" and d.get("imdbID"):
                imdb_url = f"https://www.imdb.com/title/{d['imdbID']}/"
        except Exception:
            pass

    # Fallback to IMDb Suggests API (undocumented, keyless suggestions)
    if not imdb_url:
        try:
            clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().strip()
            search_term = clean_title.replace(' ', '_')
            if search_term:
                first_letter = search_term[0]
                url = f"https://sg.media-imdb.com/suggests/{first_letter}/{urllib.parse.quote(search_term)}.json"
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                r = requests.get(url, headers=headers, timeout=5)
                if r.status_code == 200:
                    text = r.text
                    match = re.search(r'imdb\$[^(]+\((.*)\)', text)
                    if match:
                        data = json.loads(match.group(1))
                        results = data.get("d", [])
                        for item in results:
                            if item.get("id", "").startswith("tt"):
                                imdb_url = f"https://www.imdb.com/title/{item['id']}/"
                                break
        except Exception:
            pass

    # Final fallback: search IMDb results
    if not imdb_url:
        q_imdb = urllib.parse.quote_plus(title)
        imdb_url = f"https://www.imdb.com/find?q={q_imdb}&s=tt&ttype=ft"

    # Watch URL resolution
    q_watch = urllib.parse.quote_plus(title)
    if source == "Anime":
        # Free anime streaming on HiAnime (previously zoro.to)
        watch_url = f"https://hianime.to/search?keyword={q_watch}"
    else:
        # Free movie streaming on YouTube search (searching for full movie free)
        watch_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(f'{title} full movie free')}"

    return {"imdb": imdb_url, "watch": watch_url}


def _prefetch(items: list[dict]) -> list[str | None]:
    def _pre(d):
        poster = fetch_poster(d["title"], d["source"])
        try:
            # Prefetch the exact URLs to warm up the cache in parallel threads
            resolve_card_links(d["title"], d["source"])
        except Exception:
            pass
        return poster

    with ThreadPoolExecutor(max_workers=min(len(items), 10)) as ex:
        return list(ex.map(_pre, items))


# ═══════════════════════════════════════════════════════════════════════════
#  CARD RENDERER  (cinematic overlay design)
# ═══════════════════════════════════════════════════════════════════════════
def render_card(col, title: str, source: str, subtitle: str = "", poster_url=_SENTINEL, key_prefix: str = "", in_dialog: bool = False) -> None:
    if poster_url is _SENTINEL:
        poster_url = fetch_poster(title, source)

    badge_style, badge_label = _BADGE.get(
        source, ("background:rgba(255,255,255,0.12);color:rgba(255,255,255,0.8)", source.upper())
    )
    safe_title = _esc(title)
    g1, g2     = _card_gradient(title)
    inits      = _initials(title)
    sub_html   = f"<p class='rc-sub'>{_esc(subtitle)}</p>" if subtitle else ""
    
    links = resolve_card_links(title, source)
    imdb_url = links["imdb"]
    watch_url = links["watch"]

    action_bar = (
        f"<div class='rc-actions'>"
        f"<a href='{imdb_url}' target='_blank' rel='noopener' class='rc-action rc-action-det'>📖 Details</a>"
        f"<a href='{watch_url}' target='_blank' rel='noopener' class='rc-action rc-action-watch'>▶ Watch</a>"
        f"</div>"
    )

    info_html = (
        f"<div style='display:flex; align-items:center; gap:5px; margin-top:3px; margin-bottom:5px; font-size:0.65rem; color:rgba(255,255,255,0.75);'>"
        f"  <span style='color:#ff5530; font-size:0.75rem; line-height:1;'>ℹ️</span>"
        f"  <span style='letter-spacing:0.02em;'>Click card for details & trailer</span>"
        f"</div>"
    )

    if poster_url:
        safe_url = _esc(poster_url)
        html = (
            f"<div class='rc'>"
            f"<div class='rc-bg' style='background-image:url(\"{safe_url}\"),linear-gradient(135deg,{g1},{g2})'></div>"
            f"<div class='rc-overlay'>"
            f"<div class='rc-info-badge'>"
            f"<svg class='rc-info-svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.5' style='width:11px;height:11px;'><circle cx='12' cy='12' r='10'></circle><line x1='12' y1='16' x2='12' y2='12'></line><line x1='12' y1='8' x2='12.01' y2='8'></line></svg>"
            f"<span>Info</span>"
            f"</div>"
            f"<span class='rc-badge' style='{badge_style}'>{badge_label}</span>"
            f"<p class='rc-title'>{safe_title}</p>"
            f"{sub_html}"
            f"{info_html}"
            f"{action_bar}"
            f"</div>"
            f"</div>"
        )
    else:
        html = (
            f"<div class='rc'>"
            f"<div class='rc-bg rc-bg-fallback' style='background:linear-gradient(135deg,{g1},{g2})'>{inits}</div>"
            f"<div class='rc-overlay'>"
            f"<div class='rc-info-badge'>"
            f"<svg class='rc-info-svg' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.5' style='width:11px;height:11px;'><circle cx='12' cy='12' r='10'></circle><line x1='12' y1='16' x2='12' y2='12'></line><line x1='12' y1='8' x2='12.01' y2='8'></line></svg>"
            f"<span>Info</span>"
            f"</div>"
            f"<span class='rc-badge' style='{badge_style}'>{badge_label}</span>"
            f"<p class='rc-title'>{safe_title}</p>"
            f"{sub_html}"
            f"{info_html}"
            f"{action_bar}"
            f"</div>"
            f"</div>"
        )

    with col:
        st.markdown(html, unsafe_allow_html=True)
        btn_key = f"_crd_{key_prefix}_{abs(hash(title + source))}"
        if st.button("", key=btn_key):
            if not in_dialog:
                # Fresh open — clear history
                st.session_state["_dialog_history"] = []
                st.session_state["_modal_data"] = {"title": title, "source": source}
                st.session_state["_card_mode"] = "details"
                _card_action_dialog()
            else:
                # In-dialog navigation — push current to history
                history = st.session_state.setdefault("_dialog_history", [])
                current = st.session_state.get("_modal_data", {})
                if current and current.get("title") != title:
                    history.append(dict(current))
                st.session_state["_modal_data"] = {"title": title, "source": source}
                st.session_state["_card_mode"] = "details"
                st.rerun()


def render_row(cols, items: list[dict], key_prefix: str = "") -> None:
    """Pre-fetch all posters in parallel, then render."""
    urls = _prefetch(items)
    for idx, (col, item, url) in enumerate(zip(cols, items, urls)):
        render_card(col, item["title"], item["source"], item.get("subtitle", ""), poster_url=url, key_prefix=f"{key_prefix}_{idx}")


# ═══════════════════════════════════════════════════════════════════════════
#  TITLE INFO  ·  SIMILAR TITLES  ·  INFO DIALOG
# ═══════════════════════════════════════════════════════════════════════════
def get_title_info(title: str, source: str) -> dict:
    row  = combined_df[combined_df["title"] == title]
    desc = str(row["description"].values[0]) if not row.empty else ""
    genre = ""
    if "genre" in item_map.columns:
        gr = item_map[item_map["title"] == title]
        if not gr.empty:
            genre = str(gr["genre"].values[0] or "")
    return {"description": desc, "genre": genre}


@st.cache_data(ttl=3_600, show_spinner=False)
def fetch_detailed_info(title: str, source: str) -> dict:
    """Rich metadata from AniList (anime) or OMDB (movies)."""
    base: dict = {
        "description": "", "genres": [], "creator": "", "stars": "",
        "score": "", "year": "", "episodes": "", "status": "",
        "banner": "", "poster": "",
        "original_title": "", "format": "", "duration": "",
        "maturity_rating": "G", "popularity": "N/A",
    }
    search = _clean_search_title(title)

    if source == "Anime":
        _q = """
        query($s:String){Media(search:$s,type:ANIME){
          title { romaji english native }
          format duration popularity
          description(asHtml:false) genres averageScore episodes status
          startDate{year} bannerImage
          coverImage{extraLarge large}
          staff(sort:RELEVANCE,perPage:6){edges{role node{name{full}}}}
          characters(sort:ROLE,perPage:6){edges{voiceActors(language:JAPANESE){name{full}}}}
        }}"""
        for q in ([search] if search == title else [search, title]):
            try:
                r = requests.post(
                    "https://graphql.anilist.co",
                    json={"query": _q, "variables": {"s": q}},
                    timeout=8,
                )
                m = (r.json().get("data") or {}).get("Media") or {}
                if not m:
                    continue
                desc = re.sub(r"<[^>]+>", "", m.get("description") or "").strip()
                creator = ""
                for edge in (m.get("staff", {}).get("edges") or []):
                    role = edge.get("role", "").lower()
                    if any(k in role for k in ("original", "story", "creator", "author")):
                        creator = edge["node"]["name"]["full"]
                        break
                if not creator:
                    edges = m.get("staff", {}).get("edges") or []
                    if edges:
                        creator = edges[0]["node"]["name"]["full"]
                stars_list: list[str] = []
                for ce in (m.get("characters", {}).get("edges") or []):
                    for va in (ce.get("voiceActors") or []):
                        n2 = va["name"]["full"]
                        if n2 not in stars_list:
                            stars_list.append(n2)
                        if len(stars_list) >= 4:
                            break
                    if len(stars_list) >= 4:
                        break
                sc = m.get("averageScore")
                cv = m.get("coverImage") or {}
                
                eng_title = (m.get("title") or {}).get("english") or title
                native_title = (m.get("title") or {}).get("native") or (m.get("title") or {}).get("romaji") or ""
                fmt = m.get("format") or "TV"
                dur = f"{m.get('duration')}m" if m.get("duration") else ""
                pop = m.get("popularity")
                pop_str = f"{pop//1000}K" if pop and pop >= 1000 else str(pop) if pop else "N/A"
                
                base.update({
                    "description": desc[:650],
                    "genres": m.get("genres") or [],
                    "creator": creator,
                    "stars": " · ".join(stars_list),
                    "score": f"{sc/10:.1f}" if sc else "",
                    "year": str((m.get("startDate") or {}).get("year") or ""),
                    "episodes": str(m.get("episodes") or ""),
                    "status": m.get("status") or "",
                    "banner": m.get("bannerImage") or "",
                    "poster": cv.get("extraLarge") or cv.get("large") or "",
                    "original_title": native_title,
                    "format": "TV Series" if fmt == "TV" else fmt.replace("_", " ").title(),
                    "duration": dur or f"{m.get('episodes') or 12} ep",
                    "maturity_rating": "TV-14" if fmt == "TV" else "PG-13",
                    "popularity": pop_str,
                })
                return base
            except Exception:
                continue
        return base

    # ── Movie: OMDB ──────────────────────────────────────────────────────────
    if _OMDB_KEY:
        try:
            r = requests.get(
                "http://www.omdbapi.com/",
                params={"t": search, "apikey": _OMDB_KEY, "plot": "full"},
                timeout=7,
            )
            d = r.json()
            if d.get("Response") == "True":
                genres_raw = [g.strip() for g in d.get("Genre", "").split(",") if g.strip()]
                poster = d.get("Poster", "")
                votes = d.get("imdbVotes", "").replace(",", "")
                votes_str = "N/A"
                if votes.isdigit():
                    v_val = int(votes)
                    votes_str = f"{v_val//1000}K" if v_val >= 1000 else str(v_val)
                base.update({
                    "description": d.get("Plot", ""),
                    "genres": genres_raw,
                    "creator": d.get("Director", ""),
                    "stars": " · ".join(x.strip() for x in d.get("Actors", "").split(",")[:4]),
                    "score": d.get("imdbRating", ""),
                    "year": d.get("Year", ""),
                    "episodes": d.get("Runtime", ""),
                    "status": "",
                    "banner": "",
                    "poster": poster if poster and poster != "N/A" else "",
                    "original_title": d.get("Title", ""),
                    "format": d.get("Type", "").title(),
                    "duration": d.get("Runtime", ""),
                    "maturity_rating": d.get("Rated", "PG-13"),
                    "popularity": votes_str,
                })
        except Exception:
            pass
    return base


def get_similar_titles(title: str, source: str, n: int = 5) -> list[dict]:
    hit = combined_df[combined_df["title"] == title]
    if hit.empty:
        return []
    dists, idx = tfidf_model.kneighbors(tfidf_mat[hit.index[0]], n_neighbors=n + 5)
    recs, seen = [], {title}
    for dist, i in zip(dists[0], idx[0]):
        r = combined_df.iloc[i]
        if r["title"] not in seen:
            # L2 dist on normalized vecs → cosine sim = 1 - dist²/2
            cos_sim = max(0.0, 1.0 - (float(dist) ** 2) / 2.0)
            match_pct = min(100, max(0, int(cos_sim * 100)))
            recs.append({
                "title": r["title"],
                "source": r["type"],
                "subtitle": f"{match_pct}% Match",
            })
            seen.add(r["title"])
        if len(recs) == n:
            break
    return recs



@st.dialog("Title Details", width="large")
def _card_action_dialog() -> None:
    try:
        data   = st.session_state.get("_modal_data", {})
        title  = data.get("title", "")
        source = data.get("source", "")
        if not title:
            return

        # ── Back button (only when history exists) ─────────────────────────
        history = st.session_state.get("_dialog_history", [])
        if history:
            prev_title = history[-1].get("title", "")
            col_back, col_title_hint = st.columns([1, 5])
            with col_back:
                if st.button("← Back", key="dlg_back_btn", use_container_width=True):
                    prev = st.session_state["_dialog_history"].pop()
                    st.session_state["_modal_data"] = prev
                    st.session_state["_card_mode"] = "details"
                    st.rerun()
            with col_title_hint:
                st.markdown(
                    f"<p style='margin:6px 0 0; font-size:0.75rem; color:var(--t3);'>"
                    f"← {_esc(prev_title)}</p>",
                    unsafe_allow_html=True,
                )
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        mode = st.session_state.get("_card_mode", "details")

        # ── MODE: Details (IMDB-style) ─────────────────────────────────────
        if mode == "details":
            with st.spinner("Loading…"):
                info    = fetch_detailed_info(title, source)
                similar = get_similar_titles(title, source, n=5)

            has_details = bool(info.get("description") or info.get("genres") or info.get("creator") or info.get("score"))
            if not has_details:
                st.markdown(f"""
                <div style='display:flex; align-items:center; gap:6px; margin-bottom:12px;'>
                  <div class='imdb-badge-info'>
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="margin-top:-1px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                    INFO CARD
                  </div>
                  <div class='imdb-badge-source'>{_esc(source.upper())} DETAILS</div>
                </div>
                <div style='text-align:center; padding:40px 20px; background:rgba(255,255,255,0.03); border:1px solid var(--glass-border); border-radius:12px; margin:10px 0;'>
                  <div style='font-size:2.5rem; margin-bottom:12px;'>📋</div>
                  <h3 style='color:var(--t1); margin:0 0 8px; font-size:1.1rem;'>Detailed information is currently unavailable for this title</h3>
                  <p style='color:var(--t3); font-size:0.85rem; margin:0; max-width:400px; margin:0 auto;'>We could not retrieve metadata from external sources for <b>{_esc(title)}</b>. Please use the IMDb or YouTube links on the card to learn more.</p>
                </div>
                """, unsafe_allow_html=True)
                if similar:
                    st.markdown("<div class='idg-sim-label'>Similar Titles</div>", unsafe_allow_html=True)
                    sim_urls = _prefetch(similar)
                    sim_cols = st.columns(5)
                    for idx, (sim_col, item, url) in enumerate(zip(sim_cols, similar, sim_urls)):
                        render_card(sim_col, item["title"], item["source"], poster_url=url, key_prefix=f"sim_dialog_{idx}", in_dialog=True)
                return

            g1, g2 = _card_gradient(title)
            inits  = _initials(title)

            rating_val = info.get("score") or "N/A"
            popularity_val = info.get("popularity") or "N/A"
            original_title_val = info.get("original_title") or title
            format_val = info.get("format") or ("TV Series" if source == "Anime" else "Movie")
            year_val = info.get("year") or "N/A"
            maturity_val = info.get("maturity_rating") or "PG-13"
            duration_val = info.get("duration") or ("24m" if source == "Anime" else "120m")

            st.markdown(f"""
            <div class='imdb-header'>
              <div style='display:flex; align-items:center; gap:6px; margin-bottom:10px;'>
                <div class='imdb-badge-info'>
                  <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="margin-top:-1px;"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>
                  INFO CARD
                </div>
                <div class='imdb-badge-source'>{_esc(source.upper())} DETAILS</div>
              </div>
              <div style='display:flex; justify-content:space-between; align-items:flex-start;'>
                <div>
                  <h1 class='imdb-title'>{_esc(title)}</h1>
                  <p class='imdb-subtext'>Original title: {_esc(original_title_val)}</p>
                  <p class='imdb-meta-chips'>{_esc(format_val)} • {_esc(year_val)} • {_esc(maturity_val)} • {_esc(duration_val)}</p>
                </div>
                <div style='text-align:right;'>
                  <div class='imdb-rating-box'>
                    <span style='color:#f5a623; font-weight:bold; font-size:1.1rem;'>★ {rating_val}</span><span style='font-size:0.8rem; color:var(--t3);'>/10</span>
                  </div>
                  <div style='font-size:0.68rem; color:var(--t3); margin-top:4px; letter-spacing:0.05em; font-weight:600;'>POPULARITY: {popularity_val}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            col_video, col_poster = st.columns([7, 3])
            with col_video:
                safe_q = urllib.parse.quote_plus(f"{title} Official Trailer")
                trailer_url = f"https://www.youtube.com/embed?listType=search&list={safe_q}&autoplay=1"
                st.markdown(
                    f"""<iframe src="{trailer_url}" width="100%" height="280" frameborder="0" 
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
                    allowfullscreen style="border-radius:10px; border:1px solid var(--glass-border)"></iframe>""",
                    unsafe_allow_html=True
                )
            with col_poster:
                poster_url = info["poster"] or fetch_poster(title, source) or ""
                if poster_url:
                    st.markdown(
                        f"""<img src="{_esc(poster_url)}" style="width:100%; height:280px; object-fit:cover; 
                        border-radius:10px; border:1px solid var(--glass-border)">""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<div class='rc-bg-fallback' style='width:100%; height:280px; 
                        background:linear-gradient(135deg,{g1},{g2}); border-radius:10px; border:1px solid var(--glass-border)'>{inits}</div>""",
                        unsafe_allow_html=True
                    )

            genres = list(info["genres"])
            if not genres:
                local  = get_title_info(title, source)
                genres = [g.strip().title() for g in re.split(r"[,/|;]", local.get("genre", "")) if g.strip()]
            genre_pills = "".join(f"<span class='idg-genre'>{_esc(g)}</span>" for g in genres[:8])
            st.markdown(f"<div style='margin: 10px 0 16px;'>{genre_pills}</div>", unsafe_allow_html=True)

            desc = info["description"]
            if not desc:
                local = get_title_info(title, source)
                desc  = local.get("description", "")
            if desc:
                st.markdown(f"<p class='idg-desc'>{_esc(desc[:650])}{'…' if len(desc) > 650 else ''}</p>", unsafe_allow_html=True)
            else:
                st.markdown("<p class='idg-desc idg-desc-none'>No description available.</p>", unsafe_allow_html=True)

            creator_lbl = "Director" if source == "Movie" else "Creator"
            credits_summary = ""
            if info["creator"]:
                credits_summary += f"**{creator_lbl}:** {info['creator']} &nbsp;&nbsp;·&nbsp;&nbsp; "
            if info["stars"]:
                credits_summary += f"**Stars:** {info['stars']}"
            if credits_summary:
                st.markdown(f"<div style='font-size:0.85rem; color:var(--t2); margin-top:8px;'>{credits_summary}</div>", unsafe_allow_html=True)

            is_anime = source == "Anime"

            # ── Watch / Streaming & Watchlist section inside details ───────
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown("### ▶ Watch Now & Playlist")
            c_watch = st.columns(3)
            safe_q_watch = urllib.parse.quote_plus(title)
            with c_watch[0]:
                st.markdown(f"""
                <a href='https://www.youtube.com/results?search_query={urllib.parse.quote_plus(f"{title} full movie trailer")}' target='_blank' style='text-decoration:none;'>
                <div style='background:rgba(255,0,0,0.10); border:1px solid rgba(255,0,0,0.25); border-radius:10px; padding:10px; text-align:center;'>
                  <span style='font-size:1.1rem;'>▶</span>
                  <div style='font-weight:600; font-size:0.8rem; color:var(--t1);'>YouTube</div>
                  <div style='font-size:0.6rem; color:var(--t3);'>Free · Trailer & Clips</div>
                </div>
                </a>
                """, unsafe_allow_html=True)
            with c_watch[1]:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                with st.popover("🌐 All Platforms", use_container_width=True):
                    platforms = [
                        ("Netflix", "#e50914", f"https://www.netflix.com/search?q={safe_q_watch}"),
                        ("Prime Video", "#00a8e1", f"https://www.amazon.com/s?k={safe_q_watch}&i=instant-video"),
                        ("Crunchyroll", "#f47521", f"https://www.crunchyroll.com/search?q={safe_q_watch}"),
                        ("HiAnime", "#6b21a8", f"https://hianime.to/search?keyword={safe_q_watch}"),
                        ("Disney+ Hotstar", "#113ccf", f"https://www.hotstar.com/in/search?q={safe_q_watch}"),
                        ("SonyLiv", "#6ea3ff", f"https://www.sonyliv.com/search?q={safe_q_watch}"),
                    ]
                    for name, color, url in platforms:
                        st.markdown(f"""
                        <a href="{url}" target="_blank" style='text-decoration:none;'>
                        <div style='background:rgba(255,255,255,0.04); border:1px solid var(--glass-border); border-radius:8px; padding:8px 12px; margin-bottom:6px; display:flex; align-items:center; gap:10px;'>
                          <span style='font-weight:700; color:{color};'>{name[0]}</span>
                          <span style='font-weight:500; font-size:0.85rem; color:var(--t1);'>{name}</span>
                        </div>
                        </a>
                        """, unsafe_allow_html=True)
            with c_watch[2]:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                item_dict = {"title": title, "source": source}
                if _in_watchlist(item_dict):
                    if st.button("🗑️ Remove Watchlist", key=f"dlg_wl_{abs(hash(title))}", use_container_width=True):
                        st.session_state["watch_later"] = [x for x in st.session_state["watch_later"] if not (x["title"] == title and x["source"] == source)]
                        st.toast(f"Removed {title} from watchlist!")
                        st.rerun()
                else:
                    if st.button("➕ Add Watchlist", key=f"dlg_wl_{abs(hash(title))}", use_container_width=True, type="primary"):
                        st.session_state["watch_later"].append(item_dict)
                        st.toast(f"Added {title} to watchlist!")
                        st.rerun()

            st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

            with st.expander("👥 Cast & Crew"):
                prod_company = info.get("production") or ("Anime Studio" if is_anime else "Film Production")
                cast_label = "Voice Actors" if is_anime else "Cast"
                cast_val = info.get("stars") or "N/A"
                st.markdown(f"""
                <div style='font-size:0.85rem; line-height:1.6;'>
                  <p><b>{creator_lbl}:</b> {info['creator'] or 'N/A'}</p>
                  <p><b>{cast_label}:</b> {cast_val}</p>
                  <p><b>Production:</b> {prod_company}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("💬 Reviews"):
                st.markdown(
                    "<div style='font-size:0.85rem; color:var(--t2);'>Reviews are sourced externally. "
                    "Visit IMDb or AniList for community reviews on this title.</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("💡 Trivia"):
                st.markdown(
                    "<div style='font-size:0.85rem; color:var(--t2);'>Trivia for this title is available on IMDb and Wikipedia.</div>",
                    unsafe_allow_html=True,
                )

            with st.expander("❓ FAQ"):
                watch_platform = "Crunchyroll or Netflix" if is_anime else "Netflix or a streaming service near you"
                faq_html = (
                    f"<p><b>Q: Where can I watch this?</b><br>"
                    f"<span style='color:var(--t2);'>A: Check {watch_platform} for availability in your region.</span></p>"
                    f"<p><b>Q: Is it suitable for all ages?</b><br>"
                    f"<span style='color:var(--t2);'>A: Refer to the content rating shown above.</span></p>"
                )
                st.markdown(f"<div style='font-size:0.85rem; line-height:1.6;'>{faq_html}</div>", unsafe_allow_html=True)

            with st.expander("💼 IMDbPro"):
                st.markdown(f"""
                <div style='font-size:0.85rem; line-height:1.6; text-align:center; padding:10px;'>
                  <p><b>See production info, box office statistics, and agent contacts at IMDbPro.</b></p>
                  <a href='https://pro.imdb.com' target='_blank' style='display:inline-block; background:#f5a623; color:#000; font-weight:bold; padding:8px 20px; border-radius:20px; text-decoration:none; margin-top:6px;'>Go to IMDbPro</a>
                </div>
                """, unsafe_allow_html=True)

            if similar:
                st.markdown("<div class='idg-sim-label'>Similar Titles</div>", unsafe_allow_html=True)
                sim_urls = _prefetch(similar)
                sim_cols = st.columns(5)
                for idx, (sim_col, item, url) in enumerate(zip(sim_cols, similar, sim_urls)):
                    render_card(sim_col, item["title"], item["source"], poster_url=url, key_prefix=f"sim_dialog_{idx}", in_dialog=True)

    except Exception as _e:
        st.exception(_e)


# ═══════════════════════════════════════════════════════════════════════════
#  COLD-START RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════
def cold_start_recs(
    ratings: dict[str, int],
    combined_df: pd.DataFrame,
    nn_model,
    tfidf_matrix,
    n: int = 10,
) -> pd.DataFrame:
    agg, total_w = None, 0.0
    for title, rating in ratings.items():
        hit = combined_df[combined_df["title"] == title]
        if hit.empty:
            continue
        vec = tfidf_matrix[hit.index[0]]
        w   = rating / 5.0
        agg = vec * w if agg is None else agg + vec * w
        total_w += w
    if agg is None or total_w == 0:
        return pd.DataFrame()
    agg /= total_w
    _, idx = nn_model.kneighbors(agg, n_neighbors=min(n + len(ratings) + 10, len(combined_df)))
    rated = set(ratings.keys())
    recs  = [combined_df.iloc[i] for i in idx[0] if combined_df.iloc[i]["title"] not in rated]
    return pd.DataFrame(recs[:n])


# ═══════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def sec(title: str, subtitle: str = "") -> None:
    sub = f"<p class='sec-sub'>{_esc(subtitle)}</p>" if subtitle else ""
    st.markdown(f"<div class='sec-header'><h2 class='sec-title'>{_esc(title)}</h2>{sub}</div>",
                unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(f"""<div class='empty-state'>
  <div class='empty-state-icon'>{icon}</div>
  <h3>{_esc(title)}</h3>
  <p>{_esc(body)}</p>
</div>""", unsafe_allow_html=True)


def _in_watchlist(item: dict) -> bool:
    wl = st.session_state.get("watch_later", [])
    return any(x["title"] == item["title"] and x["source"] == item["source"] for x in wl)


def explain_recommendation(rec_title: str, rec_genres_str: str, user_history: list[dict], user_top_genres: dict) -> str:
    # check overlapping genres
    rec_genres = [g.strip().lower() for g in re.split(r'[,/|;]', rec_genres_str) if g.strip()]
    matching_genres = [g for g in rec_genres if g.title() in user_top_genres]
    if matching_genres:
        return f"🎯 Matches interest in {', '.join(g.title() for g in matching_genres[:2])}"
    
    # fallback to matching high ratings
    high_rated = [h for h in user_history if h.get('rating', 0) >= 4]
    if high_rated:
        ref = random.choice(high_rated)
        ref_title = ref.get('title', '')
        ref_title_clean = _clean_search_title(ref_title)
        return f"📈 Similar to your highly-rated {ref.get('source', 'item')}: {ref_title_clean}"
    return "✨ Recommended for your viewing profile"



def _genre_list(df: pd.DataFrame) -> list[str]:
    g = (df["genre"].dropna()
         .str.lower()
         .str.replace(r'[/|;]', ',', regex=True)
         .str.split(",")
         .explode()
         .str.strip()
         .unique())
    return sorted({x.title() for x in g if x})


# ═══════════════════════════════════════════════════════════════════════════
#  BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════
combined_df            = load_data()
tfidf_model, tfidf_mat = build_tfidf(combined_df["description"])
deep_model, user_map, item_map = load_deep_model()

# Load precomputed user profiles
@st.cache_data
def load_user_profiles() -> dict:
    try:
        with open("model/user_profiles.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}

user_profiles = load_user_profiles()

# Initialize watch later list
if "watch_later" not in st.session_state:
    st.session_state["watch_later"] = []

user_names = {
    1: "Arjun", 2: "Priya", 3: "Raj", 5: "Kiran", 7: "Sneha", 8: "Amit"
}
user_map = user_map.copy()
user_map["name"] = user_map["user_id"].map(user_names).fillna(
    "User " + user_map["user_id"].astype(str)
)

n_anime  = int((combined_df["type"] == "Anime").sum())
n_movies = int((combined_df["type"] == "Movie").sum())


# ═══════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div class='sidebar-logo'>
  <div class='sidebar-logo-icon'>M</div>
  <div>
    <div class='sidebar-logo-text'>Recommender</div>
    <div class='sidebar-logo-sub'>Movie & Anime · AI-Powered</div>
  </div>
 </div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div class='sidebar-stat-row'>
  <div class='sidebar-stat'>
    <div class='sidebar-stat-n'>{n_anime:,}</div>
    <div class='sidebar-stat-l'>Anime</div>
  </div>
  <div class='sidebar-stat'>
    <div class='sidebar-stat-n'>{n_movies:,}</div>
    <div class='sidebar-stat-l'>Movies</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Watch Later list widget
    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; font-weight:700; color:var(--t2); text-transform:uppercase; letter-spacing:0.05em;'>📋 Watchlist / Watch Later</p>", unsafe_allow_html=True)
    if st.session_state["watch_later"]:
        for idx, item in enumerate(st.session_state["watch_later"]):
            st.markdown(
                f"<div style='background:var(--glass-md); padding:6px 10px; border-radius:6px; border:1px solid var(--glass-border); font-size:0.75rem; margin-bottom:4px; display:flex; justify-content:space-between; align-items:center;'>"
                f"<span style='overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:140px;' title='{_esc(item['title'])}'>{_esc(item['title'])}</span>"
                f"<span style='font-size:0.6rem; opacity:0.6; text-transform:uppercase; background:rgba(255,255,255,0.08); padding:1px 4px; border-radius:3px;'>{item['source']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )
        if st.button("Clear Watchlist", key="clear_wl_btn", use_container_width=True):
            st.session_state["watch_later"] = []
            st.toast("Watchlist cleared!")
            st.rerun()
    else:
        st.markdown("<p style='font-size:0.7rem; color:var(--t3); font-style:italic; margin: 4px 0;'>Your watchlist is empty. Add items using ➕ Watch Later on any card.</p>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
    st.caption("Powered by Neural Collaborative Filtering + TF-IDF KNN")
    if not _OMDB_KEY:
        st.caption("Movie posters off — set OMDB\\_KEY to enable.")


# ═══════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='app-hero'>
  <div class='hero-eyebrow'>AI-Powered · Neural Collaborative Filtering</div>
  <h1 class='hero-title'>Movie &amp; <span>Anime</span><br>Recommender</h1>
  <p class='hero-sub'>Discover hidden gems across thousands of titles — recommendations built just for you.</p>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════
t1, t2, t3, t4, t5, t6 = st.tabs([
    "Search by Title",
    "From User",
    "Random",
    "Genre Picks",
    "Rate & Discover",
    "Watchlist 📋",
])


# ── TAB 1: Search by Title ────────────────────────────────────────────────
with t1:
    sec("Search by Title", "Content-based similarity — TF-IDF nearest neighbours")

    st.markdown("<div class='filter-label'>Content type</div>", unsafe_allow_html=True)
    ctype = st.radio("Content type", ("All", "Movie", "Anime"), horizontal=True,
                     key="t1_ctype", label_visibility="collapsed")

    st.markdown("<div class='filter-label'>Type to search title</div>", unsafe_allow_html=True)
    t1_target_df = combined_df if ctype == "All" else combined_df[combined_df["type"] == ctype]
    t1_titles_list = sorted(t1_target_df["title"].unique())
    default_idx = None
    if "Naruto" in t1_titles_list:
        default_idx = t1_titles_list.index("Naruto")
    elif len(t1_titles_list) > 0:
        default_idx = 0
        
    search_query = st.selectbox(
        "Type to search title:",
        options=t1_titles_list,
        index=default_idx,
        placeholder="Type to search (e.g. Naruto)...",
        key="t1_title_input",
        label_visibility="collapsed"
    )

    st.markdown("<div class='filter-label'>Return similar</div>", unsafe_allow_html=True)
    rec_filter = st.radio("Return similar", ("All", "Movie", "Anime"), horizontal=True,
                          key="t1_recfilter", label_visibility="collapsed")

    col_y, col_s = st.columns(2)
    with col_y:
        year_range = st.slider("Filter by Year Range:", 1900, 2026, (1980, 2026), key="t1_year_range")
    with col_s:
        sort_by = st.selectbox("Sort Results By:", ["Best Match", "Release Year (Newest)", "Release Year (Oldest)", "Alphabetical"], key="t1_sort_by")

    if st.button("🔍 Find Similar", key="t1_btn"):
        try:
            if not search_query or not search_query.strip():
                st.warning("Please enter a title to search.")
            else:
                target_df = combined_df if ctype == "All" else combined_df[combined_df["type"] == ctype]
                match_rows = target_df[target_df["title"].str.lower() == search_query.strip().lower()]
                
                if match_rows.empty:
                    st.info(f"🎬 '{search_query}' title coming soon")
                else:
                    sel_row = match_rows.iloc[[0]]
                    title = sel_row["title"].values[0]
                    sel_type = sel_row["type"].values[0]
                    
                    # Render the searched title as a beautiful card
                    col_s_card, col_space = st.columns([1, 4])
                    with col_s_card:
                        sec("Searched Title")
                        render_card(st.container(), title, sel_type, subtitle="Target Title", key_prefix="t1_search")

                    # Fetch matching items based on similarity
                    _, indices = tfidf_model.kneighbors(tfidf_mat[sel_row.index[0]], n_neighbors=100)
                    recs = []
                    for i in indices[0]:
                        if i == sel_row.index[0]:
                            continue
                        row = combined_df.iloc[i]
                        if row["title"] == title:
                            continue
                        
                        dist = float(cosine_similarity(tfidf_mat[sel_row.index[0]], tfidf_mat[i])[0][0])
                        match_pct = min(100, max(0, int(dist * 100)))
                        
                        # Parse year
                        y = extract_year(row["title"])
                        if y is not None and not (year_range[0] <= y <= year_range[1]):
                            continue
                        
                        if rec_filter == "All" or row["type"] == rec_filter:
                            recs.append({
                                "title": row["title"], 
                                "source": row["type"], 
                                "match_score": match_pct,
                                "year": y or 2000,
                                "subtitle": f"{match_pct}% Match"
                            })
    
                    # Apply sorting
                    if sort_by == "Release Year (Newest)":
                        recs.sort(key=lambda x: x["year"], reverse=True)
                    elif sort_by == "Release Year (Oldest)":
                        recs.sort(key=lambda x: x["year"])
                    elif sort_by == "Alphabetical":
                        recs.sort(key=lambda x: x["title"])
                    else: # Best Match
                        recs.sort(key=lambda x: x["match_score"], reverse=True)
    
                    final_recs = recs[:5]
    
                    if final_recs:
                        sec("Similar Titles")
                        render_row(st.columns(5), final_recs, key_prefix="t1_similar")
    
                        # 3. Compare Mode side-by-side comparison
                        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
                        with st.expander("⚖️ Compare Mode — Side-by-Side"):
                            comp_title = st.selectbox("Select a recommended title to compare:", [r["title"] for r in final_recs], key="t1_comp_title")
                            comp_item = next((r for r in final_recs if r["title"] == comp_title), None)
                            
                            col_c1, col_c2 = st.columns(2)
                            with col_c1:
                                st.markdown(f"#### 🔍 Searched: {title}")
                                info_orig = get_title_info(title, sel_type)
                                st.write(f"**Type:** {sel_type}")
                                st.write(f"**Year:** {extract_year(title) or 'N/A'}")
                                st.write(f"**Genres:** {info_orig.get('genre') or 'N/A'}")
                                st.write(f"**Overview:** {info_orig.get('description') or 'N/A'}")
                            with col_c2:
                                if comp_item:
                                    st.markdown(f"#### 🎯 Recommended: {comp_title}")
                                    info_rec = get_title_info(comp_title, comp_item["source"])
                                    st.write(f"**Type:** {comp_item['source']}")
                                    st.write(f"**Year:** {comp_item.get('year') or 'N/A'}")
                                    st.write(f"**Genres:** {info_rec.get('genre') or 'N/A'}")
                                    st.write(f"**Overview:** {info_rec.get('description') or 'N/A'}")
                    else:
                        empty_state("🔍", "No matches found",
                                    "Try a different title or change the filters above.")
        except Exception as e:
            st.error(str(e))


# ── TAB 2: For a User ─────────────────────────────────────────────────────
with t2:
    sec("Hybrid Recommendations", "Collaborative filtering blended with content-based scoring")

    if "selected_uid" not in st.session_state:
        st.session_state["selected_uid"] = 1
        
    st.markdown("<div class='user-section-title'>👤 Select Active User</div>", unsafe_allow_html=True)
    ucols = st.columns(len(user_names))
    for idx, (u_id, u_name) in enumerate(user_names.items()):
        with ucols[idx]:
            is_active = (st.session_state["selected_uid"] == u_id)
            prof = user_profiles.get(str(u_id)) or {}
            total_rated = prof.get("total_rated", 0)
            avg_rating = prof.get("avg_rating", 0.0)
            
            # Determine initials
            inits = (u_name[0] + u_name[1]) if len(u_name) > 1 else u_name[0]
            inits = inits.upper()
            
            # Gradients based on user ID for unique avatars
            g_colors = [
                ("linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)", "#b53b52"),
                ("linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%)", "#3f6db5"),
                ("linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%)", "#2d8050"),
                ("linear-gradient(135deg, #fccb90 0%, #d5d114 100%)", "#825a1e"),
                ("linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%)", "#5f3d8a"),
                ("linear-gradient(135deg, #f093fb 0%, #f5576c 100%)", "#9c2560"),
                ("linear-gradient(135deg, #5ee7df 0%, #b490ca 100%)", "#3a5f78"),
                ("linear-gradient(135deg, #f6d365 0%, #fda085 100%)", "#a8543b")
            ]
            avatar_grad, text_shade = g_colors[u_id % len(g_colors)]
            
            active_class = "uc-active" if is_active else "uc-inactive"
            active_badge = "<div class='uc-active-badge'>✓</div>" if is_active else ""
            
            card_html = textwrap.dedent(f"""
            <div class='user-card {active_class}' style='background: {avatar_grad};'>
              {active_badge}
              <div class='uc-avatar'>{inits}</div>
              <div class='uc-card-bottom'>
                <div class='uc-name'>{u_name}</div>
                <div class='uc-id'>User {u_id}</div>
                <div class='uc-stats'>{total_rated} rated</div>
              </div>
            </div>
            """)
            st.markdown(card_html, unsafe_allow_html=True)
            # Invisible overlay button to make it clickable
            if st.button(f"Select {u_name}", key=f"select_user_btn_{u_id}", use_container_width=True):
                st.session_state["selected_uid"] = u_id
                st.rerun()

    uid = st.session_state["selected_uid"]
    uname = user_names[uid]

    # Load user profile from precomputed profiles to avoid loading 530MB ratings CSV in memory
    prof = user_profiles.get(str(uid)) or user_profiles.get(uid) or {}
    total_rated = prof.get('total_rated', 0)
    avg_rating = prof.get('avg_rating', 0.0)
    history_list = prof.get('history', [])

    # --- User Profile Visualizer ---
    if prof:
        st.markdown(f"### 👤 {uname}'s Recommendation Profile")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            profile_html = textwrap.dedent(f"""
            <div style="background:var(--glass-md); border:1px solid var(--glass-border); border-radius:10px; padding:15px; height: 100%;">
              <p style="margin:0; font-size:0.8rem; color:var(--t2);">VIEWING PROFILE</p>
              <h4 style="margin:5px 0 10px; color:var(--t1);">{uname}</h4>
              <div style="display:flex; gap:15px;">
                <div>
                  <span style="font-size:1.4rem; font-weight:700; color:var(--coral);">{total_rated}</span>
                  <p style="margin:0; font-size:0.65rem; color:var(--t3); text-transform:uppercase;">Ratings</p>
                </div>
                <div style="border-left:1px solid var(--glass-border); padding-left:15px;">
                  <span style="font-size:1.4rem; font-weight:700; color:var(--cyan);">⭐ {avg_rating}</span>
                  <p style="margin:0; font-size:0.65rem; color:var(--t3); text-transform:uppercase;">Avg Rating</p>
                </div>
              </div>
            </div>
            """)
            st.markdown(profile_html, unsafe_allow_html=True)
            
        with col_m2:
            top_genres_list = list(prof.get('top_genres', {}).items())[:3]
            genres_bars = "".join(
                f'<div style="margin-bottom:8px;">'
                f'<div style="display:flex; justify-content:space-between; font-size:0.7rem; color:var(--t2); margin-bottom:2px;">'
                f'<span style="font-weight:600;">{g}</span>'
                f'<span style="margin-left:auto;">{pct}%</span>'
                f'</div>'
                f'<div style="width:100%; background:rgba(255,255,255,0.06); height:5px; border-radius:3px; overflow:hidden;">'
                f'<div style="width:{pct}%; background:linear-gradient(90deg, var(--coral), var(--magenta)); height:100%;"></div>'
                f'</div>'
                f'</div>'
                for g, pct in top_genres_list
            )
            
            affinity_html = (
                f'<div style="background:var(--glass-md); border:1px solid var(--glass-border); border-radius:10px; padding:15px; height: 100%;">'
                f'<p style="margin:0; font-size:0.8rem; color:var(--t2);">TOP GENRE AFFINITY</p>'
                f'<div style="margin-top:10px;">{genres_bars}</div>'
                f'</div>'
            )
            st.markdown(affinity_html, unsafe_allow_html=True)
            
        with st.expander("📜 View Rated History (Recent High Ratings)"):
            if history_list:
                hist_items = []
                for h in history_list[:5]:
                    raw_r = h.get("rating", 0.0)
                    is_anime = h.get("source", "").lower() == "anime" or h.get("type", "").lower() == "anime"
                    norm_r = (raw_r / 2.0) if is_anime else float(raw_r)
                    subtitle_val = f"{int(norm_r) if norm_r.is_integer() else norm_r}/5 ★"
                    hist_items.append({
                        "title": h.get("title", ""),
                        "source": h.get("source") or h.get("type") or "Movie",
                        "subtitle": subtitle_val
                    })
                render_row(st.columns(5), hist_items, key_prefix="t2_hist")
            else:
                st.markdown("<p style='font-size:0.8rem; color:var(--t3); font-style:italic;'>No rating history found for this user.</p>", unsafe_allow_html=True)

    st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)

    if "t2_alpha" not in st.session_state:
        st.session_state["t2_alpha"] = 0.6

    # Quick-set preset buttons
    st.markdown("<div class='filter-label'>Quick presets — tap to set</div>", unsafe_allow_html=True)
    _presets = [(0.0, "All Content"), (0.3, "Mostly Content"), (0.5, "50 / 50"),
                (0.7, "Mostly CF"), (1.0, "All CF")]
    _cur = st.session_state.get("t2_alpha", 0.6)
    _pcols = st.columns(len(_presets))
    for _i, (_val, _lbl) in enumerate(_presets):
        with _pcols[_i]:
            _active = abs(_cur - _val) < 0.05
            _label  = f"{'▶ ' if _active else ''}{_lbl}\n{_val:.1f}"
            if st.button(_label, key=f"t2_preset_{_i}", use_container_width=True):
                st.session_state["t2_alpha"] = _val
                st.rerun()

    alpha = st.slider(
        "Fine-tune  ·  content-based ← → collaborative",
        0.0, 1.0, step=0.1,
        help="Final score = α × CF + (1-α) × content-based",
        key="t2_alpha",
    )
    st.markdown(
        f"""<div class='blend-labels'>
          <span class='blend-chip' style='border-color:rgba(20,86,240,0.45);color:#6ea3ff'>
            Content-based&nbsp; <b>{(1-alpha)*100:.0f}%</b>
          </span>
          <span class='blend-sep'>α = {alpha:.2f}</span>
          <span class='blend-chip' style='border-color:rgba(255,85,48,0.45);color:#ff8a72'>
            Collaborative&nbsp; <b>{alpha*100:.0f}%</b>
          </span>
        </div>""",
        unsafe_allow_html=True,
    )

    # Initialize session state for Tab 2 recommendations
    if "t2_recs" not in st.session_state:
        st.session_state["t2_recs"] = None
    if "t2_show_count" not in st.session_state:
        st.session_state["t2_show_count"] = 5
    if "t2_rec_count" not in st.session_state:
        st.session_state["t2_rec_count"] = 5
    if "t2_seeds" not in st.session_state:
        st.session_state["t2_seeds"] = []

    col_slider, col_btn = st.columns([3, 1])
    with col_slider:
        rec_count = st.slider("Number of Recommendations:", 5, 20, step=5, key="t2_rec_count")
        st.session_state["t2_show_count"] = rec_count
    with col_btn:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        get_recs_clicked = st.button("🚀 Get Recommendations", key="t2_btn", use_container_width=True)

    if get_recs_clicked:
        with st.spinner("Scoring all items…"):
            history = prof.get("history", []) if prof else []
            top_genres = prof.get("top_genres", {}) if prof else {}

            if deep_model is not None and not item_map.empty:
                # ── Hybrid: CF + CB ──────────────────────────────────────
                _urow = user_map[user_map["user_id"] == uid]["user_idx"]
                if _urow.empty:
                    st.error(f"User {uid} not found in model.")
                    st.stop()
                uidx     = _urow.values[0]
                item_ids = item_map["item_idx"].values
                u_arr    = np.full(len(item_ids), uidx)

                cf_raw  = deep_model.predict([u_arr, item_ids], batch_size=4096, verbose=0).flatten()
                cf_norm = (cf_raw - cf_raw.min()) / (cf_raw.max() - cf_raw.min() + 1e-8)

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
                scores    = alpha * cf_norm + (1.0 - alpha) * cb_norm
                top_titles = item_map.iloc[scores.argsort()[::-1][:20]].to_dict("records")
                source_col = "source"
            else:
                # ── Content-based fallback (no TF model) ─────────────────
                st.caption("ℹ️ Running in content-based mode (TF model not available in cloud).")
                seeds   = history[:3] if history else []
                cb      = np.zeros(len(combined_df))
                matched = 0
                for t in seeds:
                    hit = combined_df[combined_df["title"] == t]
                    if hit.empty:
                        continue
                    cb     += cosine_similarity(tfidf_mat[hit.index[0]], tfidf_mat).flatten()
                    matched += 1
                if not matched:
                    # no history → use top-genres to seed
                    genre_seed = list(top_genres.keys())[:2] if top_genres else []
                    for t in combined_df[combined_df["description"].str.contains("|".join(genre_seed or ["Action"]), case=False, na=False)]["title"].head(3):
                        hit = combined_df[combined_df["title"] == t]
                        if not hit.empty:
                            cb += cosine_similarity(tfidf_mat[hit.index[0]], tfidf_mat).flatten()
                            matched += 1
                if matched:
                    cb /= matched
                seen = set(history)
                top_rows = sorted(
                    [(cb[i], combined_df.iloc[i]) for i in range(len(combined_df)) if combined_df.iloc[i]["title"] not in seen],
                    key=lambda x: -x[0]
                )[:20]
                top_titles = [{"title": r["title"], "type": r["type"]} for _, r in top_rows]
                source_col = "type"

            items = []
            for r in top_titles:
                item_genre = r.get("genre", "")
                expl = explain_recommendation(
                    r["title"],
                    item_genre,
                    history,
                    top_genres,
                )
                items.append({
                    "title": r["title"],
                    "source": r.get(source_col, "Movie"),
                    "subtitle": expl,
                })

            st.session_state["t2_recs"] = items
            st.session_state["t2_show_count"] = rec_count
            st.session_state["t2_seeds"] = seeds

    # Render recommendations if they exist in session state
    if st.session_state["t2_recs"] is not None:
        current_recs = st.session_state["t2_recs"]
        show_count = st.session_state["t2_show_count"]
        seeds = st.session_state["t2_seeds"]
        
        sec(f"Top {show_count} Picks from {uname}")
        st.caption(f"CF seeds: {', '.join(seeds)}")
        
        items_to_show = current_recs[:show_count]
        
        # Layout in rows of 5
        for start_idx in range(0, len(items_to_show), 5):
            chunk = items_to_show[start_idx : start_idx + 5]
            cols = st.columns(5)
            render_row(cols[:len(chunk)], chunk, key_prefix=f"t2_rec_{start_idx}")

        # Add Load More button if there are more recommendations available
        if show_count < len(current_recs):
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("🔽 Load More (+5 Recommendations)", key="t2_load_more", use_container_width=True):
                new_show = min(len(current_recs), show_count + 5)
                st.session_state["t2_show_count"] = new_show
                st.session_state["t2_rec_count"] = new_show
                st.rerun()

        # Markdown clipboard share option
        st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
        with st.expander("📋 Share recommendations list"):
            share_lines = [f"### 🎬 AI Recommendations for {uname}"]
            for ri, r in enumerate(items_to_show):
                share_lines.append(f"{ri+1}. **{r['title']}** ({r['source']}) — *{r['subtitle']}*")
            share_text = "\n".join(share_lines)
            st.text_area("Copy recommendations:", share_text, height=140, key="t2_share_area")


# ── TAB 3: Random ─────────────────────────────────────────────────────────
with t3:
    sec("Random Suggestions", "Discover something unexpected")

    st.markdown("<div class='filter-label'>Pick from</div>", unsafe_allow_html=True)
    rtype = st.radio("Pick from", ("All", "Movie", "Anime"), horizontal=True,
                     key="t3_rtype", label_visibility="collapsed")

    # 1. Popularity Mood Filter
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    mood = st.select_slider(
        "Discovery Mood Selector:",
        options=["Undiscovered Gems 💎", "Balanced Mix ⚖️", "Blockbusters & Hits 🔥"],
        value="Balanced Mix ⚖️",
        key="t3_mood"
    )

    rand_rec_count = st.slider("Number of Suggestions:", 5, 20, 5, step=5, key="t3_rec_count")

    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = 0.0
    if "rand_seed" not in st.session_state:
        st.session_state.rand_seed = random.randint(1, 10_000)

    remaining = max(0, 5 - int(time.time() - st.session_state.last_refresh))
    c1, c2 = st.columns([1.5, 4])
    with c1:
        # 2. Roulette Spin Animation
        if st.button("🎰 Spin Roulette", disabled=remaining > 0, key="t3_refresh"):
            with st.spinner("🎰 Spinning the roulette wheel..."):
                time.sleep(1.0)
            st.session_state.last_refresh = time.time()
            st.session_state.rand_seed    = random.randint(1, 10_000)
            st.balloons()
            st.rerun()
    with c2:
        if remaining > 0:
            st.caption(f"Next spin ready in {remaining}s")

    # Build and filter the pool of candidate items
    pool = combined_df if rtype == "All" else combined_df[combined_df["type"] == rtype]
    
    # Merge popularity metrics to calculate thresholds
    p_pool = _popular_pool()
    cnt_map = dict(zip(p_pool["title"], p_pool["cnt"]))
    pool_w_cnt = pool.copy()
    pool_w_cnt["cnt"] = pool_w_cnt["title"].map(cnt_map).fillna(1)
    
    cnt_vals = pool_w_cnt["cnt"].dropna()
    if not cnt_vals.empty:
        low_th = cnt_vals.quantile(0.33)
        high_th = cnt_vals.quantile(0.66)
    else:
        low_th, high_th = 5, 20
        
    if mood == "Undiscovered Gems 💎":
        pool_filtered = pool_w_cnt[pool_w_cnt["cnt"] <= low_th]
    elif mood == "Blockbusters & Hits 🔥":
        pool_filtered = pool_w_cnt[pool_w_cnt["cnt"] >= high_th]
    else:
        pool_filtered = pool_w_cnt

    if pool_filtered.empty:
        pool_filtered = pool_w_cnt # fallback if filtering returns empty

    samples = pool_filtered.sample(n=min(int(rand_rec_count), len(pool_filtered)), random_state=st.session_state.rand_seed)
    
    # Map subtitles to their popularity count
    items = []
    for r in samples.itertuples():
        cc = int(r.cnt)
        items.append({
            "title": r.title,
            "source": r.type,
            "subtitle": f"{cc} rating checkins" if cc > 1 else "Hidden Gem"
        })
        
    # Layout in rows of 5
    for start_idx in range(0, len(items), 5):
        chunk = items[start_idx : start_idx + 5]
        cols = st.columns(5)
        render_row(cols[:len(chunk)], chunk, key_prefix=f"t3_rand_{start_idx}")

    # 4. Random Trivia facts ticker
    trivia_facts = [
        "🎬 Akira (1988) was composed of 2,212 individual shots and 160,000 single pictures, using 327 different colors.",
        "🎬 Toy Story (1995) was the first fully computer-animated feature film.",
        "🎬 Spirited Away (2001) is the only hand-drawn and non-English-language animated film to win the Academy Award for Best Animated Feature.",
        "🎬 In Pulp Fiction (1994), the word 'f**k' is used 265 times.",
        "🎬 The skeleton model used in Jason and the Argonauts (1963) took animator Ray Harryhausen 4 months to animate.",
        "🎬 Studio Ghibli's name comes from the Italian noun ghibli, based on the Libyan-Arabic name for the hot desert wind, referring to the wind of change."
    ]
    # We use rand_seed to pick a consistent trivia fact per refresh
    trivia_fact = trivia_facts[st.session_state.rand_seed % len(trivia_facts)]
    st.markdown(f"""
    <div style="background:linear-gradient(90deg, rgba(255,85,48,0.06), rgba(20,86,240,0.06)); border:1px solid var(--glass-border); border-radius:10px; padding:12px; text-align:center; margin-top:25px;">
      <span style="font-size:1.1rem; vertical-align:middle;">💡</span> <span style="font-size:0.8rem; font-weight:600; color:var(--t2); font-style:italic;">{trivia_fact}</span>
    </div>
    """, unsafe_allow_html=True)


# ── TAB 4: Genre Picks ────────────────────────────────────────────────────
with t4:
    sec("Browse by Genre", "Hand-picked titles filtered by your favourite genre")

    gdf    = item_map.copy() if "genre" in item_map.columns else combined_df.copy()
    genres = _genre_list(gdf)

    col_g, col_f, col_m = st.columns([3, 2, 2])
    with col_g:
        sel_g = st.multiselect("Genres:", genres, default=[genres[0]] if genres else [], key="t4_genre")
    with col_f:
        gfilter = st.radio("Filter:", ("All", "Movie", "Anime/OVA"), horizontal=True, key="t4_filter")
    with col_m:
        match_mode = st.radio("Match Logic:", ("Any (OR)", "All (AND)"), horizontal=True, key="t4_match_mode")

    col_s, col_p = st.columns(2)
    with col_s:
        t4_sort = st.selectbox("Sort Results By:", ["Popularity", "Alphabetical", "Random"], key="t4_sort")
    with col_p:
        if "t4_page" not in st.session_state:
            st.session_state.t4_page = 0
            
    # Reset page offset if genres, filters, match mode, or sort options change
    current_key = f"{sel_g}_{gfilter}_{match_mode}_{t4_sort}"
    if st.session_state.get("t4_last_key") != current_key:
        st.session_state.t4_page = 0
        st.session_state.t4_last_key = current_key

    if sel_g:
        df = gdf.dropna(subset=["genre"]).copy()
        df["_gl"] = df["genre"].str.lower().str.split(",").apply(lambda x: [g.strip() for g in x])
        
        # Apply match mode logic
        sel_g_lower = [g.lower() for g in sel_g]
        if match_mode == "All (AND)":
            matched = df[df["_gl"].apply(lambda g: all(x in g for x in sel_g_lower))].copy()
        else: # Any (OR)
            matched = df[df["_gl"].apply(lambda g: any(x in g for x in sel_g_lower))].copy()

        if gfilter == "Movie":
            matched = matched[matched["type"].str.lower() == "movie"].copy()
        elif gfilter == "Anime/OVA":
            matched = matched[matched["type"].str.lower().isin(["anime", "tv", "ova", "ona", "special"])].copy()

        if matched.empty:
            empty_state("🎭", "No titles found",
                        f"No titles match the selected genre combination under this filter. Try 'Any (OR)' match logic.")
        else:
            # Subset distribution analytics
            tot_matches = len(matched)
            pct_movie = int((matched["type"].str.lower() == "movie").sum() / tot_matches * 100)
            pct_anime = 100 - pct_movie
            
            st.markdown(f"""
            <div style="background:var(--glass-md); border:1px solid var(--glass-border); border-radius:10px; padding:12px; margin-bottom:15px; display:flex; justify-content:space-around; align-items:center;">
              <div><span style="font-size:0.75rem; color:var(--t3); text-transform:uppercase;">Matches Found</span><br><b style="font-size:1.1rem; color:var(--t1);">{tot_matches}</b></div>
              <div style="border-left:1px solid var(--glass-border); padding-left:20px;"><span style="font-size:0.75rem; color:var(--t3); text-transform:uppercase;">Movies</span><br><b style="font-size:1.1rem; color:var(--cyan);">{pct_movie}%</b></div>
              <div style="border-left:1px solid var(--glass-border); padding-left:20px;"><span style="font-size:0.75rem; color:var(--t3); text-transform:uppercase;">Anime</span><br><b style="font-size:1.1rem; color:var(--coral);">{pct_anime}%</b></div>
            </div>
            """, unsafe_allow_html=True)

            # Apply sorting
            p_pool = _popular_pool()
            cnt_map = dict(zip(p_pool["title"], p_pool["cnt"]))
            matched["cnt"] = matched["title"].map(cnt_map).fillna(1)
            
            if t4_sort == "Popularity":
                matched = matched.sort_values("cnt", ascending=False)
            elif t4_sort == "Alphabetical":
                matched = matched.sort_values("title")
            else: # Random
                matched = matched.sample(frac=1, random_state=st.session_state.t4_page + 42)

            start_idx = st.session_state.t4_page * 5
            if start_idx >= len(matched):
                st.session_state.t4_page = 0
                start_idx = 0

            picks = matched.iloc[start_idx : start_idx + 5]
            
            genres_desc = " + ".join(sel_g)
            sec(f"{genres_desc} (Batch {st.session_state.t4_page + 1})")
            
            items = [
                {"title": r["title"],
                 "source": r.get("source", r.get("type", "Movie")),
                 "subtitle": (r.get("genre") or "")[:35]}
                for _, r in picks.iterrows()
            ]
            
            render_row(st.columns(5), items, key_prefix=f"t4_genre_{start_idx}")

            # Batch Pagination navigation
            st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("⬅️ Previous Batch", disabled=st.session_state.t4_page == 0, key="t4_prev_btn", use_container_width=True):
                    st.session_state.t4_page = max(0, st.session_state.t4_page - 1)
                    st.rerun()
            with col_next:
                if st.button("➡️ Next Batch", disabled=start_idx + 5 >= len(matched), key="t4_next_btn", use_container_width=True):
                    st.session_state.t4_page += 1
                    st.rerun()
    else:
        empty_state("🎭", "Select Genres", "Please select at least one genre from the multiselect dropdown.")


# ── TAB 5: Rate & Discover ────────────────────────────────────────────────
with t5:
    sec("Rate & Discover", "Step 1 · Rate titles below (1–5 ★)   ·   Step 2 · Click 'Get My Recommendations'")

    if "rate_seed" not in st.session_state:
        st.session_state.rate_seed    = 42
    if "user_ratings" not in st.session_state:
        st.session_state.user_ratings = {}

    # 1. Custom Autocomplete Search Title & Rate
    st.markdown("### 🔍 Search & Rate Any Title")
    fdf_all = combined_df.copy()
    fdf_all["label"] = fdf_all["title"] + " (" + fdf_all["type"] + ")"
    all_titles_sorted = sorted(fdf_all["label"].unique())
    
    col_s1, col_s2, col_s3 = st.columns([4, 2, 2])
    with col_s1:
        st.markdown("<div style='font-size:0.875rem; font-weight:500; margin-bottom:4px; color:var(--t1);'>Type to search title:</div>", unsafe_allow_html=True)
        custom_sel = st.selectbox("Type to search title:", all_titles_sorted, key="t5_custom_sel_title", label_visibility="collapsed")
    with col_s2:
        st.markdown("<div style='font-size:0.875rem; font-weight:500; margin-bottom:4px; color:var(--t1);'>Your Rating:</div>", unsafe_allow_html=True)
        custom_rating = st.selectbox("Your Rating:", ["5 ★", "4 ★", "3 ★", "2 ★", "1 ★"], key="t5_custom_rating", label_visibility="collapsed")
    with col_s3:
        st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
        if st.button("⭐ Add Rating", key="t5_add_custom_btn", use_container_width=True):
            try:
                title, stype = custom_sel.rsplit(" (", 1)
                stype = stype.strip(")")
                val = int(custom_rating[0])
                st.session_state.user_ratings[title] = val
                st.toast(f"Added {title} ({val}★) to your ratings!")
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)

    # 2. Rating History Drawer & Progress Tracker
    rated_n = len(st.session_state.user_ratings)
    progress_val = min(1.0, rated_n / 5.0)
    st.markdown(f"**Rating Progress:** {rated_n} / 5 rated items")
    st.progress(progress_val)
    
    if rated_n >= 5:
        st.markdown("""
        <div style="background:rgba(234,94,193,0.15); border:1px solid rgba(234,94,193,0.3); border-radius:10px; padding:10px 15px; margin-bottom:15px; text-align:center;">
          <span style="color:#ea5ec1; font-weight:bold; font-size:0.9rem;">⭐ Master Critic Mode Enabled</span>
          <p style="font-size:0.75rem; color:var(--t2); margin:4px 0 0;">You've rated 5+ items! We are using fine-grained matching for your recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
    if rated_n > 0:
        with st.expander("📋 Your Rated History"):
            for r_title, r_val in list(st.session_state.user_ratings.items()):
                col_r1, col_r2 = st.columns([5, 1])
                with col_r1:
                    st.markdown(f"⭐ **{r_val} ★** &nbsp;&nbsp;&middot;&nbsp;&nbsp; {r_title}")
                with col_r2:
                    if st.button("🗑️", key=f"del_rate_{r_title}", use_container_width=True):
                        del st.session_state.user_ratings[r_title]
                        st.rerun()

    st.markdown("### Rate Popular Suggestions")
    if st.button("🔄 Try Different Items", key="t5_shuffle"):
        st.session_state.rate_seed    = random.randint(1, 9_999)
        st.session_state.user_ratings = {}
        st.rerun()

    items = get_popular_items(n_anime=3, n_movie=2, seed=st.session_state.rate_seed)
    if not items:
        empty_state("⚠️", "Data not found",
                    "Ensure data/combined_ratings.csv exists and re-run preprocess.py.")
        st.stop()

    # Pre-fetch poster batch in parallel
    urls = _prefetch(items)

    cols_cards = st.columns(5)
    for idx, (col, item, url) in enumerate(zip(cols_cards, items, urls)):
        render_card(col, item["title"], item["source"], poster_url=url, key_prefix=f"t5_pop_{idx}")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    cols_sliders = st.columns(5)
    for col_s, item in zip(cols_sliders, items):
        with col_s:
            rkey = f"rate_{item['title']}_{st.session_state.rate_seed}"
            default_val = "Skip"
            if item["title"] in st.session_state.user_ratings:
                default_val = f"{st.session_state.user_ratings[item['title']]} ★"
                
            _opts = ["Skip", "1 ★", "2 ★", "3 ★", "4 ★", "5 ★"]
            try:
                _idx = _opts.index(default_val)
            except ValueError:
                _idx = 0
            rating = st.selectbox(
                "Rating",
                options=_opts,
                index=_idx,
                key=rkey,
                label_visibility="collapsed",
            )
            if rating != "Skip":
                try:
                    st.session_state.user_ratings[item["title"]] = int(rating[0])
                except ValueError:
                    pass
            elif item["title"] in st.session_state.user_ratings and rating == "Skip":
                del st.session_state.user_ratings[item["title"]]

    st.markdown("<hr style='margin:15px 0;'>", unsafe_allow_html=True)

    # 3. Discovery Bias Slider
    st.markdown("### Discovery Bias Settings")
    bias_option = st.select_slider(
        "Recommendation Bias:",
        options=["Deep Cut Gems 💎", "Balanced ⚖️", "Popular Hits 🔥"],
        value="Balanced ⚖️",
        key="t5_bias_slider"
    )

    if rated_n == 0:
        st.caption("Rate at least one title above to unlock recommendations.")
    else:
        lbl = "cold-start content-based" if rated_n < 5 else "content-based"
        st.caption(f"{rated_n} rated &middot; mode: {lbl}")

        if st.button("✨ Get My Recommendations", key="t5_recs"):
            with st.spinner("Finding your picks…"):
                recs = cold_start_recs(
                    st.session_state.user_ratings,
                    combined_df, tfidf_model, tfidf_mat, n=20,
                )
            if recs.empty:
                empty_state("🤷", "No results found",
                            "Try rating different titles for better results.")
            else:
                p_pool = _popular_pool()
                cnt_map = dict(zip(p_pool["title"], p_pool["cnt"]))
                recs["cnt"] = recs["title"].map(cnt_map).fillna(1)

                # Calculate similarities for sorting
                sims = []
                for _, r in recs.iterrows():
                    agg = None
                    total_w = 0.0
                    for t, rating_val in st.session_state.user_ratings.items():
                        hit = combined_df[combined_df["title"] == t]
                        if hit.empty:
                            continue
                        vec = tfidf_mat[hit.index[0]]
                        w   = rating_val / 5.0
                        agg = vec * w if agg is None else agg + vec * w
                        total_w += w
                    if agg is not None:
                        agg /= total_w
                        hit_rec = combined_df[combined_df["title"] == r["title"]]
                        if not hit_rec.empty:
                            sim = float(cosine_similarity(agg, tfidf_mat[hit_rec.index[0]])[0][0])
                        else:
                            sim = 0.1
                    else:
                        sim = 0.1
                    sims.append(sim)
                recs["sim"] = sims

                # Apply bias weighting
                if bias_option == "Deep Cut Gems 💎":
                    recs["final_score"] = recs["sim"] / (np.log1p(recs["cnt"]) + 1e-5)
                elif bias_option == "Popular Hits 🔥":
                    recs["final_score"] = recs["sim"] * np.log1p(recs["cnt"])
                else: # Balanced
                    recs["final_score"] = recs["sim"]

                recs = recs.sort_values("final_score", ascending=False)

                based_on = ", ".join(
                    f"{t} ({r}★)" for t, r in st.session_state.user_ratings.items()
                )
                sec("Your Personalized Picks")
                st.caption(f"Based on: {based_on}")

                top5 = [
                    {"title": r["title"], "source": r.get("type", "Movie"), "subtitle": f"{int(r['sim']*100)}% match"}
                    for _, r in recs.head(5).iterrows()
                ]
                render_row(st.columns(5), top5, key_prefix="t5_rec")

                if len(recs) > 5:
                    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                    more = [
                        {"title": r["title"], "source": r.get("type", "Movie"), "subtitle": f"{int(r['sim']*100)}% match"}
                        for _, r in recs.iloc[5:10].iterrows()
                    ]
                    render_row(st.columns(5), more, key_prefix="t5_rec_more")


# ── TAB 6: Watchlist ──────────────────────────────────────────────────────
with t6:
    sec("My Watchlist", "Manage your saved movies and anime titles")
    
    if "watch_later" not in st.session_state:
        st.session_state["watch_later"] = []
        
    watchlist = st.session_state["watch_later"]
    
    if not watchlist:
        empty_state("📋", "Your Watchlist is empty",
                    "Add titles to your watchlist by clicking on any card and using the '➕ Add Watchlist' button.")
    else:
        st.markdown(f"You have **{len(watchlist)}** saved titles. Here are your saved picks:")
        
        # Display saved titles in rows of 5
        urls = _prefetch(watchlist)
        
        for start_idx in range(0, len(watchlist), 5):
            chunk = watchlist[start_idx : start_idx + 5]
            cols = st.columns(5)
            
            # Render each card
            for col_idx, (item, url) in enumerate(zip(chunk, urls[start_idx : start_idx + 5])):
                render_card(cols[col_idx], item["title"], item["source"], poster_url=url, key_prefix=f"t6_wl_{start_idx+col_idx}")
                
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            col_dels = st.columns(5)
            for col_idx, item in enumerate(chunk):
                with col_dels[col_idx]:
                    if st.button("❌ Remove", key=f"t6_del_{start_idx+col_idx}_{abs(hash(item['title']))}", use_container_width=True):
                        # Remove item from watchlist
                        st.session_state["watch_later"] = [x for x in st.session_state["watch_later"] if x["title"] != item["title"]]
                        st.toast(f"Removed {item['title']} from watchlist!")
                        st.rerun()
                        
        st.markdown("<hr style='margin:20px 0;'>", unsafe_allow_html=True)
        
        # Clear Watchlist button and Watchlist-based Recommendations
        c_clear, c_rec = st.columns([1, 2])
        with c_clear:
            if st.button("🗑️ Clear Entire Watchlist", key="t6_clear_all", use_container_width=True):
                st.session_state["watch_later"] = []
                st.toast("Watchlist cleared!")
                st.rerun()
                
        with c_rec:
            if st.button("✨ Get Recommendations based on Watchlist", key="t6_recs_btn", use_container_width=True):
                st.session_state["t6_show_recs"] = True
                
        if st.session_state.get("t6_show_recs") and watchlist:
            sec("Watchlist-Based Recommendations")
            try:
                # Find indices of watchlist items in combined_df
                watch_indices = []
                for item in watchlist:
                    hit = combined_df[combined_df["title"] == item["title"]]
                    if not hit.empty:
                        watch_indices.append(hit.index[0])
                        
                if watch_indices:
                    # Average the TF-IDF vectors
                    avg_vec = tfidf_mat[watch_indices].mean(axis=0)
                    dist = cosine_similarity(avg_vec, tfidf_mat).flatten()
                    
                    # Sort indices by distance descending
                    rec_indices = dist.argsort()[::-1]
                    
                    wl_titles = {x["title"] for x in watchlist}
                    recs = []
                    for idx in rec_indices:
                        row = combined_df.iloc[idx]
                        if row["title"] not in wl_titles:
                            match_pct = min(100, max(0, int(dist[idx] * 100)))
                            recs.append({
                                "title": row["title"],
                                "source": row["type"],
                                "subtitle": f"{match_pct}% Match"
                            })
                            if len(recs) == 5:
                                break
                                
                    if recs:
                        render_row(st.columns(5), recs, key_prefix="t6_rec")
                    else:
                        st.info("No recommendations found.")
                else:
                    st.warning("Titles in watchlist could not be matched in local catalog.")
            except Exception as e:
                st.error(f"Error generating watchlist recommendations: {str(e)}")

