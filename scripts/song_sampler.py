import os
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.cache_handler import CacheFileHandler
from googleapiclient.discovery import build as build_youtube_client
from dotenv import load_dotenv

load_dotenv()

# ── Config ──
DATA_DIR = "notebooks/data_filtered"
YEARS = range(2000, 2024)
SAMPLE_SIZE = 15
OUTPUT_DIR = "samples"

FEATURE_COLS = [
    "id", "name", "album_name", "artists",
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "valence", "tempo", "duration_ms",
    "lyric_sentiment", "alignment_gap", "genre", "popularity",
]

# ── Step 1: Load + combine the 260k dataset (2000–2023) ────────────────────
def load_combined():
    dfs = []
    for yr in YEARS:
        path = os.path.join(DATA_DIR, f"songs_{yr}.csv")
        dfs.append(pd.read_csv(path))
    return pd.concat(dfs, ignore_index=True)

# ── Step 2: Build per-bucket candidate pools, ranked by popularity ─────────
def build_candidate_pools(df):
    df = df[FEATURE_COLS].dropna(subset=["id", "name", "artists"]).copy()

    df["alignment_bucket"] = pd.cut(
        df["alignment_gap"],
        bins=[-float("inf"), -0.15, 0.15, float("inf")],
        labels=["mismatch", "near_zero", "aligned"],
    )

    pools = {}
    for key, g in df.groupby(["alignment_bucket", "genre"], observed=True):
        pools[key] = g.sort_values("popularity", ascending=False).to_dict("records")
    return pools

# ── Step 3: Lookups — a candidate only counts if BOTH resolve ──────────────
def spotify_match(sp, name, artists):
    query = f"track:{name} artist:{artists}"
    results = sp.search(q=query, type="track", limit=1)
    items = results.get("tracks", {}).get("items", [])
    if not items:
        return None
    return items[0]["external_urls"]["spotify"]

def youtube_match(yt, name, artists):
    query = f"{name} {artists} lyrics"
    try:
        response = yt.search().list(
            q=query, part="id", type="video", maxResults=1
        ).execute()
    except Exception:
        return None

    items = response.get("items", [])
    if not items:
        return None
    video_id = items[0]["id"].get("videoId")
    if not video_id:
        return None
    return f"https://www.youtube.com/watch?v={video_id}"

def resolve_candidate(sp, yt, candidate):
    """Returns (spotify_url, youtube_url) only if BOTH are found, else None."""
    spotify_url = spotify_match(sp, candidate["name"], candidate["artists"])
    if not spotify_url:
        return None

    youtube_url = youtube_match(yt, candidate["name"], candidate["artists"])
    if not youtube_url:
        return None

    return spotify_url, youtube_url

# ── Step 4: Round-robin across buckets, validating as we go ────────────────
def build_validated_sample(pools, sp, yt, n=SAMPLE_SIZE):
    bucket_keys = [k for k, v in pools.items() if v]
    cursor = {k: 0 for k in bucket_keys}
    sample_rows = []
    active_keys = list(bucket_keys)

    while active_keys and len(sample_rows) < n:
        still_active = []
        for key in active_keys:
            if len(sample_rows) >= n:
                break

            pool = pools[key]
            idx = cursor[key]
            matched = False

            while idx < len(pool):
                candidate = pool[idx]
                idx += 1
                result = resolve_candidate(sp, yt, candidate)
                if result:
                    spotify_url, youtube_url = result
                    candidate["spotify_url"] = spotify_url
                    candidate["youtube_url"] = youtube_url
                    sample_rows.append(candidate)
                    matched = True
                    break
                # missing Spotify match OR missing YouTube match — discard, try next

            cursor[key] = idx
            if matched and idx < len(pool):
                still_active.append(key)
            elif not matched:
                pass  # pool exhausted for this bucket

        active_keys = still_active

    return pd.DataFrame(sample_rows)

def main():
    combined = load_combined()
    print(f"Loaded {len(combined):,} total rows across {len(list(YEARS))} years")

    pools = build_candidate_pools(combined)
    print(f"Built {len(pools)} stratification buckets")

    spotify_auth = SpotifyClientCredentials(
        client_id=os.environ["SPOTIFY_CLIENT_ID"],
        client_secret=os.environ["SPOTIFY_CLIENT_SECRET"],
        cache_handler=CacheFileHandler(cache_path="scripts/spotipy_cache.json")
    )
    sp = spotipy.Spotify(client_credentials_manager=spotify_auth)

    yt = build_youtube_client(
        "youtube", "v3", developerKey=os.environ["YOUTUBE_API_KEY"]
    )

    sample = build_validated_sample(pools, sp, yt, n=SAMPLE_SIZE)
    print(f"Validated sample of {len(sample)} tracks (target was {SAMPLE_SIZE})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUTPUT_DIR, "validation_sample.csv")
    sample.to_csv(out_csv, index=False)
    print(f"Saved sample metadata to {out_csv}")

if __name__ == "__main__":
    main()