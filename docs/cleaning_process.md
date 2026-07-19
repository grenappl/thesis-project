# Data Cleaning Process (Validation + Lyrics Filtering)

This document describes the cleaning/validation steps used to prepare the dataset for downstream analysis (e.g., NLP sentiment/alignment) using:

- `notebooks/1_validation.ipynb`
- `notebooks/2_lyrics_filtering.ipynb`

The pipeline focuses on ensuring:

- dataset integrity (missing values, duplicates, valid ranges)
- sufficient lyrics coverage for NLP
- lyrics text cleanliness and validity (English, non-trivial length, no metadata artifacts)

---

## 1) Dataset validation (`notebooks/1_validation.ipynb`)

### 1.1 Dataset acquisition and basic structure
- Load `dataset/songs.csv` into `songs` and `dataset/artists.csv` into `artists`.
- If `dataset/songs.csv` and `dataset/artists.csv` are missing, download the Kaggle dataset:
  - `serkantysz/490k-spotify-song-audio-embeddings-and-metadata`
  - Output directory: `dataset/`

The notebook also prints:
- row/column counts
- sample rows (`songs.head()`)
- column data types
- basic year preview (e.g., `year >= 2024`)

### 1.2 Missing values audit
For both tables, the notebook computes:
- missing counts per column (`isnull().sum()`)
- missing percentages

It reports only columns with at least one missing value.

### 1.3 Lyrics coverage check
The notebook estimates how many tracks contain usable lyrics:
- `null_lyrics`: number of rows where `songs['lyrics']` is null
- `empty_lyrics`: number of rows where `lyrics` is blank after stripping whitespace
- `usable_lyrics = total - empty_lyrics`

Decision rule:
- If `usable_lyrics / total >= 0.70`, lyrics coverage is considered sufficient for NLP analysis.
- Otherwise, a warning is emitted.

A small visual spot-check prints the first few non-empty lyrics samples.

### 1.4 Popularity distribution sanity checks
The notebook analyzes `songs['popularity']`:
- descriptive statistics
- percentage of tracks with `popularity == 0`
- number of tracks with `popularity > 50`

Decision rule:
- If the fraction of zero popularity tracks is high (`zero_pct > 25`), recommend filtering out `popularity=0`.

It also visualizes:
- histogram of popularity
- horizontal boxplot

### 1.5 Audio feature distribution validation
Audio features are enumerated as:

- `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence`, `tempo`

The notebook:
- prints summary statistics (`describe()`)
- plots histograms for each feature

Additionally, it performs an **out-of-range check** using expected Spotify ranges (per Spotify documentation), e.g.:
- `danceability` / `energy` / `speechiness` / `acousticness` / `instrumentalness` / `liveness` / `valence` in `[0, 1]`
- `loudness` in `[-60, 5]`
- `tempo` in `[0, 300]`
- `popularity` in `[0, 100]`
- `mode` in `[0, 1]`

### 1.6 Temporal coverage / recency bias inspection
For `songs['year']`, the notebook:
- prints descriptive stats
- plots bar chart of track counts per year
- creates a scatter plot of `year` vs `popularity`

It also prints summary comparisons:
- tracks from `2010+` vs before 2010
- tracks from `2000+` vs overall (via a secondary “2ts” visualization)

This is used to detect possible recency bias, with a suggestion to filter/adjust in modeling if necessary.

### 1.7 Genre distribution and imbalance warning
The notebook:
- computes `value_counts()` for `songs['genre']`
- visualizes a horizontal bar chart

It computes an imbalance ratio:
- `ratio = max_genre / min_genre`

Decision rule:
- If `ratio > 5`, it warns about genre imbalance and suggests a stratified split.

### 1.8 Correlation analysis
Computes correlation matrix for:
- all audio features + `popularity`

Visualized via a heatmap, and additionally prints correlations sorted by absolute value with `popularity`.

### 1.9 Duplicate and ID integrity checks
The notebook checks duplicates in:
- `songs['id']` (duplicate track IDs)
- `(songs['name'], songs['artists'])` (duplicate track identity)
- `artists['id']` (duplicate artist IDs)

It warns if duplicates exist and shows example duplicated rows.

### 1.10 Linkage between `songs` and `artists`
Because `songs['artist_ids']` may store a list as a string (e.g. `"['abc','def']"`), the notebook:

1. Parses `songs['artist_ids']` via `ast.literal_eval`.
2. Extracts a `primary_artist_id` (first artist ID in the list).
3. Checks how many song rows have a `primary_artist_id` present in `artists['id']`.

Decision rule:
- If matched fraction is `>= 0.90`, linkage is considered strong.
- Otherwise, it warns about potential data loss from joining.

### 1.11 Thesis feasibility report
A consolidated checklist evaluates whether the dataset is suitable for the thesis, including:
- sufficient dataset size (>10k tracks)
- lyrics usable (>50% coverage)
- valence present and within `[0,1]`
- popularity has meaningful variance (std > 10)
- genre coverage (>90% non-null)
- year spans >10 years
- audio features have no missing values
- artist linkage ≥ 90%

### 1.12 Initial filtered dataset used downstream
At the end of the notebook, a preliminary `filtered` dataset is built by keeping tracks that satisfy:
- `popularity > 0`
- `lyrics` is non-null
- `lyrics` is not blank after stripping
- `year <= 2023`

It sorts by popularity (descending) and removes duplicates by:
- `drop_duplicates(subset=['name', 'artists'], keep='first')`

This serves as the input candidate set for lyrics filtering.

---

## 2) Lyrics cleaning and filtering (`notebooks/2_lyrics_filtering.ipynb`)

This notebook takes the initial `songs` dataframe and applies stricter, lyrics-specific cleaning and validation.

### 2.1 Filtering selection (baseline)
The notebook constructs `filtered` using the same core criteria as above:
- `popularity > 0`
- non-null lyrics
- lyrics not blank
- `year <= 2023`

It then:
- sorts by popularity
- deduplicates by `['name', 'artists']`
- copies/reset index

Afterwards, it enforces a time window:
- keep only tracks where `year >= 2000`

### 2.2 Detecting suspicious escape sequences
Some datasets contain literal escape sequences (e.g., `\n`, `\t`, `\\`, `\uXXXX`).

A helper `scan_escape_sequences()`:
- scans a lyrics column for known patterns
- counts occurrences
- prints example snippets around detected matches

This step is run **before** and **after** cleaning to confirm improvements.

### 2.3 Lyrics normalization/cleaning
The notebook cleans `filtered['lyrics']` via a series of transformations:
1. Lowercase (`str.lower()`)
2. Decode unicode escape sequences:
   - `\\u([0-9a-fA-F]{4})` → converts to the actual Unicode character
3. Replace literal escape sequences with spaces:
   - `\\n`, `\\r`, `\\t` → `' '`
4. Unescape quoted escapes:
   - `\\"` → `"`
   - `\\'` → `'`
5. Remove stray literal backslashes:
   - `\\\\` → `''`
6. Collapse whitespace:
   - `\s+` → single space
7. Strip leading/trailing whitespace

This produces a more consistent text representation for validation.

### 2.4 Lyrics structural cleanup (metadata/timestamps)
Before validation, `clean_lyrics()` removes common non-lyric artifacts:
- removes bracketed sections like `[Chorus]`, `[Verse 1]` using regex `\[.*?\]`
- strips LRC timestamps such as `00:12.34` with regex `\d{1,2}:\d{2}(\.\d+)?`

### 2.5 Language and “looks English” fallback
To avoid rejecting valid English lyrics incorrectly, the notebook uses:
- `langdetect.detect()` for primary language detection
- a fallback heuristic `looks_english()`

Heuristic:
- extract words matching `[a-z]+`
- count how many appear in a set of common English stop words
- require a minimum ratio (`min_ratio`, default 0.15)

### 2.6 Comprehensive lyrics validity function
Core filtering happens in `is_lyrics_valid(lyrics, index, ...)`.

A lyrics string is marked **valid** only if it passes all checks:

1. **Type/emptiness**
   - lyrics must be a `str`
   - not empty after stripping

2. **Metadata stripping must leave content**
   - if cleaning removes everything → invalid

3. **Placeholder text rejection**
   - rejects exactly-known placeholder terms such as:
     - `instrumental`
     - `no lyrics`
     - `lyrics not available`

4. **Minimum length**
   - requires at least `min_words` (default 20)
   - prevents very short junk entries from contaminating sentiment/alignment scoring

5. **Non-ASCII ratio threshold**
   - computes fraction of characters with `ord(c) > 127`
   - invalid if above `max_non_ascii_ratio` (default 0.15)
   - designed to catch mojibake/non-Latin noise

6. **Language detection**
   - `detect(cleaned)` must return `'en'`
   - if not `'en'`, it can still pass if `looks_english(cleaned)` is true

If it fails any condition, the function prints a reason (used for logs).

### 2.7 Applying lyrics validation across years and writing outputs
To reduce memory/time overhead and enable reproducible logging, the notebook validates per release year:

- Creates output directories:
  - `data_filtered/` (for per-year cleaned CSVs)
  - `logs/` (for per-year validity decision logs)

For each year (performed in descending blocks, e.g. 2021–2023, then 2018–2020, etc.):

1. Compute mask `filtered['year'] == year`
2. If the log file doesn’t exist, open `logs/valid_lyrics_{year}.txt` and temporarily redirect `stdout`
3. Apply the validity function row-wise:
   - `filtered.loc[mask, 'is_valid_lyrics'] = ...apply(lambda row: is_lyrics_valid(...))`
4. Save the subset to `data_filtered/songs_{year}.csv` if it doesn’t exist yet:
   - `df[df['is_valid_lyrics']].to_csv(..., index=False)`

At the end, it reports total track counts before vs after filtering by aggregating all per-year CSVs.

---

## 3) Resulting dataset semantics

After these notebooks complete:

- The dataset has been validated for structural integrity (missing values, ranges, duplicates, linkage).
- Only tracks with non-empty, sufficiently long, mostly-English lyrics survive the filtering.
- Lyrics have been normalized to reduce escape-sequence artifacts and to remove LRC/timestamp/section metadata.

The per-year outputs in `notebooks/data_filtered/` represent the cleaned training/evaluation corpus for subsequent NLP steps.

