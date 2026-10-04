# Data Cleaning Process (Validation → Lyrics Filtering → Sentiment)

This document describes the cleaning/validation steps used to prepare the dataset for downstream analysis (e.g., NLP sentiment/alignment) using:

- `notebooks/1_validation.ipynb`
- `notebooks/2_lyrics_filtering.ipynb`
- `notebooks/3_sentiment_analysis.ipynb`

The pipeline focuses on ensuring:

- dataset integrity (missing values, duplicates, valid ranges)
- sufficient lyrics coverage for NLP
- lyrics text cleanliness and validity (English, non-trivial length, no metadata artifacts)
- a trustworthy lyric-sentiment signal that can be compared against audio valence (alignment)

---

## 1) Dataset validation (`notebooks/1_validation.ipynb`)

### 1.1 Dataset acquisition and basic structure
- In the setup cell, if **both** `dataset/songs.csv` and `dataset/artists.csv` are missing, the Kaggle dataset is downloaded via `kagglehub` (and a stale `dataset/.complete` marker is removed if present):
  - `serkantysz/490k-spotify-song-audio-embeddings-and-metadata`
  - Output directory: `dataset/`
- Seaborn theme (`whitegrid`/`muted`) and figure DPI are configured.
- Load `dataset/songs.csv` into `songs` and `dataset/artists.csv` into `artists`.

The notebook also prints:
- row/column counts
- sample rows (`songs.head()` / `artists.head()`)
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
- histogram and horizontal boxplot of the full popularity range
- counts/percentages of zero-popularity vs non-zero-popularity tracks

It then investigates whether zero-popularity is systematic rather than random:
- compares the genre mix of zero-popularity vs non-zero-popularity tracks (`value_counts(normalize=True)` per group, plus the percentage-point difference)
- plots the year distribution of zero vs non-zero popularity tracks
- runs a chi-square test of independence on the `genre × is_zero_pop` contingency table (`chi2_contingency`; p < 0.05 → zero-popularity status is *not* independent of genre)
- computes the zero-popularity rate per release year (`groupby('year')`), inspecting the most recent years

This evidence supports the later decision to keep only `popularity > 0` tracks (reducing label noise from unranked tracks).

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

### 1.13 Valence distribution — key NLP alignment feature
The notebook closes with a dedicated look at `songs['valence']`, since it is the audio counterpart to lyric sentiment:
- prints the valence mean, standard deviation, and `Corr(valence, popularity)`
- plots the valence distribution (with a mean line) and a valence-vs-popularity scatter

It notes that valence will be normalized to the −1 to +1 range before the alignment gap is computed in the sentiment notebook (§3).

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
- a fallback heuristic `looks_english()` used only when `detect()` does not return `'en'`

The English word set is built from NLTK stopwords:
- `ENGLISH_COMMON_WORDS = set(stopwords.words('english')) ∪ {'like', 'hate'}`
  (`DetectorFactory.seed = 0` is set for reproducible `langdetect` output, and `nltk.download('stopwords')` is run.)

Heuristic `looks_english(text, min_ratio=0.15)`:
- extracts words matching `\b[a-z]+\b`
- counts how many appear in `ENGLISH_COMMON_WORDS`
- requires the matched ratio to be `>= min_ratio` (default 0.15)

### 2.6 Validating the English-fallback threshold
Because the `looks_english` fallback decides the fate of lyrics that `langdetect` mislabels, the notebook validates the `min_ratio=0.15` threshold instead of assuming it:
1. take a random sample of `5000` tracks from `filtered` **first** (sampling before detection keeps this cheap)
2. compute `cleaned_lyrics`, `langdetect_result`, and `common_word_ratio` on the sample only
3. narrow down to the fallback-triggered subset (tracks where `langdetect_result != 'en'`)
4. add a blank `manual_label` column (`en` / non-English / garbage) to hand-label these tracks
5. sweep thresholds `0.15 → 0.50` in steps of `0.05`, computing precision and recall of `common_word_ratio >= t` against the manual ground truth

This confirms (or tunes) the fallback threshold. A companion markdown cell lists track indices that were flagged wrongly (both those corrected by the fallback and those still wrong).

### 2.7 Sample validation and runtime estimate
Before the long full-corpus pass, the notebook de-risks it:
- validates a random sample (`sample_size = 5000`) and logs per-track decisions to `logs/valid_lyrics_test_samples.txt` (stdout redirected to file)
- displays the flagged (`is_valid_lyrics == False`) tracks with their IDs/names for manual spot-checks
- estimates total runtime from an observed per-record rate (e.g. from a 1000-track test run) and formats the estimate as hours/minutes/seconds

### 2.8 Comprehensive lyrics validity function
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

### 2.9 Applying lyrics validation across years and writing outputs
To reduce memory/time overhead and enable reproducible logging, the notebook validates per release year.

- Creates output directories:
  - `data_filtered/` (for per-year cleaned CSVs)
  - `logs/` (for per-year validity decision logs and a consolidated `logs/summary_valid_lyrics.txt`)

Years are processed in descending blocks, one notebook cell per block:
- `2021–2023`, `2018–2020`, `2015–2017`, `2012–2014`, `2008–2011`, `2004–2007`, `2000–2003`

For each year within a block:

1. Compute mask `filtered['year'] == year`
2. If the log file doesn’t exist, open `logs/valid_lyrics_{year}.txt` and temporarily redirect `stdout`
3. Apply the validity function row-wise:
   - `filtered.loc[mask, 'is_valid_lyrics'] = ...apply(lambda row: is_lyrics_valid(...))`
4. Save the subset to `data_filtered/songs_{year}.csv` if it doesn’t exist yet:
   - `df[df['is_valid_lyrics']].to_csv(..., index=False)`
5. Append the per-year BEFORE/AFTER counts to `logs/summary_valid_lyrics.txt` via `write_to_summary_log(...)` (idempotent — a year block is only written once)

The helper `data_filtered_len(year)` re-reads the per-year CSV (falling back to the in-memory frame) so the notebook can total the surviving tracks, and `print_track_count()` reports old vs new counts and the number of deleted tracks.

---

## 3) Sentiment & alignment scoring (`notebooks/3_sentiment_analysis.ipynb`)

This notebook consumes the per-year cleaned CSVs in `notebooks/data_filtered/` and produces the lyric-sentiment and audio–lyric alignment signals used downstream.

### 3.1 Loading the filtered corpus
- Reads every file in `data_filtered/`, reverses the list, then concatenates them into a single `songs` frame (`pd.concat`).
- Prints the combined row count and inspects which tracks already have `lyric_sentiment` / `alignment_gap`.
- Re-applies `clean_lyrics()` (strips `[...]` sections and `00:12.34` LRC timestamps) and keeps only `is_valid_lyrics == True`, reporting how many tracks enter the sentiment pipeline.

### 3.2 Sentiment tooling
Four tools are compared:
- **VADER (primary)** — rule/lexicon tool producing a continuous `compound` score in `[-1, 1]`.
- **AFINN (secondary)** — lexicon tool whose raw score is summed/averaged, so it is *not* on the VADER scale.
- **pysentimiento** — transformer classifier returning `POS/NEU/NEG` probabilities.
- **`siebert/sentiment-roberta-large-english`** — binary classifier used as a directional/confidence check.

Objects are instantiated as `SentimentIntensityAnalyzer`, `Afinn()`, `create_analyzer(task="sentiment", lang="en")`, and a Hugging Face `pipeline("sentiment-analysis", ...)`.

Per-tool scoring helpers (first exercised on a single test track):
- VADER: `polarity_scores(text)['compound']`
- AFINN: `afinn_normalized_matched()` — raw AFINN score divided by the number of lexicon-matched words (`0.0` if none match)
- pysentimiento: `POS − NEG` probability
- RoBERTa: label mapped to `'POS'` / `'NEG'`

### 3.3 Valence normalization
- `valence_norm = valence * 2 - 1` maps audio valence from `[0, 1]` to `[-1, 1]` so it is directly comparable to VADER.

### 3.4 Sentiment on a sample (VADER + AFINN)
To keep runtime manageable, sentiment is first computed on a random subsample:
- `sample_songs = songs.sample(len(songs) // 25)` (~4%)
- `vader_compound` = VADER compound per track
- `afinn_raw` = `afinn_normalized_matched()` per track, then min–max rescaled to `[-1, 1]` as `afinn_rescaled`
- the sample is persisted to `test/samples_1.csv` so later cells can reload it

A (currently commented-out) pysentimiento scoring block is retained for reference.

### 3.5 Cross-tool / RoBERTa checks
- Runs the RoBERTa classifier on a further 1/10 subsample (`random_state=42`), storing `roberta_label` (`POS`/`NEG`) and saving `test/samples_with_roberta_1.csv`.
- Computes Pearson and Spearman correlations across `valence_norm`, `vader_compound`, `afinn_rescaled`, and `popularity` (`scipy.stats.pearsonr` / `spearmanr`), plus histograms of the three sentiment distributions.

Observation: VADER and AFINN correlate strongly with each other, while their correlation with audio `valence_norm` and with `popularity` is weak — which motivates the alignment-gap analysis.

### 3.6 Audio–lyric alignment on samples
For the sample, two alignment features are computed by halving the signed difference between lyric sentiment and audio valence (so the result stays in `[-1, 1]`):
- `audio_vader_alignment = (vader_compound − valence_norm) / 2`
- `audio_afinn_alignment = (afinn_rescaled − valence_norm) / 2`

Histograms of both alignments (alongside `popularity`) are plotted, and the augmented sample is written back to `test/samples_1.csv`.

### 3.7 Full-corpus VADER sentiment & alignment gap
- Splits `songs` into a dict keyed by release year (`songs_by_year`, years `2000–2023`).
- For each year, computes `lyric_sentiment` = VADER `compound` for every track.
- For each year, computes the alignment gap:
  - `alignment_gap = lyric_sentiment − valence_norm`
- Sanity-checks that per-year row counts match the saved CSVs and that the re-concatenated frame (`songs_new`) matches the original length.
- Plots distributions of `valence_norm`, `lyric_sentiment`, and `alignment_gap`.

### 3.8 Writing outputs
After an interactive confirmation prompt, the enriched frames are written back to `data_filtered/songs_{year}.csv`, so each per-year file now also carries `valence_norm`, `lyric_sentiment`, and `alignment_gap`.

---

## 4) Resulting dataset semantics

After these notebooks complete:

- The dataset has been validated for structural integrity (missing values, ranges, duplicates, linkage).
- Only tracks with non-empty, sufficiently long, mostly-English lyrics survive the filtering.
- Lyrics have been normalized to reduce escape-sequence artifacts and to remove LRC/timestamp/section metadata.
- Each surviving track carries an audio `valence_norm` (∈ `[-1, 1]`), a VADER `lyric_sentiment` (∈ `[-1, 1]`), and their `alignment_gap = lyric_sentiment − valence_norm`.

The per-year outputs in `notebooks/data_filtered/` represent the cleaned, sentiment-scored training/evaluation corpus for subsequent modeling steps.

