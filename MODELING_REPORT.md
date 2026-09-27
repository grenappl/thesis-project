# Modeling Stage — Analysis Report and Build Plan

Status as of 2026-09-18. Covers Objectives 4 and 5 (train/evaluate the ladder, SHAP attribution)
and flags what remains for Objective 6 (demo application).

Nothing has been executed. Every number in Sections 2–4 comes from read-only probes of
`notebooks/data_filtered/`; the modeling notebooks described in Section 8 have been written but
not run.

---

## 1. Summary

The methodology in the final paper is specified tightly enough that modeling is mostly
implementation to spec rather than design. Three things needed resolving before code:

| Question | Resolution |
|---|---|
| 11 or 12 audio descriptors? | **Twelve.** The final paper is consistent (Table 2, Table 5, §4.9.3B, REQ-4.4-2). The "eleven" in `revisions/models_metrics_draft.md` was a draft artifact. |
| How to group artists for the split? | **Option C** — primary-artist grouping plus an explicit edge cut. A strict reading is infeasible; see Section 4. |
| Notebook or package? | **Notebooks**, per your call, with one shared constants module. See Section 8. |

One methodological gap the paper leaves open is flagged in Section 7, and six internal
inconsistencies in the submitted document are listed in Section 6.

---

## 2. The corpus is modeling-ready

`notebooks/data_filtered/songs_2000.csv` … `songs_2023.csv`, concatenated:

| Property | Measured |
|---|---|
| Retained tracks | **259,458** (all IDs unique) |
| Year range | 2000–2023 |
| `popularity` | mean 24.51, median 23, **σ = 15.94**, range 1–96 |
| Zero-popularity rows | 0 — the §4.2 exclusion already applied |
| Invalid-lyric rows | 0 — `is_valid_lyrics` is True throughout |
| Missing values in modeling columns | 0 (only `album_name` has 10 nulls, not a feature) |
| Genres | 10, from Rock 96,321 to Jazz 4,511 |
| **Genre imbalance** | **21.4 : 1** — matches §4.1.1's "approximately 21:1" |

Objectives 1–3 are therefore complete in the data. `valence_norm`, `lyric_sentiment` and
`alignment_gap` are all present and precomputed.

---

## 3. The engineered feature verifies clean

This is the §4.9.1 *correctness of the model's engineered input* check, run against the real
corpus rather than hand-computed cases alone:

- `valence_norm == 2·valence − 1` — exact
- `alignment_gap == lyric_sentiment − valence_norm` — **max deviation 4.4 × 10⁻¹⁶**
- Subtraction order correct; **not** inverted
- Bounded in [−1.988, 1.948] ⊂ [−2, 2] as §4.2.1 claims by construction

The tails behave as §4.9.3(D) predicts. Most positive gaps are sad-sounding production against
elated lyrics (valence ≈ 0.035, sentiment ≈ 0.998); most negative are bright, high-valence
production against bleak lyrics (valence ≈ 0.976, sentiment ≈ −0.999).

---

## 4. The artist-split investigation — why Option C

§4.9.1 asserts that *"the intersection of artist identifiers between any two partitions is
empty."* Read literally, collaborations chain artists together transitively. Measured on this
corpus:

- 18.5% of tracks credit 2+ artists; 48,206 unique artists
- Union-find over co-appearance yields **19,348 connected components**
- **The largest holds 132,076 tracks — 50.9% of the entire corpus**

A 70/15/15 split remains arithmetically possible (the giant component fits inside train), but it
forces train to *be* the mainstream collaboration network:

| | Train (giant component) | Val/test pool (remainder) |
|---|---|---|
| Tracks | 132,076 | 127,382 |
| Mean popularity | 29.1 | **19.8** |
| Rock | 16.9% | **58.1%** |
| Hip-Hop | 15.7% | **1.9%** |
| Electronic | 22.1% | 7.9% |
| R&B | 7.6% | 2.1% |

That is not a train/test split of one population. Genre stratification becomes impossible in any
meaningful sense, and a 9.3-point popularity gap between train and test depresses R² on its own,
independent of any feature — which would likely trip Table 11's *"Control R² below 0.05 →
investigate the pipeline"* diagnostic for a reason having nothing to do with the pipeline.

By contrast, grouping on primary artist alone gives 30,496 groups with the largest at **0.39%**
of the corpus — stratification works, distributions match — but a featured artist can then appear
on both sides of the split, violating the assertion §4.9.1 actually makes.

### Option C, as implemented in `5_splits.ipynb`

1. Group by primary artist (`artist_ids[0]`).
2. Split 70/15/15 with `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)`,
   stratifying on genre, grouping on primary artist; folds 0–13 → train, 14–16 → val, 17–19 → test.
3. **Edge cut:** train keeps every track; validation drops any track crediting an artist present
   anywhere in train; test drops any track crediting an artist in train *or* validation.
4. Assert all three pairwise artist intersections are empty; record the dropped counts by reason
   in the manifest.

The §4.9.1 invariant then holds on the literal reading, stratification survives, and the cost is
a bounded, *reported* exclusion instead of a silently skewed test set. Upper bound on the cost:
18.5% of val/test rows (only multi-artist tracks are ever at risk); the notebook prints the actual
figure.

**§4.2.1 needs one added sentence** describing the edge cut, and §4.1.1's existing phrase
— *"genre proportions are preserved as closely as the artist-grouping constraint allows"* —
already licenses it.

---

## 5. What the central result is likely to be

Two measurements set expectations honestly, before anything is trained:

- `corr(alignment_gap, popularity)` = **0.019**
- `corr(alignment_gap, lyric_sentiment)` = **0.857**; `corr(alignment_gap, valence)` = −0.393

The gap is largely a rescaled sentiment score, and it is an exact linear combination of two
columns the Control model already receives. Expect ΔR² at or below the bottom of Table 11's
0.001–0.01 band, quite possibly with a confidence interval straddling zero.

This is a publishable outcome and the paper is already built for it: §4.9.3(A) pre-commits to
reporting a non-surviving gain as *no detectable contribution*, and §4.9.3(B) makes SHAP a
co-primary result precisely because it *"remains informative even where that aggregate difference
is small."* The §4.9.3(D) behavioral checks — distribution, genre patterning against Attia,
tails, input correlations — stand on their own regardless of what the model does.

The one wording change that makes this path fully coherent is item 2 in the next section.

---

## 6. Internal inconsistencies in the submitted paper

Ordered by how much a panelist would care.

1. **SRS Figure B.2** (p. 107) labels the alignment-score box **`1 − |z(valence) − z(sentiment)|`**
   — the z-scored formula §4.2.1 explicitly argues *against* (*"unlike a z-scored distance would
   need"*). Chapter 4's Figure 6 carries no such subtitle. Stale figure; regenerate.
2. **NFR1 (§4.4.1) and REQ-5.4-4 vs §4.9.3(A).** The requirement says the Alignment-Augmented
   model **"shall achieve"** a measurable R² improvement, at High priority. §4.9.3(A) says a gain
   that fails resampling is reported as no detectable contribution. As written, an honest null
   result *fails a High-priority requirement*. Reword NFR1 as a test to be applied, not an outcome
   guaranteed.
3. **§4.2.2, opening sentence:** "the twelve **self-computed** audio descriptors" — contradicts
   §4.1.4, §4.3, Table 4 and FR2, all of which state the reference-dataset descriptors are
   dataset-provided. One-word fix.
4. **Table 11:** Control RMSE given as 15–20 popularity points. With σ = 15.94, an RMSE of 20
   implies R² ≈ −0.58, contradicting the same table's 0.05–0.20. The consistent band is
   **14.3–15.5**.
5. **§4.2.1** still excludes records *"whose raw audio cannot be retrieved or decoded"* — a
   leftover from the pre-correction design; §4.2 states plainly that the reference dataset needs
   no raw audio.
6. **§4.1.1** justifies genre stratification by reference to *"the per-genre performance reporting
   described in Section 4.2.2"* — §4.2.2 describes no such reporting. Either add it or drop the
   cross-reference. (Table 1 separately implies popularity-band-stratified metrics.) Notebook 7
   produces both breakdowns either way.

---

## 7. Open decision the paper does not settle

§4.2.2 says the hyperparameter configuration is selected by grid search on the validation
partition, but never says **which feature configuration the search runs on**. Tuning on
Alignment-Augmented would let the hyperparameters be chosen in the treatment's favour.

`6_tuning.ipynb` therefore tunes on **Control** — the reference point of the central comparison,
and the conservative choice. It is switchable via `TUNE_ON`. Expect the Data Scientist reviewer
(§4.9.3 C) to ask; the answer belongs in the chapter.

---

## 8. The notebooks

Written, not run. They follow the existing `notebooks/` numbering.

| Notebook | Covers | Runtime |
|---|---|---|
| `4_behavioral_checks.ipynb` | §4.9.3(D) in full, §4.9.1 engineered-input check, Table 1 profile | minutes |
| `5_splits.ipynb` | §4.1.1, §4.2.1 split; §4.9.1 partition integrity; manifest | minutes |
| `6_tuning.ipynb` | §4.2.2 grids — XGB 3×3, RF 2×2 | hours (RF dominates) |
| `7_model_ladder.ipynb` | Table 5 nine models, §4.9.1 verification, §4.9.3(A) bootstrap, per-genre/band | 1–2 hours |
| `8_shap.ipynb` | §4.9.3(B) co-primary result | ~30 min |
| `9_robustness.ipynb` | §4.9.2 all four checks (26 XGB fits) | hours |

`notebooks/modeling_config.py` holds the shared constants — the twelve descriptors, the Table 5
ladder, the forbidden-column list, the metric functions. It is not a package; it exists because
§4.9.1's feature-configuration isolation and configuration-parity checks are only meaningful if
every notebook builds its matrices from one definition. Copying the ladder into six notebooks is
exactly how that check silently stops being true.

**Prerequisite before running anything:**

```bash
uv add xgboost shap
```

`scikit-learn`, `pandas`, `numpy`, `scipy`, `joblib` and `matplotlib` are already present.

Artifacts land in `notebooks/artifacts/`: the manifest and splits, frozen configs, the nine
trained models, test predictions, the ladder metrics table, the central comparison with its
bootstrap interval, SHAP tables, robustness tables, and the figures.

### Columns deliberately excluded from every model

Asserted in code, not just documented: `year`, `genre` (filter and stratification only — NFR1
excludes recency), `total_artist_followers`, `avg_artist_popularity` (leakage, §4.9.3 A),
and **`valence_norm`** — the throwaway intermediate from §4.2.1. Feeding that to Control would
hand it half the alignment relationship the ablation is supposed to withhold.

---

## 9. Not built yet: the demo prediction path

`api/routers/` contains only `demo_features` and `tracks`. There is **no `/predict` endpoint and
no popularity model in the application**; the demo stops at feature extraction. That leaves
unbuilt:

- **FR6, FR7, REQ-4.4-8** — process inputs through XGBoost, return and display predicted
  popularity with the alignment breakdown
- **Figure 7 / Figure B.4** — the `POST /predict` sequence
- **Objective 6** — the demonstration application as specified

Phase 7 of the plan covers it: freeze `xgboost__alignment_augmented.joblib`, add the endpoint,
wire it into `app/app/test-feature-extraction/`. Worth scheduling explicitly — it is a
deliverable in Chapter 1, not a detail.

Separately, `api/models/track.py`'s `audio_sentiment` / `emotional_alignment` fields predate and
do not match the `lyric_sentiment` / `alignment_gap` naming, and are not currently wired to
anything.

---

## 10. Suggested order of work

1. `uv add xgboost shap`
2. Run `4_behavioral_checks.ipynb` — cheap, and it tells you within the hour roughly where ΔR²
   will land
3. Run `5_splits.ipynb`; check the genre-mix table and the train-vs-test popularity gap before
   proceeding
4. `6_tuning.ipynb` overnight
5. `7_model_ladder.ipynb` → the headline results table and the decision rule
6. `8_shap.ipynb`, `9_robustness.ipynb`
7. Package the outputs for the two §4.9.3(C) reviewers; build the `/predict` endpoint
8. Apply the Section 6 fixes to the document
