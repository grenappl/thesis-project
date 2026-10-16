# Data Scientist Validation — Review Guide

**Thesis:** *Lyric and Audio Emotional Alignment as a Predictor of Spotify Track Popularity: A Feature Engineering Approach*
**Researchers:** Joenell Conrad C. de Pedro, Ivan Paul V. Ruelan — BS Computer Science, University of San Carlos
**Adviser:** Christian V. Maderazo, M.Eng
**Reviewer:** Neriah "BJ" Ato, Data Scientist
**Please return the completed review form (Section 9) by:** `[date — fill in before sending]`

Thank you for agreeing to review our modeling pipeline. This guide tells you what we are asking you to judge, how to check it, and where everything is. Expect roughly **2–3 hours** of review time, plus about **40 minutes of unattended compute** if you choose to reproduce the results yourself.

### At a glance

| | |
|---|---|
| **Your role** | Independent validator of the modeling stage — thesis Section 4.9.3(C) |
| **The two questions** | Is the hyperparameter configuration defensible? Do the reported R², RMSE and MAE figures follow correctly and reproducibly from it? |
| **Not your role** | Finding a better model, or judging the audio features themselves (a separate reviewer covers those) |
| **Your time** | About 2–3 hours of reading and judgement |
| **Optional compute** | A 2-minute quick check, or about 40 minutes for a full re-run. CPU only, no GPU |
| **What you return** | The review form in Section 9: one verdict and a short comment per statement, signed and dated |
| **Deadline** | `[date — fill in before sending]` |

---

## 0. What to expect

**What you will receive from us**

- This guide.
- The thesis manuscript as a PDF. Chapter 4 is the part that matters for your review.
- Read access to the code repository, which contains the dataset, the six modeling notebooks, every result file the thesis cites, and a one-command reproduction script.

**How the review runs**

1. You read the methodology (Sections 4.2.2 and 4.9 of the thesis) and this guide.
2. You decide how deep to go: review the reported figures as given, run the 2-minute quick check, or re-run the whole pipeline. Any of the three is acceptable. Please record which one you chose (form item B1).
3. You work through the questions in Steps 4 and 5 and fill in the form.
4. If something is unclear partway through, please ask us rather than guess. Contact details are at the end of this guide.
5. You return the completed form by the deadline above.

**What we will do with your review**

- Your verdicts and comments are reported **qualitatively** in Section 4.9.3(C) of the thesis, as an independent expert assessment. We will not convert them into a score.
- Where you mark *Agree with reservations* or *Disagree*, we will report the reservation as you stated it, together with our response or the correction we made. We will not leave it out.
- If you find an error, we would much rather fix it now than at the defense. Section 8 lists errors we have already found and corrected ourselves.
- We may contact you with one round of short follow-up questions after reading your comments.

**What you do not need to do**

- No database, WSL, GPU or audio processing is needed. The modeling stage reads only CSV files committed to the repository.
- You do not need to look at the web application or the audio-feature extraction pipeline. Section 2 explains why they are out of scope.
- You do not need to write any code.

---

## 1. What we are asking you to judge

Your role is defined in Section 4.9.3(C) of the thesis. It asks two questions:

1. **Is the hyperparameter configuration defensible?** Given the grid search described in Section 4.2.2, is the configuration selected for each algorithm a reasonable choice for this dataset's scale and feature composition?
2. **Are the reported figures correctly and reproducibly derived?** Do the R², RMSE and MAE figures reported for each model follow, correctly and reproducibly, from that configuration and its training runs?

**What we are *not* asking:** the thesis is explicit that this is *"not a second hyperparameter search or an attempt to improve on the reported figures."* You do not need to find a better model. We are asking whether the figures we report can be trusted, given how they were produced.

Your findings will be reported **qualitatively** in the thesis, separately from the music teacher's review, which checks the audio features themselves.

---

## 2. The study in one page

**Research question.** Does the emotional agreement or mismatch between a song's lyrics and its music help predict its Spotify popularity, beyond what the audio and lyric features already provide?

**The engineered feature.** Lyric sentiment is scored with VADER (compound score, range −1 to +1). Spotify's audio `valence` (range 0 to 1) is rescaled to the same range, and the feature is their signed difference:

```
valence_normalized = 2 × valence − 1
alignment_gap      = lyric_sentiment − valence_normalized        range [−2, 2]
```

Positive means the lyrics read brighter than the music sounds; negative means the reverse.

**Data.** 259,458 English-language tracks released 2000–2023, from the Kaggle *490K Spotify Song Audio Embeddings & Metadata* dataset. Tracks with zero popularity or unusable lyrics were excluded. The target is Spotify `popularity` (0–100), treated as a regression problem.

**The model ladder.** Five feature configurations, which differ only in which inputs they receive:

| Model | Features | Purpose |
|---|---|---|
| Baseline | none; predicts the training-set mean | establishes the R² = 0 floor |
| Audio-Only | the twelve Spotify audio descriptors (key one-hot encoded) | audio's standalone contribution |
| Lyric-Only | VADER lyric sentiment | lyrics' standalone contribution |
| **Control** | audio + lyric sentiment | the comparison point |
| **Alignment-Augmented** | audio + lyric sentiment + `alignment_gap` | the full model |

The four models with features are each trained twice: with **XGBoost** (the primary algorithm) and with **Random Forest** (a robustness check). With the shared Baseline, that makes **nine trained models**.

**The decision rule.** The central comparison is Control vs Alignment-Augmented, which differ by exactly one column. A contribution counts as established only if the **95% paired bootstrap confidence interval on ΔR²** (10,000 resamples of the test set) excludes zero.

**The result.** The interval contains zero under both algorithms, so the thesis reports **no detectable contribution** from the alignment feature. The reference figures are in Section 6.

**A note on the null result.** The thesis committed in advance (Section 4.9.3(A)) to reporting a gain that does not survive resampling as "no detectable contribution". The headline figure is therefore not something we are trying to defend as a positive finding. What we need from you is a judgement on whether the *null* is trustworthy: whether the pipeline could plausibly have hidden a real effect, or manufactured the absence of one.

**Out of scope.** The repository also contains a demonstration web application. It estimates the twelve audio descriptors from raw audio for new songs (`api/pipelines/demo/`) and serves the trained model through an API. None of that feeds the figures you are reviewing. The modeling stage uses the descriptors Spotify itself provides in the dataset. The audio-estimation pipeline is validated separately, so you can ignore `api/`, `app/` and everything under `scripts/` except `export_popularity_model.py`.

---

## 3. What you need

| | |
|---|---|
| Access | the GitHub repository `grenappl/thesis-project` — `[access method: collaborator invite / zip — fill in before sending]` |
| The thesis document | Chapter 4, especially Sections 4.2.2 and 4.9 — `[provided separately as PDF]` |
| Software | Git, Python 3.13, and [`uv`](https://docs.astral.sh/uv/). Windows, macOS and Linux all work |
| Hardware | any recent CPU; **no GPU needed**. 16 GB RAM recommended |
| Disk | about 15 GB free: roughly 0.5 GB for the repository and dataset, several GB for the Python environment, and several GB more for the Random Forest models if you do the full re-run |
| Not needed | No database, no WSL, and no audio processing. The modeling pipeline reads only the committed CSV files |

---

## 4. Setup (about 15–20 minutes, mostly downloading)

```bash
git clone https://github.com/grenappl/thesis-project.git
cd thesis-project
uv sync
```

The dataset (`notebooks/data_filtered/songs_2000.csv` … `songs_2023.csv`, about 400 MB) is part of the repository, so no separate download is needed.

**Why `uv sync` downloads so much.** The project's environment also serves the demo application, so it pulls in PyTorch, Hugging Face Transformers and (on Linux) TensorFlow. The modeling notebooks don't use them, but installing the locked environment as a whole is what makes the reproduction exact. Please use `uv sync` rather than installing packages by hand. `uv.lock` pins every version, including `xgboost`, `scikit-learn` and `networkx`, and the artist split uses `networkx`'s Louvain implementation, which is only reproducible under a fixed version.

**Which documents to trust.** For your review, the authoritative sources are the thesis manuscript, this guide, and the files in `notebooks/` and `notebooks/artifacts/`. The repository also contains `MODELING_REPORT.md`, a planning document written *before* the pipeline was run. Its Section 4 describes an earlier split design (primary-artist grouping with an edge cut) that was later replaced by the community-detection split described here. Please disregard that section.

---

## 5. Step-by-step review

### Step 1 — Read the methodology (about 30 minutes)

In the thesis document, read **Section 4.2.2** (model training, the hyperparameter grids, the ladder and the evaluation metrics) and **Section 4.9** (verification, testing and validation). Table 11 states the result ranges we expected before training.

### Step 2 — Get oriented in the code (about 20 minutes)

The modeling pipeline is six Jupyter notebooks plus one shared module, all in `notebooks/`. They run in order; each one reads what the previous one wrote to `notebooks/artifacts/`.

| File | What it does |
|---|---|
| `modeling_config.py` | **Read this first.** Shared definitions: the twelve descriptors, the ladder (`LADDER`), the columns that must never reach a model (`FORBIDDEN_COLS`), the metric functions (`regression_metrics`), and the artist-disjoint split (`community_partition`) |
| `4_behavioral_checks.ipynb` | Sanity checks on the engineered feature (formula correctness, distribution). Optional for your review |
| `5_splits.ipynb` | The 70/15/15 train/validation/test split. No artist appears in more than one partition; this is asserted in code |
| `6_tuning.ipynb` | The two grid searches, selected on validation RMSE |
| `7_model_ladder.ipynb` | Trains the nine models, verifies them, and computes the bootstrap confidence interval. **This is where the headline figures come from** |
| `8_shap.ipynb` | SHAP feature attribution on the XGBoost Alignment-Augmented model |
| `9_robustness.ipynb` | Seed, cross-validation, hyperparameter and cross-algorithm robustness checks |

Things worth checking directly in the code:

- **Feature isolation:** the Control model's columns must not include `alignment_gap`. This is asserted in notebook 7, in the verification cell.
- **Metrics:** R², RMSE and MAE are computed directly in `regression_metrics`, not through a library helper, so the values don't depend on the scikit-learn version.
- **The bootstrap:** `paired_bootstrap_delta_r2` in notebook 7. Each resample scores both models on the same rows, and R² is recomputed within each resample.

### Step 3 — Reproduce the results (optional, but the strongest check)

**Quick check, about 2 minutes.** This retrains the primary model (XGBoost Alignment-Augmented) from the frozen configuration and compares its predictions on all 37,632 test tracks against the predictions saved by notebook 7:

```bash
uv run python scripts/export_popularity_model.py
```

Expected output:

```
trained on 179,218 rows, 25 features, stopped at round 517
test R2 +0.06783  RMSE 15.0927  MAE 12.1144
max |retrained - notebook 7| over 37,632 test tracks: ~1e-06
wrote models/popularity/xgboost_alignment_augmented.json
wrote models/popularity/model_metadata.json
```

A maximum difference of around 10⁻⁶ is float32 rounding. Anything above 10⁻⁴ means the result did not reproduce. In that case the script prints `FAIL` and writes nothing. This check has passed on two separate machines; the most recent run (16 October 2026) gave a maximum difference of 1.9 × 10⁻⁶ and took under 2 minutes.

The model stops at round 517 here, not the 549 recorded in `config_xgb.json`. That is expected: 549 is where the *Control* model stopped during tuning, and each model early-stops on its own (Section 7, item 2). The two `wrote` lines refer to the copy of the model used by the demo application. Writing them changes nothing you are reviewing.

**Full check, about 40 minutes on CPU.** Run notebooks 5 to 9 in order. Either open them in VS Code or Jupyter and choose **Run All**, or run them from the terminal:

```bash
for nb in 5_splits 6_tuning 7_model_ladder 8_shap 9_robustness; do
  uv run --with nbconvert --with ipykernel jupyter nbconvert \
    --to notebook --execute --inplace "notebooks/$nb.ipynb"
done
```

That loop is bash (macOS, Linux, or Git Bash on Windows). In PowerShell, run the `uv run ... nbconvert ...` line once per notebook, in the same order.

Approximate times on our machine: notebook 5 about 2 minutes, notebook 6 about 10, notebook 7 about 15, notebook 8 about 3, notebook 9 about 10. Random Forest fits take most of that time; individual XGBoost fits take seconds.

This regenerates `notebooks/artifacts/` in your copy. That's intended; your copy is disposable. Then compare the new values against the reference figures in Section 6. They should agree to **4 decimal places for R²** and **2 decimal places for RMSE and MAE**. The last digits of individual predictions can differ slightly because of float32 rounding, so compare the summary figures rather than running `git diff` on the prediction files.

### Step 4 — Assess the configuration

The evidence for this step is in `notebooks/artifacts/tuning_results.csv` (every grid run), `config_xgb.json` and `config_rf.json` (the selected configurations), and Section 7 below, which lists the judgement calls we made.

Questions to consider:

- Is the XGBoost grid (`max_depth` {4, 6, 8} × `learning_rate` {0.01, 0.05, 0.1}, up to 1,000 rounds with 50-round early stopping, `subsample` and `colsample_bytree` fixed at 0.8) reasonable for about 179,000 training rows and 25 features?
- Is the Random Forest grid (`n_estimators` {200, 500} × `min_samples_leaf` {1, 5}, unrestricted depth, `max_features` = ⌊p/3⌋) reasonable?
- Is selecting on a single fixed 15% validation partition, rather than k-fold cross-validation, defensible at this scale?
- Is it appropriate to tune on the **Control** configuration and then reuse that configuration for every model in the ladder? See Section 7.

### Step 5 — Assess the derivation

The evidence for this step is the reproduction in Step 3 (or the reference figures, if you skip it), notebook 7's verification cell, and notebook 5's split assertions.

Questions to consider:

- Do the reported figures reproduce?
- Is the split free of leakage? (Artist-disjoint, with popularity and genre balanced across partitions.)
- Is the paired bootstrap on ΔR² a valid decision rule for this comparison?
- Do the robustness results support the reported conclusion of "no detectable contribution"?

### Step 6 — Fill in the review form

Use Section 9 of this guide. Short comments are fine. Where you disagree or have reservations, please say why, because that is what we will report.

---

## 6. Reference figures

These are the figures reported in the thesis. All come from `notebooks/artifacts/`.

### The split (`manifest.json`)

| Partition | Tracks | Share | Mean popularity |
|---|---|---|---|
| Train | 179,218 | 70.4% | 24.53 |
| Validation | 37,784 | 14.8% | 23.78 |
| Test | 37,632 | 14.8% | 23.97 |
| Excluded (collaborations spanning partitions) | 4,824 | 1.86% of corpus | — |

No artist appears in more than one partition, counting featured artists as well as headliners. Genre shares stay within 1.95 percentage points of the full corpus.

### Tuning (`tuning_results.csv`, selected on validation RMSE)

| Algorithm | Selected configuration | Validation RMSE |
|---|---|---|
| XGBoost | `max_depth` 4, `learning_rate` 0.05 (best round 549) | 14.6408 |
| Random Forest | `n_estimators` 500, `min_samples_leaf` 5 | 14.7127 |

All nine XGBoost configurations fell between 14.641 and 14.690 validation RMSE, so the grid is close to flat.

### The nine models, on the held-out test set (`ladder_metrics.csv`)

| Algorithm | Model | Features | R² | RMSE | MAE |
|---|---|---|---|---|---|
| shared | Baseline | 0 | −0.0013 | 15.642 | 12.766 |
| XGBoost | Audio-Only | 23 | 0.0671 | 15.098 | 12.121 |
| XGBoost | Lyric-Only | 1 | 0.0067 | 15.580 | 12.688 |
| XGBoost | Control | 24 | 0.0683 | 15.089 | 12.109 |
| XGBoost | Alignment-Augmented | 25 | 0.0678 | 15.093 | 12.114 |
| Random Forest | Audio-Only | 23 | 0.0619 | 15.141 | 12.169 |
| Random Forest | Lyric-Only | 1 | −0.0370 | 15.919 | 12.916 |
| Random Forest | Control | 24 | 0.0623 | 15.137 | 12.166 |
| Random Forest | Alignment-Augmented | 25 | 0.0626 | 15.135 | 12.160 |

The 23 audio columns are the eleven non-key descriptors plus twelve one-hot key columns.

### The central comparison (`central_comparison.json`)

| Algorithm | ΔR² (Augmented − Control) | 95% bootstrap CI | Excludes zero? |
|---|---|---|---|
| XGBoost | −0.00042 | [−0.00115, +0.00032] | No |
| Random Forest | +0.00027 | [−0.00072, +0.00123] | No |

### Robustness (`robustness_summary.json`)

| Check | Result |
|---|---|
| Seed sensitivity (5 seeds) | ΔR² per seed: −0.00042, +0.00058, −0.00106, +0.00096, −0.00084. Its mean (−0.00016) is smaller than the seed-to-seed R² standard deviation (0.00051) |
| 5-fold artist-disjoint CV | ΔR² per fold: +0.00151, −0.00086, −0.00049, −0.00039, +0.00063. Control R² per fold ranges from 0.023 to 0.165 |
| Hyperparameter settings (4) | ΔR²: −0.00042 (selected), +0.00031, −0.00098, −0.00018. The sign is not consistent |
| XGBoost vs Random Forest | opposite signs |

### SHAP attribution (`shap_group_contributions.csv`, `shap_feature_ranks.csv`)

The twelve audio descriptors account for 91.5% of mean |SHAP|, lyric sentiment for 6.7%, and the alignment score for 1.7%. `alignment_gap` ranks 13th of 14 individual features.

---

## 7. Decisions we would like you to scrutinise

These are the judgement calls in the pipeline. Some are not fully documented in the thesis text yet. We would rather you examine them directly than come across them.

1. **Tuning was done on the Control configuration.** Section 4.2.2 does not say which configuration the grid search runs on. We chose Control so the hyperparameters could not be selected in the treatment model's favour. The selected configuration was then reused, unchanged, for all nine models.
2. **Early stopping runs separately for each model.** All models share the frozen configuration, but each XGBoost model early-stops against the same validation partition, so the number of boosting rounds differs between models. We treat the round count as a result of early stopping rather than a tuned hyperparameter.
3. **Random Forest's `max_features` changes with the feature count.** It is ⌊p/3⌋ (Breiman's convention for regression), and p differs between models; Lyric-Only has p = 1. This follows from the rule rather than being re-tuned, but it does mean the parameter value differs between models.
4. **The artist-disjoint split uses community detection.** Taken literally, Section 4.9.1 requires no artist to appear in two partitions. Collaborations chain artists together: one connected group holds 51% of all tracks, so grouping on connected groups would make a 70/15/15 split badly unrepresentative. Instead, we detect collaboration communities (Louvain), place each community whole into one partition while balancing size, genre and popularity, and exclude the 1.86% of tracks whose artists span partitions. These excluded tracks are not random: they are all collaborations, more often Electronic and Hip-Hop, and more popular on average (33.6 vs 24.3).
5. **No feature scaling or target transformation.** Tree ensembles don't need scaling. Popularity is modeled on its raw 0–100 scale.
6. **Features deliberately excluded:** release year (recency is excluded by design), genre (used only for stratification), and artist follower and popularity counts (they would leak artist identity). These exclusions are asserted in code.
7. **The bootstrap resamples a single test partition.** It cannot capture variation that comes from the split itself. The 5-fold CV in notebook 9 covers that, and absolute R² turns out to depend heavily on the partition (0.023 to 0.165).
8. **Two grid points hit the round cap.** Both `learning_rate = 0.01` configurations stopped at round 998 or 999 of 1,000 (`tuning_results.csv`), so early stopping never triggered for them, and their validation RMSE may be slightly pessimistic. We did not raise the cap. The whole XGBoost grid spans only 0.05 RMSE points, and the robustness check's "deeper, slower" setting (`max_depth` 6, `learning_rate` 0.01) leaves the conclusion unchanged. You may still judge that the cap should have been higher.
9. **Absolute R² is low, and that is expected.** About 0.07 on the test set is at the low end of Table 11's expected range. Popularity depends heavily on things no model here receives, such as artist reach, marketing and playlisting. Artist-identity features are excluded on purpose (item 6), and the artist-disjoint split stops the model from memorising artists. The thesis's question is the *difference* between Control and Alignment-Augmented, not the absolute level.

## 8. Corrections already made

For transparency, here are two errors we found in our own pipeline and corrected before this review:

1. **Per-popularity-band metrics.** The first version reported R² within narrow popularity bands. Within a 20-point band the target barely varies, so R² collapsed to meaningless values (−320 for the top band). The band table now reports RMSE and MAE only.
2. **A duplicated hyperparameter setting.** The robustness check's list of alternative settings was written before tuning ran, and one "alternative" turned out to be identical to the selected configuration. It was replaced with a genuinely different setting. This **changed one conclusion**: the hyperparameter check had appeared to show a consistent direction, and with the fix it no longer does. The notebook now asserts that no alternative can equal the selected configuration.

Neither correction required retraining the nine reported models, and neither changes the headline result.

---

## 9. Review form

For each item, enter one verdict: **Agree**, **Agree with reservations**, **Disagree**, or **Not assessed**. Add a comment wherever the verdict isn't a plain *Agree*.

### A. The hyperparameter configuration

| # | Statement | Verdict | Comments |
|---|---|---|---|
| A1 | The XGBoost search space is appropriate for this dataset's size and features. | | |
| A2 | The Random Forest search space is appropriate for this dataset's size and features. | | |
| A3 | Selecting the configuration on a single fixed 15% validation partition is defensible at this scale. | | |
| A4 | Tuning on the Control configuration, and reusing that configuration for every model, keeps the central comparison fair. | | |
| A5 | The selected configurations are a defensible choice overall. | | |

### B. Derivation of the reported figures

| # | Statement | Verdict | Comments |
|---|---|---|---|
| B1 | The reported R², RMSE and MAE figures reproduce from the configuration. *(Note whether you ran the quick check, the full check, or reviewed the figures only.)* | | |
| B2 | The metrics are computed correctly. | | |
| B3 | The train/validation/test split is free of artist-identity leakage. | | |
| B4 | The 1.86% exclusion of cross-partition collaborations is acceptable and adequately reported. | | |
| B5 | The paired bootstrap confidence interval on ΔR² is a valid decision rule for this comparison. | | |
| B6 | The robustness checks support the conclusion of "no detectable contribution". | | |

### C. Overall

| # | Statement | Verdict | Comments |
|---|---|---|---|
| C1 | The reported model accuracy can be trusted, given how it was produced. | | |

**Concerns, or anything we should change or state more clearly:**

>

**Reviewer:** Neriah "BJ" Ato
**Signature:** ____________________
**Date:** ____________________

---

## 10. Where everything is

| What | Where |
|---|---|
| Pipeline notebooks | `notebooks/5_splits.ipynb` … `notebooks/9_robustness.ipynb` |
| Shared definitions | `notebooks/modeling_config.py` |
| All result files | `notebooks/artifacts/` |
| Split record | `notebooks/artifacts/manifest.json` |
| Every grid run | `notebooks/artifacts/tuning_results.csv` |
| Selected configurations | `notebooks/artifacts/config_xgb.json`, `config_rf.json` |
| The nine models' metrics | `notebooks/artifacts/ladder_metrics.csv` |
| Bootstrap comparison | `notebooks/artifacts/central_comparison.json` |
| Robustness checks | `notebooks/artifacts/robustness_*.csv`, `robustness_summary.json` |
| SHAP results | `notebooks/artifacts/shap_*.csv`, `fig_shap_*.png` |
| Test-set predictions, all nine models | `notebooks/artifacts/test_predictions.csv` |
| Quick reproduction script | `scripts/export_popularity_model.py` |

**Questions while reviewing:** `[contact details — fill in before sending]`

Thank you again for your time and expertise.
