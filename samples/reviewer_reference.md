# Reviewer reference: the figures we report and the judgement calls we made

**Open this after you have finished Part 1 of the review form** (see `data_scientist_review_guide.md`, Section 5). It contains the answers the review is meant to test, so reading it first would make your Part 1 findings less independent.

`scripts/verify_reference_figures.py --compare-reference` checks every table in Section A against what it measures from the raw predictions. If you edit a figure here, the script will report it.

---

## A. Reference figures

These are the figures reported in the thesis. All come from `notebooks/artifacts/`.

**The result.** The 95% paired bootstrap interval on ΔR² (Control vs Alignment-Augmented) contains zero under both algorithms, so the thesis reports **no detectable contribution** from the alignment feature. Table 11 of the thesis states the result ranges we expected before training.

### The split (`manifest.json`)

| Partition | Tracks | Share | Mean popularity |
|---|---|---|---|
| Train | 179,218 | 70.4% | 24.53 |
| Validation | 37,784 | 14.8% | 23.78 |
| Test | 37,632 | 14.8% | 23.97 |
| Excluded (collaborations spanning partitions) | 4,824 | 1.86% of corpus | n/a |

No artist appears in more than one partition, counting featured artists as well as headliners. Genre shares stay within 1.95 percentage points of the full corpus.

*A note if you check this yourself:* match on `artist_ids`, not on artist names. 235 names (generic ones such as "Alias" or "8") appear in more than one partition, but each is a different artist who happens to share a name; no artist ID is shared.

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

## B. Decisions we would like you to scrutinise

These are the judgement calls in the pipeline. Some are not fully documented in the thesis text yet. We would rather you examine them directly than come across them.

1. **Tuning was done on the Control configuration.** Section 4.2.2 does not say which configuration the grid search runs on. We chose Control so the hyperparameters could not be selected in the treatment model's favour. The selected configuration was then reused, unchanged, for all nine models.
2. **Early stopping runs separately for each model.** All models share the frozen configuration, but each XGBoost model early-stops against the same validation partition, so the number of boosting rounds differs between models. We treat the round count as a result of early stopping rather than a tuned hyperparameter.
3. **Random Forest's `max_features` changes with the feature count.** It is ⌊p/3⌋ (Breiman's convention for regression), and p differs between models; Lyric-Only has p = 1. This follows from the rule rather than being re-tuned, but it does mean the parameter value differs between models.
4. **The artist-disjoint split uses community detection.** Taken literally, Section 4.9.1 requires no artist to appear in two partitions. Collaborations chain artists together: one connected group holds 51% of all tracks, so grouping on connected groups would make a 70/15/15 split badly unrepresentative. Instead, we detect collaboration communities (Louvain), place each community whole into one partition while balancing size, genre and popularity, and exclude the 1.86% of tracks whose artists span partitions. These excluded tracks are not random: they are all collaborations, more often Electronic and Hip-Hop, and more popular on average (33.6 vs 24.3).
5. **No feature scaling or target transformation.** Tree ensembles don't need scaling. Popularity is modeled on its raw 0–100 scale.
6. **Features deliberately excluded:** release year (recency is excluded by design), genre (used only for stratification), and artist follower and popularity counts (they would leak artist identity). These exclusions are asserted in code.
7. **The bootstrap resamples a single test partition.** It cannot capture variation that comes from the split itself. The 5-fold CV in notebook 9 covers that, and absolute R² turns out to depend heavily on the partition (0.023 to 0.165).
8. **Subgroup bias is measured and reported, not corrected.** `validator_package/bias_report.md` breaks the primary model's residuals down by popularity, genre, release year, alignment gap and other slices. Overall bias is about zero, but three things are worth your scrutiny: (a) predictions are narrow (the model cannot separate hits from misses) and slightly over-confident at the tails (calibration slope 0.77); (b) the model over-predicts Hip-Hop by about 4 points, more than the Baseline would, and its Hip-Hop R² is negative; (c) residuals trend with release year, but that trend is already in the target (it correlates 0.95 with the Baseline's), because year is excluded by design. We did not tune the model to remove these, since that would change the reported figures; tell us whether the report should say more about them.

## C. Corrections already made

For transparency, here are errors we found in our own work and corrected before this review:

1. **Per-popularity-band metrics.** The first version reported R² within narrow popularity bands. Within a 20-point band the target barely varies, so R² collapsed to meaningless values (−320 for the top band). The band table now reports RMSE and MAE only.
2. **A duplicated hyperparameter setting.** The robustness check's list of alternative settings was written before tuning ran, and one "alternative" turned out to be identical to the selected configuration. It was replaced with a genuinely different setting. This **changed one conclusion**: the hyperparameter check had appeared to show a consistent direction, and with the fix it no longer does. The notebook now asserts that no alternative can equal the selected configuration.
3. **The demo model used extra trees.** The exported demo model contained the trees trained after XGBoost's best early-stopping round, so the demo app's predictions differed from the reported model by up to 2.15 popularity points (mean 0.11). The export now keeps only the trees up to the best round, and a test replays held-out tracks against notebook 7's saved predictions. This affected the demo app only; the nine reported models and their figures were not affected.

Corrections 1 and 2 did not require retraining the nine reported models, and none of the three changes the headline result.
