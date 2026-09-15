# Sentiment Analysis Pipeline — Algorithm Aversion Study

Analyzes interview transcripts to compare how participants talk about
algorithm-recommended vs. peer-recommended music, in two contexts: reacting
to a specific recommended song ("Study") and describing their general music
habits ("Life").

## Study design

The four transcript topics form a 2×2 factorial design:

|            | **Algorithm**    | **Peer**       |
|------------|------------------|----------------|
| **Study**  | `Computer Music` | `Peer Music`   |
| **Life**   | `Algorithms`     | `Music Sharing`|

- **Source** (Algorithm vs. Peer): who recommended the music.
- **Context** (Study vs. Life): whether the participant is reacting to one
  specific recommendation they just heard, or speaking generally about how
  they find music day-to-day.

## Pipeline overview

Two scripts, run in order:

1. **`sent-analysis2.0.py`** — the core pipeline. Reads raw `.txt`
   transcripts, splits them into sentences, scores each sentence's
   sentiment, and aggregates by theme and by participant.
2. **`context_source_comparison.py`** — a downstream analysis. Reads that
   pipeline's `participant_profiles.csv` output and tests/visualizes
   whether the algorithm-vs-peer sentiment gap changes between the Study
   and Life contexts.

> **Note:** `sentiment-analysis1.0.py` and `ABSA_Interview_Results.csv` are
> an earlier prototype (RoBERTa + aspect extraction). They're kept for
> reference but are **not** the current pipeline — use `sent-analysis2.0.py`.

For a stage-by-stage explanation of *why* each step works the way it does,
see [`PIPELINE_EXPLAINED.md`](PIPELINE_EXPLAINED.md).

## Setup

```bash
python3 -m venv .venv-run          # or reuse an existing venv
source .venv-run/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Important:** create the venv *outside* any iCloud-Drive-synced folder
(e.g. not under `~/Documents` if you have "Desktop & Documents Folders"
sync enabled). A venv is thousands of small files, and iCloud's file
provider intercepting every file access made package installation and
imports painfully slow for us during setup. Putting it anywhere else
(`~/venvs/...`, `/opt/...`, etc.) avoids the problem entirely.

The first run of `sent-analysis2.0.py` will also prompt NLTK to download the
VADER lexicon, tokenizer, and stopword data automatically if it isn't
already cached in `~/nltk_data`.

## Running it

Both scripts resolve their input/output folders relative to their own file
location, so they run correctly regardless of your working directory.

### 1. Run the core pipeline

Open `sent-analysis2.0.py` and update the two paths at the top of
`PipelineConfig` to point at the transcript folder and a fresh output
folder for this run:

```python
data_dir = _base_dir / "All-transcriptions9.13.26"  # <- your dataset folder
output_dir = _base_dir / "2026-13-09-exports"  # <- new folder for this run's results
```

Then:

```bash
python sent-analysis2.0.py
```

This produces (inside `output_dir`):
- `sentences_scored.csv` — one row per sentence, full audit trail.
- `theme_aggregates.csv` — one row per theme, summary stats.
- `participant_profiles.csv` — one row per participant × theme.
- `sentiment_by_theme.png`, `theme_heatmap.png`, `violin_plots.png`,
  `score_distributions.png`.
- `stats_report.txt` — Kruskal-Wallis + pairwise Mann-Whitney results.

### 2. Run the Context × Source comparison

Open `context_source_comparison.py` and update the exports folder name to
match the run you just produced:

```python
profiles_csv: Path = _base_dir / "2026-13-09-exports" / "participant_profiles.csv"
output_dir: Path = _base_dir / "2026-13-09-exports" / "context_comparison"
```

Then:

```bash
python context_source_comparison.py
```

This produces (inside `output_dir`):
- `interaction_plot.png` — mean sentiment by Context × Source, the
  headline chart.
- `gap_distribution.png` — distribution of each participant's
  (Algorithm − Peer) sentiment gap, by context.
- `participant_slopegraph.png` — per-participant shift in that gap from
  Study to Life.
- `context_comparison_report.txt` — both statistical tests, effect sizes,
  and sample-size breakdown.

## Methodology summary

- **Segmentation**: spaCy's sentencizer (sentence-boundary detection only,
  no parsing/NER — fast). Sentences under 20 characters are dropped as
  filler ("Yeah.", "Mm-hmm.").
- **Sentiment scoring**: VADER (rule-based, from NLTK). Chosen over a
  transformer model for this pipeline because it's fast, deterministic,
  and its scoring logic is fully inspectable — important for a method
  reviewers will need to trust. A RoBERTa option exists in the config for
  a second opinion but isn't the default.
- **Aggregation**: mean/median/std sentiment score and % positive/neutral/
  negative, computed per theme and per participant × theme.
- **Significance testing**: non-parametric throughout (Kruskal-Wallis,
  Mann-Whitney U, Wilcoxon signed-rank), because sentiment scores are not
  normally distributed. Multiple comparisons are Bonferroni-corrected.
  All effect sizes are rank-biserial correlation.

## Current results (run: `2026-13-09-exports`, doubled dataset)

- 385 transcript files → 108 participants → 3,592 sentences (36 empty
  files were skipped — worth confirming these are genuinely blank
  recordings rather than an export issue).
- Overall sentiment skews positive: 77.7% positive, 16.1% neutral, 6.2%
  negative.
- Theme means: Music Sharing (+0.479) > Peer Music (+0.455) > Algorithms
  (+0.414) > Computer Music (+0.397). Kruskal-Wallis is significant
  overall (p < 0.0001), but **all pairwise effect sizes are negligible**
  (r ≤ 0.12) — statistically real, practically small differences.
- The Algorithm-vs-Peer sentiment gap is negative (favors the peer/human
  source) in **both** contexts — Study: −0.085, Life: −0.061 — but neither
  test found the difference between contexts to be statistically
  significant (Mann-Whitney p=0.32; paired Wilcoxon on the 71 fully-crossed
  participants, p=0.16). **Algorithm aversion shows up in both settings;
  this dataset doesn't yet show strong evidence that its size differs
  between them.**

## Known limitations

- **Raggedness**: not every participant has a transcript in all four
  conditions. Of 100 participants with at least one valid gap, only 71
  have both a Study-gap and a Life-gap. `context_source_comparison.py`
  runs two tests to account for this (see `PIPELINE_EXPLAINED.md`).
- **VADER validity**: a general-purpose lexicon, not tuned to this
  domain's vocabulary. Before treating results as final, consider manually
  labeling ~50–100 sentences and computing agreement (Cohen's Kappa) as a
  validity check — noted as a next step in `sent-analysis2.0.py`'s own
  docstring.
- **Effect sizes are small**: with several thousand sentences, even tiny
  differences reach statistical significance. Read the effect sizes
  alongside the p-values, not instead of them.
