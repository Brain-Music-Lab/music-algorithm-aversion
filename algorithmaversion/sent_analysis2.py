#!/usr/bin/env python3
"""
Sentiment Analysis Pipeline — Qualitative Interview Research
=============================================================

Processes .txt transcript files organized in topic folders, runs sentence-level
sentiment analysis, aggregates by theme, and produces publication-ready outputs.

Expected directory structure:
    data/
    ├── topic_A/
    │   ├── participant_01.txt
    │   └── participant_02.txt
    └── topic_B/
        ├── participant_01.txt
        └── participant_03.txt

Setup (run once in your terminal):
    pip install pandas spacy nltk vaderSentiment matplotlib seaborn scipy openpyxl
    python -m spacy download en_core_web_sm

Optional — RoBERTa scorer (higher accuracy, more compute):
    pip install transformers torch

Usage:
    python sentiment_pipeline.py
"""

import re
import warnings
from itertools import combinations
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)


# =============================================================================
# CONFIGURATION
# =============================================================================
# All settings live here. Edit this block before running — no need to touch
# the functions below. Each option is explained inline.


class PipelineConfig:
    # ── Paths ─────────────────────────────────────────────────────────────────
    # Resolved relative to this script's own location, not the working
    # directory — so it runs the same whether launched from a terminal, the
    # VS Code Run button, or any other cwd. Only the two folder names below
    # need updating for a new run; leave _base_dir alone.
    _base_dir = Path(__file__).resolve().parent
    data_dir = _base_dir / "All-transcriptions9.13.26"  # Root folder containing your topic subfolders
    output_dir = _base_dir / "2026-13-09-exports"  # Where all CSV, image, and report files go

    # ── Stage 2: Sentence segmentation ───────────────────────────────────────
    # "spacy"  → recommended. Handles informal speech, ellipses, and missing
    #            punctuation better than NLTK. Uses only the sentencizer
    #            component (fast — no POS tagging or NER loaded).
    # "nltk"   → simpler Punkt tokenizer. Faster but more brittle on messy text.
    segmenter: Literal["spacy", "nltk"] = "spacy"

    # spaCy model. "en_core_web_sm" is fast and sufficient for segmentation.
    # Upgrade to "en_core_web_md" or "en_core_web_lg" for marginally better
    # sentence boundary detection on very complex sentences.
    spacy_model: str = "en_core_web_sm"

    # Minimum sentence length (characters) to keep. Filters filler like
    # "Yeah.", "Mm-hmm.", "Right." that add noise without sentiment signal.
    # ➜ OPTIMIZATION: raise to 30–50 if your transcripts have a lot of filler.
    min_sentence_chars: int = 20

    # ── Stage 3: Sentiment scoring ────────────────────────────────────────────
    # "vader"   → rule-based, instant, interpretable. Handles capitalization,
    #             punctuation emphasis, negations, and booster words. Designed
    #             for informal/spoken text — excellent fit for interviews.
    #             Best choice for transparency with research reviewers.
    #
    # "roberta" → transformer model (Cardiff NLP, ~500 MB download on first run).
    #             Higher accuracy on ambiguous or domain-specific sentences.
    #             Requires 'transformers' and 'torch'. Slow on CPU (~1–3 s/sentence).
    #             Best choice when accuracy outweighs interpretability concerns.
    scorer: Literal["vader", "roberta"] = "vader"

    # RoBERTa model name (only used if scorer = "roberta").
    # "cardiffnlp/twitter-roberta-base-sentiment-latest" is trained on ~58M
    # tweets — its informal register matches interview speech well.
    # ➜ OPTIMIZATION: fine-tune on ~200 manually labelled sentences from your
    #   own corpus for domain-specific accuracy gains.
    roberta_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Reduce to 8 if running out of memory on CPU.
    roberta_batch_size: int = 16

    # VADER label thresholds (applied to the compound score).
    # Default VADER convention: ≥ +0.05 = positive, ≤ −0.05 = negative.
    # ➜ OPTIMIZATION: tighten to ±0.2 if you want to reduce the neutral bucket
    #   and focus analysis on more strongly valenced sentences.
    vader_pos_threshold: float = 0.05
    vader_neg_threshold: float = -0.05

    # ── Stage 5: Outputs ─────────────────────────────────────────────────────
    fig_dpi: int = 150  # Use 300 for publication-quality figures
    run_stats: bool = True  # Set False to skip statistical tests
    alpha: float = 0.05  # Significance level for all tests


CONFIG = PipelineConfig()


# =============================================================================
# STAGE 1 — DATA INGESTION
# =============================================================================


def ingest_data(config: PipelineConfig) -> pd.DataFrame:
    """
    Walk the topic folder tree and return a flat DataFrame of raw texts.

    Theme  = folder name   (e.g. "healthcare_access")
    Participant ID = filename without extension (e.g. "participant_01")

    Returns
    -------
    pd.DataFrame with columns:
        participant_id, theme, raw_text, file_path

    ➜ OPTIMIZATION — filename metadata:
        If your filenames encode extra info (e.g. "P01_F_35.txt" for gender/age),
        parse those fields here and add them as extra columns. They can then be
        used as grouping variables in Stage 4 aggregation and Stage 5 plots.

    ➜ OPTIMIZATION — theme filtering:
        Add a TOPICS_TO_INCLUDE = ["topic_A", "topic_B"] list and filter
        topic_dirs to only those names if you want to run the pipeline on a
        subset of your folders.
    """
    print(f"\n{'=' * 60}")
    print("STAGE 1: Data ingestion")
    print(f"{'=' * 60}")

    data_path = config.data_dir
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data directory '{data_path}' not found.\n"
            f"Set CONFIG.data_dir to the folder containing your topic subfolders."
        )

    topic_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    if not topic_dirs:
        raise ValueError(f"No subdirectories found in '{data_path}'.\nEach topic must have its own subfolder.")

    print(f"Root: {data_path.resolve()}")
    print(f"Topic folders found: {len(topic_dirs)}")

    records = []
    for topic_dir in topic_dirs:
        theme = topic_dir.name
        txt_files = sorted(topic_dir.glob("*.txt"))

        if not txt_files:
            print(f"  [WARN] No .txt files in '{theme}', skipping.")
            continue

        for txt_file in txt_files:
            try:
                raw = txt_file.read_text(encoding="utf-8").strip()
            except UnicodeDecodeError:
                # Fallback for files saved by transcription software with
                # non-UTF-8 encoding (latin-1 covers most Western encodings).
                raw = txt_file.read_text(encoding="latin-1").strip()
                print(f"  [WARN] Non-UTF-8 in '{txt_file.name}' — used latin-1.")

            if not raw:
                print(f"  [WARN] Empty file: '{txt_file.name}', skipping.")
                continue

            records.append(
                {
                    "participant_id": txt_file.stem.split("_")[0],  # e.g. "0M3LDX" from "0M3LDX_algorithms.txt"
                    "theme": theme,
                    "raw_text": raw,
                    "file_path": str(txt_file),
                }
            )

    if not records:
        raise ValueError("No valid .txt files found. Check your data directory.")

    df = pd.DataFrame(records)
    print(f"\nFiles loaded   : {len(df)}")
    print(f"Themes         : {df['theme'].nunique()} → {sorted(df['theme'].unique())}")
    print(f"Participants   : {df['participant_id'].nunique()}")
    return df


# =============================================================================
# STAGE 2 — SENTENCE SEGMENTATION
# =============================================================================


def segment_sentences(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Split each participant's raw text into individual sentences.

    Every sentence inherits its parent file's participant_id and theme.
    Short fragments below config.min_sentence_chars are dropped.

    Returns
    -------
    pd.DataFrame with columns:
        participant_id, theme, sentence_idx, sentence

    ➜ OPTIMIZATION — abbreviation handling:
        spaCy/NLTK can false-split on "Dr.", "Fig.", "U.S." etc.
        Add a custom list to the sentencizer to prevent this:
            nlp.get_pipe("sentencizer").punct_chars.add("...")

    ➜ OPTIMIZATION — speaker filtering:
        If your .txt files contain both interviewer and participant speech
        (e.g. lines prefixed "INT:" or "P01:"), filter to participant lines
        only by adding a regex check inside _clean_sentence().

    ➜ OPTIMIZATION — contraction expansion:
        Run `pip install contractions` and expand contractions before scoring
        ("don't" → "do not") for slightly improved VADER accuracy.
    """
    print(f"\n{'=' * 60}")
    print(f"STAGE 2: Sentence segmentation  [{config.segmenter}]")
    print(f"{'=' * 60}")

    if config.segmenter == "spacy":
        return _segment_spacy(df, config)
    elif config.segmenter == "nltk":
        return _segment_nltk(df, config)
    else:
        raise ValueError(f"Unknown segmenter '{config.segmenter}'. Choose 'spacy' or 'nltk'.")


def _clean_sentence(text: str) -> str:
    """
    Light cleaning applied to every sentence before scoring.

    Collapses whitespace and removes common transcription artifacts.
    Punctuation is intentionally preserved — VADER uses it as a signal
    (e.g. "great!!!" scores higher than "great").

    ➜ OPTIMIZATION — domain-specific artifacts:
        Extend the `artifacts` list with patterns from your transcription
        software (e.g. speaker labels, timestamp markers, confidence flags).
    """
    text = re.sub(r"\s+", " ", text).strip()

    # Remove transcription non-verbal markers — add your own as needed
    artifacts = [
        r"\[inaudible\]",
        r"\[laughter\]",
        r"\[laughs\]",
        r"\[pause\]",
        r"\[crosstalk\]",
        r"\[unclear\]",
    ]
    for pattern in artifacts:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()


def _segment_spacy(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    spaCy-based segmentation using only the sentencizer component.

    We skip the dependency parser and NER (enabled by default) because
    we only need sentence boundaries — this makes it 5–10× faster.
    nlp.pipe() processes all texts in a single batched pass.
    """
    try:
        import spacy

        # Load model with only the sentencizer — avoids loading NER, parser, etc.
        nlp = spacy.load(config.spacy_model, enable=["sentencizer"])
    except OSError:
        raise OSError(
            f"spaCy model '{config.spacy_model}' not found.\nFix: python -m spacy download {config.spacy_model}"
        )

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    records = []
    texts = df["raw_text"].tolist()

    # nlp.pipe() is batched and much faster than calling nlp() in a loop
    docs = list(nlp.pipe(texts, batch_size=50))

    for doc, (_, row) in zip(docs, df.iterrows(), strict=False):
        for idx, sent in enumerate(doc.sents):
            cleaned = _clean_sentence(sent.text)
            if len(cleaned) >= config.min_sentence_chars:
                records.append(
                    {
                        "participant_id": row["participant_id"],
                        "theme": row["theme"],
                        "sentence_idx": idx,
                        "sentence": cleaned,
                    }
                )

    return _finalize_sentences(records, config)


def _segment_nltk(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    NLTK Punkt tokenizer fallback.

    Simpler and faster than spaCy, but less robust on transcripts with
    missing sentence-ending punctuation or non-standard abbreviations.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("  Downloading NLTK punkt tokenizer...")
        nltk.download("punkt", quiet=True)

    from nltk.tokenize import sent_tokenize

    records = []
    for _, row in df.iterrows():
        for idx, sent in enumerate(sent_tokenize(row["raw_text"])):
            cleaned = _clean_sentence(sent)
            if len(cleaned) >= config.min_sentence_chars:
                records.append(
                    {
                        "participant_id": row["participant_id"],
                        "theme": row["theme"],
                        "sentence_idx": idx,
                        "sentence": cleaned,
                    }
                )

    return _finalize_sentences(records, config)


def _finalize_sentences(records: list, config: PipelineConfig) -> pd.DataFrame:
    df = pd.DataFrame(records)
    print(f"\nTotal sentences kept : {len(df)}  (min length: {config.min_sentence_chars} chars)")
    print("\nSentences per theme:")
    print(df.groupby("theme")["sentence"].count().to_string())
    return df


# =============================================================================
# STAGE 3 — SENTIMENT SCORING
# =============================================================================


def score_sentiment(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    Assign sentiment labels and scores to every sentence.

    Adds columns:
        sentiment_label : "positive" | "neutral" | "negative"
        sentiment_score : float in [−1, +1]
                          VADER   → compound score
                          RoBERTa → P(positive) − P(negative)

    Both scorers produce comparable signed scores, so you can switch between
    them without changing any downstream aggregation or visualization code.

    ➜ OPTIMIZATION — dual scoring for validation:
        Run both scorers, store both score columns, then compute agreement rate:
            agreement = (df['vader_label'] == df['roberta_label']).mean()
        Flag sentences where they disagree for manual review. High disagreement
        on a specific theme may indicate domain vocabulary that VADER misses.

    ➜ OPTIMIZATION — inter-rater reliability:
        Manually label ~50–100 sentences (stratified across themes), then
        compute Cohen's Kappa against the model labels. Report this as your
        validation metric in the paper.
    """
    print(f"\n{'=' * 60}")
    print(f"STAGE 3: Sentiment scoring  [{config.scorer}]")
    print(f"{'=' * 60}")

    if config.scorer == "vader":
        return _score_vader(df, config)
    elif config.scorer == "roberta":
        return _score_roberta(df, config)
    else:
        raise ValueError(f"Unknown scorer '{config.scorer}'. Choose 'vader' or 'roberta'.")


def _score_vader(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    VADER (Valence Aware Dictionary and sEntiment Reasoner) scoring.

    VADER returns four values per sentence:
        neg, neu, pos  → proportion of text with each valence (sum to 1.0)
        compound       → weighted composite score, normalized to [−1, +1]

    The compound score is our primary metric. It handles:
        • Negations        — "not good" scores negative
        • Capitalization   — "TERRIBLE" scores more negative than "terrible"
        • Punctuation      — "great!!!" scores higher than "great"
        • Booster words    — "very bad" scores more negative than "bad"

    We store the raw neg/neu/pos components alongside compound so you can
    inspect them in sentences_scored.csv if a score looks unexpected.

    ➜ OPTIMIZATION — custom lexicon:
        VADER allows adding or overriding word scores. If your domain has
        terms VADER misclassifies (e.g. clinical terms, domain jargon),
        add them via: sia.lexicon.update({"your_word": 2.5})
    """
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        print("  Downloading VADER lexicon...")
        nltk.download("vader_lexicon", quiet=True)

    sia = SentimentIntensityAnalyzer()

    print(f"\n  Scoring {len(df)} sentences with VADER...")

    scores, labels, raw_pos, raw_neu, raw_neg = [], [], [], [], []

    for sentence in df["sentence"]:
        result = sia.polarity_scores(sentence)
        compound = result["compound"]

        # Assign label using configurable thresholds (see PipelineConfig)
        if compound >= config.vader_pos_threshold:
            label = "positive"
        elif compound <= config.vader_neg_threshold:
            label = "negative"
        else:
            label = "neutral"

        scores.append(compound)
        labels.append(label)
        raw_pos.append(result["pos"])
        raw_neu.append(result["neu"])
        raw_neg.append(result["neg"])

    df = df.copy()
    df["sentiment_score"] = scores
    df["sentiment_label"] = labels

    # Raw components stored for inspection/debugging — not used in aggregation
    df["vader_pos"] = raw_pos
    df["vader_neu"] = raw_neu
    df["vader_neg"] = raw_neg

    return _report_scores(df)


def _score_roberta(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """
    RoBERTa-based scoring via HuggingFace Transformers.

    Model: Cardiff NLP twitter-roberta-base-sentiment-latest
    Training data: ~58M tweets — informal register, good fit for speech.
    Output: 3-class softmax [negative, neutral, positive]

    We convert to a signed score for direct comparability with VADER:
        signed_score = P(positive) − P(negative)

    Notes
    -----
        • Sentences > 512 tokens are truncated (rare in interview data).
        • First run downloads ~500 MB model to HuggingFace cache.
        • With CUDA GPU: ~10–50× faster than CPU.

    ➜ OPTIMIZATION — GPU acceleration:
        If you have an NVIDIA GPU with CUDA, install torch with CUDA support
        and the model will automatically use it (device=0 is set below).

    ➜ OPTIMIZATION — fine-tuning:
        Label ~200 sentences from your corpus manually, fine-tune the model
        for 3–5 epochs on your data. Even small fine-tuning datasets produce
        meaningful accuracy gains for domain-specific vocabulary.
    """
    try:
        import torch
        from transformers import pipeline as hf_pipeline
    except ImportError:
        raise ImportError(
            "HuggingFace Transformers not installed.\n"
            "Fix: pip install transformers torch\n"
            "Or switch CONFIG.scorer to 'vader'."
        )

    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU (slow — consider GPU)"
    print(f"\n  Device: {device_name}")
    print(f"  Loading '{config.roberta_model}' (first run downloads ~500 MB)...")

    classifier = hf_pipeline(
        "text-classification",
        model=config.roberta_model,
        top_k=None,  # Return all three class probabilities
        truncation=True,  # Silently truncate sentences > 512 tokens
        max_length=512,
        device=device,
    )

    sentences = df["sentence"].tolist()
    all_results = []

    print(f"  Scoring {len(sentences)} sentences in batches of {config.roberta_batch_size}...")
    for i in range(0, len(sentences), config.roberta_batch_size):
        batch = sentences[i : i + config.roberta_batch_size]
        all_results.extend(classifier(batch))
        pct = min(100, int((i + len(batch)) / len(sentences) * 100))
        if pct % 20 == 0:
            print(f"    {pct}% complete...")

    scores, labels = [], []

    for result in all_results:
        # Normalize label names — some model versions use LABEL_0/1/2,
        # others use negative/neutral/positive
        probs = {item["label"].lower(): item["score"] for item in result}
        neg = probs.get("negative", probs.get("label_0", 0.0))
        neu = probs.get("neutral", probs.get("label_1", 0.0))
        pos = probs.get("positive", probs.get("label_2", 0.0))

        # Signed compound-style score for consistency with VADER output
        signed = pos - neg

        # Label = argmax of the three class probabilities
        if pos > neu and pos > neg:
            label = "positive"
        elif neg > neu and neg > pos:
            label = "negative"
        else:
            label = "neutral"

        scores.append(signed)
        labels.append(label)

    df = df.copy()
    df["sentiment_score"] = scores
    df["sentiment_label"] = labels
    return _report_scores(df)


def _report_scores(df: pd.DataFrame) -> pd.DataFrame:
    label_counts = df["sentiment_label"].value_counts()
    total = len(df)
    print("\n  Scoring complete.")
    print("  Label distribution:")
    for label in ["positive", "neutral", "negative"]:
        n = label_counts.get(label, 0)
        pct = n / total * 100
        print(f"    {label:<10} {n:>5}  ({pct:.1f}%)")
    print("\n  Score statistics:")
    print(f"    Mean   {df['sentiment_score'].mean():>+.3f}")
    print(f"    Std    {df['sentiment_score'].std():>.3f}")
    print(f"    Min    {df['sentiment_score'].min():>+.3f}")
    print(f"    Max    {df['sentiment_score'].max():>+.3f}")
    return df


# =============================================================================
# STAGE 4 — AGGREGATION
# =============================================================================


def aggregate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize sentiment at two levels of analysis:

        theme_agg        → One row per theme
        participant_agg  → One row per (participant × theme) pair

    Metrics computed for each group:
        n_sentences    — number of sentences in the group
        n_participants — unique participants in the group
        mean_score     — mean compound/signed score (primary stat)
        median_score   — robust to outliers; useful for skewed distributions
        std_score      — within-group variability (high = ambivalent responses)
        pct_positive   — % of sentences labelled positive
        pct_neutral    — % neutral
        pct_negative   — % negative

    Both outputs are returned and saved as CSVs in Stage 5A.

    ➜ OPTIMIZATION — participant-level covariate joins:
        If you parsed demographic fields in Stage 1 (age, gender, etc.),
        merge them onto participant_agg here and add them as groupby keys
        for sub-group comparisons in Stage 5C.

    ➜ OPTIMIZATION — sentiment variability as a feature:
        High std_score within a theme + participant may indicate ambivalence.
        Consider flagging (participant, theme) pairs where std > 0.4 for
        follow-up qualitative review.

    ➜ OPTIMIZATION — temporal arc (if your data is ordered):
        Add a sentence_position column (0.0 to 1.0) and compute rolling
        mean sentiment to capture how feelings evolve across an interview.
    """
    print(f"\n{'=' * 60}")
    print("STAGE 4: Aggregation")
    print(f"{'=' * 60}")

    def summarize(group):
        counts = group["sentiment_label"].value_counts(normalize=True) * 100
        return pd.Series(
            {
                "n_sentences": len(group),
                "n_participants": group["participant_id"].nunique() if "participant_id" in group.columns else 1,
                "mean_score": group["sentiment_score"].mean(),
                "median_score": group["sentiment_score"].median(),
                "std_score": group["sentiment_score"].std(),
                "pct_positive": counts.get("positive", 0.0),
                "pct_neutral": counts.get("neutral", 0.0),
                "pct_negative": counts.get("negative", 0.0),
            }
        )

    theme_agg = (
        df.groupby("theme")
        .apply(summarize, include_groups=False)
        .reset_index()
        .sort_values("mean_score", ascending=False)
    )

    participant_agg = (
        df.groupby(["participant_id", "theme"])
        .apply(summarize, include_groups=False)
        .reset_index()
        .sort_values(["theme", "mean_score"], ascending=[True, False])
    )

    print("\nTheme-level summary (sorted by mean score):")
    cols = ["theme", "n_sentences", "mean_score", "pct_positive", "pct_negative"]
    print(theme_agg[cols].to_string(index=False))
    print(f"\nParticipant × theme profiles: {len(participant_agg)} rows")

    return theme_agg, participant_agg


# =============================================================================
# STAGE 5A — CSV OUTPUTS
# =============================================================================


def save_outputs(
    df_sentences: pd.DataFrame,
    theme_agg: pd.DataFrame,
    participant_agg: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """
    Write all data files to config.output_dir.

    Files produced:
        sentences_scored.csv      One row per sentence — your full audit trail.
                                  Include this as a supplementary file with the paper.
        theme_aggregates.csv      Theme-level summary stats — the main result table.
        participant_profiles.csv  Per-participant breakdown — useful for variance analysis.

    ➜ OPTIMIZATION — Excel export:
        Replace the .csv saves with an .xlsx export using openpyxl to produce
        a single, multi-sheet workbook for collaborators who prefer Excel:

            with pd.ExcelWriter(out / "results.xlsx", engine="openpyxl") as writer:
                df_sentences.to_excel(writer, sheet_name="Sentences", index=False)
                theme_agg.to_excel(writer, sheet_name="Themes", index=False)
                participant_agg.to_excel(writer, sheet_name="Participants", index=False)

    ➜ OPTIMIZATION — manual review sample:
        Export a stratified random sample for validation:
            sample = df_sentences.groupby(["theme","sentiment_label"]).sample(n=5)
            sample.to_csv(out / "review_sample.csv", index=False)
    """
    out = config.output_dir
    out.mkdir(parents=True, exist_ok=True)

    df_sentences.to_csv(out / "sentences_scored.csv", index=False)
    theme_agg.to_csv(out / "theme_aggregates.csv", index=False)
    participant_agg.to_csv(out / "participant_profiles.csv", index=False)

    print(f"\n{'=' * 60}")
    print(f"STAGE 5A: CSV outputs → {out.resolve()}")
    print(f"{'=' * 60}")
    print(f"  sentences_scored.csv      ({len(df_sentences)} rows)")
    print(f"  theme_aggregates.csv      ({len(theme_agg)} rows)")
    print(f"  participant_profiles.csv  ({len(participant_agg)} rows)")


# =============================================================================
# STAGE 5B — VISUALIZATIONS
# =============================================================================


def create_visualizations(
    df_sentences: pd.DataFrame,
    theme_agg: pd.DataFrame,
    participant_agg: pd.DataFrame,
    config: PipelineConfig,
) -> None:
    """
    Generate and save four publication-ready figures.

    Figure 1 — sentiment_by_theme.png
        Grouped bar chart: % positive / neutral / negative per theme.
        Best for: showing label distribution at a glance.

    Figure 2 — theme_heatmap.png
        Heatmap: mean score per participant × theme.
        Best for: spotting individual-level patterns and outliers.

    Figure 3 — violin_plots.png
        Violin + strip plot: full score distribution per theme.
        Best for: showing within-theme variance alongside central tendency.

    Figure 4 — score_distributions.png
        Overlapping KDE density curves per theme.
        Best for: comparing distributional shape across themes in one view.

    ➜ OPTIMIZATION — demographic hue:
        If you have a demographic covariate (e.g. group = ["A","B","A",...]),
        add it as a column in df_sentences and set hue="group" in the
        violinplot/stripplot calls below.

    ➜ OPTIMIZATION — interactive HTML:
        Replace matplotlib with Plotly for interactive figures:
            import plotly.express as px
            fig = px.violin(df_sentences, x="theme", y="sentiment_score",
                            box=True, points="all")
            fig.write_html(out / "violin_plots.html")
    """
    print(f"\n{'=' * 60}")
    print("STAGE 5B: Visualizations")
    print(f"{'=' * 60}")

    out = config.output_dir
    themes = sorted(df_sentences["theme"].unique())
    n_themes = len(themes)
    fig_w = max(8, n_themes * 1.6)

    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.4,
        }
    )

    # ── Figure 1: Grouped bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    bar_data = theme_agg.set_index("theme")[["pct_positive", "pct_neutral", "pct_negative"]]
    bar_data.plot(
        kind="bar",
        ax=ax,
        color=["#4caf50", "#9e9e9e", "#f44336"],
        width=0.7,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title("Sentiment label distribution by theme", fontsize=14, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of sentences (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(["Positive", "Neutral", "Negative"], loc="upper right", framealpha=0.9)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(out / "sentiment_by_theme.png", dpi=config.fig_dpi)
    plt.close(fig)
    print("  Saved: sentiment_by_theme.png")

    # ── Figure 2: Participant × theme heatmap ────────────────────────────────
    pivot = participant_agg.pivot_table(
        index="participant_id",
        columns="theme",
        values="mean_score",
        aggfunc="mean",
    )

    if pivot.shape[0] > 1 and pivot.shape[1] > 1:
        h = max(6, len(pivot) * 0.35)
        fig, ax = plt.subplots(figsize=(fig_w, h))
        sns.heatmap(
            pivot,
            ax=ax,
            cmap="RdYlGn",
            center=0,
            vmin=-1,
            vmax=1,
            linewidths=0.4,
            linecolor="white",
            # Show numeric annotations only when the grid isn't too crowded
            annot=(len(pivot) <= 30),
            fmt=".2f",
            cbar_kws={"label": "Mean sentiment score", "shrink": 0.8},
        )
        ax.set_title("Mean sentiment score: participant × theme", fontsize=14, pad=12)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        fig.savefig(out / "theme_heatmap.png", dpi=config.fig_dpi)
        plt.close(fig)
        print("  Saved: theme_heatmap.png")
    else:
        print("  Skipped heatmap (needs >1 theme and >1 participant)")

    # ── Figure 3: Violin + strip plot ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    sns.violinplot(
        data=df_sentences,
        x="theme",
        y="sentiment_score",
        ax=ax,
        palette="RdYlGn",
        inner="quartile",
        cut=0,
        order=themes,
    )
    # Overlay individual points — shows actual data density alongside the KDE.
    # ➜ OPTIMIZATION: set alpha lower (0.05) or remove stripplot entirely if
    #   N is large (>500 sentences per theme) to avoid overplotting.
    sns.stripplot(
        data=df_sentences,
        x="theme",
        y="sentiment_score",
        ax=ax,
        color="black",
        alpha=0.15,
        size=2.5,
        jitter=True,
        order=themes,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Sentiment score distribution by theme", fontsize=14, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Sentiment score (−1 to +1)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    fig.savefig(out / "violin_plots.png", dpi=config.fig_dpi)
    plt.close(fig)
    print("  Saved: violin_plots.png")

    # ── Figure 4: KDE density curves ─────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    palette = sns.color_palette("tab10", n_colors=n_themes)

    for i, theme in enumerate(themes):
        subset = df_sentences[df_sentences["theme"] == theme]["sentiment_score"]
        subset.plot.kde(ax=ax, label=theme, color=palette[i], linewidth=1.8)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Sentiment score density by theme", fontsize=14, pad=12)
    ax.set_xlabel("Sentiment score")
    ax.set_ylabel("Density")
    ax.set_xlim(-1.1, 1.1)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out / "score_distributions.png", dpi=config.fig_dpi)
    plt.close(fig)
    print("  Saved: score_distributions.png")


# =============================================================================
# STAGE 5C — STATISTICAL TESTS
# =============================================================================


def run_statistical_tests(df_sentences: pd.DataFrame, config: PipelineConfig) -> None:
    """
    Non-parametric tests comparing sentiment distributions across themes.

    Why non-parametric?
        Sentiment scores are rarely normally distributed — they tend to be
        bimodal (many positive + many negative, few neutral) or skewed.
        Kruskal-Wallis and Mann-Whitney make no distributional assumptions.

    Tests performed:

        1. Kruskal-Wallis H-test
           "Is there any significant difference across all themes?"
           Equivalent to a one-way ANOVA but rank-based.

        2. Pairwise Mann-Whitney U tests
           "Which specific theme pairs differ significantly?"
           With Bonferroni correction for multiple comparisons.
           Bonferroni is conservative — if you have many themes (>6),
           consider Benjamini-Hochberg FDR correction instead (see note below).

        3. Effect size: rank-biserial correlation r
           |r| < 0.1 → negligible  |  0.1–0.3 → small
           0.3–0.5  → medium       |  > 0.5   → large

    Results are written to stats_report.txt.

    ➜ OPTIMIZATION — Benjamini-Hochberg FDR (less conservative than Bonferroni):
        from statsmodels.stats.multitest import multipletests
        _, p_corrected, _, _ = multipletests(raw_p_values, method="fdr_bh")

    ➜ OPTIMIZATION — repeated measures:
        If the same participants appear across all themes (fully crossed design),
        use Friedman's test instead of Kruskal-Wallis:
            from scipy.stats import friedmanchisquare
            friedmanchisquare(*groups)

    ➜ OPTIMIZATION — demographic subgroups:
        If you have a covariate column (e.g. "gender"), run a Mann-Whitney
        between groups within each theme and report effect sizes.
    """
    print(f"\n{'=' * 60}")
    print("STAGE 5C: Statistical tests")
    print(f"{'=' * 60}")

    themes = sorted(df_sentences["theme"].unique())
    groups = {t: df_sentences[df_sentences["theme"] == t]["sentiment_score"].values for t in themes}

    # Drop themes with fewer than 3 sentences — not enough for reliable testing
    valid = {t: g for t, g in groups.items() if len(g) >= 3}
    if len(valid) < 2:
        print("  Not enough themes with ≥3 sentences. Skipping statistical tests.")
        return

    lines = []

    # ── Test 1: Kruskal-Wallis ───────────────────────────────────────────────
    h_stat, p_kw = stats.kruskal(*valid.values())
    sig_overall = p_kw < config.alpha

    lines += [
        "=" * 60,
        "KRUSKAL-WALLIS TEST  (overall — any difference across themes?)",
        "=" * 60,
        f"H statistic  : {h_stat:.4f}",
        f"p-value      : {p_kw:.4f}",
        f"Significant  : {'YES ✓' if sig_overall else 'NO'} (α = {config.alpha})",
        f"Themes tested: {list(valid.keys())}",
        "",
    ]

    # ── Test 2: Pairwise Mann-Whitney U with Bonferroni correction ───────────
    valid_themes = list(valid.keys())
    valid_groups = list(valid.values())
    pairs = list(combinations(range(len(valid_themes)), 2))
    n_comp = len(pairs)
    bonferroni_alpha = config.alpha / n_comp

    lines += [
        "=" * 60,
        "PAIRWISE MANN-WHITNEY U TESTS  (which themes differ?)",
        f"Bonferroni-corrected α = {config.alpha} / {n_comp} = {bonferroni_alpha:.4f}",
        "=" * 60,
    ]

    for i, j in pairs:
        t1, g1 = valid_themes[i], valid_groups[i]
        t2, g2 = valid_themes[j], valid_groups[j]

        u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        # Rank-biserial correlation: r = 1 − (2U / n1*n2), range [−1, +1]
        r = 1 - (2 * u_stat) / (len(g1) * len(g2))

        sig = "  *  ← significant" if p_val < bonferroni_alpha else ""
        lines.append(f"  {t1:28s} vs {t2:28s} | p={p_val:.4f}  r={r:+.3f}{sig}")

    lines += [
        "",
        "* = significant after Bonferroni correction",
        "Effect size |r|:  <0.1 negligible | 0.1–0.3 small | 0.3–0.5 medium | >0.5 large",
    ]

    report = "\n".join(lines)
    print()
    print(report)

    report_path = config.output_dir / "stats_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print("\n  Full report saved: stats_report.txt")


# =============================================================================
# MAIN
# =============================================================================


def main():
    """
    Run all pipeline stages end-to-end.

    Each stage function is independent — during development you can run them
    individually or comment out stages you want to skip:

        df_raw       = ingest_data(CONFIG)
        df_sentences = segment_sentences(df_raw, CONFIG)
        df_scored    = score_sentiment(df_sentences, CONFIG)
        theme_agg, participant_agg = aggregate(df_scored)
        save_outputs(df_scored, theme_agg, participant_agg, CONFIG)
        create_visualizations(df_scored, theme_agg, participant_agg, CONFIG)
        run_statistical_tests(df_scored, CONFIG)

    ➜ NEXT STEPS after running the pipeline:

        1. Inspect sentences_scored.csv — read a sample of each label per theme
           and verify the scores make sense for your domain.

        2. Check the heatmap for outlier participants — anyone with a very
           different profile may warrant a closer qualitative read.

        3. If VADER is producing many neutrals (>50%), try tightening
           vader_pos_threshold / vader_neg_threshold in CONFIG, or switch to
           scorer="roberta" for a second opinion.

        4. Use stats_report.txt to identify which theme pairs differ
           significantly — those are the most interesting for interpretation.

        5. For the paper: report scorer used, threshold settings, N sentences
           per theme, Kruskal-Wallis result, and inter-rater reliability
           (manually validate ~50–100 sentences and compute Cohen's Kappa).
    """
    print("\n" + "=" * 60)
    print("  SENTIMENT ANALYSIS PIPELINE")
    print(f"  Scorer    : {CONFIG.scorer.upper()}")
    print(f"  Segmenter : {CONFIG.segmenter.upper()}")
    print(f"  Data dir  : {CONFIG.data_dir.resolve()}")
    print("=" * 60)

    df_raw = ingest_data(CONFIG)
    df_sentences = segment_sentences(df_raw, CONFIG)
    df_scored = score_sentiment(df_sentences, CONFIG)
    theme_agg, p_agg = aggregate(df_scored)

    save_outputs(df_scored, theme_agg, p_agg, CONFIG)
    create_visualizations(df_scored, theme_agg, p_agg, CONFIG)

    if CONFIG.run_stats:
        run_statistical_tests(df_scored, CONFIG)

    print(f"\n{'=' * 60}")
    print("  Pipeline complete.")
    print(f"  All outputs → {CONFIG.output_dir.resolve()}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
