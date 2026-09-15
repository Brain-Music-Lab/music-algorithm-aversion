#!/usr/bin/env python3
"""
Context x Source Comparison — Study vs Life, Algorithm vs Peer
=================================================================

Downstream analysis of sent-analysis2.0.py's output. Reframes the four
transcript themes as a 2x2 factorial design:

                    Algorithm          Peer
    Study        computer_music     peer_music
    Life         algorithms         music_sharing

...and tests/visualizes whether the algorithm-vs-peer sentiment gap
changes between the Study context (reacting to a specific recommended
song) and the Life context (talking generally about how they find music).

Does NOT rerun sentence segmentation or scoring — it reads
participant_profiles.csv, which sent-analysis2.0.py already produces.

Usage:
    python context_source_comparison.py
"""

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

import os

# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class ComparisonConfig:
    # Point this at the participant_profiles.csv from the sent-analysis2.0.py
    # run you want to analyze. Resolved relative to this script's own
    # location (like sent-analysis2.0.py's CONFIG) rather than a hardcoded
    # absolute path — portable across machines/usernames, and immune to
    # working-directory differences. Only the exports folder name needs
    # updating for a new run.
    _base_dir = "./interviews"
    profiles_csv = os.path.join(_base_dir, "2026-13-09-exports", "participant_profiles.csv")
    output_dir = os.path.join(_base_dir, "2026-13-09-exports", "context_comparison")

    # Maps each `theme` value (topic folder name) onto the 2x2 design.
    # Keys MUST exactly match the theme strings in participant_profiles.csv —
    # confirmed against the actual run: Algorithms, Computer Music,
    # Music Sharing, Peer Music.
    condition_map: dict = field(
        default_factory=lambda: {
            "Computer Music": ("Study", "Algorithm"),
            "Peer Music": ("Study", "Peer"),
            "Algorithms": ("Life", "Algorithm"),
            "Music Sharing": ("Life", "Peer"),
        }
    )

    alpha: float = 0.05
    n_bootstrap: int = 2000  # resamples for interaction-plot CIs
    fig_dpi: int = 150


CONFIG = ComparisonConfig()


# =============================================================================
# STAGE 1 — LOAD + MAP
# =============================================================================


def load_profiles(config: ComparisonConfig) -> pd.DataFrame:
    """
    Read participant_profiles.csv produced by sent-analysis2.0.py.

    Returns
    -------
    DataFrame with (at least): participant_id, theme, mean_score, n_sentences
    """
    print(f"\n{'=' * 60}")
    print("STAGE 1: Load participant profiles")
    print(f"{'=' * 60}")

    path = config.profiles_csv
    if not Path(path).exists():
        raise FileNotFoundError(
            f"'{path}' not found.\n"
            f"Set CONFIG.profiles_csv to the participant_profiles.csv from "
            f"the sent-analysis2.0.py run you want to analyze."
        )

    df = pd.read_csv(path)
    required = {"participant_id", "theme", "mean_score", "n_sentences"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"'{path}' is missing expected columns: {missing}")

    print(f"Loaded: {path.resolve()}")
    print(f"Rows  : {len(df)}")
    print(f"Themes: {sorted(df['theme'].unique())}")
    return df


def map_conditions(df: pd.DataFrame, config: ComparisonConfig) -> pd.DataFrame:
    """
    Attach `context` ("Study"/"Life") and `source` ("Algorithm"/"Peer")
    columns via config.condition_map[theme].

    Rows whose theme isn't a key in condition_map are dropped — printed
    explicitly so nothing disappears silently.
    """
    print(f"\n{'=' * 60}")
    print("STAGE 1B: Map themes onto Context x Source")
    print(f"{'=' * 60}")

    known = set(config.condition_map.keys())
    present = set(df["theme"].unique())
    unmapped = present - known

    if unmapped:
        dropped_rows = df[df["theme"].isin(unmapped)]
        print(f"  [WARN] Dropping {len(dropped_rows)} rows from unmapped theme(s): {sorted(unmapped)}")
        print(
            "  [WARN] If any of these SHOULD be one of the four conditions, "
            "fix CONFIG.condition_map — theme strings must match exactly."
        )

    missing_map_keys = known - present
    if missing_map_keys:
        print(f"  [WARN] condition_map expects theme(s) not found in the data: {sorted(missing_map_keys)}")

    mapped = df[df["theme"].isin(known)].copy()
    mapped["context"] = mapped["theme"].map(lambda t: config.condition_map[t][0])
    mapped["source"] = mapped["theme"].map(lambda t: config.condition_map[t][1])

    print(f"\n  Rows kept: {len(mapped)} / {len(df)}")
    print(mapped.groupby(["context", "source"])["participant_id"].nunique().rename("n_participants").to_string())

    return mapped


# =============================================================================
# STAGE 2 — PER-PARTICIPANT GAPS
# =============================================================================


def compute_gaps(mapped: pd.DataFrame) -> pd.DataFrame:
    """
    Per participant x context: gap = Algorithm.mean_score - Peer.mean_score.

    Only computed where BOTH sources exist for that participant in that
    context. Ragged participants simply get no row for a context they're
    missing a side of.

    Sign convention: gap > 0 -> sentiment skews toward the algorithm;
                      gap < 0 -> skews toward the peer (the "aversion" direction).

    Returns
    -------
    DataFrame: participant_id, context, gap, n_sentences_algorithm, n_sentences_peer
    """
    print(f"\n{'=' * 60}")
    print("STAGE 2: Compute per-participant Algorithm-minus-Peer gaps")
    print(f"{'=' * 60}")

    pivot = mapped.pivot_table(
        index=["participant_id", "context"],
        columns="source",
        values=["mean_score", "n_sentences"],
    )

    # Keep only participant x context cells where both sources are present.
    has_both = pivot["mean_score"][["Algorithm", "Peer"]].notna().all(axis=1)
    pivot = pivot[has_both]

    gaps = pd.DataFrame(
        {
            "gap": pivot["mean_score"]["Algorithm"] - pivot["mean_score"]["Peer"],
            "n_sentences_algorithm": pivot["n_sentences"]["Algorithm"],
            "n_sentences_peer": pivot["n_sentences"]["Peer"],
        }
    ).reset_index()

    print(f"\n  Gaps computed: {len(gaps)} (participant x context) pairs")
    print(gaps.groupby("context")["gap"].agg(n="count", mean="mean", median="median").to_string())

    return gaps


def build_crossed_subset(gaps: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to one row per participant with study_gap and life_gap columns,
    keeping only participants present in BOTH contexts.

    Returns
    -------
    DataFrame: participant_id, study_gap, life_gap
    """
    print(f"\n{'=' * 60}")
    print("STAGE 2B: Build fully-crossed subset (for paired test + slopegraph)")
    print(f"{'=' * 60}")

    wide = gaps.pivot(index="participant_id", columns="context", values="gap")
    crossed = wide.dropna(subset=["Study", "Life"]).reset_index()
    crossed = crossed.rename(columns={"Study": "study_gap", "Life": "life_gap"})

    n_total = wide.shape[0]
    n_crossed = len(crossed)
    print(f"  Participants with a gap in at least one context: {n_total}")
    print(f"  Participants with a gap in BOTH contexts (crossed): {n_crossed}")
    print(f"  Dropped (ragged, missing one side): {n_total - n_crossed}")

    return crossed[["participant_id", "study_gap", "life_gap"]]


# =============================================================================
# STAGE 3 — STATISTICAL TESTS
# =============================================================================


def run_tests(gaps: pd.DataFrame, crossed: pd.DataFrame, config: ComparisonConfig) -> dict:
    """
    Two complementary tests of "does the gap differ by context?":

      1. Mann-Whitney U on all Study-gaps vs all Life-gaps (full N from
         `gaps`, independent-samples — the correct call given raggedness,
         since not every participant has both sides).

      2. Wilcoxon signed-rank on crossed.study_gap vs crossed.life_gap
         (paired, participant-matched, but restricted to the fully-crossed
         subset — the robustness check).

    Both report a rank-biserial effect size.

    Returns
    -------
    dict with keys: mwu_stat, mwu_p, mwu_r, n_study, n_life,
                     wilcoxon_stat, wilcoxon_p, wilcoxon_r, n_crossed
    """
    print(f"\n{'=' * 60}")
    print("STAGE 3: Statistical tests")
    print(f"{'=' * 60}")

    study_gaps = gaps.loc[gaps["context"] == "Study", "gap"].values
    life_gaps = gaps.loc[gaps["context"] == "Life", "gap"].values

    results = {"n_study": len(study_gaps), "n_life": len(life_gaps), "n_crossed": len(crossed)}

    # ── Test 1: Mann-Whitney U (full sample, independent) ───────────────────
    if len(study_gaps) >= 3 and len(life_gaps) >= 3:
        u_stat, p_mwu = stats.mannwhitneyu(study_gaps, life_gaps, alternative="two-sided")
        r_mwu = 1 - (2 * u_stat) / (len(study_gaps) * len(life_gaps))
        results.update(mwu_stat=u_stat, mwu_p=p_mwu, mwu_r=r_mwu)
        print(f"\n  Mann-Whitney U (Study-gaps n={len(study_gaps)} vs Life-gaps n={len(life_gaps)}):")
        print(f"    U={u_stat:.2f}  p={p_mwu:.4f}  r={r_mwu:+.3f}")
    else:
        results.update(mwu_stat=None, mwu_p=None, mwu_r=None)
        print("\n  Skipped Mann-Whitney U — fewer than 3 gaps in one context.")

    # ── Test 2: Wilcoxon signed-rank (paired, crossed subset) ───────────────
    if len(crossed) >= 3:
        diffs = crossed["study_gap"].values - crossed["life_gap"].values
        nonzero = diffs[diffs != 0]

        if len(nonzero) >= 3:
            w_stat, p_wil = stats.wilcoxon(crossed["study_gap"], crossed["life_gap"])

            # Matched-pairs rank-biserial correlation: rank |diff|, split by sign.
            ranks = stats.rankdata(np.abs(nonzero))
            w_plus = ranks[nonzero > 0].sum()
            w_minus = ranks[nonzero < 0].sum()
            r_wil = (w_plus - w_minus) / (w_plus + w_minus)

            results.update(wilcoxon_stat=w_stat, wilcoxon_p=p_wil, wilcoxon_r=r_wil)
            print(f"\n  Wilcoxon signed-rank (crossed subset, n={len(crossed)}):")
            print(f"    W={w_stat:.2f}  p={p_wil:.4f}  r={r_wil:+.3f}")
        else:
            results.update(wilcoxon_stat=None, wilcoxon_p=None, wilcoxon_r=None)
            print("\n  Skipped Wilcoxon — fewer than 3 non-tied pairs in the crossed subset.")
    else:
        results.update(wilcoxon_stat=None, wilcoxon_p=None, wilcoxon_r=None)
        print("\n  Skipped Wilcoxon — fewer than 3 participants in the crossed subset.")

    return results


# =============================================================================
# STAGE 4 — FIGURES
# =============================================================================


def _bootstrap_ci(values: np.ndarray, config: ComparisonConfig):
    """Percentile bootstrap 95% CI for the mean. Returns (low, high) or (nan, nan)."""
    if len(values) < 2:
        return np.nan, np.nan
    res = stats.bootstrap(
        (values,),
        np.mean,
        confidence_level=1 - config.alpha,
        n_resamples=config.n_bootstrap,
        method="percentile",
    )
    return res.confidence_interval.low, res.confidence_interval.high


def plot_interaction(mapped: pd.DataFrame, config: ComparisonConfig) -> None:
    """
    Figure A — 2x2 interaction plot. x = Context (Study, Life), one line per
    Source (Algorithm, Peer), y = mean of participant-level mean_score per
    cell, with a bootstrap 95% CI band.
    Saves: interaction_plot.png
    """
    print(f"\n{'=' * 60}")
    print("STAGE 4A: Interaction plot")
    print(f"{'=' * 60}")

    contexts = ["Study", "Life"]
    sources = ["Algorithm", "Peer"]
    colors = {"Algorithm": "#e74c3c", "Peer": "#3498db"}

    fig, ax = plt.subplots(figsize=(6, 5))

    for source in sources:
        means, los, his = [], [], []
        for context in contexts:
            vals = mapped.loc[
                (mapped["context"] == context) & (mapped["source"] == source),
                "mean_score",
            ].values
            m = vals.mean() if len(vals) else np.nan
            lo, hi = _bootstrap_ci(vals, config)
            means.append(m)
            los.append(m - lo if not np.isnan(lo) else 0)
            his.append(hi - m if not np.isnan(hi) else 0)

        ax.errorbar(
            contexts,
            means,
            yerr=[los, his],
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=5,
            label=source,
            color=colors[source],
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_title("Sentiment by Context x Source (95% bootstrap CI)", fontsize=13, pad=12)
    ax.set_ylabel("Mean sentiment score")
    ax.set_xlabel("")
    ax.legend(title="Source", loc="best", framealpha=0.9)
    fig.tight_layout()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_dir / "interaction_plot.png", dpi=config.fig_dpi)
    plt.close(fig)
    print("  Saved: interaction_plot.png")


def plot_gap_distribution(gaps: pd.DataFrame, test_results: dict, config: ComparisonConfig) -> None:
    """
    Figure B — Study-gap vs Life-gap distributions (violin + strip), dashed
    line at gap=0, Mann-Whitney result annotated.
    Saves: gap_distribution.png
    """
    print(f"\n{'=' * 60}")
    print("STAGE 4B: Gap distribution plot")
    print(f"{'=' * 60}")

    order = ["Study", "Life"]
    fig, ax = plt.subplots(figsize=(6, 5))

    sns.violinplot(
        data=gaps,
        x="context",
        y="gap",
        order=order,
        ax=ax,
        palette=["#f39c12", "#8e44ad"],
        inner="quartile",
        cut=0,
    )
    sns.stripplot(
        data=gaps,
        x="context",
        y="gap",
        order=order,
        ax=ax,
        color="black",
        alpha=0.3,
        size=3,
        jitter=True,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Algorithm-minus-Peer sentiment gap, by context", fontsize=13, pad=12)
    ax.set_ylabel("Gap (Algorithm − Peer sentiment)")
    ax.set_xlabel("")

    if test_results.get("mwu_p") is not None:
        p, r = test_results["mwu_p"], test_results["mwu_r"]
        sig = "significant" if p < config.alpha else "not significant"
        ax.text(
            0.5,
            0.02,
            f"Mann-Whitney U: p={p:.4f} ({sig}), r={r:+.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    fig.tight_layout()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_dir / "gap_distribution.png", dpi=config.fig_dpi)
    plt.close(fig)
    print("  Saved: gap_distribution.png")


def plot_slopegraph(crossed: pd.DataFrame, test_results: dict, config: ComparisonConfig) -> None:
    """
    Figure C — one line per participant, study_gap -> life_gap, colored by
    direction of change. Title states the crossed-subset N explicitly.
    Saves: participant_slopegraph.png
    """
    print(f"\n{'=' * 60}")
    print("STAGE 4C: Participant slopegraph")
    print(f"{'=' * 60}")

    if len(crossed) == 0:
        print("  Skipped — no participants have data in all four conditions.")
        return

    fig, ax = plt.subplots(figsize=(5.5, 6))
    x = [0, 1]

    for _, row in crossed.iterrows():
        # "Aversion deepens" = gap moves more negative (toward peer) in Life vs Study.
        deepens = row["life_gap"] < row["study_gap"]
        color = "#c0392b" if deepens else "#2980b9"
        ax.plot(x, [row["study_gap"], row["life_gap"]], color=color, alpha=0.4, linewidth=1.2)

    mean_study, mean_life = crossed["study_gap"].mean(), crossed["life_gap"].mean()
    ax.plot(x, [mean_study, mean_life], color="black", linewidth=3, marker="o", markersize=8, label="Mean")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Study gap", "Life gap"])
    ax.set_ylabel("Gap (Algorithm − Peer sentiment)")

    n_crossed = len(crossed)
    max(test_results.get("n_study", 0), test_results.get("n_life", 0))
    ax.set_title(
        f"Per-participant shift, Study -> Life\n(n={n_crossed} of participants with data in all four conditions)",
        fontsize=12,
        pad=12,
    )

    if test_results.get("wilcoxon_p") is not None:
        p, r = test_results["wilcoxon_p"], test_results["wilcoxon_r"]
        sig = "significant" if p < config.alpha else "not significant"
        ax.text(
            0.5,
            -0.14,
            f"Wilcoxon signed-rank: p={p:.4f} ({sig}), r={r:+.3f}",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
        )

    ax.legend(loc="best")
    fig.tight_layout()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.output_dir / "participant_slopegraph.png", dpi=config.fig_dpi)
    plt.close(fig)
    print("  Saved: participant_slopegraph.png")


# =============================================================================
# STAGE 5 — REPORT
# =============================================================================


def write_report(test_results: dict, config: ComparisonConfig) -> None:
    """
    Plain-text summary: both test results, effect sizes, N breakdown.
    Saves: context_comparison_report.txt
    """
    print(f"\n{'=' * 60}")
    print("STAGE 5: Report")
    print(f"{'=' * 60}")

    n_study, n_life, n_crossed = (test_results["n_study"], test_results["n_life"], test_results["n_crossed"])
    n_dropped = (n_study + n_life) - 2 * n_crossed  # each crossed participant counted once per context

    lines = [
        "=" * 60,
        "CONTEXT (Study/Life) x SOURCE (Algorithm/Peer) COMPARISON",
        "=" * 60,
        "",
        f"Study-gaps available : {n_study}",
        f"Life-gaps available  : {n_life}",
        f"Fully-crossed subset : {n_crossed}",
        f"Ragged (one side only): ~{n_dropped}",
        "",
        "-" * 60,
        "TEST 1 — Mann-Whitney U (independent samples, full N)",
        "  'Are Study-gaps and Life-gaps drawn from different distributions?'",
        "-" * 60,
    ]

    if test_results.get("mwu_p") is not None:
        p, r = test_results["mwu_p"], test_results["mwu_r"]
        lines += [
            f"  U = {test_results['mwu_stat']:.2f}",
            f"  p = {p:.4f}",
            f"  r = {r:+.3f}  (rank-biserial)",
            f"  Significant: {'YES' if p < config.alpha else 'NO'} (alpha={config.alpha})",
        ]
    else:
        lines.append("  Skipped — insufficient N.")

    lines += [
        "",
        "-" * 60,
        "TEST 2 — Wilcoxon signed-rank (paired, fully-crossed subset)",
        "  'Within the same participants, does their gap shift between contexts?'",
        "-" * 60,
    ]

    if test_results.get("wilcoxon_p") is not None:
        p, r = test_results["wilcoxon_p"], test_results["wilcoxon_r"]
        lines += [
            f"  W = {test_results['wilcoxon_stat']:.2f}",
            f"  p = {p:.4f}",
            f"  r = {r:+.3f}  (matched-pairs rank-biserial)",
            f"  Significant: {'YES' if p < config.alpha else 'NO'} (alpha={config.alpha})",
        ]
    else:
        lines.append("  Skipped — insufficient N.")

    lines += [
        "",
        "Effect size |r|: <0.1 negligible | 0.1-0.3 small | 0.3-0.5 medium | >0.5 large",
        "",
        "Interpretation notes:",
        "  gap = Algorithm mean sentiment - Peer mean sentiment.",
        "  gap < 0 means sentiment skews toward the peer (algorithm aversion).",
        "  If Test 1 and Test 2 agree, that's a robust context effect.",
        "  If they disagree, the ragged (non-crossed) participants likely differ",
        "  systematically from the fully-crossed ones — worth a closer look.",
    ]

    report = "\n".join(lines)
    print("\n" + report)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = config.output_dir / "context_comparison_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n  Saved: {report_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("\n" + "=" * 60)
    print("  CONTEXT x SOURCE COMPARISON")
    print(f"  Profiles  : {CONFIG.profiles_csv}")
    print(f"  Output    : {CONFIG.output_dir}")
    print("=" * 60)

    df = load_profiles(CONFIG)
    mapped = map_conditions(df, CONFIG)
    gaps = compute_gaps(mapped)
    crossed = build_crossed_subset(gaps)
    results = run_tests(gaps, crossed, CONFIG)

    plot_interaction(mapped, CONFIG)
    plot_gap_distribution(gaps, results, CONFIG)
    plot_slopegraph(crossed, results, CONFIG)
    write_report(results, CONFIG)

    print(f"\n{'=' * 60}")
    print(f"  Done. All outputs -> {CONFIG.output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
