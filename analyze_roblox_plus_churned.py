#!/usr/bin/env python3
"""
Roblox Plus churned subscriber survey analysis.

This script loads an SPSS .sav file with pyreadstat, preserves embedded value
labels for human-readable outputs, cleans the target analysis fields, and writes
CSV tables plus seaborn charts for product and retention analysis.

Example:
    python analyze_roblox_plus_churned.py \
        --sav-path "Roblox Plus Churned_July 8, 2026_15.57.sav" \
        --output-dir outputs
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable


DEFAULT_SAV_PATH = "Roblox Plus Churned_July 8, 2026_15.57.sav"

Q2_ORDER = ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]
Q10_ORDER = ["Very Unlikely", "Unlikely", "Maybe", "Likely", "Very Likely"]
Q12_ORDER = [
    "AI backdrops",
    "App themes",
    "Frames",
    "Rich text",
    "Particles",
    "AI items",
    "Spawn effects",
]

EXPECTED_COLUMNS = [
    "USERID",
    "TENURE",
    "GENDER",
    "Q1",
    "Q2",
    "Q3",
    "Q3_9_TEXT",
    "Q4",
    "Q5",
    "Q7",
    "Q9",
    "Q9_5_TEXT",
    "Q10",
    "Q12",
]

KNOWN_VALUE_LABELS = {
    "GENDER": {1: "Female", 2: "Male", 3: "Unknown"},
    "Gender": {1: "Female", 2: "Male", 3: "Unknown"},
    "Q2": {1: "Love it", 2: "Like it", 3: "Neutral", 4: "Dislike it", 5: "Hate it"},
    "Q10": {1: "Very Unlikely", 2: "Unlikely", 3: "Maybe", 4: "Likely", 5: "Very Likely"},
    "Q12": dict(enumerate(Q12_ORDER, start=1)),
}

COLUMN_ALIASES = {
    "USERID": ["USERID", "userId", "UserID", "userid", "user_id"],
    "TENURE": ["TENURE", "Tenure", "tenure"],
    "GENDER": ["GENDER", "Gender", "gender"],
}

CHURN_KEYWORDS = {
    "Robux Absence / Premium Rebrand": [
        "robux",
        "premium",
        "stipend",
        "builders club",
        "builder's club",
        "bc",
    ],
    "Financial Strain": ["broke", "poor", "money", "afford", "college", "funds"],
    "Verification / Age Restrictions": ["parent", "permission", "verify", "verified", "verification", "id", "18", "16"],
}

BACKLASH_KEYWORDS = {
    "slop": ["slop"],
    "AI garbage": ["ai garbage", "garbage ai"],
    "investor": ["investor", "investors"],
    "greed": ["greed", "greedy"],
}


def require_dependencies() -> None:
    """Import runtime dependencies after argparse so --help works in a clean env."""
    global chi2_contingency, np, pd, plt, pyreadstat, sns, spearmanr

    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import pyreadstat
        import seaborn as sns
        from scipy.stats import chi2_contingency, spearmanr
    except ImportError as exc:
        missing = exc.name or "a required package"
        raise SystemExit(
            f"Missing Python dependency: {missing}\n"
            "Install the analysis dependencies with:\n"
            "  python3 -m pip install pandas pyreadstat seaborn matplotlib scipy"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Roblox Plus churned subscriber SPSS survey data."
    )
    parser.add_argument(
        "--sav-path",
        default=DEFAULT_SAV_PATH,
        help="Path to the SPSS .sav file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where CSV tables and charts will be written.",
    )
    parser.add_argument(
        "--min-age",
        default=13,
        type=int,
        help="Minimum self-reported age to retain after survey screen-out validation.",
    )
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, canonical_column: str) -> str | None:
    """Resolve survey schema aliases such as USERID/userId and TENURE/Tenure."""
    candidates = COLUMN_ALIASES.get(canonical_column, [canonical_column])
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    lowered = {column.lower(): column for column in df.columns}
    return lowered.get(canonical_column.lower())


def configure_runtime(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "analysis.log", mode="w"),
        ],
    )
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.bbox"] = "tight"


def read_sav_with_metadata(sav_path: Path) -> tuple[pd.DataFrame, pyreadstat.metadata_container]:
    """Load raw SPSS codes plus metadata so labels remain available for analysis."""
    if not sav_path.exists():
        raise FileNotFoundError(f"SPSS file not found: {sav_path}")

    df, meta = pyreadstat.read_sav(
        str(sav_path),
        apply_value_formats=False,
        user_missing=True,
    )
    logging.info("Loaded %s rows and %s columns from %s", len(df), len(df.columns), sav_path)
    return df, meta


def metadata_value_map(meta: pyreadstat.metadata_container, column: str) -> dict:
    """Return code-to-label mapping for a column, if pyreadstat metadata provides one."""
    if column in getattr(meta, "variable_value_labels", {}):
        return meta.variable_value_labels[column]

    variable_to_label = getattr(meta, "variable_to_label", {})
    value_labels = getattr(meta, "value_labels", {})
    label_set_name = variable_to_label.get(column)
    if label_set_name and label_set_name in value_labels:
        return value_labels[label_set_name]

    return KNOWN_VALUE_LABELS.get(column, {})


def labeled_series(df: pd.DataFrame, meta: pyreadstat.metadata_container, column: str) -> pd.Series:
    """Map SPSS numeric/string codes to human-readable labels without mutating raw data."""
    resolved_column = resolve_column(df, column) or column
    if resolved_column not in df.columns:
        return pd.Series(index=df.index, dtype="object")

    value_map = metadata_value_map(meta, resolved_column) or metadata_value_map(meta, column)
    raw = df[resolved_column]
    if not value_map:
        return raw.astype("string")

    # SPSS codes can arrive as floats while metadata keys are ints or strings.
    normalized_map = {}
    for key, value in value_map.items():
        normalized_map[key] = value
        if isinstance(key, (int, float)) and not pd.isna(key):
            normalized_map[float(key)] = value
            normalized_map[int(key)] = value
            normalized_map[str(int(key)) if float(key).is_integer() else str(key)] = value

    return raw.map(normalized_map).fillna(raw).astype("string")


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    resolved_column = resolve_column(df, column) or column
    if resolved_column not in df.columns:
        return pd.Series(index=df.index, dtype="float64")
    return pd.to_numeric(df[resolved_column], errors="coerce")


def find_columns(df: pd.DataFrame, prefix: str, suffix_range: Iterable[int] | None = None) -> list[str]:
    """Find wide matrix columns such as Q11_0 through Q11_6 while preserving numeric order."""
    if suffix_range is not None:
        candidates = [f"{prefix}_{suffix}" for suffix in suffix_range]
        return [column for column in candidates if column in df.columns]

    pattern = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?$")

    def sort_key(column: str) -> tuple[int, str]:
        match = pattern.match(column)
        if match and match.group(1) is not None:
            return int(match.group(1)), column
        return -1, column

    return sorted([column for column in df.columns if pattern.match(column)], key=sort_key)


def combine_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Combine sparse wide text matrix columns into one clean text field per respondent."""
    if not columns:
        return pd.Series("", index=df.index, dtype="string")

    text_df = df[columns].copy()
    for column in columns:
        text_df[column] = (
            text_df[column]
            .astype("string")
            .replace({"<NA>": pd.NA, "nan": pd.NA, "None": pd.NA})
            .str.strip()
        )

    return text_df.apply(
        lambda row: " | ".join(value for value in row.dropna().astype(str) if value.strip()),
        axis=1,
    ).astype("string")


def clean_analysis_frame(
    df: pd.DataFrame,
    meta: pyreadstat.metadata_container,
    min_age: int,
) -> pd.DataFrame:
    """Create a labeled and analysis-ready dataframe while retaining raw columns."""
    analysis = df.copy()

    user_id_column = resolve_column(analysis, "USERID")
    if user_id_column and "USERID" not in analysis.columns:
        analysis["USERID"] = analysis[user_id_column]

    analysis["AGE"] = numeric_series(analysis, "Q1")
    before = len(analysis)
    analysis = analysis[analysis["AGE"].isna() | (analysis["AGE"] >= min_age)].copy()
    logging.info("Dropped %s rows where Q1 < %s", before - len(analysis), min_age)

    analysis["TENURE_NUMERIC"] = numeric_series(analysis, "TENURE")
    analysis["GENDER_LABEL"] = labeled_series(analysis, meta, "GENDER")
    analysis["Q2_LABEL"] = labeled_series(analysis, meta, "Q2")
    analysis["Q9_LABEL"] = labeled_series(analysis, meta, "Q9")
    analysis["Q10_SCORE"] = numeric_series(analysis, "Q10")
    analysis["Q10_LABEL"] = labeled_series(analysis, meta, "Q10")
    analysis["Q12_LABEL"] = labeled_series(analysis, meta, "Q12")

    # If labels were not present, keep Q10 human-readable from the known scale.
    q10_score_map = {
        1: "Very Unlikely",
        2: "Unlikely",
        3: "Maybe",
        4: "Likely",
        5: "Very Likely",
    }
    analysis["Q10_LABEL"] = analysis["Q10_SCORE"].map(q10_score_map).fillna(analysis["Q10_LABEL"])
    analysis["RETENTION_GROUP"] = pd.cut(
        analysis["Q10_SCORE"],
        bins=[0, 2, 3, 5],
        labels=["Low Intent", "Conditional Intent", "High Intent"],
        include_lowest=True,
    )

    q11_columns = find_columns(analysis, "Q11")
    q13_columns = find_columns(analysis, "Q13")
    analysis["Q11_TEXT_COMBINED"] = combine_text_columns(analysis, q11_columns)
    analysis["Q13_TEXT_COMBINED"] = combine_text_columns(analysis, q13_columns)

    if "Q9_5_TEXT" in analysis.columns:
        q9_other = analysis["Q9_5_TEXT"].astype("string").fillna("")
    else:
        q9_other = pd.Series("", index=analysis.index, dtype="string")
    analysis["CHURN_TEXT_COMBINED"] = (
        q9_other.astype("string").fillna("").str.strip()
        + " | "
        + analysis["Q11_TEXT_COMBINED"].fillna("").str.strip()
    ).str.strip(" |")

    return analysis


def validate_schema_and_quality(
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    output_dir: Path,
    min_age: int,
) -> None:
    """Export a lightweight audit trail for expected columns, screen-outs, and IDs."""
    q6_expected = [f"Q6_{idx}" for idx in range(1, 10)]
    q8_expected = [f"Q8_{idx}" for idx in range(1, 6)]
    q11_expected = ["Q11"] + [f"Q11_{idx}" for idx in range(0, 7)]
    q13_expected = ["Q13"] + [f"Q13_{idx}" for idx in range(0, 7)]
    expected = EXPECTED_COLUMNS + q6_expected + q8_expected + q11_expected + q13_expected

    rows = []
    for column in expected:
        resolved_column = resolve_column(df, column) or column
        rows.append(
            {
                "check": "expected_column_present",
                "field": column,
                "value": resolved_column in df.columns,
            }
        )

    age = numeric_series(df, "Q1")
    rows.extend(
        [
            {"check": "raw_rows", "field": "dataset", "value": len(df)},
            {"check": "analysis_rows_after_age_filter", "field": "dataset", "value": len(analysis)},
            {"check": "rows_dropped_q1_under_min_age", "field": f"Q1 < {min_age}", "value": int(age.lt(min_age).sum())},
            {"check": "rows_missing_age_retained", "field": "Q1", "value": int(age.isna().sum())},
        ]
    )

    user_id_column = resolve_column(df, "USERID")
    if user_id_column:
        rows.append(
            {
                "check": "duplicate_userid_count",
                "field": user_id_column,
                "value": int(df[user_id_column].duplicated().sum()),
            }
        )

    save_table(pd.DataFrame(rows).set_index(["check", "field"]), output_dir, "dataset_quality_summary.csv")


def save_table(df: pd.DataFrame, output_dir: Path, filename: str) -> None:
    path = output_dir / filename
    df.to_csv(path, index=True)
    logging.info("Wrote %s", path)


def plot_stacked_bar(table: pd.DataFrame, output_path: Path, title: str, ylabel: str) -> None:
    ax = table.plot(kind="bar", stacked=True, figsize=(12, 7), colormap="viridis")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.legend(title="Likelihood to Re-subscribe", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logging.info("Wrote %s", output_path)


def annotate_horizontal_bars(ax, values: Iterable[float], formatter) -> None:
    """Add labels to seaborn horizontal bars without depending on container internals."""
    for patch, value in zip(ax.patches, values):
        width = patch.get_width()
        if pd.isna(value) or pd.isna(width):
            continue
        ax.annotate(
            formatter(value),
            xy=(width, patch.get_y() + patch.get_height() / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
        )


def demographics_and_brand_health(analysis: pd.DataFrame, output_dir: Path) -> None:
    age_tenure = analysis[["AGE", "TENURE_NUMERIC"]].describe(percentiles=[0.25, 0.5, 0.75]).T
    gender_distribution = (
        analysis["GENDER_LABEL"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("gender")
        .to_frame("n")
    )
    gender_distribution["pct"] = gender_distribution["n"] / gender_distribution["n"].sum()

    save_table(age_tenure, output_dir, "demographics_age_tenure_summary.csv")
    save_table(gender_distribution, output_dir, "demographics_gender_distribution.csv")

    q2 = pd.Categorical(analysis["Q2_LABEL"], categories=Q2_ORDER, ordered=True)
    q10 = pd.Categorical(analysis["Q10_LABEL"], categories=Q10_ORDER, ordered=True)
    crosstab_counts = pd.crosstab(q2, q10, dropna=False)
    crosstab_pct = crosstab_counts.div(crosstab_counts.sum(axis=1).replace(0, np.nan), axis=0)

    save_table(crosstab_counts, output_dir, "q2_by_q10_crosstab_counts.csv")
    save_table(crosstab_pct, output_dir, "q2_by_q10_crosstab_row_pct.csv")

    chi_square_input = crosstab_counts.fillna(0)
    chi_square_input = chi_square_input.loc[
        chi_square_input.sum(axis=1).gt(0),
        chi_square_input.sum(axis=0).gt(0),
    ]
    if chi_square_input.shape[0] >= 2 and chi_square_input.shape[1] >= 2:
        chi2, p_value, dof, expected = chi2_contingency(chi_square_input)
    else:
        chi2, p_value, dof = np.nan, np.nan, np.nan
        logging.warning("Skipped chi-square because the Q2 x Q10 table has insufficient non-empty categories")
    spearman_data = analysis[["Q10_SCORE"]].copy()
    sentiment_order = {label: score for score, label in enumerate(reversed(Q2_ORDER), start=1)}
    spearman_data["Q2_SCORE"] = analysis["Q2_LABEL"].map(sentiment_order)
    spearman_data = spearman_data.dropna()
    rho, rho_p = spearmanr(spearman_data["Q2_SCORE"], spearman_data["Q10_SCORE"]) if len(spearman_data) > 1 else (np.nan, np.nan)

    stats = pd.DataFrame(
        {
            "metric": ["chi_square", "chi_square_p_value", "degrees_of_freedom", "spearman_rho", "spearman_p_value"],
            "value": [chi2, p_value, dof, rho, rho_p],
        }
    )
    save_table(stats.set_index("metric"), output_dir, "q2_q10_association_stats.csv")

    plot_stacked_bar(
        crosstab_pct,
        output_dir / "q2_by_q10_stacked_bar.png",
        "Platform Sentiment vs. Likelihood to Re-subscribe",
        "Row share of respondents",
    )


def category_distribution(series: pd.Series, label_name: str) -> pd.DataFrame:
    distribution = (
        series.fillna("Missing")
        .replace({"": "Missing"})
        .value_counts(dropna=False)
        .rename_axis(label_name)
        .to_frame("n")
    )
    distribution["pct"] = distribution["n"] / distribution["n"].sum()
    return distribution


def conversion_and_subscription_motivations(
    analysis: pd.DataFrame,
    meta: pyreadstat.metadata_container,
    output_dir: Path,
) -> None:
    """Summarize how churned users found Plus and what initially motivated purchase."""
    if "Q5" in analysis.columns:
        q5_distribution = category_distribution(labeled_series(analysis, meta, "Q5"), "conversion_channel")
        save_table(q5_distribution, output_dir, "q5_conversion_channel_distribution.csv")

    if "Q7" in analysis.columns:
        q7_distribution = category_distribution(labeled_series(analysis, meta, "Q7"), "main_subscription_reason")
        save_table(q7_distribution, output_dir, "q7_main_subscription_reason_distribution.csv")

    q6_columns = [f"Q6_{idx}" for idx in range(1, 10) if f"Q6_{idx}" in analysis.columns]
    if q6_columns:
        rows = []
        for column in q6_columns:
            values = analysis[column]
            if values.dropna().empty:
                selected = pd.Series(False, index=analysis.index)
            elif pd.api.types.is_numeric_dtype(values):
                selected = pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
            else:
                selected = values.astype("string").str.strip().str.lower().isin(["1", "true", "yes", "selected"])

            rows.append(
                {
                    "motivation_variable": column,
                    "motivation": clean_rank_label(column, meta),
                    "selected_n": int(selected.sum()),
                    "respondents": int(selected.notna().sum()),
                    "selected_pct": selected.mean(),
                }
            )

        q6_summary = pd.DataFrame(rows).sort_values(["selected_pct", "selected_n"], ascending=[False, False])
        save_table(q6_summary.set_index("motivation_variable"), output_dir, "q6_initial_subscription_motivations.csv")

        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=q6_summary, y="motivation", x="selected_pct", palette="flare")
        ax.set_title("Q6 Initial Subscription Motivations")
        ax.set_xlabel("Share of respondents selecting motivation")
        ax.set_ylabel("")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
        annotate_horizontal_bars(ax, q6_summary["selected_pct"], lambda value: f"{value:.1%}")
        plt.tight_layout()
        plt.savefig(output_dir / "q6_initial_subscription_motivations.png")
        plt.close()


def clean_rank_label(column: str, meta: pyreadstat.metadata_container) -> str:
    label = getattr(meta, "column_names_to_labels", {}).get(column, "") or column
    label = re.sub(r"^\s*Q[68][_\d]*\s*[-:.)]?\s*", "", label, flags=re.IGNORECASE)
    label = re.sub(r"\s+", " ", label).strip()
    return label or column


def ranking_optimization(analysis: pd.DataFrame, meta: pyreadstat.metadata_container, output_dir: Path) -> None:
    q8_columns = [f"Q8_{idx}" for idx in range(1, 6) if f"Q8_{idx}" in analysis.columns]
    if not q8_columns:
        logging.warning("No Q8 rank columns found; skipping ranking analysis")
        return

    rank_df = analysis[q8_columns].apply(pd.to_numeric, errors="coerce")
    summary = (
        rank_df.agg(["count", "mean", "median", "std"])
        .T
        .rename_axis("feature_variable")
        .reset_index()
    )
    summary["feature"] = summary["feature_variable"].apply(lambda column: clean_rank_label(column, meta))
    summary["importance_score"] = 6 - summary["mean"]
    summary = summary.sort_values(["importance_score", "mean"], ascending=[False, True])
    save_table(summary.set_index("feature_variable"), output_dir, "q8_feature_rank_summary.csv")

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=summary, y="feature", x="importance_score", palette="mako")
    ax.set_title("Q8 Feature Importance Ranking")
    ax.set_xlabel("Importance score (6 - mean rank; higher is better)")
    ax.set_ylabel("")
    annotate_horizontal_bars(ax, summary["importance_score"], lambda value: f"{value:.2f}")
    plt.tight_layout()
    plt.savefig(output_dir / "q8_feature_importance_scores.png")
    plt.close()


def keyword_match(text: str, keywords: list[str]) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False

    lowered = text.lower()
    return any(re.search(rf"\b{re.escape(keyword.lower())}\b", lowered) for keyword in keywords)


def keyword_share_of_voice(
    df: pd.DataFrame,
    text_column: str,
    group_column: str,
    keyword_dict: dict[str, list[str]],
) -> pd.DataFrame:
    output_columns = [
        "group",
        "keyword_category",
        "matching_respondents",
        "text_respondents",
        "share_of_voice",
    ]
    rows = []
    working = df[[group_column, text_column]].copy()
    working[text_column] = working[text_column].fillna("").astype(str)
    working = working[working[text_column].str.strip().ne("")]

    for group, group_df in working.groupby(group_column, dropna=False):
        denominator = len(group_df)
        for category, keywords in keyword_dict.items():
            matches = group_df[text_column].apply(lambda value: keyword_match(value, keywords))
            rows.append(
                {
                    "group": group,
                    "keyword_category": category,
                    "matching_respondents": int(matches.sum()),
                    "text_respondents": denominator,
                    "share_of_voice": matches.mean() if denominator else np.nan,
                }
            )

    return pd.DataFrame(rows, columns=output_columns)


def churn_segmentation(analysis: pd.DataFrame, output_dir: Path) -> None:
    q9_distribution = (
        pd.crosstab(analysis["RETENTION_GROUP"], analysis["Q9_LABEL"], normalize="index")
        .sort_index()
    )
    save_table(q9_distribution, output_dir, "q9_churn_reason_by_retention_group_row_pct.csv")

    keyword_sov = keyword_share_of_voice(
        analysis,
        text_column="CHURN_TEXT_COMBINED",
        group_column="RETENTION_GROUP",
        keyword_dict=CHURN_KEYWORDS,
    )
    save_table(keyword_sov.set_index(["group", "keyword_category"]), output_dir, "churn_text_keyword_share_of_voice.csv")

    if not keyword_sov.empty:
        heatmap_data = keyword_sov.pivot(index="keyword_category", columns="group", values="share_of_voice")
        plt.figure(figsize=(11, 6))
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap="Reds", linewidths=0.5)
        ax.set_title("Churn Pain Point Share of Voice by Retention Intent")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "churn_keyword_share_heatmap.png")
        plt.close()


def feature_evaluation(analysis: pd.DataFrame, output_dir: Path) -> None:
    q12_counts = (
        analysis["Q12_LABEL"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("feature")
        .to_frame("n")
    )
    q12_counts["pick_rate"] = q12_counts["n"] / q12_counts["n"].sum()
    q12_counts = q12_counts.sort_values(["n", "feature"], ascending=[False, True])
    save_table(q12_counts, output_dir, "q12_feature_pick_rates.csv")

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=q12_counts.reset_index(),
        y="feature",
        x="pick_rate",
        palette="crest",
    )
    ax.set_title("Q12 Future Feature Pick Rates")
    ax.set_xlabel("Pick rate")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    annotate_horizontal_bars(ax, q12_counts["pick_rate"], lambda value: f"{value:.1%}")
    plt.tight_layout()
    plt.savefig(output_dir / "q12_feature_pick_rates.png")
    plt.close()

    backlash_sov = keyword_share_of_voice(
        analysis,
        text_column="Q13_TEXT_COMBINED",
        group_column="Q12_LABEL",
        keyword_dict=BACKLASH_KEYWORDS,
    ).rename(columns={"group": "feature", "keyword_category": "backlash_keyword"})
    save_table(backlash_sov.set_index(["feature", "backlash_keyword"]), output_dir, "q13_backlash_keywords_by_feature.csv")

    if not backlash_sov.empty:
        heatmap_data = backlash_sov.pivot(index="feature", columns="backlash_keyword", values="share_of_voice").fillna(0)
        # Preserve product relevance by sorting feature rows by Q12 pick rate.
        ordered_features = [feature for feature in q12_counts.index if feature in heatmap_data.index]
        heatmap_data = heatmap_data.loc[ordered_features]

        plt.figure(figsize=(11, max(5, 0.55 * len(heatmap_data))))
        ax = sns.heatmap(heatmap_data, annot=True, fmt=".1%", cmap="rocket_r", linewidths=0.5)
        ax.set_title("Q13 Corporate Backlash Language by Selected Feature")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(output_dir / "q13_backlash_keywords_by_feature_heatmap.png")
        plt.close()


def export_metadata_reference(meta: pyreadstat.metadata_container, output_dir: Path, columns: list[str]) -> None:
    """Write a compact reference of SPSS labels used by the script."""
    rows = []
    column_labels = getattr(meta, "column_names_to_labels", {})
    for column in columns:
        value_map = metadata_value_map(meta, column)
        if value_map:
            for code, label in value_map.items():
                rows.append(
                    {
                        "variable": column,
                        "variable_label": column_labels.get(column, ""),
                        "value_code": code,
                        "value_label": label,
                    }
                )
        else:
            rows.append(
                {
                    "variable": column,
                    "variable_label": column_labels.get(column, ""),
                    "value_code": "",
                    "value_label": "",
                }
            )

    save_table(pd.DataFrame(rows).set_index(["variable", "value_code"]), output_dir, "spss_metadata_reference.csv")


def main() -> None:
    args = parse_args()
    require_dependencies()
    sav_path = Path(args.sav_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    configure_runtime(output_dir)

    df, meta = read_sav_with_metadata(sav_path)
    analysis = clean_analysis_frame(df, meta, min_age=args.min_age)
    validate_schema_and_quality(df, analysis, output_dir, min_age=args.min_age)

    metadata_columns = [
        "GENDER",
        "Q2",
        "Q3",
        "Q4",
        "Q5",
        "Q6_1",
        "Q6_2",
        "Q6_3",
        "Q6_4",
        "Q6_5",
        "Q6_6",
        "Q6_7",
        "Q6_8",
        "Q6_9",
        "Q7",
        "Q8_1",
        "Q8_2",
        "Q8_3",
        "Q8_4",
        "Q8_5",
        "Q9",
        "Q10",
        "Q12",
    ]
    export_metadata_reference(meta, output_dir, [column for column in metadata_columns if column in df.columns])

    demographics_and_brand_health(analysis, output_dir)
    conversion_and_subscription_motivations(analysis, meta, output_dir)
    ranking_optimization(analysis, meta, output_dir)
    churn_segmentation(analysis, output_dir)
    feature_evaluation(analysis, output_dir)

    analysis_columns = [
        "USERID",
        "AGE",
        "TENURE_NUMERIC",
        "GENDER_LABEL",
        "Q2_LABEL",
        "Q9_LABEL",
        "Q10_SCORE",
        "Q10_LABEL",
        "RETENTION_GROUP",
        "Q12_LABEL",
        "Q11_TEXT_COMBINED",
        "Q13_TEXT_COMBINED",
        "CHURN_TEXT_COMBINED",
    ]
    export_columns = [column for column in analysis_columns if column in analysis.columns]
    analysis[export_columns].to_csv(output_dir / "analysis_ready_respondent_level.csv", index=False)

    logging.info("Analysis complete. Outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
