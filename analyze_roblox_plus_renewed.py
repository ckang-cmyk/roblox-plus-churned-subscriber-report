#!/usr/bin/env python3
"""
Roblox Plus renewed subscriber survey analysis.

Creates the same report-ready CSV outputs as the churned-subscriber analysis,
adapted to the renewed-subscriber survey schema:

- Q8 is the structured "most valuable" Plus benefit.
- Q9_1 to Q9_5 are the current-benefit ranking fields.
- Q10 is likelihood to stay subscribed.

Run:
    python analyze_roblox_plus_renewed.py
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat
from scipy.stats import chi2_contingency, spearmanr


SAV_PATH = Path("Roblox Plus Renewed subs_July 8, 2026_16.48.sav")
OUTPUT_DIR = Path("renewed_outputs")

SENTIMENT_ORDER = ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]
LIKELIHOOD_ORDER = ["Very Unlikely", "Unlikely", "Maybe", "Likely", "Very Likely"]
RETENTION_ORDER = ["Low Intent", "Conditional Intent", "High Intent"]

GENDER_LABELS = {1: "Female", 2: "Male", 3: "Unknown"}
Q7_LABELS = {
    1: "10-20% item discounts",
    2: "Free & unlimited private servers",
    3: "Free Robux transfers",
    4: "Exclusive Plus badge",
    5: "Trade & resell avatar items",
    6: "Publish avatar items",
    7: "Wanted to try it out",
    8: "Free trial",
    9: "Other",
}

RENEWAL_VALUE_KEYWORDS = {
    "Robux Absence / Premium Rebrand": ["robux", "premium", "stipend", "builders club", "builder's club"],
    "Financial Strain": ["price", "cost", "money", "afford", "expensive", "deal"],
    "Savings / Discount Value": ["discount", "save", "saving", "savings", "deal", "cheap", "cheaper"],
    "Private Server Utility": ["private server", "private servers", "server", "servers"],
    "Robux Transfer Utility": ["robux", "transfer", "send", "give", "donate"],
    "Creator / Trading Utility": ["trade", "resell", "publish", "ugc", "create", "creator", "marketplace"],
    "Status / Exclusivity": ["badge", "exclusive", "recognized", "status", "profile"],
}

BACKLASH_KEYWORDS = {
    "slop": ["slop"],
    "AI garbage": ["ai garbage", "garbage ai"],
    "investor": ["investor", "investors"],
    "greed": ["greed", "greedy"],
}


def normalize_label(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    text = str(value).strip()
    return text if text else "Missing"


def normalized_code(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def metadata_value_map(meta: pyreadstat.metadata_container, column: str) -> dict:
    return getattr(meta, "variable_value_labels", {}).get(column, {})


def labeled_series(df: pd.DataFrame, meta: pyreadstat.metadata_container, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(index=df.index, dtype="string")

    value_map = metadata_value_map(meta, column)
    if column == "Gender" and not value_map:
        value_map = GENDER_LABELS
    if column == "Q7" and not value_map:
        value_map = Q7_LABELS

    normalized_map: dict[object, object] = {}
    for key, label in value_map.items():
        normalized_map[key] = label
        code = normalized_code(key)
        if code is not None:
            normalized_map[code] = label
            normalized_map[float(code)] = label
            normalized_map[str(code)] = label

    if not normalized_map:
        return df[column].map(normalize_label).astype("string")

    return df[column].map(normalized_map).fillna(df[column]).map(normalize_label).astype("string")


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def combine_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series("", index=df.index, dtype="string")
    return (
        df[columns]
        .fillna("")
        .astype(str)
        .apply(lambda row: " | ".join(value.strip() for value in row if value.strip()), axis=1)
        .astype("string")
    )


def save_table(df: pd.DataFrame, filename: str, index: bool = True) -> None:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=index)
    logging.info("Wrote %s", path)


def category_distribution(series: pd.Series, label_name: str, count_name: str = "n", pct_name: str = "pct") -> pd.DataFrame:
    distribution = (
        series.fillna("Missing")
        .replace({"": "Missing"})
        .value_counts(dropna=False)
        .rename_axis(label_name)
        .reset_index(name=count_name)
    )
    distribution[pct_name] = distribution[count_name] / distribution[count_name].sum()
    return distribution


def clean_rank_label(column: str, meta: pyreadstat.metadata_container) -> str:
    label = getattr(meta, "column_names_to_labels", {}).get(column, column)
    label = re.sub(r"^Rank_", "", str(label))
    return re.sub(r"\s+", " ", label).strip()


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
    rows = []
    working = df[[group_column, text_column]].copy()
    working[text_column] = working[text_column].fillna("").astype(str)
    working = working[working[text_column].str.strip().ne("")]

    for group, group_df in working.groupby(group_column, dropna=False, observed=False):
        denominator = len(group_df)
        for category, keywords in keyword_dict.items():
            matches = group_df[text_column].map(lambda value: keyword_match(value, keywords))
            rows.append(
                {
                    "group": group,
                    "keyword_category": category,
                    "matching_respondents": int(matches.sum()),
                    "text_respondents": denominator,
                    "share_of_voice": matches.mean() if denominator else np.nan,
                }
            )

    return pd.DataFrame(
        rows,
        columns=["group", "keyword_category", "matching_respondents", "text_respondents", "share_of_voice"],
    )


def build_analysis_frame(df: pd.DataFrame, meta: pyreadstat.metadata_container, min_age: int = 13) -> pd.DataFrame:
    analysis = df.copy()
    analysis["USERID"] = analysis.get("userId")
    analysis["AGE"] = numeric_series(analysis, "Q1")
    before = len(analysis)
    analysis = analysis[analysis["AGE"].isna() | (analysis["AGE"] >= min_age)].copy()
    logging.info("Dropped %s rows where Q1 < %s", before - len(analysis), min_age)

    analysis["TENURE_NUMERIC"] = numeric_series(analysis, "Tenure")
    analysis["GENDER_LABEL"] = labeled_series(analysis, meta, "Gender")
    analysis["Q2_LABEL"] = labeled_series(analysis, meta, "Q2")
    analysis["Q3_LABEL"] = labeled_series(analysis, meta, "Q3")
    analysis["Q4_LABEL"] = labeled_series(analysis, meta, "Q4")
    analysis["Q8_LABEL"] = labeled_series(analysis, meta, "Q8")

    analysis["Q9_LABEL"] = analysis["Q8_LABEL"]
    analysis["Q10_SCORE"] = numeric_series(analysis, "Q10")
    q10_score_map = {1: "Very Unlikely", 2: "Unlikely", 3: "Maybe", 4: "Likely", 5: "Very Likely"}
    analysis["Q10_LABEL"] = analysis["Q10_SCORE"].map(q10_score_map).fillna(labeled_series(analysis, meta, "Q10"))
    analysis["Q12_LABEL"] = labeled_series(analysis, meta, "Q12")

    analysis["RETENTION_GROUP"] = pd.cut(
        analysis["Q10_SCORE"],
        bins=[0, 2, 3, 5],
        labels=RETENTION_ORDER,
        include_lowest=True,
    )
    analysis["Q11_TEXT_COMBINED"] = combine_text_columns(analysis, ["Q11"])
    analysis["Q13_TEXT_COMBINED"] = combine_text_columns(analysis, ["Q13"])
    analysis["CHURN_TEXT_COMBINED"] = combine_text_columns(analysis, ["Q8_9_TEXT", "Q11"])
    return analysis


def export_quality(df: pd.DataFrame, analysis: pd.DataFrame) -> None:
    expected = [
        "userId",
        "Tenure",
        "Gender",
        "Q1",
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
        "Q8",
        "Q9_1",
        "Q9_2",
        "Q9_3",
        "Q9_4",
        "Q9_5",
        "Q10",
        "Q11",
        "Q12",
        "Q13",
    ]
    rows = [{"check": "expected_column_present", "field": column, "value": column in df.columns} for column in expected]
    rows.extend(
        [
            {"check": "raw_rows", "field": "dataset", "value": len(df)},
            {"check": "analysis_rows_after_age_filter", "field": "dataset", "value": len(analysis)},
            {"check": "duplicate_userid_count", "field": "userId", "value": int(df["userId"].duplicated().sum())},
        ]
    )
    save_table(pd.DataFrame(rows).set_index(["check", "field"]), "dataset_quality_summary.csv")


def export_demographics_and_brand_health(analysis: pd.DataFrame) -> None:
    age_tenure = analysis[["AGE", "TENURE_NUMERIC"]].describe(percentiles=[0.25, 0.5, 0.75]).T
    gender = category_distribution(analysis["GENDER_LABEL"], "gender")
    save_table(age_tenure, "demographics_age_tenure_summary.csv")
    save_table(gender, "demographics_gender_distribution.csv", index=False)

    q2 = pd.Categorical(analysis["Q2_LABEL"], categories=SENTIMENT_ORDER, ordered=True)
    q10 = pd.Categorical(analysis["Q10_LABEL"], categories=LIKELIHOOD_ORDER, ordered=True)
    counts = pd.crosstab(q2, q10, dropna=False)
    row_pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)
    save_table(counts, "q2_by_q10_crosstab_counts.csv")
    save_table(row_pct, "q2_by_q10_crosstab_row_pct.csv")

    chi_square_input = counts.loc[counts.sum(axis=1).gt(0), counts.sum(axis=0).gt(0)]
    chi2, p_value, dof, _ = chi2_contingency(chi_square_input) if chi_square_input.shape[0] >= 2 and chi_square_input.shape[1] >= 2 else (np.nan, np.nan, np.nan, None)
    sentiment_order = {label: score for score, label in enumerate(reversed(SENTIMENT_ORDER), start=1)}
    spearman_data = pd.DataFrame(
        {"Q2_SCORE": analysis["Q2_LABEL"].map(sentiment_order), "Q10_SCORE": analysis["Q10_SCORE"]}
    ).dropna()
    rho, rho_p = spearmanr(spearman_data["Q2_SCORE"], spearman_data["Q10_SCORE"]) if len(spearman_data) > 1 else (np.nan, np.nan)
    save_table(
        pd.DataFrame(
            {
                "metric": ["chi_square", "chi_square_p_value", "degrees_of_freedom", "spearman_rho", "spearman_p_value"],
                "value": [chi2, p_value, dof, rho, rho_p],
            }
        ).set_index("metric"),
        "q2_q10_association_stats.csv",
    )

    q2_distribution = category_distribution(analysis["Q2_LABEL"], "platform_sentiment", "respondents", "share")
    q2_distribution["sort_order"] = q2_distribution["platform_sentiment"].map({value: i for i, value in enumerate(SENTIMENT_ORDER)})
    q2_distribution = q2_distribution.sort_values(["sort_order", "platform_sentiment"]).drop(columns="sort_order")
    save_table(q2_distribution, "q2_platform_sentiment_distribution.csv", index=False)

    q4_order = ["Not familiar at all", "Barely familiar", "Somewhat familiar", "Very familiar", "Extremely familiar"]
    q4_distribution = category_distribution(analysis["Q4_LABEL"], "plus_familiarity", "respondents", "share")
    q4_distribution["sort_order"] = q4_distribution["plus_familiarity"].map({value: i for i, value in enumerate(q4_order)})
    q4_distribution = q4_distribution.sort_values(["sort_order", "plus_familiarity"]).drop(columns="sort_order")
    save_table(q4_distribution, "q4_plus_familiarity_distribution.csv", index=False)

    q3_distribution = category_distribution(analysis["Q3_LABEL"], "primary_reason_for_using_roblox", "respondents", "share")
    save_table(q3_distribution, "q3_primary_reason_for_using_roblox.csv", index=False)


def export_subscription_path(analysis: pd.DataFrame, meta: pyreadstat.metadata_container) -> None:
    q5 = category_distribution(labeled_series(analysis, meta, "Q5"), "conversion_channel")
    save_table(q5, "q5_conversion_channel_distribution.csv", index=False)

    q7 = category_distribution(labeled_series(analysis, meta, "Q7"), "main_subscription_reason")
    save_table(q7, "q7_main_subscription_reason_distribution.csv", index=False)

    rows = []
    for column in [f"Q6_{idx}" for idx in range(1, 10)]:
        values = analysis[column] if column in analysis.columns else pd.Series(index=analysis.index, dtype="float64")
        selected = pd.to_numeric(values, errors="coerce").fillna(0).ne(0)
        label_map = metadata_value_map(meta, column)
        label = next(iter(label_map.values()), column)
        rows.append(
            {
                "motivation_variable": column,
                "motivation": label,
                "selected_n": int(selected.sum()),
                "respondents": int(len(selected)),
                "selected_pct": selected.mean(),
            }
        )

    q6 = pd.DataFrame(rows).sort_values(["selected_pct", "selected_n"], ascending=[False, False])
    save_table(q6.set_index("motivation_variable"), "q6_initial_subscription_motivations.csv")


def export_ranking(analysis: pd.DataFrame, meta: pyreadstat.metadata_container) -> None:
    rank_columns = [f"Q9_{idx}" for idx in range(1, 6)]
    rank_df = analysis[rank_columns].apply(pd.to_numeric, errors="coerce")
    summary = rank_df.agg(["count", "mean", "median", "std"]).T.rename_axis("feature_variable").reset_index()
    summary["feature"] = summary["feature_variable"].map(lambda column: clean_rank_label(column, meta))
    summary["importance_score"] = 6 - summary["mean"]
    summary = summary.sort_values(["importance_score", "mean"], ascending=[False, True])
    save_table(summary.set_index("feature_variable"), "q8_feature_rank_summary.csv")

    rows = []
    long_rows = []
    for column in rank_columns:
        benefit = clean_rank_label(column, meta)
        counts = rank_df[column].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        valid_n = int(rank_df[column].notna().sum())
        row = {"Benefit": benefit, "Valid Ranking N": valid_n}
        for rank in range(1, 6):
            count = int(counts.loc[rank])
            pct = count / valid_n if valid_n else np.nan
            row[f"#{rank} Count"] = count
            row[f"#{rank} %"] = pct
            long_rows.append({"Benefit": benefit, "Rank": f"#{rank}", "Count": count, "Percent": pct})
        row["Top Box #1 %"] = row["#1 %"]
        rows.append(row)

    save_table(pd.DataFrame(rows), "q8_benefit_ranking_distribution.csv", index=False)
    save_table(pd.DataFrame(long_rows), "q8_benefit_ranking_long.csv", index=False)


def export_value_segmentation(analysis: pd.DataFrame) -> None:
    value_mix = pd.crosstab(analysis["RETENTION_GROUP"], analysis["Q8_LABEL"], normalize="index").sort_index()
    save_table(value_mix, "q9_churn_reason_by_retention_group_row_pct.csv")

    keywords = keyword_share_of_voice(
        analysis,
        text_column="CHURN_TEXT_COMBINED",
        group_column="RETENTION_GROUP",
        keyword_dict=RENEWAL_VALUE_KEYWORDS,
    )
    save_table(keywords.set_index(["group", "keyword_category"]), "churn_text_keyword_share_of_voice.csv")


def export_feature_evaluation(analysis: pd.DataFrame) -> None:
    q12 = (
        analysis["Q12_LABEL"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("feature")
        .reset_index(name="n")
    )
    q12["pick_rate"] = q12["n"] / q12["n"].sum()
    q12 = q12.sort_values(["n", "feature"], ascending=[False, True])
    save_table(q12, "q12_feature_pick_rates.csv", index=False)

    backlash = keyword_share_of_voice(
        analysis,
        text_column="Q13_TEXT_COMBINED",
        group_column="Q12_LABEL",
        keyword_dict=BACKLASH_KEYWORDS,
    ).rename(columns={"group": "feature", "keyword_category": "backlash_keyword"})
    save_table(backlash.set_index(["feature", "backlash_keyword"]), "q13_backlash_keywords_by_feature.csv")


def export_metadata(meta: pyreadstat.metadata_container, columns: list[str]) -> None:
    rows = []
    column_labels = getattr(meta, "column_names_to_labels", {})
    for column in columns:
        value_map = metadata_value_map(meta, column)
        if column == "Q7" and not value_map:
            value_map = Q7_LABELS
        for code, label in value_map.items():
            rows.append(
                {
                    "variable": column,
                    "variable_label": column_labels.get(column, ""),
                    "value_code": code,
                    "value_label": label,
                }
            )
    save_table(pd.DataFrame(rows).set_index(["variable", "value_code"]), "spss_metadata_reference.csv")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df, meta = pyreadstat.read_sav(str(SAV_PATH), apply_value_formats=False, user_missing=True)
    logging.info("Loaded %s rows and %s columns from %s", len(df), len(df.columns), SAV_PATH)

    analysis = build_analysis_frame(df, meta)
    export_quality(df, analysis)
    export_metadata(meta, [column for column in df.columns if column.startswith("Q") or column == "Gender"])
    export_demographics_and_brand_health(analysis)
    export_subscription_path(analysis, meta)
    export_ranking(analysis, meta)
    export_value_segmentation(analysis)
    export_feature_evaluation(analysis)

    analysis_columns = [
        "USERID",
        "AGE",
        "TENURE_NUMERIC",
        "GENDER_LABEL",
        "Q2_LABEL",
        "Q3_LABEL",
        "Q4_LABEL",
        "Q8_LABEL",
        "Q9_LABEL",
        "Q10_SCORE",
        "Q10_LABEL",
        "RETENTION_GROUP",
        "Q11_TEXT_COMBINED",
        "Q12_LABEL",
        "Q13_TEXT_COMBINED",
        "CHURN_TEXT_COMBINED",
    ]
    save_table(analysis[analysis_columns], "analysis_ready_respondent_level.csv", index=False)


if __name__ == "__main__":
    main()
