#!/usr/bin/env python3
"""
Generate a static comparison report across Roblox Plus non-subscribers,
churned subscribers, and renewed subscribers.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pyreadstat
from scipy.stats import chi2_contingency


REPORT_PATH = Path("roblox_plus_three_segment_comparison_report.html")

SEGMENT_ORDER = ["Non-subscribers", "Churned subscribers", "Renewed subscribers"]
SENTIMENT_ORDER = ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]
FAMILIARITY_ORDER = ["Not familiar at all", "Barely familiar", "Somewhat familiar", "Very familiar", "Extremely familiar"]
LIKELIHOOD_ORDER = ["Very Unlikely", "Unlikely", "Maybe", "Likely", "Very Likely"]
GENDER_ORDER = ["Male", "Female", "Unknown", "Missing"]

SEGMENT_COLORS = {
    "Non-subscribers": "#6c8ebf",
    "Churned subscribers": "#d98282",
    "Renewed subscribers": "#6fbf8b",
}
BENEFIT_COLORS = {
    "Discounts": "#4e79a7",
    "Private servers": "#f28e2b",
    "Free Robux transfers": "#59a14f",
    "Trade & resell avatar items": "#e15759",
    "Publish avatar items": "#b07aa1",
}
LIKELIHOOD_COLORS = {
    "Very Unlikely": "#c0392b",
    "Unlikely": "#f28e8e",
    "Maybe": "#9e9e9e",
    "Likely": "#8fd19e",
    "Very Likely": "#2e8b57",
}
GENDER_COLORS = {
    "Male": "#9ecae1",
    "Female": "#f7b6d2",
    "Unknown": "#c5b0d5",
    "Missing": "#d9d9d9",
}

Q2_LABELS = {1: "Love it", 2: "Like it", 3: "Neutral", 4: "Dislike it", 5: "Hate it"}
Q3_LABELS = {
    1: "Spend time with friends",
    2: "Connect with new people",
    3: "Communicate with others",
    4: "Variety of games / experiences",
    5: "Create",
    6: "Entertain myself",
    7: "Spend time with family",
    8: "Express myself",
    9: "Other",
}
Q4_LABELS = {
    1: "Not familiar at all",
    2: "Barely familiar",
    3: "Somewhat familiar",
    4: "Very familiar",
    5: "Extremely familiar",
}
Q5_LABELS = {
    1: "Buy Robux page",
    2: "Buying an item",
    3: "Home / More tab",
    4: "Plus badge on profile",
    5: "Social media",
    6: "Friends",
    7: "Somewhere else",
    8: "Don't remember",
}
FEATURE_LABELS = {
    1: "AI avatar backgrounds",
    2: "App themes",
    3: "Profile frames",
    4: "Rich text and emojis",
    5: "Particle effect avatar items",
    6: "AI-generated avatars/items",
    7: "Spawn / despawn effects",
}
BENEFIT_LABELS = {
    1: "Discounts",
    2: "Private servers",
    3: "Free Robux transfers",
    4: "Trade & resell avatar items",
    5: "Publish avatar items",
}
NON_SUB_REASON_LABELS = {
    1: "Not familiar with benefits",
    2: "Don't play enough",
    3: "Don't buy enough items",
    4: "Don't use private servers",
    5: "Don't transfer Robux often",
    6: "No monthly Robux stipend",
    7: "Don't like monthly subscription",
    8: "Haven't gotten around to it",
    9: "No payment method",
    11: "Too expensive / not worth cost",
    12: "Other",
}
RENEWED_VALUE_LABELS = {
    "Saving Robux with the 10-20% item discounts": "Discounts",
    "Free & unlimited private servers": "Private servers",
    "I feel like I’m getting a better deal": "Better deal",
    "Send Robux for free to anyone": "Free Robux transfers",
    "Other (Please specify)": "Other",
    "I feel more exclusive/recognized with the Plus badge on my profile": "Feel more exclusive with Plus badge",
    "Trade & resell avatar items": "Trade & resell avatar items",
    "I feel more connected with friends": "Friend connection",
    "Publish avatar items": "Publish avatar items",
}
SUBSCRIPTION_REASON_LABELS = {
    "WhySub_Free Trial": "Free trial",
    "WhySub_Private servers": "Private servers",
    "WhySub_Try it out": "Wanted to try it out",
    "WhySub_Discount": "Discounts",
    "WhySub_Robux transfers": "Free Robux transfers",
    "WhySub_Plus badge": "Plus badge",
    "WhySub_Trade & resell": "Trade & resell avatar items",
    "WhySub_Publish avatar items": "Publish avatar items",
    "WhySub_Other": "Other",
    "Discount": "Discounts",
    "10% discount on items (increases to 20% after 2 months)": "Discounts",
    "Free & unlimited private servers": "Private servers",
    "Robux transfers": "Free Robux transfers",
    "Send Robux for free to anyone": "Free Robux transfers",
    "There was a free trial": "Free trial",
    "I just wanted to try it out": "Wanted to try it out",
    "Having exclusive Plus badge on my profile": "Plus badge",
    "Trade and resell": "Trade & resell avatar items",
    "Trade & resell": "Trade & resell avatar items",
    "Trade & resell avatar items": "Trade & resell avatar items",
    "Publish avatar items": "Publish avatar items",
    "Other (Please specify)": "Other",
    "10-20% item discounts": "Discounts",
    "Exclusive Plus badge": "Plus badge",
}
Q7_LABELS = {
    "1.0": "Discounts",
    "2.0": "Private servers",
    "3.0": "Free Robux transfers",
    "4.0": "Plus badge",
    "5.0": "Trade & resell avatar items",
    "6.0": "Publish avatar items",
    "7.0": "Wanted to try it out",
    "8.0": "Free trial",
    "9.0": "Other",
}
BENEFIT_CANONICAL_LABELS = {
    "Discount": "Discounts",
    "Discounts": "Discounts",
    "Item discounts": "Discounts",
    "10-20% item discounts": "Discounts",
    "Private servers": "Private servers",
    "Free / unlimited private servers": "Private servers",
    "Robux transfers": "Free Robux transfers",
    "Free Robux transfers": "Free Robux transfers",
    "Zero-fee Robux transfers": "Free Robux transfers",
    "Trade and resell": "Trade & resell avatar items",
    "Trade & resell": "Trade & resell avatar items",
    "Trade & resell avatar items": "Trade & resell avatar items",
    "Publish avatar items": "Publish avatar items",
}
CHURN_REASON_LABELS = {
    "It did not come with monthly Robux": "No monthly Robux",
    "Too expensive / not enough value": "Too expensive / weak value",
    "I only wanted the free trial": "Only wanted free trial",
    "I did not use the benefits enough": "Did not use benefits enough",
    "Other (Please specify)": "Other",
}


def normalize_code(value: object) -> int | None:
    try:
        if pd.isna(value) or str(value).strip() == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def map_codes(series: pd.Series, labels: dict[int, str]) -> pd.Series:
    return series.map(lambda value: labels.get(normalize_code(value), "Missing"))


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def wilson_ci(successes: float, total: float, z: float = 1.959963984540054) -> tuple[float, float]:
    if pd.isna(successes) or pd.isna(total) or total <= 0:
        return (np.nan, np.nan)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return max(0, center - half), min(1, center + half)


def add_ci_columns(df: pd.DataFrame, success_col: str, total_col: str, pct_col: str) -> pd.DataFrame:
    out = df.copy()
    bounds = out.apply(lambda row: wilson_ci(row[success_col], row[total_col]), axis=1, result_type="expand")
    out["ci_lower"] = bounds[0]
    out["ci_upper"] = bounds[1]
    out["ci_error_plus"] = out["ci_upper"] - out[pct_col]
    out["ci_error_minus"] = out[pct_col] - out["ci_lower"]
    return out


def keep_labels_clear(fig) -> None:
    fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=11, cliponaxis=False)
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")


def fig_html(fig, include_plotlyjs: bool = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
        default_height="500px",
    )


def table_html(df: pd.DataFrame) -> str:
    escaped = df.copy()
    for col in escaped.columns:
        escaped[col] = escaped[col].map(lambda value: escape(str(value)))
    return escaped.to_html(index=False, classes="data-table", border=0, escape=False)


def section(title: str, body: str) -> str:
    return f"""
    <section class="report-section">
      <h2>{escape(title)}</h2>
      {body}
    </section>
    """


def make_kpi(label: str, value: str, note: str) -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{escape(label)}</div>
      <div class="kpi-value">{escape(value)}</div>
      <div class="kpi-note">{escape(note)}</div>
    </div>
    """


def cramers_v(table: pd.DataFrame) -> tuple[float, float, int, float]:
    table = table.loc[table.sum(axis=1).gt(0), table.sum(axis=0).gt(0)]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan, 0, np.nan
    chi2, p_value, dof, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    v = np.sqrt((chi2 / n) / min(k - 1, r - 1)) if min(k - 1, r - 1) else np.nan
    return chi2, p_value, dof, v


def load_non_subscribers() -> pd.DataFrame:
    raw = pd.read_csv("Non-subsFINAL_June 25, 2026_13.44.csv")
    df = pd.DataFrame(index=raw.index)
    df["USERID"] = pd.to_numeric(raw["userId"], errors="coerce")
    df["AGE"] = pd.to_numeric(raw["Q1"], errors="coerce")
    raw = raw[df["AGE"].isna() | (df["AGE"] >= 13)].copy()
    df = df.loc[raw.index].copy()
    df["Segment"] = "Non-subscribers"
    df["AGE"] = pd.to_numeric(raw["Q1"], errors="coerce")
    df["TENURE_NUMERIC"] = pd.to_numeric(raw["Tenure"], errors="coerce")
    df["GENDER_LABEL"] = map_codes(raw["Gender"], {0: "Male", 1: "Female", 2: "Unknown"})
    df["Q2_LABEL"] = map_codes(raw["Q2"], Q2_LABELS)
    df["Q3_LABEL"] = map_codes(raw["Q3"], Q3_LABELS)
    df["Q4_LABEL"] = map_codes(raw["Q4"], Q4_LABELS)
    df["Q5_LABEL"] = map_codes(raw["Q5"], Q5_LABELS)
    df["Q10_SCORE"] = pd.to_numeric(raw["Q17_Combined"], errors="coerce")
    df["Q10_LABEL"] = df["Q10_SCORE"].map({1: "Very Unlikely", 2: "Unlikely", 3: "Maybe", 4: "Likely", 5: "Very Likely"}).fillna("Missing")
    df["Intent Type"] = "Likelihood to subscribe"
    df["Q12_LABEL"] = map_codes(raw["Q26"], FEATURE_LABELS)
    df["Awareness Group"] = np.where(pd.to_numeric(raw["Q4"], errors="coerce").ge(3), "Aware", "Unaware")
    df["Source Row"] = raw.index
    return df


def value_label_map(meta: pyreadstat.metadata_container, column: str) -> dict:
    value_map = meta.variable_value_labels.get(column, {})
    out = {}
    for key, value in value_map.items():
        code = normalize_code(key)
        if code is not None:
            out[code] = value
    return out


def load_sav_segment(path: str, segment: str, intent_type: str) -> pd.DataFrame:
    raw, meta = pyreadstat.read_sav(path, apply_value_formats=False, user_missing=True)
    df = pd.DataFrame(index=raw.index)
    df["USERID"] = raw["userId"]
    df["AGE"] = pd.to_numeric(raw["Q1"], errors="coerce")
    raw = raw[df["AGE"].isna() | (df["AGE"] >= 13)].copy()
    df = df.loc[raw.index].copy()
    df["Segment"] = segment
    df["AGE"] = pd.to_numeric(raw["Q1"], errors="coerce")
    df["TENURE_NUMERIC"] = pd.to_numeric(raw["Tenure"], errors="coerce")
    df["GENDER_LABEL"] = raw["Gender"].map({1: "Female", 2: "Male", 3: "Unknown"}).fillna("Missing")
    df["Q2_LABEL"] = raw["Q2"].map(value_label_map(meta, "Q2")).fillna("Missing")
    df["Q3_LABEL"] = raw["Q3"].map(value_label_map(meta, "Q3")).fillna("Missing")
    df["Q4_LABEL"] = raw["Q4"].map(value_label_map(meta, "Q4")).fillna("Missing")
    df["Q5_LABEL"] = raw["Q5"].map(value_label_map(meta, "Q5")).fillna("Missing")
    df["Q10_SCORE"] = pd.to_numeric(raw["Q10"], errors="coerce")
    df["Q10_LABEL"] = df["Q10_SCORE"].map({1: "Very Unlikely", 2: "Unlikely", 3: "Maybe", 4: "Likely", 5: "Very Likely"}).fillna("Missing")
    df["Intent Type"] = intent_type
    df["Q12_LABEL"] = raw["Q12"].map(value_label_map(meta, "Q12")).fillna("Missing").replace(FEATURE_LABELS)
    return df


def load_segments() -> pd.DataFrame:
    non = load_non_subscribers()
    churn = load_sav_segment(
        "Roblox Plus Churned_July 8, 2026_15.57.sav",
        "Churned subscribers",
        "Likelihood to re-subscribe",
    )
    renewed = load_sav_segment(
        "Roblox Plus Renewed subs_July 8, 2026_16.48.sav",
        "Renewed subscribers",
        "Likelihood to stay subscribed",
    )
    combined = pd.concat([non, churn, renewed], ignore_index=True)
    combined["Age Range"] = pd.cut(
        combined["AGE"],
        bins=[12, 17, 24, 200],
        labels=["13-17", "18-24", "25+"],
        include_lowest=True,
    ).astype("string").fillna("Missing")
    combined["Intent Group"] = pd.cut(
        combined["Q10_SCORE"],
        bins=[0, 2, 3, 5],
        labels=["Low Intent", "Maybe", "High Intent"],
        include_lowest=True,
    ).astype("string").fillna("Missing")
    return combined


def distribution(df: pd.DataFrame, group_cols: list[str], value_col: str, label: str) -> pd.DataFrame:
    out = (
        df[group_cols + [value_col]]
        .fillna("Missing")
        .groupby(group_cols + [value_col], observed=False)
        .size()
        .reset_index(name="Respondents")
        .rename(columns={value_col: label})
    )
    out["Total"] = out.groupby(group_cols, observed=False)["Respondents"].transform("sum")
    out["Share"] = out["Respondents"] / out["Total"]
    return add_ci_columns(out, "Respondents", "Total", "Share")


def overall_distribution(df: pd.DataFrame, value_col: str, label: str) -> pd.DataFrame:
    out = df[value_col].fillna("Missing").value_counts(dropna=False).rename_axis(label).reset_index(name="Respondents")
    out["Total"] = out["Respondents"].sum()
    out["Share"] = out["Respondents"] / out["Total"]
    return add_ci_columns(out, "Respondents", "Total", "Share")


def feature_short(series: pd.Series) -> pd.Series:
    reverse = {v: v for v in FEATURE_LABELS.values()}
    long_map = {
        "Roblox avatar items with particle effects: Get exclusive accessories made by Roblox that glow, sparkle or have moving animations": "Particle effect avatar items",
        "Spawn/despawn effects: Customize how you appear/disappear in games": "Spawn / despawn effects",
        "App themes: Customize Roblox app with color themes": "App themes",
        "Profile frames: Apply a frame to profile": "Profile frames",
        "Rich text and emojis: Use color, gradients and animations for text, including access to a broader set of emojis": "Rich text and emojis",
        "AI-generated avatars & avatar items: Use text to generate original avatars, clothing & accessories": "AI-generated avatars/items",
        "Custom AI-generated avatar backgrounds: Generate background from a text prompt": "AI avatar backgrounds",
    }
    return series.replace(long_map).replace(reverse)


def load_non_rank_distribution() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv("Non-subsFINAL_June 25, 2026_13.44.csv")
    rank_cols = [f"Q15_Combined_{i}" for i in range(1, 6)]
    rows = []
    long_rows = []
    for idx, column in enumerate(rank_cols, start=1):
        benefit = BENEFIT_LABELS[idx]
        ranks = pd.to_numeric(raw[column], errors="coerce")
        counts = ranks.value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
        valid_n = int(ranks.notna().sum())
        row = {"Segment": "Non-subscribers", "Benefit": benefit, "Valid Ranking N": valid_n}
        for rank in range(1, 6):
            count = int(counts.loc[rank])
            share = count / valid_n if valid_n else np.nan
            row[f"#{rank} Count"] = count
            row[f"#{rank} %"] = share
            long_rows.append({"Segment": "Non-subscribers", "Benefit": benefit, "Rank": f"#{rank}", "Count": count, "Percent": share, "Valid Ranking N": valid_n})
        row["Top Box #1 %"] = row["#1 %"]
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(long_rows)


def load_rank_distribution(path: str, segment: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    wide = pd.read_csv(path)
    wide["Segment"] = segment
    for col in ["Benefit"]:
        wide[col] = wide[col].replace(BENEFIT_CANONICAL_LABELS)
    long_rows = []
    for _, row in wide.iterrows():
        for rank in range(1, 6):
            long_rows.append(
                {
                    "Segment": segment,
                    "Benefit": row["Benefit"],
                    "Rank": f"#{rank}",
                    "Count": row[f"#{rank} Count"],
                    "Percent": row[f"#{rank} %"],
                    "Valid Ranking N": row["Valid Ranking N"],
                }
            )
    return wide, pd.DataFrame(long_rows)


def load_non_specific() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv("Non-subsFINAL_June 25, 2026_13.44.csv")
    rows = []
    for code, label in NON_SUB_REASON_LABELS.items():
        column = f"Q19_{code}"
        if column not in raw.columns:
            continue
        selected = raw[column].astype(str).str.strip().eq("1")
        rows.append({"Reason": label, "Respondents": int(selected.sum()), "Total": int(len(raw)), "Share": selected.mean()})
    select_all = add_ci_columns(pd.DataFrame(rows), "Respondents", "Total", "Share").sort_values("Share", ascending=False)

    primary = raw["Q20_Combined"].map(lambda v: NON_SUB_REASON_LABELS.get(normalize_code(v), "Missing"))
    primary_dist = overall_distribution(pd.DataFrame({"reason": primary}), "reason", "Reason").sort_values("Share", ascending=False)
    return select_all, primary_dist


def load_churn_specific() -> pd.DataFrame:
    analysis = pd.read_csv("outputs/analysis_ready_respondent_level.csv")
    q9 = analysis["Q9_LABEL"].fillna("Missing").replace(CHURN_REASON_LABELS)
    return overall_distribution(pd.DataFrame({"reason": q9}), "reason", "Reason").sort_values("Share", ascending=False)


def load_renewed_specific() -> pd.DataFrame:
    analysis = pd.read_csv("renewed_outputs/analysis_ready_respondent_level.csv")
    values = analysis["Q8_LABEL"].fillna("Missing").replace(RENEWED_VALUE_LABELS)
    return overall_distribution(pd.DataFrame({"value": values}), "value", "Value Driver").sort_values("Share", ascending=False)


def normalize_subscription_reason(value: object) -> str:
    text = str(value).strip()
    return SUBSCRIPTION_REASON_LABELS.get(text, Q7_LABELS.get(text, text if text else "Missing"))


def load_subscription_reasons() -> tuple[pd.DataFrame, pd.DataFrame]:
    select_frames = []
    primary_frames = []
    sources = [
        ("Churned subscribers", Path("outputs")),
        ("Renewed subscribers", Path("renewed_outputs")),
    ]
    for segment, output_dir in sources:
        q6 = pd.read_csv(output_dir / "q6_initial_subscription_motivations.csv")
        q6["Segment"] = segment
        q6["Reason"] = q6["motivation"].map(normalize_subscription_reason)
        q6 = q6.rename(columns={"selected_n": "Respondents", "respondents": "Total", "selected_pct": "Share"})
        select_frames.append(q6[["Segment", "Reason", "Respondents", "Total", "Share"]])

        q7 = pd.read_csv(output_dir / "q7_main_subscription_reason_distribution.csv")
        q7["Segment"] = segment
        q7["Reason"] = q7["main_subscription_reason"].map(normalize_subscription_reason)
        q7 = q7.rename(columns={"n": "Respondents", "pct": "Share"})
        q7["Total"] = q7["Respondents"].sum()
        primary_frames.append(q7[["Segment", "Reason", "Respondents", "Total", "Share"]])

    select_all = add_ci_columns(pd.concat(select_frames, ignore_index=True), "Respondents", "Total", "Share")
    primary = add_ci_columns(pd.concat(primary_frames, ignore_index=True), "Respondents", "Total", "Share")
    return select_all, primary


def main() -> None:
    combined = load_segments()
    total_n = len(combined)
    segment_counts = combined["Segment"].value_counts().reindex(SEGMENT_ORDER)

    # Core distributions
    sentiment = distribution(combined, ["Segment"], "Q2_LABEL", "Roblox Sentiment")
    familiarity = distribution(combined, ["Segment"], "Q4_LABEL", "Plus Familiarity")
    motivation = distribution(combined, ["Segment"], "Q3_LABEL", "Primary Roblox Motivation")
    discovery = distribution(combined[combined["Q5_LABEL"].ne("Missing")], ["Segment"], "Q5_LABEL", "Discovery Channel")
    intent = distribution(combined[combined["Q10_LABEL"].ne("Missing")], ["Segment"], "Q10_LABEL", "Likelihood")
    intent_group = distribution(combined[combined["Intent Group"].ne("Missing")], ["Segment"], "Intent Group", "Intent Group")
    future = distribution(combined[combined["Q12_LABEL"].ne("Missing")], ["Segment"], "Q12_LABEL", "Future Feature")
    future["Future Feature"] = feature_short(future["Future Feature"])

    sample_summary = (
        combined.groupby("Segment", observed=False)
        .agg(
            Respondents=("Segment", "size"),
            Mean_Age=("AGE", "mean"),
            Median_Age=("AGE", "median"),
            Mean_Tenure_Days=("TENURE_NUMERIC", "mean"),
            Median_Tenure_Days=("TENURE_NUMERIC", "median"),
        )
        .reindex(SEGMENT_ORDER)
        .reset_index()
    )
    sample_table = sample_summary.assign(
        Mean_Age=sample_summary["Mean_Age"].map(lambda v: num(v, 2)),
        Median_Age=sample_summary["Median_Age"].map(lambda v: num(v, 0)),
        Mean_Tenure_Days=sample_summary["Mean_Tenure_Days"].map(lambda v: num(v, 0)),
        Median_Tenure_Days=sample_summary["Median_Tenure_Days"].map(lambda v: num(v, 0)),
    ).rename(columns={"Mean_Age": "Mean age", "Median_Age": "Median age", "Mean_Tenure_Days": "Mean tenure days", "Median_Tenure_Days": "Median tenure days"})

    sentiment_table = pd.crosstab(combined["Segment"], combined["Q2_LABEL"]).reindex(index=SEGMENT_ORDER, columns=SENTIMENT_ORDER).fillna(0)
    sentiment_chi, sentiment_p, sentiment_dof, sentiment_v = cramers_v(sentiment_table)
    feature_table = (
        future.pivot_table(
            index="Segment",
            columns="Future Feature",
            values="Respondents",
            aggfunc="sum",
            fill_value=0,
            observed=False,
        )
        .reindex(index=SEGMENT_ORDER)
        .fillna(0)
    )
    feature_chi, feature_p, feature_dof, feature_v = cramers_v(feature_table)
    intent_table = pd.crosstab(combined["Segment"], combined["Q10_LABEL"]).reindex(index=SEGMENT_ORDER, columns=LIKELIHOOD_ORDER).fillna(0)
    intent_chi, intent_p, intent_dof, intent_v = cramers_v(intent_table)

    # Charts
    fig_sentiment = px.bar(
        sentiment,
        x="Roblox Sentiment",
        y="Share",
        color="Segment",
        category_orders={"Roblox Sentiment": SENTIMENT_ORDER, "Segment": SEGMENT_ORDER},
        color_discrete_map=SEGMENT_COLORS,
        barmode="group",
        title="Roblox Platform Sentiment by Segment",
        text=sentiment["Share"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_sentiment.update_layout(yaxis_tickformat=".0%")
    keep_labels_clear(fig_sentiment)

    fig_familiarity = px.bar(
        familiarity,
        x="Plus Familiarity",
        y="Share",
        color="Segment",
        category_orders={"Plus Familiarity": FAMILIARITY_ORDER, "Segment": SEGMENT_ORDER},
        color_discrete_map=SEGMENT_COLORS,
        barmode="group",
        title="Roblox Plus Familiarity by Segment",
        text=familiarity["Share"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_familiarity.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-25, xaxis_tickfont_size=10, margin=dict(b=120))
    keep_labels_clear(fig_familiarity)

    top_motivation_categories = motivation.groupby("Primary Roblox Motivation")["Respondents"].sum().sort_values(ascending=False).head(7).index
    fig_motivation = px.bar(
        motivation[motivation["Primary Roblox Motivation"].isin(top_motivation_categories)],
        x="Primary Roblox Motivation",
        y="Share",
        color="Segment",
        category_orders={"Segment": SEGMENT_ORDER},
        color_discrete_map=SEGMENT_COLORS,
        barmode="group",
        title="Primary Reason for Using Roblox by Segment",
        text=motivation[motivation["Primary Roblox Motivation"].isin(top_motivation_categories)]["Share"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_motivation.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30, xaxis_tickfont_size=10, margin=dict(b=130))
    keep_labels_clear(fig_motivation)

    fig_intent = px.bar(
        intent,
        x="Segment",
        y="Share",
        color="Likelihood",
        category_orders={"Segment": SEGMENT_ORDER, "Likelihood": LIKELIHOOD_ORDER},
        color_discrete_map=LIKELIHOOD_COLORS,
        title="Subscription Intent by Segment",
        labels={"Share": "Share within segment"},
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_intent.update_layout(barmode="stack", yaxis_tickformat=".0%")

    fig_intent_group = px.bar(
        intent_group,
        x="Segment",
        y="Share",
        color="Intent Group",
        category_orders={"Segment": SEGMENT_ORDER, "Intent Group": ["Low Intent", "Maybe", "High Intent"]},
        title="Low / Maybe / High Intent Summary",
        labels={"Share": "Share within segment"},
        text=intent_group["Share"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_intent_group.update_layout(barmode="stack", yaxis_tickformat=".0%")
    keep_labels_clear(fig_intent_group)

    feature_order = future.groupby("Future Feature")["Respondents"].sum().sort_values(ascending=False).index.tolist()
    fig_future = px.bar(
        future,
        x="Future Feature",
        y="Share",
        color="Segment",
        category_orders={"Future Feature": feature_order, "Segment": SEGMENT_ORDER},
        color_discrete_map=SEGMENT_COLORS,
        barmode="group",
        title="Future Plus Feature Demand by Segment",
        text=future["Share"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_future.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30, xaxis_tickfont_size=10, margin=dict(b=130))
    keep_labels_clear(fig_future)

    top_discovery = discovery.groupby("Discovery Channel")["Respondents"].sum().sort_values(ascending=False).head(8).index
    fig_discovery = px.bar(
        discovery[discovery["Discovery Channel"].isin(top_discovery)],
        x="Discovery Channel",
        y="Share",
        color="Segment",
        category_orders={"Segment": SEGMENT_ORDER},
        color_discrete_map=SEGMENT_COLORS,
        barmode="group",
        title="Top Discovery Channels by Segment",
        text=discovery[discovery["Discovery Channel"].isin(top_discovery)]["Share"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_discovery.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30, xaxis_tickfont_size=10, margin=dict(b=130))
    keep_labels_clear(fig_discovery)

    non_wide, non_long = load_non_rank_distribution()
    churn_wide, churn_long = load_rank_distribution("outputs/q8_benefit_ranking_distribution.csv", "Churned subscribers")
    renewed_wide, renewed_long = load_rank_distribution("renewed_outputs/q8_benefit_ranking_distribution.csv", "Renewed subscribers")
    rank_wide = pd.concat([non_wide, churn_wide, renewed_wide], ignore_index=True)
    rank_long = pd.concat([non_long, churn_long, renewed_long], ignore_index=True)
    rank_wide["Benefit"] = rank_wide["Benefit"].replace(BENEFIT_CANONICAL_LABELS)
    rank_long["Benefit"] = rank_long["Benefit"].replace(BENEFIT_CANONICAL_LABELS)
    rank_long = (
        rank_long.groupby(["Segment", "Benefit", "Rank"], observed=False, as_index=False)
        .agg({"Count": "sum", "Valid Ranking N": "first"})
    )
    rank_long["Percent"] = rank_long["Count"] / rank_long["Valid Ranking N"]
    rank_long_ci = add_ci_columns(rank_long, "Count", "Valid Ranking N", "Percent")
    fig_rank = px.bar(
        rank_long_ci,
        x="Rank",
        y="Percent",
        color="Benefit",
        facet_col="Segment",
        category_orders={
            "Segment": SEGMENT_ORDER,
            "Rank": ["#1", "#2", "#3", "#4", "#5"],
            "Benefit": list(BENEFIT_COLORS),
        },
        color_discrete_map=BENEFIT_COLORS,
        title="Benefit Ranking Distribution Across Segments",
        labels={"Percent": "Share of valid rankings"},
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_rank.update_layout(barmode="group", yaxis_tickformat=".0%")

    rank_top_ci = rank_long_ci[rank_long_ci["Rank"].eq("#1")].copy()
    rank_top_ci["Benefit"] = pd.Categorical(rank_top_ci["Benefit"], categories=list(BENEFIT_COLORS), ordered=True)
    rank_top_ci = rank_top_ci.sort_values(["Segment", "Benefit"])
    fig_rank_top = px.bar(
        rank_top_ci,
        x="Benefit",
        y="Percent",
        color="Benefit",
        facet_col="Segment",
        category_orders={"Segment": SEGMENT_ORDER, "Benefit": list(BENEFIT_COLORS)},
        color_discrete_map=BENEFIT_COLORS,
        title="Top-Ranked Plus Benefit Share by Segment",
        labels={"Percent": "Share ranking benefit #1"},
        text=rank_top_ci["Percent"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_rank_top.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30, xaxis_tickfont_size=10, margin=dict(b=120))
    keep_labels_clear(fig_rank_top)

    top_rank_heatmap = (
        rank_long_ci[rank_long_ci["Rank"].eq("#1")]
        .pivot(index="Benefit", columns="Segment", values="Percent")
        .reindex(index=list(BENEFIT_COLORS), columns=SEGMENT_ORDER)
    )
    fig_rank_top_heatmap = px.imshow(
        top_rank_heatmap,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Heatmap: Share Ranking Each Benefit #1",
        labels={"x": "Segment", "y": "Benefit", "color": "#1 share"},
    )
    fig_rank_top_heatmap.update_layout(coloraxis_colorbar_tickformat=".0%")

    bottom_rank_heatmap = (
        rank_long_ci[rank_long_ci["Rank"].eq("#5")]
        .pivot(index="Benefit", columns="Segment", values="Percent")
        .reindex(index=list(BENEFIT_COLORS), columns=SEGMENT_ORDER)
    )
    fig_rank_bottom_heatmap = px.imshow(
        bottom_rank_heatmap,
        text_auto=".0%",
        aspect="auto",
        color_continuous_scale="Reds",
        title="Heatmap: Share Ranking Each Benefit #5",
        labels={"x": "Segment", "y": "Benefit", "color": "#5 share"},
    )
    fig_rank_bottom_heatmap.update_layout(coloraxis_colorbar_tickformat=".0%")

    top_rank_table = (
        rank_wide.sort_values(["Segment", "#1 %"], ascending=[True, False])
        .groupby("Segment", as_index=False)
        .head(1)[["Segment", "Benefit", "#1 %", "#5 %"]]
        .assign(**{"#1 %": lambda d: d["#1 %"].map(pct), "#5 %": lambda d: d["#5 %"].map(pct)})
        .rename(columns={"Benefit": "Top #1 ranked benefit"})
    )

    subscribe_select_all, subscribe_primary = load_subscription_reasons()
    subscribe_reason_order = (
        subscribe_select_all.groupby("Reason")["Respondents"].sum().sort_values(ascending=False).index.tolist()
    )
    subscribe_select_all["Reason"] = pd.Categorical(
        subscribe_select_all["Reason"],
        categories=subscribe_reason_order,
        ordered=True,
    )
    subscribe_select_all = subscribe_select_all.sort_values(["Reason", "Segment"])
    fig_subscribe_select = px.bar(
        subscribe_select_all,
        x="Reason",
        y="Share",
        color="Segment",
        barmode="group",
        category_orders={"Segment": ["Churned subscribers", "Renewed subscribers"], "Reason": subscribe_reason_order},
        color_discrete_map=SEGMENT_COLORS,
        title="Reasons for Initially Subscribing: Select All",
        text=subscribe_select_all["Share"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_subscribe_select.update_layout(
        yaxis_tickformat=".0%",
        xaxis_tickangle=-30,
        xaxis_tickfont_size=10,
        xaxis_categoryorder="array",
        xaxis_categoryarray=subscribe_reason_order,
        margin=dict(b=130),
    )
    keep_labels_clear(fig_subscribe_select)

    primary_reason_order = (
        subscribe_primary.groupby("Reason")["Respondents"].sum().sort_values(ascending=False).index.tolist()
    )
    subscribe_primary["Reason"] = pd.Categorical(
        subscribe_primary["Reason"],
        categories=primary_reason_order,
        ordered=True,
    )
    subscribe_primary = subscribe_primary.sort_values(["Reason", "Segment"])
    fig_subscribe_primary = px.bar(
        subscribe_primary,
        x="Reason",
        y="Share",
        color="Segment",
        barmode="group",
        category_orders={"Segment": ["Churned subscribers", "Renewed subscribers"], "Reason": primary_reason_order},
        color_discrete_map=SEGMENT_COLORS,
        title="Primary Reason for Initially Subscribing: Choose One",
        text=subscribe_primary["Share"].map(lambda v: pct(v, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_subscribe_primary.update_layout(
        yaxis_tickformat=".0%",
        xaxis_tickangle=-30,
        xaxis_tickfont_size=10,
        xaxis_categoryorder="array",
        xaxis_categoryarray=primary_reason_order,
        margin=dict(b=130),
    )
    keep_labels_clear(fig_subscribe_primary)

    primary_reason_table = subscribe_primary.pivot_table(
        index="Segment",
        columns="Reason",
        values="Respondents",
        aggfunc="sum",
        fill_value=0,
        observed=False,
    )
    _, subscribe_primary_p, _, subscribe_primary_v = cramers_v(primary_reason_table)
    top_subscribe_reasons = (
        subscribe_primary.sort_values(["Segment", "Share"], ascending=[True, False])
        .groupby("Segment", as_index=False)
        .head(1)[["Segment", "Reason", "Share"]]
    )

    non_select_all, non_primary = load_non_specific()
    fig_non_select = px.bar(
        non_select_all,
        x="Reason",
        y="Share",
        title="Non-Subscribers: Reasons for Not Subscribing (Select All)",
        text=non_select_all["Share"].map(lambda v: pct(v)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_non_select.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35, xaxis_tickfont_size=10, margin=dict(b=150))
    keep_labels_clear(fig_non_select)

    fig_non_primary = px.bar(
        non_primary,
        x="Reason",
        y="Share",
        title="Non-Subscribers: Primary Reason for Not Subscribing",
        text=non_primary["Share"].map(lambda v: pct(v)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_non_primary.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35, xaxis_tickfont_size=10, margin=dict(b=150))
    keep_labels_clear(fig_non_primary)

    churn_reasons = load_churn_specific()
    fig_churn = px.bar(
        churn_reasons,
        x="Reason",
        y="Share",
        title="Churned Subscribers: Main Reason for Not Renewing",
        text=churn_reasons["Share"].map(lambda v: pct(v)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_churn.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30, xaxis_tickfont_size=10, margin=dict(b=130))
    keep_labels_clear(fig_churn)

    renewed_values = load_renewed_specific()
    fig_renewed = px.bar(
        renewed_values,
        x="Value Driver",
        y="Share",
        title="Renewed Subscribers: Most Valuable Part of Roblox Plus",
        text=renewed_values["Share"].map(lambda v: pct(v)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_renewed.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30, xaxis_tickfont_size=10, margin=dict(b=140))
    keep_labels_clear(fig_renewed)

    high_intent = (
        intent_group[intent_group["Intent Group"].eq("High Intent")]
        .set_index("Segment")
        .reindex(SEGMENT_ORDER)["Share"]
    )
    top_features = (
        future.sort_values(["Segment", "Share"], ascending=[True, False])
        .groupby("Segment", as_index=False)
        .head(1)[["Segment", "Future Feature", "Share"]]
    )
    top_features["Share"] = top_features["Share"].map(pct)
    top_values = renewed_values.iloc[0]
    top_churn = churn_reasons.iloc[0]
    top_non = non_primary.iloc[0]

    hero = f"""
    <header class="hero">
      <p class="eyebrow">Roblox Plus Post-Launch Readout · Three Segment Comparison</p>
      <h1>Roblox Plus Segment Comparison Report</h1>
      <p class="lede">This report compares non-subscribers, churned subscribers, and renewed subscribers on shared baseline questions, subscription intent, current benefit value, and future roadmap demand. Segment-specific tabs cover the questions unique to each user group.</p>
      <div class="kpi-grid">
        {make_kpi("Total Sample", f"n={total_n:,}", "Across all three segment surveys")}
        {make_kpi("Non-Sub High Intent", pct(high_intent["Non-subscribers"]), "Likely / very likely to subscribe")}
        {make_kpi("Churned High Intent", pct(high_intent["Churned subscribers"]), "Likely / very likely to re-subscribe")}
        {make_kpi("Renewed High Intent", pct(high_intent["Renewed subscribers"]), "Likely / very likely to stay subscribed")}
      </div>
    </header>
    """

    executive_tab = f"""
    <div class="takeaway">
      <h2>Executive Summary</h2>
      <ol>
        <li><strong>Subscription intent separates the segments sharply:</strong> high intent is {pct(high_intent["Renewed subscribers"])} among renewed subscribers, {pct(high_intent["Churned subscribers"])} among churned subscribers, and {pct(high_intent["Non-subscribers"])} among non-subscribers.</li>
        <li><strong>Shared roadmap demand is not identical:</strong> the top future feature is {top_features.iloc[0]["Future Feature"]} for {top_features.iloc[0]["Segment"]}, {top_features.iloc[1]["Future Feature"]} for {top_features.iloc[1]["Segment"]}, and {top_features.iloc[2]["Future Feature"]} for {top_features.iloc[2]["Segment"]}.</li>
        <li><strong>Each segment has a distinct business problem:</strong> non-subscribers need conversion barriers reduced, churned users need value gaps repaired, and renewed users need reinforcement of the benefits they already value.</li>
      </ol>
      <p>Chi-square tests show segment differences are significant for platform sentiment (p={sentiment_p:.2e}, Cramer's V={sentiment_v:.3f}), subscription intent (p={intent_p:.2e}, Cramer's V={intent_v:.3f}), and future feature demand (p={feature_p:.2e}, Cramer's V={feature_v:.3f}). These are statistically clear segment differences, with effect sizes ranging from small to moderate depending on the outcome.</p>
    </div>
    {fig_html(fig_intent, include_plotlyjs=True)}
    {fig_html(fig_intent_group)}
    """

    baseline_tab = f"""
    {section("Sample Characteristics", f'''
      <p>The three samples are similar in age profile but differ meaningfully in product relationship: non-subscribers are prospects, churned subscribers are former trial/adopters, and renewed subscribers are the retained subscriber base.</p>
      {table_html(sample_table)}
    ''')}
    {section("Platform Sentiment and Plus Familiarity", f'''
      <p>Platform sentiment and Plus familiarity provide the baseline context for conversion, churn, and retention. Familiarity is expectedly lowest among non-subscribers and highest among subscriber segments.</p>
      {fig_html(fig_sentiment)}
      {fig_html(fig_familiarity)}
    ''')}
    {section("Core Roblox Motivation and Discovery", f'''
      <p>Entertainment and social use cases dominate across segments, while discovery channels show where users first encountered Roblox Plus.</p>
      {fig_html(fig_motivation)}
      {fig_html(fig_discovery)}
    ''')}
    """

    ranking_tab = f"""
    {section("Current Plus Benefit Ranking Across Segments", f'''
      <p>The ranking exercise shows how the current Plus bundle is valued before purchase, after churn, and among retained subscribers. To reduce visual clutter, this view focuses first on each benefit’s #1 share, then uses heatmaps to show both top-rank pull and bottom-rank rejection.</p>
      {fig_html(fig_rank_top)}
      {fig_html(fig_rank_top_heatmap)}
      {fig_html(fig_rank_bottom_heatmap)}
      {table_html(top_rank_table)}
    ''')}
    """

    subscribing_tab = f"""
    {section("Reasons for Subscribing: Churned vs. Renewed", f'''
      <p>This tab compares why former subscribers and retained subscribers initially signed up for Roblox Plus. The select-all chart captures the broad motivation set, while the single-choice chart identifies the primary hook.</p>
      <p><strong>Key insight:</strong> churned subscribers were more trial-led, with <strong>{top_subscribe_reasons[top_subscribe_reasons["Segment"].eq("Churned subscribers")]["Reason"].iloc[0]}</strong> as their top primary reason at {pct(top_subscribe_reasons[top_subscribe_reasons["Segment"].eq("Churned subscribers")]["Share"].iloc[0])}. Renewed subscribers were more value-led, with <strong>{top_subscribe_reasons[top_subscribe_reasons["Segment"].eq("Renewed subscribers")]["Reason"].iloc[0]}</strong> as their top primary reason at {pct(top_subscribe_reasons[top_subscribe_reasons["Segment"].eq("Renewed subscribers")]["Share"].iloc[0])}. The primary-reason mix differs significantly between churned and renewed subscribers (chi-square p={subscribe_primary_p:.2e}, Cramer's V={subscribe_primary_v:.3f}).</p>
      {fig_html(fig_subscribe_select)}
      {fig_html(fig_subscribe_primary)}
    ''')}
    """

    roadmap_tab = f"""
    {section("Future Plus Feature Demand Across Segments", f'''
      <p>Future feature choice differs significantly by segment (chi-square p={feature_p:.2e}). Non-subscribers lean more toward conversion catalysts such as spawn/despawn effects and app themes, while renewed users are especially strong on particle effect avatar items.</p>
      {fig_html(fig_future)}
      {table_html(top_features.rename(columns={"Future Feature": "Top future feature", "Share": "Pick rate"}))}
    ''')}
    """

    non_tab = f"""
    {section("Non-Subscriber Conversion Barriers", f'''
      <p>The non-subscriber path is dominated by subscription friction, payment access, affordability, and uncertainty about practical value. The primary-reason chart separates the single biggest barrier from the broader select-all barrier set.</p>
      {fig_html(fig_non_select)}
      {fig_html(fig_non_primary)}
      <p><strong>Key insight:</strong> the top primary reason is {top_non["Reason"]} at {pct(top_non["Share"])}, while select-all barriers show the broader set of objections users hold simultaneously.</p>
    ''')}
    """

    churn_tab = f"""
    {section("Churned Subscriber Deep Dive", f'''
      <p>Churned subscribers already tried Plus, so the diagnostic focus is why the current bundle failed to sustain subscription. The chart below shows the main structured reason for not renewing.</p>
      {fig_html(fig_churn)}
      <p><strong>Key insight:</strong> the top churn reason is {top_churn["Reason"]} at {pct(top_churn["Share"])}, making it the clearest repair opportunity for winback messaging and bundle design.</p>
    ''')}
    """

    renewed_tab = f"""
    {section("Renewed Subscriber Deep Dive", f'''
      <p>Renewed subscribers show what is already working in the Plus bundle. This tab focuses on the one benefit respondents say is most valuable.</p>
      {fig_html(fig_renewed)}
      <p><strong>Key insight:</strong> {top_values["Value Driver"]} are the dominant value driver at {pct(top_values["Share"])}, followed by {renewed_values.iloc[1]["Value Driver"]} at {pct(renewed_values.iloc[1]["Share"])}. Retention messaging should reinforce these tangible benefits while testing whether future cosmetic features can add emotional upside.</p>
    ''')}
    """

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roblox Plus Three Segment Comparison Report</title>
  <style>
    :root {{
      --bg: #f6f7fb;
      --paper: #ffffff;
      --ink: #14171f;
      --muted: #5b6472;
      --line: #dfe3ea;
      --accent: #2f5fd0;
      --accent-soft: #eef3ff;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.55;
    }}
    .page {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 56px; }}
    .hero, .tab-panel, .report-section, .takeaway {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    .hero {{ padding: 36px; margin-bottom: 18px; }}
    .eyebrow {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    h1 {{ margin: 0; font-size: 34px; line-height: 1.15; letter-spacing: -0.03em; }}
    h2 {{ margin: 0 0 12px; font-size: 22px; letter-spacing: -0.02em; }}
    .lede {{ max-width: 860px; color: var(--muted); font-size: 17px; margin: 14px 0 28px; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    .kpi-card {{ border: 1px solid var(--line); border-radius: 14px; padding: 16px; background: var(--accent-soft); }}
    .kpi-label {{ color: var(--muted); font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .05em; }}
    .kpi-value {{ margin-top: 8px; font-size: 23px; font-weight: 800; line-height: 1.15; }}
    .kpi-note {{ margin-top: 8px; color: var(--muted); font-size: 13px; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 18px 0; }}
    .tab-button {{
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    .tab-button.active {{ background: var(--ink); color: white; border-color: var(--ink); }}
    .tab-panel {{ display: none; padding: 24px; }}
    .tab-panel.active {{ display: block; }}
    .takeaway, .report-section {{ padding: 22px; margin-bottom: 18px; }}
    .takeaway p, .report-section p {{ color: var(--muted); margin-top: 0; }}
    table.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; background: white; }}
    .data-table th {{ text-align: left; color: var(--muted); background: #f3f5f9; border-bottom: 1px solid var(--line); padding: 10px; }}
    .data-table td {{ border-bottom: 1px solid var(--line); padding: 10px; vertical-align: top; }}
    .footer {{ color: var(--muted); font-size: 12px; margin-top: 24px; text-align: center; }}
    @media (max-width: 860px) {{
      .kpi-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 28px; }}
      .hero {{ padding: 24px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    {hero}
    <nav class="tabs" aria-label="Report sections">
      <button class="tab-button active" data-tab="executive">Executive Summary</button>
      <button class="tab-button" data-tab="baseline">Shared Baseline</button>
      <button class="tab-button" data-tab="subscribing">Reasons for Subscribing</button>
      <button class="tab-button" data-tab="ranking">Benefit Rankings</button>
      <button class="tab-button" data-tab="roadmap">Future Roadmap</button>
      <button class="tab-button" data-tab="non-subs">Non-Subscriber Barriers</button>
      <button class="tab-button" data-tab="churned">Churned Deep Dive</button>
      <button class="tab-button" data-tab="renewed">Renewed Deep Dive</button>
    </nav>
    <section id="executive" class="tab-panel active">{executive_tab}</section>
    <section id="baseline" class="tab-panel">{baseline_tab}</section>
    <section id="subscribing" class="tab-panel">{subscribing_tab}</section>
    <section id="ranking" class="tab-panel">{ranking_tab}</section>
    <section id="roadmap" class="tab-panel">{roadmap_tab}</section>
    <section id="non-subs" class="tab-panel">{non_tab}</section>
    <section id="churned" class="tab-panel">{churn_tab}</section>
    <section id="renewed" class="tab-panel">{renewed_tab}</section>
    <div class="footer">Sources: Roblox Plus non-subscriber CSV, churned subscriber SPSS survey, renewed subscriber SPSS survey · Chart error bars show 95% Wilson confidence intervals for proportions.</div>
  </main>
  <script>
    document.querySelectorAll(".tab-button").forEach((button) => {{
      button.addEventListener("click", () => {{
        const tab = button.dataset.tab;
        document.querySelectorAll(".tab-button").forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(tab).classList.add("active");
        window.dispatchEvent(new Event("resize"));
      }});
    }});
  </script>
</body>
</html>
"""
    REPORT_PATH.write_text(html, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
