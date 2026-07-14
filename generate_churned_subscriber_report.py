#!/usr/bin/env python3
"""
Generate a static HTML readout for the Roblox Plus churned subscriber survey.

Run:
    python generate_churned_subscriber_report.py

Output:
    roblox_plus_churned_subscriber_report.html
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio


OUTPUT_DIR = Path("outputs")
REPORT_PATH = Path("roblox_plus_churned_subscriber_report.html")

LIKELIHOOD_ORDER = ["Very Unlikely", "Unlikely", "Maybe", "Likely", "Very Likely"]
SENTIMENT_ORDER = ["Love it", "Like it", "Neutral", "Dislike it", "Hate it"]
RETENTION_ORDER = ["Low Intent", "Conditional Intent", "High Intent"]
LIKELIHOOD_COLORS = {
    "Very Unlikely": "#c0392b",
    "Unlikely": "#f28e8e",
    "Maybe": "#9e9e9e",
    "Likely": "#8fd19e",
    "Very Likely": "#2e8b57",
}

FEATURE_SHORT_NAMES = {
    "Roblox avatar items with particle effects: Get exclusive accessories made by Roblox that glow, sparkle or have moving animations": "Particle effect avatar items",
    "Spawn/despawn effects: Customize how you appear/disappear in games": "Spawn / despawn effects",
    "App themes: Customize Roblox app with color themes": "App themes",
    "Profile frames: Apply a frame to profile": "Profile frames",
    "Rich text and emojis: Use color, gradients and animations for text, including access to a broader set of emojis": "Rich text and emojis",
    "AI-generated avatars & avatar items: Use text to generate original avatars, clothing & accessories": "AI-generated avatars/items",
    "Custom AI-generated avatar backgrounds: Generate background from a text prompt": "AI avatar backgrounds",
}

Q7_LABELS = {
    "1.0": "10-20% item discounts",
    "2.0": "Free & unlimited private servers",
    "3.0": "Free Robux transfers",
    "4.0": "Exclusive Plus badge",
    "5.0": "Trade & resell avatar items",
    "6.0": "Publish avatar items",
    "7.0": "Wanted to try it out",
    "8.0": "Free trial",
    "9.0": "Other",
}

Q6_SHORT_NAMES = {
    "WhySub_Free Trial": "Free trial",
    "WhySub_Private servers": "Private servers",
    "WhySub_Try it out": "Wanted to try it out",
    "WhySub_Discount": "Item discounts",
    "WhySub_Robux transfers": "Robux transfers",
    "WhySub_Plus badge": "Plus badge",
    "WhySub_Trade & resell": "Trade & resell",
    "WhySub_Publish avatar items": "Publish avatar items",
    "WhySub_Other": "Other",
}

Q8_SHORT_NAMES = {
    "Rank_Publish avatar items": "Publish avatar items",
    "Rank_Trade & resell": "Trade & resell",
    "Rank_Robux transfers": "Robux transfers",
    "Rank_Private servers": "Private servers",
    "Rank_Discount": "Item discounts",
}


def read_csv(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    return pd.read_csv(path)


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 0) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def wilson_ci(successes: float, total: float, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return Wilson 95% confidence interval bounds for a binomial proportion."""
    if pd.isna(successes) or pd.isna(total) or total <= 0:
        return (pd.NA, pd.NA)

    p = successes / total
    denominator = 1 + (z**2 / total)
    center = (p + (z**2 / (2 * total))) / denominator
    half_width = z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5) / denominator
    return (max(0, center - half_width), min(1, center + half_width))


def add_ci_columns(
    df: pd.DataFrame,
    success_column: str,
    total_column: str,
    proportion_column: str,
) -> pd.DataFrame:
    """Add Plotly-ready 95% CI error deltas around an existing proportion column."""
    working = df.copy()
    bounds = working.apply(
        lambda row: wilson_ci(row[success_column], row[total_column]),
        axis=1,
        result_type="expand",
    )
    working["ci_lower"] = bounds[0]
    working["ci_upper"] = bounds[1]
    working["ci_error_plus"] = working["ci_upper"] - working[proportion_column]
    working["ci_error_minus"] = working[proportion_column] - working["ci_lower"]
    return working


def short_feature(label: str) -> str:
    return FEATURE_SHORT_NAMES.get(str(label), str(label))


def fig_html(fig, include_plotlyjs: bool = False) -> str:
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif", size=13),
        margin=dict(l=40, r=24, t=70, b=45),
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
    )


def keep_percent_labels_off_error_bars(fig) -> None:
    """Place percent labels inside bars so they do not overlap CI whiskers."""
    fig.update_traces(
        textposition="inside",
        insidetextanchor="middle",
        cliponaxis=False,
    )
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")


def table_html(df: pd.DataFrame, classes: str = "data-table") -> str:
    if df.empty:
        return ""
    safe = df.copy()
    return safe.to_html(index=False, escape=True, classes=classes, border=0)


def section(title: str, body: str) -> str:
    return f"""
    <section class="report-section">
      <h2>{escape(title)}</h2>
      {body}
    </section>
    """


def make_kpi(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{escape(label)}</div>
      <div class="kpi-value">{escape(value)}</div>
      <div class="kpi-note">{escape(note)}</div>
    </div>
    """


def main() -> None:
    analysis = read_csv("analysis_ready_respondent_level.csv")
    age_tenure = read_csv("demographics_age_tenure_summary.csv").rename(columns={"Unnamed: 0": "metric"})
    gender = read_csv("demographics_gender_distribution.csv")
    q2_q10_counts = read_csv("q2_by_q10_crosstab_counts.csv").rename(columns={"row_0": "Platform Sentiment"})
    q2_q10_pct = read_csv("q2_by_q10_crosstab_row_pct.csv").rename(columns={"row_0": "Platform Sentiment"})
    q2_stats = read_csv("q2_q10_association_stats.csv")
    q3 = read_csv("q3_primary_reason_for_using_roblox.csv")
    q5 = read_csv("q5_conversion_channel_distribution.csv")
    q6 = read_csv("q6_initial_subscription_motivations.csv")
    q7 = read_csv("q7_main_subscription_reason_distribution.csv")
    q8 = read_csv("q8_feature_rank_summary.csv")
    q8_rank_dist = read_csv("q8_benefit_ranking_distribution.csv")
    q8_rank_long = read_csv("q8_benefit_ranking_long.csv")
    q9 = read_csv("q9_churn_reason_by_retention_group_row_pct.csv")
    churn_keywords = read_csv("churn_text_keyword_share_of_voice.csv")
    q12 = read_csv("q12_feature_pick_rates.csv")
    q13_backlash = read_csv("q13_backlash_keywords_by_feature.csv")

    total_n = len(analysis)
    age_row = age_tenure.loc[age_tenure["metric"].eq("AGE")].iloc[0]
    tenure_row = age_tenure.loc[age_tenure["metric"].eq("TENURE_NUMERIC")].iloc[0]
    high_intent_n = int(analysis["Q10_SCORE"].isin([4, 5]).sum())
    low_intent_n = int(analysis["Q10_SCORE"].isin([1, 2]).sum())
    maybe_n = int(analysis["Q10_SCORE"].eq(3).sum())

    q9_total = analysis["Q9_LABEL"].fillna("Missing").value_counts(normalize=True)
    top_churn_reason = q9_total.index[0]
    top_churn_pct = q9_total.iloc[0]
    top_feature = q12.iloc[0]["feature"]
    top_feature_pct = q12.iloc[0]["pick_rate"]
    top_motivation = q6.iloc[0]["motivation"]
    top_motivation_pct = q6.iloc[0]["selected_pct"]

    q6["motivation_short"] = q6["motivation"].map(Q6_SHORT_NAMES).fillna(q6["motivation"])
    q7["reason_label"] = q7["main_subscription_reason"].astype(str).map(Q7_LABELS).fillna(q7["main_subscription_reason"].astype(str))
    q8["feature_short"] = q8["feature"].map(Q8_SHORT_NAMES).fillna(q8["feature"])
    q12["feature_short"] = q12["feature"].map(short_feature)
    q13_backlash["feature_short"] = q13_backlash["feature"].map(short_feature)
    feature_order = q12["feature_short"].tolist()

    q10_overall = (
        analysis["Q10_LABEL"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("Likelihood to Re-subscribe")
        .reset_index(name="Respondents")
    )
    q10_overall["Share"] = q10_overall["Respondents"] / q10_overall["Respondents"].sum()
    q10_overall["Total"] = q10_overall["Respondents"].sum()
    q10_overall = add_ci_columns(q10_overall, "Respondents", "Total", "Share")
    fig_q10_overall = px.bar(
        q10_overall,
        x="Likelihood to Re-subscribe",
        y="Share",
        color="Likelihood to Re-subscribe",
        category_orders={"Likelihood to Re-subscribe": LIKELIHOOD_ORDER},
        color_discrete_map=LIKELIHOOD_COLORS,
        title="Overall Likelihood to Re-subscribe",
        labels={"Share": "Share of responses", "Likelihood to Re-subscribe": "Likelihood to re-subscribe"},
        text=q10_overall["Share"].map(lambda value: pct(value)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q10_overall.update_layout(yaxis_tickformat=".0%", showlegend=False)
    keep_percent_labels_off_error_bars(fig_q10_overall)

    crosstab_counts_plot = q2_q10_counts.melt(
        id_vars="Platform Sentiment",
        var_name="Likelihood to Re-subscribe",
        value_name="Respondents",
    )
    sentiment_totals = q2_q10_counts.set_index("Platform Sentiment")[LIKELIHOOD_ORDER].sum(axis=1)
    crosstab_counts_plot["Total"] = crosstab_counts_plot["Platform Sentiment"].map(sentiment_totals)
    crosstab_counts_plot["Row Share"] = crosstab_counts_plot["Respondents"] / crosstab_counts_plot["Total"]
    crosstab_plot = add_ci_columns(crosstab_counts_plot, "Respondents", "Total", "Row Share")
    fig_sentiment = px.bar(
        crosstab_plot,
        x="Platform Sentiment",
        y="Row Share",
        color="Likelihood to Re-subscribe",
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
        category_orders={
            "Platform Sentiment": SENTIMENT_ORDER,
            "Likelihood to Re-subscribe": LIKELIHOOD_ORDER,
        },
        title="Likelihood to Re-subscribe by Roblox Platform Sentiment",
        labels={"Row Share": "Share within sentiment group", "Platform Sentiment": "Platform sentiment"},
        color_discrete_map=LIKELIHOOD_COLORS,
    )
    fig_sentiment.update_layout(barmode="stack", yaxis_tickformat=".0%")

    q6_ci = add_ci_columns(q6, "selected_n", "respondents", "selected_pct")
    fig_q6 = px.bar(
        q6_ci.sort_values("selected_pct", ascending=False),
        x="motivation_short",
        y="selected_pct",
        title="Initial Subscription Motivations Among Churned Subscribers",
        labels={"selected_pct": "Share selecting motivation", "motivation_short": "Initial motivation"},
        text=q6_ci.sort_values("selected_pct", ascending=False)["selected_pct"].map(lambda value: pct(value)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q6.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35)
    keep_percent_labels_off_error_bars(fig_q6)

    q7_ci = q7.copy()
    q7_ci["Total"] = q7_ci["n"].sum()
    q7_ci = add_ci_columns(q7_ci, "n", "Total", "pct")
    fig_q7 = px.bar(
        q7_ci.sort_values("pct", ascending=False),
        x="reason_label",
        y="pct",
        title="Primary Reason for Subscribing to Roblox Plus",
        labels={"pct": "Share of respondents", "reason_label": "Primary subscription reason"},
        text=q7_ci.sort_values("pct", ascending=False)["pct"].map(lambda value: pct(value)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q7.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35)
    keep_percent_labels_off_error_bars(fig_q7)

    q3_ci = q3.copy()
    q3_ci["Total"] = q3_ci["respondents"].sum()
    q3_ci = add_ci_columns(q3_ci, "respondents", "Total", "share")
    fig_q3 = px.bar(
        q3_ci.sort_values("share", ascending=False),
        x="primary_reason_for_using_roblox",
        y="share",
        title="Primary Reason for Using Roblox",
        labels={
            "share": "Share of respondents",
            "primary_reason_for_using_roblox": "Primary reason for using Roblox",
        },
        text=q3_ci.sort_values("share", ascending=False)["share"].map(lambda value: pct(value)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q3.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35)
    keep_percent_labels_off_error_bars(fig_q3)

    q8_rank_dist_ci = add_ci_columns(q8_rank_dist, "#1 Count", "Valid Ranking N", "Top Box #1 %")
    fig_q8 = px.bar(
        q8_rank_dist_ci.sort_values("Top Box #1 %", ascending=False),
        x="Benefit",
        y="Top Box #1 %",
        title="Plus Benefit #1 Ranking Share",
        labels={"Top Box #1 %": "Share ranking benefit #1", "Benefit": "Plus benefit"},
        text=q8_rank_dist_ci.sort_values("Top Box #1 %", ascending=False)["Top Box #1 %"].map(lambda value: pct(value)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q8.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30)
    keep_percent_labels_off_error_bars(fig_q8)

    rank_order = ["#1", "#2", "#3", "#4", "#5"]
    q8_rank_long_ci = q8_rank_long.merge(
        q8_rank_dist[["Benefit", "Valid Ranking N"]],
        on="Benefit",
        how="left",
    )
    q8_rank_long_ci = add_ci_columns(q8_rank_long_ci, "Count", "Valid Ranking N", "Percent")
    fig_q8_rank_dist = px.bar(
        q8_rank_long_ci,
        x="Rank",
        y="Percent",
        color="Benefit",
        category_orders={"Rank": rank_order},
        title="Benefit Ranking Distribution by Benefit",
        labels={"Rank": "Rank position", "Percent": "Share of valid rankings", "Benefit": "Plus benefit"},
        text=q8_rank_long_ci["Percent"].map(lambda value: pct(value, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q8_rank_dist.update_layout(barmode="group", yaxis_tickformat=".0%")
    keep_percent_labels_off_error_bars(fig_q8_rank_dist)

    q9_segment_counts = (
        analysis[["RETENTION_GROUP", "Q9_LABEL"]]
        .fillna("Missing")
        .groupby(["RETENTION_GROUP", "Q9_LABEL"], observed=False)
        .size()
        .reset_index(name="Respondents")
    )
    q9_segment_counts["Total"] = q9_segment_counts.groupby("RETENTION_GROUP", observed=False)["Respondents"].transform("sum")
    q9_segment_counts["Share"] = q9_segment_counts["Respondents"] / q9_segment_counts["Total"]
    q9_plot = add_ci_columns(
        q9_segment_counts.rename(columns={"RETENTION_GROUP": "Retention Group", "Q9_LABEL": "Churn Reason"}),
        "Respondents",
        "Total",
        "Share",
    )
    q9_overall = (
        analysis["Q9_LABEL"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("Churn Reason")
        .reset_index(name="Respondents")
    )
    q9_overall["Share"] = q9_overall["Respondents"] / q9_overall["Respondents"].sum()
    q9_overall["Total"] = q9_overall["Respondents"].sum()
    q9_overall = add_ci_columns(q9_overall, "Respondents", "Total", "Share")
    fig_q9_overall = px.bar(
        q9_overall.sort_values("Share", ascending=False),
        x="Churn Reason",
        y="Share",
        title="Main Reason for Not Renewing Roblox Plus",
        labels={"Churn Reason": "Main reason for not renewing", "Share": "Share of responses"},
        text=q9_overall.sort_values("Share", ascending=False)["Share"].map(lambda value: pct(value)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q9_overall.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30)
    keep_percent_labels_off_error_bars(fig_q9_overall)

    fig_q9 = px.bar(
        q9_plot,
        x="Retention Group",
        y="Share",
        color="Churn Reason",
        category_orders={"Retention Group": RETENTION_ORDER},
        title="Core Churn Reason Mix by Return Intent Segment",
        labels={"Share": "Share within segment", "Retention Group": "Return intent segment"},
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q9.update_layout(barmode="stack", yaxis_tickformat=".0%")

    churn_keywords_ci = add_ci_columns(
        churn_keywords,
        "matching_respondents",
        "text_respondents",
        "share_of_voice",
    )
    fig_churn_keywords = px.bar(
        churn_keywords_ci,
        x="keyword_category",
        y="share_of_voice",
        color="group",
        category_orders={"group": RETENTION_ORDER},
        title="Open-End Churn Pain Point Share of Voice",
        labels={"group": "Return intent segment", "keyword_category": "Pain point", "share_of_voice": "Share of open-end respondents"},
        text=churn_keywords_ci["share_of_voice"].map(lambda value: pct(value, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_churn_keywords.update_layout(barmode="group", yaxis_tickformat=".0%", xaxis_tickangle=-25)
    keep_percent_labels_off_error_bars(fig_churn_keywords)

    q12_ci = q12.copy()
    q12_ci["Total"] = q12_ci["n"].sum()
    q12_ci = add_ci_columns(q12_ci, "n", "Total", "pick_rate")
    fig_q12 = px.bar(
        q12_ci.sort_values("pick_rate", ascending=False),
        x="feature_short",
        y="pick_rate",
        title="Future Roadmap Demand: Q12 Feature Pick Rate",
        labels={"pick_rate": "Pick rate", "feature_short": "Future feature"},
        text=q12_ci.sort_values("pick_rate", ascending=False)["pick_rate"].map(lambda value: pct(value)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q12.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35)
    keep_percent_labels_off_error_bars(fig_q12)

    q12_segments = analysis[["Q12_LABEL", "GENDER_LABEL", "AGE"]].copy()
    q12_segments["feature_short"] = q12_segments["Q12_LABEL"].map(short_feature)
    q12_segments["GENDER_LABEL"] = q12_segments["GENDER_LABEL"].fillna("Missing")
    q12_segments["AGE_GROUP"] = pd.cut(
        q12_segments["AGE"],
        bins=[12, 17, 24, 34, 200],
        labels=["13-17", "18-24", "25-34", "35+"],
        include_lowest=True,
    ).astype("string").fillna("Missing")

    q12_gender = (
        q12_segments.dropna(subset=["feature_short"])
        .groupby(["GENDER_LABEL", "feature_short"], observed=False)
        .size()
        .reset_index(name="Respondents")
    )
    q12_gender["Total"] = q12_gender.groupby("GENDER_LABEL", observed=False)["Respondents"].transform("sum")
    q12_gender["Pick Rate"] = q12_gender["Respondents"] / q12_gender["Total"]
    q12_gender = add_ci_columns(q12_gender, "Respondents", "Total", "Pick Rate")
    fig_q12_gender = px.bar(
        q12_gender,
        x="feature_short",
        y="Pick Rate",
        color="GENDER_LABEL",
        barmode="group",
        category_orders={"feature_short": feature_order, "GENDER_LABEL": ["Male", "Female", "Unknown", "Missing"]},
        title="Future Feature Demand by Gender",
        labels={"feature_short": "Future feature", "Pick Rate": "Pick rate within gender", "GENDER_LABEL": "Gender"},
        text=q12_gender["Pick Rate"].map(lambda value: pct(value, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q12_gender.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35)
    keep_percent_labels_off_error_bars(fig_q12_gender)

    q12_age = (
        q12_segments.dropna(subset=["feature_short"])
        .groupby(["AGE_GROUP", "feature_short"], observed=False)
        .size()
        .reset_index(name="Respondents")
    )
    q12_age["Total"] = q12_age.groupby("AGE_GROUP", observed=False)["Respondents"].transform("sum")
    q12_age["Pick Rate"] = q12_age["Respondents"] / q12_age["Total"]
    q12_age = add_ci_columns(q12_age, "Respondents", "Total", "Pick Rate")
    fig_q12_age = px.bar(
        q12_age,
        x="feature_short",
        y="Pick Rate",
        color="AGE_GROUP",
        barmode="group",
        category_orders={"feature_short": feature_order, "AGE_GROUP": ["13-17", "18-24", "25-34", "35+", "Missing"]},
        title="Future Feature Demand by Age Group",
        labels={"feature_short": "Future feature", "Pick Rate": "Pick rate within age group", "AGE_GROUP": "Age group"},
        text=q12_age["Pick Rate"].map(lambda value: pct(value, 0)),
        error_y="ci_error_plus",
        error_y_minus="ci_error_minus",
    )
    fig_q12_age.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35)
    keep_percent_labels_off_error_bars(fig_q12_age)

    backlash_plot = q13_backlash[q13_backlash["matching_respondents"].gt(0)].copy()
    fig_backlash_html = ""
    if not backlash_plot.empty:
        backlash_plot = add_ci_columns(
            backlash_plot,
            "matching_respondents",
            "text_respondents",
            "share_of_voice",
        )
        fig_backlash = px.bar(
            backlash_plot,
            x="feature_short",
            y="share_of_voice",
            color="backlash_keyword",
            title="Corporate Backlash Language in Feature Appeal Open-Ends",
            labels={"feature_short": "Selected future feature", "share_of_voice": "Share of feature open-ends", "backlash_keyword": "Backlash keyword"},
            text=backlash_plot["share_of_voice"].map(lambda value: pct(value, 1)),
            error_y="ci_error_plus",
            error_y_minus="ci_error_minus",
        )
        fig_backlash.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-35)
        keep_percent_labels_off_error_bars(fig_backlash)
        fig_backlash_html = fig_html(fig_backlash)

    age_summary_table = pd.DataFrame(
        [
            ["Mean age", num(age_row["mean"], 2)],
            ["Median age", num(age_row["50%"], 0)],
            ["Age range", f"{num(age_row['min'], 0)} to {num(age_row['max'], 0)}"],
            ["Mean tenure", f"{num(tenure_row['mean'], 0)} days ({num(tenure_row['mean'] / 365.25, 1)} years)"],
            ["Median tenure", f"{num(tenure_row['50%'], 0)} days ({num(tenure_row['50%'] / 365.25, 1)} years)"],
        ],
        columns=["Metric", "Value"],
    )
    gender_table = gender.assign(pct=gender["pct"].map(pct)).rename(columns={"gender": "Gender", "n": "Respondents", "pct": "Share"})
    q5_table = q5.head(8).assign(pct=q5.head(8)["pct"].map(pct)).rename(columns={"conversion_channel": "Discovery channel", "n": "Respondents", "pct": "Share"})
    q3_table = q3.assign(
        share=q3["share"].map(pct)
    ).rename(columns={"primary_reason_for_using_roblox": "Primary reason", "respondents": "Respondents", "share": "Share"})
    q6_table = q6[["motivation_short", "selected_n", "selected_pct"]].head(9).assign(
        selected_pct=q6["selected_pct"].head(9).map(pct)
    ).rename(columns={"motivation_short": "Initial motivation", "selected_n": "Respondents", "selected_pct": "Share"})
    q7_table = q7[["reason_label", "n", "pct"]].head(9).assign(
        pct=q7["pct"].head(9).map(pct)
    ).rename(columns={"reason_label": "Main subscription reason", "n": "Respondents", "pct": "Share"})
    q8_table = q8[["feature_short", "mean", "importance_score"]].assign(
        mean=q8["mean"].map(lambda value: f"{value:.2f}"),
        importance_score=q8["importance_score"].map(lambda value: f"{value:.2f}"),
    ).rename(columns={"feature_short": "Plus benefit", "mean": "Mean rank", "importance_score": "Importance score"})
    q8_rank_table = q8_rank_dist[
        [
            "Benefit",
            "Valid Ranking N",
            "#1 Count",
            "#1 %",
            "#2 Count",
            "#2 %",
            "#3 Count",
            "#3 %",
            "#4 Count",
            "#4 %",
            "#5 Count",
            "#5 %",
        ]
    ].copy()
    for column in ["#1 %", "#2 %", "#3 %", "#4 %", "#5 %"]:
        q8_rank_table[column] = q8_rank_table[column].map(pct)
    q12_table = q12[["feature_short", "n", "pick_rate"]].assign(
        pick_rate=q12["pick_rate"].map(pct)
    ).rename(columns={"feature_short": "Future feature", "n": "Respondents", "pick_rate": "Pick rate"})

    chi = q2_stats.set_index("metric")["value"]
    low_robux = churn_keywords[
        churn_keywords["group"].eq("Low Intent")
        & churn_keywords["keyword_category"].eq("Robux Absence / Premium Rebrand")
    ]["share_of_voice"].iloc[0]
    low_finance = churn_keywords[
        churn_keywords["group"].eq("Low Intent")
        & churn_keywords["keyword_category"].eq("Financial Strain")
    ]["share_of_voice"].iloc[0]
    top_ranked_benefit = q8_rank_dist.sort_values("#1 %", ascending=False).iloc[0]
    most_bottom_ranked_benefit = q8_rank_dist.sort_values("#5 %", ascending=False).iloc[0]
    middle_default_benefit = q8_rank_dist.sort_values("#3 %", ascending=False).iloc[0]

    hero = f"""
    <header class="hero">
      <p class="eyebrow">Month-1 Post-Launch Readout · Churned Subscribers</p>
      <h1>Roblox Plus Churned Subscriber Survey</h1>
      <p class="lede">This report summarizes why early Roblox Plus subscribers cancelled, which benefits still carry value, and which future roadmap ideas can help recover hesitant users.</p>
      <div class="kpi-grid">
        {make_kpi("Sample", f"n={total_n:,}", "13+ churned Roblox Plus subscribers")}
        {make_kpi("Top Churn Reason", str(top_churn_reason), pct(top_churn_pct))}
        {make_kpi("High Return Intent", pct(high_intent_n / total_n), f"{high_intent_n:,} likely / very likely")}
        {make_kpi("Top Future Feature", short_feature(top_feature), pct(top_feature_pct))}
      </div>
    </header>
    """

    executive_tab = f"""
    <div class="takeaway">
      <h2>Executive Summary</h2>
      <p>Churn is not primarily a lack-of-awareness problem. Respondents are long-tenured Roblox users who tried Plus, then cancelled because the bundle did not meet expectations around Robux, financial value, or ongoing need.</p>
      <ol>
        <li><strong>Robux absence is the clearest product gap:</strong> “It did not come with monthly Robux” is the top structured churn reason at {pct(top_churn_pct)}.</li>
        <li><strong>Return intent is recoverable but conditional:</strong> {pct((maybe_n + high_intent_n) / total_n)} are at least maybe willing to re-subscribe, while {pct(low_intent_n / total_n)} are unlikely or very unlikely.</li>
        <li><strong>Future demand skews cosmetic and expressive:</strong> {short_feature(top_feature)} leads the roadmap choices at {pct(top_feature_pct)}, followed by spawn effects and app themes.</li>
      </ol>
    </div>
    <div class="two-column">
      <div>
        <h3>Return Intent Funnel</h3>
        <div class="funnel">
          <div><span>All churned respondents</span><strong>{total_n:,}</strong></div>
          <div><span>Maybe to return</span><strong>{maybe_n:,}</strong></div>
          <div><span>Likely / very likely</span><strong>{high_intent_n:,}</strong></div>
          <div><span>Low intent</span><strong>{low_intent_n:,}</strong></div>
        </div>
      </div>
      <div>
        <h3>Signal Strength</h3>
        <p>Platform sentiment and likelihood to re-subscribe are meaningfully associated: chi-square p-value {chi['chi_square_p_value']:.2e}, Spearman rho {chi['spearman_rho']:.3f}. More positive Roblox sentiment translates into higher return intent, but does not eliminate product-specific churn concerns.</p>
        <p>Among Low Intent open-end respondents, financial language appears in {pct(low_finance)} and Robux/Premium language appears in {pct(low_robux)}.</p>
      </div>
    </div>
    {fig_html(fig_q10_overall, include_plotlyjs=True)}
    {fig_html(fig_sentiment)}
    """

    sample_tab = f"""
    {section("Sample Characteristics", f'''
      <p>The churned subscriber sample is young by age but mature by account history. Median tenure is {num(tenure_row["50%"], 0)} days, or roughly {num(tenure_row["50%"] / 365.25, 1)} years.</p>
      <div class="table-grid">
        <div><h3>Age and Tenure</h3>{table_html(age_summary_table)}</div>
        <div><h3>Gender</h3>{table_html(gender_table)}</div>
      </div>
    ''')}
    {section("Primary Reason for Using Roblox", f'''
      <p>Entertainment is the largest core platform use case among churned Roblox Plus subscribers, followed by spending time with friends and the variety of games and experiences.</p>
      <div class="chart-block">{fig_html(fig_q3)}</div>
      {table_html(q3_table)}
    ''')}
    {section("Discovery and Initial Subscription Path", f'''
      <p>Social media and the Buy Robux page were the largest discovery channels. This helps show where churned subscribers first encountered Plus before their initial subscription decision.</p>
      <div class="table-grid">
        <div><h3>Top Discovery Channels</h3>{table_html(q5_table)}</div>
      </div>
    ''')}
    """

    reasons_tab = f"""
    <div class="takeaway">
      <h2>Reasons for Subscribing</h2>
      <p>Trialing was the strongest acquisition hook for churned subscribers, while private servers, “wanted to try it out,” and discounts formed the next tier of motivation. This suggests that the front-door pitch was effective enough to drive trial, but not durable enough to prevent churn.</p>
    </div>
    {section("Primary Reasons for Subscribing: Select All", f'''
      <p>Respondents could select multiple initial motivations. Free trial leads by a wide margin at {pct(top_motivation_pct)}, followed by private servers and general curiosity.</p>
      <div class="chart-block">{fig_html(fig_q6)}</div>
      {table_html(q6_table)}
    ''')}
    {section("Primary Reason for Subscribing: Single Choice", f'''
      <p>When forced to choose one primary reason, free trial remains the top driver. This makes trial design and post-trial perceived value central to improving retention.</p>
      <div class="chart-block">{fig_html(fig_q7)}</div>
      {table_html(q7_table)}
    ''')}
    """

    ranking_tab = f"""
    <div class="takeaway">
      <h2>Ranking Plus Benefits</h2>
      <p>The churned subscriber audience is divided on which current Plus benefit matters most. No benefit dominates the ranking exercise, which reinforces that the current bundle lacks a single universally compelling anchor.</p>
      <ol>
        <li><strong>Top ranked benefit:</strong> {top_ranked_benefit["Benefit"]} has the highest #1 share at {pct(top_ranked_benefit["#1 %"])}.</li>
        <li><strong>Most polarizing benefit:</strong> {most_bottom_ranked_benefit["Benefit"]} has the highest #5 share at {pct(most_bottom_ranked_benefit["#5 %"])}, even though it also receives meaningful #1 votes.</li>
        <li><strong>Default middle benefit:</strong> {middle_default_benefit["Benefit"]} is most often ranked #3 at {pct(middle_default_benefit["#3 %"])}, suggesting it is useful but rarely the headline reason to subscribe.</li>
      </ol>
    </div>
    {section("Benefit Ranking Distribution by Benefit", f'''
      <p>Rank 1 means “most important” and rank 5 means “least important.” The chart below shows each benefit’s full distribution across all rank positions.</p>
      {fig_html(fig_q8_rank_dist)}
      {table_html(q8_rank_table)}
    ''')}
    {section("Interpretation for Product Strategy", f'''
      <p>Publish avatar items and trade/resell features score slightly better on mean rank, but the gap across shipped benefits is narrow. Private servers and discounts have clear value but also draw large bottom-rank shares, indicating they are not enough to keep many churned subscribers attached to the bundle by themselves.</p>
      <div class="chart-block">{fig_html(fig_q8)}</div>
      {table_html(q8_table)}
    ''')}
    """

    churn_tab = f"""
    {section("Why Users Cancelled", f'''
      <p>The structured churn data points to a product-value mismatch: monthly Robux is the most common cancellation reason across all return-intent groups. Open-end keywords show financial pressure is even more common than Robux language among Low Intent respondents.</p>
      {fig_html(fig_q9_overall)}
      {fig_html(fig_q9)}
      {fig_html(fig_churn_keywords)}
    ''')}
    """

    roadmap_tab = f"""
    {section("Future Roadmap Demand", f'''
      <p>The strongest roadmap candidates are visible identity and in-experience expression features. AI concepts sit at the bottom of explicit demand, and the backlash scan shows negative language is rare but concentrated around AI and customization categories.</p>
      {fig_html(fig_q12)}
      {table_html(q12_table)}
    ''')}
    {section("Feature Demand Differences by Gender and Age", f'''
      <p>These cuts compare pick rates within each demographic segment. Use them to identify whether roadmap demand is broad-based or concentrated among specific user groups.</p>
      {fig_html(fig_q12_gender)}
      {fig_html(fig_q12_age)}
    ''')}
    {section("AI and Corporate Backlash Watchouts", f'''
      <p>Backlash keywords are not broadly prevalent, but terms like “slop,” “AI garbage,” “investor,” and “greed” do appear in Q13 open-ends. Treat AI features as higher-risk roadmap bets unless paired with strong creator-quality positioning.</p>
      {fig_backlash_html}
    ''')}
    """

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Roblox Plus Churned Subscriber Survey Report</title>
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
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 24px 56px;
    }}
    .hero, .tab-panel, .report-section, .takeaway {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    .hero {{
      padding: 36px;
      margin-bottom: 18px;
    }}
    .eyebrow {{
      margin: 0 0 10px;
      color: var(--accent);
      font-size: 13px;
      font-weight: 800;
      letter-spacing: .08em;
      text-transform: uppercase;
    }}
    h1 {{
      margin: 0;
      font-size: 34px;
      line-height: 1.15;
      letter-spacing: -0.03em;
    }}
    h2 {{
      margin: 0 0 12px;
      font-size: 22px;
      letter-spacing: -0.02em;
    }}
    h3 {{
      margin: 0 0 10px;
      font-size: 16px;
    }}
    .lede {{
      max-width: 820px;
      color: var(--muted);
      font-size: 17px;
      margin: 14px 0 28px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }}
    .kpi-card {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      background: var(--accent-soft);
    }}
    .kpi-label {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .05em;
    }}
    .kpi-value {{
      margin-top: 8px;
      font-size: 23px;
      font-weight: 800;
      line-height: 1.15;
    }}
    .kpi-note {{
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
    }}
    .tabs {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 18px 0;
    }}
    .tab-button {{
      border: 1px solid var(--line);
      background: var(--paper);
      color: var(--ink);
      border-radius: 999px;
      padding: 10px 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    .tab-button.active {{
      background: var(--ink);
      color: white;
      border-color: var(--ink);
    }}
    .tab-panel {{
      display: none;
      padding: 24px;
    }}
    .tab-panel.active {{ display: block; }}
    .takeaway {{
      padding: 22px;
      margin-bottom: 18px;
    }}
    .takeaway p, .report-section p {{
      color: var(--muted);
      margin-top: 0;
    }}
    .report-section {{
      padding: 22px;
      margin-bottom: 18px;
    }}
    .two-column, .table-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin: 18px 0;
    }}
    .funnel {{
      display: grid;
      gap: 10px;
    }}
    .funnel div {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 14px;
      background: #fafbfe;
    }}
    .funnel span {{ color: var(--muted); }}
    .funnel strong {{ font-size: 18px; }}
    .chart-block {{
      margin: 16px 0;
    }}
    table.data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: white;
    }}
    .data-table th {{
      text-align: left;
      color: var(--muted);
      background: #f3f5f9;
      border-bottom: 1px solid var(--line);
      padding: 10px;
    }}
    .data-table td {{
      border-bottom: 1px solid var(--line);
      padding: 10px;
      vertical-align: top;
    }}
    .footer {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 24px;
      text-align: center;
    }}
    @media (max-width: 860px) {{
      .kpi-grid, .two-column, .table-grid {{ grid-template-columns: 1fr; }}
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
      <button class="tab-button" data-tab="sample">Sample & Subscription Path</button>
      <button class="tab-button" data-tab="reasons">Reasons for Subscribing</button>
      <button class="tab-button" data-tab="ranking">Ranking Plus Benefits</button>
      <button class="tab-button" data-tab="churn">Churn Deep Dive</button>
      <button class="tab-button" data-tab="roadmap">Future Roadmap</button>
    </nav>
    <section id="executive" class="tab-panel active">{executive_tab}</section>
    <section id="sample" class="tab-panel">{sample_tab}</section>
    <section id="reasons" class="tab-panel">{reasons_tab}</section>
    <section id="ranking" class="tab-panel">{ranking_tab}</section>
    <section id="churn" class="tab-panel">{churn_tab}</section>
    <section id="roadmap" class="tab-panel">{roadmap_tab}</section>
    <div class="footer">Source: Roblox Plus churned subscriber SPSS survey · Analysis generated from outputs/analysis_ready_respondent_level.csv · Chart error bars show 95% Wilson confidence intervals for proportions.</div>
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
    Path("index.html").write_text(html, encoding="utf-8")
    print(f"Wrote {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()
