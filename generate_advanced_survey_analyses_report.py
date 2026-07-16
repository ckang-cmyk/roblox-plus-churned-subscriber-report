#!/usr/bin/env python3
"""
Generate advanced Roblox Plus survey analyses across non-subscribers,
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

import generate_three_segment_comparison_report as base


REPORT_PATH = Path("roblox_plus_advanced_survey_analyses_report.html")
SEGMENT_ORDER = base.SEGMENT_ORDER

THEME_RULES = {
    "Cost / affordability": ["money", "afford", "expensive", "cost", "price", "broke", "funds", "pay"],
    "Monthly subscription resistance": ["monthly", "subscription", "recurring", "renew", "cancel"],
    "Robux stipend / currency value": ["robux", "stipend", "premium", "transfer"],
    "Low usage / low need": ["need", "use", "enough", "play", "often", "worth"],
    "Platform trust / protest": ["greed", "lawsuit", "safety", "predator", "corporation", "moderation", "trust", "ai"],
    "Age / verification restrictions": ["age", "verify", "verification", "id", "parent", "permission"],
    "Private server utility": ["private server", "server"],
    "Customization / identity": ["custom", "theme", "profile", "frame", "avatar", "particle", "exclusive"],
}

VARIABLE_DISPLAY_NAMES = {
    "Q2_LABEL": "Roblox sentiment",
    "Q3_LABEL": "Primary motivation for using Roblox",
    "Q4_LABEL": "Familiarity with Plus",
    "Q12_LABEL": "New Plus feature picked",
}


def pct(value: float, digits: int = 1) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value * 100:.{digits}f}%"


def num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{digits}f}"


def table_html(df: pd.DataFrame) -> str:
    escaped = df.copy()
    for col in escaped.columns:
        escaped[col] = escaped[col].map(lambda value: escape(str(value)))
    return escaped.to_html(index=False, classes="data-table", border=0, escape=False)


def fig_html(fig, include_plotlyjs: bool = False) -> str:
    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={"displayModeBar": False, "responsive": True},
        default_height="520px",
    )


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


def driver_test(df: pd.DataFrame, predictor: str, outcome: str = "High Intent") -> dict:
    working = df[[predictor, outcome]].copy()
    working[predictor] = working[predictor].fillna("Missing").astype(str)
    working = working[working[predictor].ne("Missing")]
    table = pd.crosstab(working[predictor], working[outcome])
    chi2, p_value, dof, v = cramers_v(table)
    rates = working.groupby(predictor, observed=False)[outcome].agg(["mean", "count"]).reset_index()
    rates = rates[rates["count"].ge(30)]
    if rates.empty:
        return {
            "Predictor": predictor,
            "Respondents": len(working),
            "Cramer's V": v,
            "p-value": p_value,
            "High-intent spread": np.nan,
            "Highest high-intent category": "N/A",
            "Lowest high-intent category": "N/A",
        }
    highest = rates.sort_values("mean", ascending=False).iloc[0]
    lowest = rates.sort_values("mean", ascending=True).iloc[0]
    return {
        "Predictor": predictor,
        "Respondents": len(working),
        "Cramer's V": v,
        "p-value": p_value,
        "High-intent spread": highest["mean"] - lowest["mean"],
        "Highest high-intent category": f"{highest[predictor]} ({pct(highest['mean'])}, n={int(highest['count'])})",
        "Lowest high-intent category": f"{lowest[predictor]} ({pct(lowest['mean'])}, n={int(lowest['count'])})",
    }


def top_category_profiles(df: pd.DataFrame, group_col: str, profile_cols: list[str]) -> pd.DataFrame:
    rows = []
    for value, value_df in df.groupby(group_col, dropna=False, observed=False):
        row = {group_col: value, "Respondents": len(value_df), "High Intent": pct(value_df["High Intent"].mean())}
        for col in profile_cols:
            if col not in value_df.columns:
                continue
            top = value_df[col].fillna("Missing").value_counts(normalize=True).head(1)
            row[col] = f"{top.index[0]} ({pct(top.iloc[0])})" if not top.empty else "N/A"
        rows.append(row)
    return pd.DataFrame(rows)


def keyword_theme_table(texts: pd.Series) -> pd.DataFrame:
    clean = texts.fillna("").astype(str).map(lambda value: " ".join(value.split()))
    clean = clean[clean.str.len().ge(8)]
    rows = []
    for theme, keywords in THEME_RULES.items():
        mask = clean.str.lower().map(lambda value: any(keyword in value for keyword in keywords))
        quotes = clean[mask].head(3).tolist()
        rows.append(
            {
                "Theme": theme,
                "Mentions": int(mask.sum()),
                "Share of text respondents": mask.mean() if len(clean) else np.nan,
                "Representative quotes": " | ".join(f'"{quote[:180]}"' for quote in quotes),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("Mentions", ascending=False)


def load_non_raw() -> pd.DataFrame:
    raw = pd.read_csv("Non-subsFINAL_June 25, 2026_13.44.csv")
    raw["AGE"] = pd.to_numeric(raw["Q1"], errors="coerce")
    return raw[raw["AGE"].isna() | raw["AGE"].ge(13)].copy()


def load_sav_raw(path: str) -> tuple[pd.DataFrame, pyreadstat.metadata_container]:
    raw, meta = pyreadstat.read_sav(path, apply_value_formats=False, user_missing=True)
    raw["AGE"] = pd.to_numeric(raw["Q1"], errors="coerce")
    return raw[raw["AGE"].isna() | raw["AGE"].ge(13)].copy(), meta


def normalize_reason_series(series: pd.Series, label_map: dict) -> pd.Series:
    return series.map(lambda value: label_map.get(base.normalize_code(value), "Missing"))


def make_percent_bar(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None):
    fig = px.bar(df, x=x, y=y, color=color, title=title, text=df[y].map(lambda value: pct(value, 0)))
    fig.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30, xaxis_tickfont_size=10, margin=dict(b=130))
    fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=11, cliponaxis=False)
    fig.update_layout(uniformtext_minsize=9, uniformtext_mode="hide")
    return fig


def apply_variable_display_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Predictor" in out.columns:
        out["Predictor"] = out["Predictor"].replace(VARIABLE_DISPLAY_NAMES)
    return out.rename(columns=VARIABLE_DISPLAY_NAMES)


def main() -> None:
    combined = base.load_segments()
    combined["High Intent"] = combined["Q10_SCORE"].ge(4)
    combined["Low Intent"] = combined["Q10_SCORE"].le(2)
    combined["Q12_LABEL"] = base.feature_short(combined["Q12_LABEL"])

    predictors = ["Segment", "Q2_LABEL", "Q4_LABEL", "Q3_LABEL", "Age Range", "GENDER_LABEL", "Q12_LABEL"]
    combined_drivers = pd.DataFrame([driver_test(combined, predictor) for predictor in predictors])
    combined_drivers = combined_drivers.sort_values("Cramer's V", ascending=False)

    segment_driver_rows = []
    for segment, segment_df in combined.groupby("Segment", observed=False):
        for predictor in ["Q2_LABEL", "Q4_LABEL", "Q3_LABEL", "Age Range", "GENDER_LABEL", "Q12_LABEL"]:
            result = driver_test(segment_df, predictor)
            result["Segment"] = segment
            segment_driver_rows.append(result)
    segment_drivers = pd.DataFrame(segment_driver_rows).sort_values(["Segment", "Cramer's V"], ascending=[True, False])

    driver_plot = apply_variable_display_names(combined_drivers)
    fig_combined_drivers = px.bar(
        driver_plot,
        x="Predictor",
        y="Cramer's V",
        title="Cross-Segment High-Intent Driver Strength",
        text=driver_plot["Cramer's V"].map(lambda value: num(value, 3)),
    )
    fig_combined_drivers.update_layout(xaxis_tickangle=-25, yaxis_title="Cramer's V")

    segment_driver_plot = apply_variable_display_names(segment_drivers.groupby("Segment", as_index=False).head(4))
    fig_segment_drivers = px.bar(
        segment_driver_plot,
        x="Predictor",
        y="Cramer's V",
        color="Segment",
        facet_col="Segment",
        title="Top Segment-Specific High-Intent Drivers",
        text=segment_driver_plot["Cramer's V"].map(lambda value: num(value, 3)),
    )
    fig_segment_drivers.update_layout(showlegend=False, xaxis_tickangle=-25)

    future_priority = (
        combined[combined["Q12_LABEL"].notna() & combined["Q12_LABEL"].ne("Missing")]
        .groupby(["Segment", "Q12_LABEL"], observed=False)
        .agg(Respondents=("Q12_LABEL", "size"), High_Intent_Rate=("High Intent", "mean"))
        .reset_index()
    )
    future_priority["Segment Total"] = future_priority.groupby("Segment", observed=False)["Respondents"].transform("sum")
    future_priority["Pick Rate"] = future_priority["Respondents"] / future_priority["Segment Total"]
    fig_future_priority = px.scatter(
        future_priority,
        x="Pick Rate",
        y="High_Intent_Rate",
        color="Segment",
        size="Respondents",
        hover_name="Q12_LABEL",
        title="Future Feature Prioritization Matrix",
        labels={"Q12_LABEL": "Future feature", "High_Intent_Rate": "High-intent rate among pickers"},
    )
    fig_future_priority.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%")

    intent_profile = top_category_profiles(
        combined,
        "Intent Group",
        ["Segment", "Q2_LABEL", "Q4_LABEL", "Q3_LABEL", "Q12_LABEL", "Age Range"],
    ).sort_values("Respondents", ascending=False)
    intent_profile = apply_variable_display_names(intent_profile)

    # Non-subscriber analyses
    non_raw = load_non_raw()
    non_segment = combined[combined["Segment"].eq("Non-subscribers")].copy()
    non_segment["Awareness Group"] = np.where(non_segment["Q4_LABEL"].isin(["Somewhat familiar", "Very familiar", "Extremely familiar"]), "Aware", "Unaware")
    aware_intent = (
        non_segment.groupby("Awareness Group", observed=False)
        .agg(Respondents=("High Intent", "size"), High_Intent=("High Intent", "mean"), Low_Intent=("Low Intent", "mean"))
        .reset_index()
    )
    fig_aware_intent = make_percent_bar(aware_intent, "Awareness Group", "High_Intent", "Non-Subscriber High Intent by Awareness")

    non_select_all, non_primary = base.load_non_specific()
    barrier_groups = {
        "Awareness gap": ["Not familiar with benefits"],
        "Low utility": ["Don't play enough", "Don't buy enough items", "Don't use private servers", "Don't transfer Robux often"],
        "Bundle value gap": ["No monthly Robux stipend", "Too expensive / not worth cost"],
        "Subscription/payment friction": ["Don't like monthly subscription", "No payment method"],
        "Timing / inertia": ["Haven't gotten around to it"],
        "Other": ["Other"],
    }
    barrier_rows = []
    for group, reasons in barrier_groups.items():
        subset = non_select_all[non_select_all["Reason"].isin(reasons)]
        barrier_rows.append({"Barrier Group": group, "Estimated Share": subset["Share"].sum()})
    barrier_summary = pd.DataFrame(barrier_rows).sort_values("Estimated Share", ascending=False)
    fig_barriers = make_percent_bar(barrier_summary, "Barrier Group", "Estimated Share", "Non-Subscriber Barrier Group Sizing")

    non_text = pd.concat(
        [
            non_raw.get("Q19_12_TEXT", pd.Series(dtype="object")),
            non_raw.get("Q18_Combined", pd.Series(dtype="object")),
            non_raw.get("Q27", pd.Series(dtype="object")),
        ],
        ignore_index=True,
    )
    non_themes = keyword_theme_table(non_text)

    # Churned analyses
    churn_analysis = pd.read_csv("outputs/analysis_ready_respondent_level.csv")
    churn_analysis["High Intent"] = pd.to_numeric(churn_analysis["Q10_SCORE"], errors="coerce").ge(4)
    churn_analysis["Maybe Intent"] = pd.to_numeric(churn_analysis["Q10_SCORE"], errors="coerce").eq(3)
    churn_analysis["Q9_LABEL"] = churn_analysis["Q9_LABEL"].replace(base.CHURN_REASON_LABELS)
    churn_winback = (
        churn_analysis.groupby("Q9_LABEL", observed=False)
        .agg(Respondents=("Q9_LABEL", "size"), High_Intent=("High Intent", "mean"), Maybe_Intent=("Maybe Intent", "mean"))
        .reset_index()
        .rename(columns={"Q9_LABEL": "Churn Reason"})
    )
    churn_winback["Recoverable Intent"] = churn_winback["High_Intent"] + churn_winback["Maybe_Intent"]
    churn_winback = churn_winback.sort_values("Respondents", ascending=False)
    fig_churn_winback = px.scatter(
        churn_winback,
        x="Respondents",
        y="Recoverable Intent",
        size="Respondents",
        hover_name="Churn Reason",
        title="Churned Winback Opportunity by Churn Reason",
    )
    fig_churn_winback.update_layout(yaxis_tickformat=".0%")
    churn_themes = keyword_theme_table(churn_analysis.get("CHURN_TEXT_COMBINED", pd.Series(dtype="object")))

    # Renewed analyses
    renewed = pd.read_csv("renewed_outputs/analysis_ready_respondent_level.csv")
    renewed["High Intent"] = pd.to_numeric(renewed["Q10_SCORE"], errors="coerce").ge(4)
    renewed["Low_or_Maybe"] = pd.to_numeric(renewed["Q10_SCORE"], errors="coerce").le(3)
    renewed["Value Driver"] = renewed["Q8_LABEL"].replace(base.RENEWED_VALUE_LABELS)
    renewed_value_profile = (
        renewed.groupby("Value Driver", observed=False)
        .agg(Respondents=("Value Driver", "size"), High_Intent=("High Intent", "mean"), At_Risk=("Low_or_Maybe", "mean"))
        .reset_index()
        .sort_values("Respondents", ascending=False)
    )
    fig_renewed_value = px.scatter(
        renewed_value_profile,
        x="Respondents",
        y="High_Intent",
        size="Respondents",
        hover_name="Value Driver",
        title="Renewed Retention Strength by Value Driver",
    )
    fig_renewed_value.update_layout(yaxis_tickformat=".0%")
    renewed_risk_profile = top_category_profiles(
        renewed.assign(**{"Intent Risk": np.where(renewed["Low_or_Maybe"], "Low / Maybe Stay Intent", "High Stay Intent")}),
        "Intent Risk",
        ["Value Driver", "Q2_LABEL", "Q4_LABEL", "Q12_LABEL"],
    )
    renewed_risk_profile = apply_variable_display_names(renewed_risk_profile)
    renewed_themes = keyword_theme_table(renewed.get("Q11_TEXT_COMBINED", pd.Series(dtype="object")))

    # Formatting tables
    combined_driver_table = apply_variable_display_names(combined_drivers).assign(
        **{
            "Cramer's V": lambda d: d["Cramer's V"].map(lambda value: num(value, 3)),
            "p-value": lambda d: d["p-value"].map(lambda value: f"{value:.2e}" if pd.notna(value) else "N/A"),
            "High-intent spread": lambda d: d["High-intent spread"].map(pct),
        }
    )
    segment_driver_table = apply_variable_display_names(segment_drivers.groupby("Segment", as_index=False).head(5)).assign(
        **{
            "Cramer's V": lambda d: d["Cramer's V"].map(lambda value: num(value, 3)),
            "p-value": lambda d: d["p-value"].map(lambda value: f"{value:.2e}" if pd.notna(value) else "N/A"),
            "High-intent spread": lambda d: d["High-intent spread"].map(pct),
        }
    )
    aware_table = aware_intent.assign(
        High_Intent=aware_intent["High_Intent"].map(pct),
        Low_Intent=aware_intent["Low_Intent"].map(pct),
    ).rename(columns={"High_Intent": "High intent", "Low_Intent": "Low intent"})
    barrier_table = barrier_summary.assign(**{"Estimated Share": barrier_summary["Estimated Share"].map(pct)})
    non_theme_table = non_themes.head(8).assign(**{"Share of text respondents": non_themes["Share of text respondents"].head(8).map(pct)})
    churn_winback_table = churn_winback.assign(
        High_Intent=churn_winback["High_Intent"].map(pct),
        Maybe_Intent=churn_winback["Maybe_Intent"].map(pct),
        **{"Recoverable Intent": churn_winback["Recoverable Intent"].map(pct)},
    ).rename(columns={"High_Intent": "High intent", "Maybe_Intent": "Maybe intent"})
    churn_theme_table = churn_themes.head(8).assign(**{"Share of text respondents": churn_themes["Share of text respondents"].head(8).map(pct)})
    renewed_value_table = renewed_value_profile.assign(
        High_Intent=renewed_value_profile["High_Intent"].map(pct),
        At_Risk=renewed_value_profile["At_Risk"].map(pct),
    ).rename(columns={"High_Intent": "High stay intent", "At_Risk": "Low/maybe risk"})
    renewed_theme_table = renewed_themes.head(8).assign(**{"Share of text respondents": renewed_themes["Share of text respondents"].head(8).map(pct)})

    top_combined_driver = combined_drivers.iloc[0]
    top_segment_drivers = segment_drivers.groupby("Segment", as_index=False).head(1)
    top_non_barrier = barrier_summary.iloc[0]
    top_churn_opportunity = churn_winback.sort_values("Recoverable Intent", ascending=False).iloc[0]
    top_renewed_value = renewed_value_profile.iloc[0]
    top_combined_driver_v = top_combined_driver["Cramer's V"]
    top_combined_driver_name = VARIABLE_DISPLAY_NAMES.get(
        top_combined_driver["Predictor"],
        top_combined_driver["Predictor"],
    )

    hero = f"""
    <header class="hero">
      <p class="eyebrow">Roblox Plus · Advanced Survey Analyses</p>
      <h1>Advanced Roblox Plus Survey Analysis</h1>
      <p class="lede">This report runs the next layer of analyses across non-subscribers, churned subscribers, and renewed subscribers: high-intent driver tests, segment-specific profiles, barrier sizing, winback opportunity, retention value drivers, future feature prioritization, and open-end keyword themes.</p>
      <div class="kpi-grid">
        {make_kpi("Top Cross-Segment Driver", str(top_combined_driver_name), f"Cramer's V {num(top_combined_driver_v, 3)}")}
        {make_kpi("Top Non-Sub Barrier", str(top_non_barrier["Barrier Group"]), pct(top_non_barrier["Estimated Share"]))}
        {make_kpi("Best Winback Signal", str(top_churn_opportunity["Churn Reason"]), f"{pct(top_churn_opportunity['Recoverable Intent'])} maybe/high intent")}
        {make_kpi("Top Retention Value", str(top_renewed_value["Value Driver"]), f"{top_renewed_value['Respondents']:,} renewed users")}
      </div>
    </header>
    """

    executive_tab = f"""
    <div class="takeaway">
      <h2>Executive Summary</h2>
      <ol>
        <li><strong>Intent is most strongly structured by segment relationship:</strong> the strongest cross-segment driver is {top_combined_driver_name} (Cramer's V={num(top_combined_driver["Cramer's V"], 3)}), confirming that acquisition, winback, and retention should be managed as distinct funnels.</li>
        <li><strong>Within segments, platform sentiment and Plus familiarity are recurring intent signals:</strong> they repeatedly appear among the top univariate drivers, meaning product value messaging has to work alongside broader Roblox sentiment.</li>
        <li><strong>The biggest strategic split is practical value vs. emotional/cosmetic upside:</strong> discounts, private servers, and Robux-related benefits anchor current value, while future features help create conversion or retention upside for specific audiences.</li>
      </ol>
    </div>
    {fig_html(fig_combined_drivers, include_plotlyjs=True)}
    {table_html(combined_driver_table)}
    """

    drivers_tab = f"""
    {section("Segment-Specific Intent Drivers", f'''
      <p>These are univariate driver tests for high intent within each segment. They are not causal models, but they identify which respondent characteristics most strongly separate high-intent users from everyone else.</p>
      {fig_html(fig_segment_drivers)}
      {table_html(segment_driver_table)}
    ''')}
    {section("Intent Group Profiles", f'''
      <p>This table profiles low, maybe, and high intent groups by their most common segment, sentiment, familiarity, platform motivation, future feature pick, and age range.</p>
      {table_html(intent_profile)}
    ''')}
    """

    roadmap_tab = f"""
    {section("Future Feature Prioritization Matrix", f'''
      <p>The prioritization matrix plots feature demand against high-intent concentration among respondents who chose that feature. Use the upper-right area for features that are both popular and attached to higher-intent users.</p>
      {fig_html(fig_future_priority)}
    ''')}
    """

    non_tab = f"""
    {section("Non-Subscriber Barrier Segmentation", f'''
      <p>Barrier groups combine select-all reasons into broader strategic categories. Because this is select-all data, percentages can sum above 100%.</p>
      {fig_html(fig_barriers)}
      {table_html(barrier_table)}
    ''')}
    {section("Aware vs. Unaware Non-Subscribers", f'''
      <p>This compares high subscription intent among non-subscribers who are aware versus unaware of Roblox Plus, using Plus familiarity as the awareness proxy.</p>
      {fig_html(fig_aware_intent)}
      {table_html(aware_table)}
    ''')}
    {section("Non-Subscriber Open-End Themes", f'''
      <p>Keyword coding of non-subscriber open-ends highlights affordability, subscription resistance, low value, platform protest, and Robux stipend concerns.</p>
      {table_html(non_theme_table)}
    ''')}
    """

    churn_tab = f"""
    {section("Churned Winback Opportunity", f'''
      <p>This combines main churn reason with re-subscribe intent. Reasons with high respondent count and high maybe/high intent are the most actionable winback pools.</p>
      {fig_html(fig_churn_winback)}
      {table_html(churn_winback_table)}
    ''')}
    {section("Churned Open-End Themes", f'''
      <p>Keyword coding of churned open-ends surfaces the most common pain points behind cancellation and return intent.</p>
      {table_html(churn_theme_table)}
    ''')}
    """

    renewed_tab = f"""
    {section("Renewed Retention Value Drivers", f'''
      <p>This maps each renewed value driver by size and high stay intent. Large groups with lower high-intent rates are the highest-priority retention watchouts.</p>
      {fig_html(fig_renewed_value)}
      {table_html(renewed_value_table)}
    ''')}
    {section("Renewed Risk Profile", f'''
      <p>This compares low/maybe stay intent against high stay intent among renewed subscribers.</p>
      {table_html(renewed_risk_profile)}
    ''')}
    {section("Renewed Open-End Themes", f'''
      <p>Keyword coding of renewed open-ends highlights what current subscribers say when explaining stay intent.</p>
      {table_html(renewed_theme_table)}
    ''')}
    """

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Advanced Roblox Plus Survey Analyses</title>
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
    .kpi-value {{ margin-top: 8px; font-size: 22px; font-weight: 800; line-height: 1.15; }}
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
      <button class="tab-button" data-tab="drivers">Intent Drivers</button>
      <button class="tab-button" data-tab="roadmap">Future Roadmap</button>
      <button class="tab-button" data-tab="non-subs">Non-Subscribers</button>
      <button class="tab-button" data-tab="churned">Churned</button>
      <button class="tab-button" data-tab="renewed">Renewed</button>
    </nav>
    <section id="executive" class="tab-panel active">{executive_tab}</section>
    <section id="drivers" class="tab-panel">{drivers_tab}</section>
    <section id="roadmap" class="tab-panel">{roadmap_tab}</section>
    <section id="non-subs" class="tab-panel">{non_tab}</section>
    <section id="churned" class="tab-panel">{churn_tab}</section>
    <section id="renewed" class="tab-panel">{renewed_tab}</section>
    <div class="footer">Sources: Roblox Plus non-subscriber CSV, churned subscriber SPSS survey, renewed subscriber SPSS survey · Driver tests use chi-square and Cramer's V; text themes use keyword coding.</div>
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
