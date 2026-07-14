#!/usr/bin/env python3
"""
Generate the Roblox Plus renewed-subscriber report.

This intentionally reuses the churned report's layout and chart logic so both
subscriber-segment reports stay visually and structurally aligned.
"""

from __future__ import annotations

from pathlib import Path


SOURCE_REPORT_GENERATOR = Path("generate_churned_subscriber_report.py")


REPLACEMENTS = {
    'OUTPUT_DIR = Path("outputs")': 'OUTPUT_DIR = Path("renewed_outputs")',
    'REPORT_PATH = Path("roblox_plus_churned_subscriber_report.html")': 'REPORT_PATH = Path("roblox_plus_renewed_subscriber_report.html")',
    'Path("index.html").write_text(html, encoding="utf-8")': '# Keep the churned GitHub Pages index intact for now.',
    "Roblox Plus Churned Subscriber Survey Report": "Roblox Plus Renewed Subscriber Survey Report",
    "Roblox Plus Churned Subscriber Survey": "Roblox Plus Renewed Subscriber Survey",
    "Month-1 Post-Launch Readout · Churned Subscribers": "Month-1 Post-Launch Readout · Renewed Subscribers",
    "13+ churned Roblox Plus subscribers": "13+ renewed Roblox Plus subscribers",
    "churned Roblox Plus subscribers": "renewed Roblox Plus subscribers",
    "churned subscriber": "renewed subscriber",
    "churned subscribers": "renewed subscribers",
    "churned respondents": "renewed respondents",
    "churned subscriber sample": "renewed subscriber sample",
    "Churned Subscribers": "Renewed Subscribers",
    "Churned Subscriber": "Renewed Subscriber",
    "churned": "renewed",
    "Top Churn Reason": "Top Value Driver",
    "Top churn reason": "Top value driver",
    "High Return Intent": "High Stay Intent",
    "Return Intent Funnel": "Stay Intent Funnel",
    "All renewed respondents": "All renewed respondents",
    "Maybe to return": "Maybe to stay",
    "likely / very likely": "likely / very likely to stay",
    "Return intent": "Stay intent",
    "return intent": "stay intent",
    "likelihood to re-subscribe": "likelihood to stay subscribed",
    "Likelihood to Re-subscribe": "Likelihood to Stay Subscribed",
    "Likelihood to re-subscribe": "Likelihood to stay subscribed",
    "Overall Likelihood to Re-subscribe": "Overall Likelihood to Stay Subscribed",
    "Likelihood to Re-subscribe by Roblox Platform Sentiment": "Likelihood to Stay Subscribed by Roblox Platform Sentiment",
    "re-subscribe": "stay subscribed",
    "re-sub": "stay sub",
    "cancelled": "renewed",
    "cancel": "renew",
    "cancelled because": "renewed because",
    "cancelled, which benefits still carry value": "renewed, which benefits carry the most ongoing value",
    "Churn Deep Dive": "Renewal Deep Dive",
    "Why Users Cancelled": "Why Users Stay",
    "Why Users renewed": "Why Users Stay",
    "Main Reason for Not Renewing Roblox Plus": "Most Valuable Part of Roblox Plus",
    "Main reason for not renewing": "Most valuable Plus benefit",
    "Core Churn Reason Mix by Return Intent Segment": "Most Valuable Benefit Mix by Stay Intent Segment",
    "Core renewal Reason Mix by Stay Intent Segment": "Most Valuable Benefit Mix by Stay Intent Segment",
    "Churn Reason": "Most Valuable Benefit",
    "renewal Reason": "Most Valuable Benefit",
    "Open-End Churn Pain Point Share of Voice": "Open-End Renewal Value Share of Voice",
    "Open-End renewal Pain Point Share of Voice": "Open-End Renewal Value Share of Voice",
    "Pain point": "Value theme",
    "Source: Roblox Plus renewed subscriber SPSS survey": "Source: Roblox Plus renewed subscriber SPSS survey",
    "Source: Roblox Plus churned subscriber SPSS survey": "Source: Roblox Plus renewed subscriber SPSS survey",
}

POST_REPLACEMENTS = {
    "Churn is not primarily a lack-of-awareness problem. Respondents are long-tenured Roblox users who tried Plus, then renewed because the bundle did not meet expectations around Robux, financial value, or ongoing need.": "Renewal is anchored in practical value. Respondents are long-tenured Roblox users who kept Plus because at least one part of the bundle still delivers clear ongoing utility.",
    "Robux absence is the clearest product gap:": "Ongoing value is the clearest retention signal:",
    "“It did not come with monthly Robux” is the top structured churn reason": "the top structured value driver is shown below",
    "Return intent is recoverable but conditional:": "Stay intent is positive but still conditional:",
    "are at least maybe willing to stay subscribed": "are at least maybe willing to stay subscribed",
    "while ": "while ",
    "are unlikely or very unlikely.": "are unlikely or very unlikely to stay.",
    "More positive Roblox sentiment translates into higher stay intent, but does not eliminate product-specific churn concerns.": "More positive Roblox sentiment translates into higher stay intent, while lower platform sentiment still creates retention risk.",
    "Trialing was the strongest acquisition hook for renewed subscribers, while private servers, “wanted to try it out,” and discounts formed the next tier of motivation. This suggests that the front-door pitch was effective enough to drive trial, but not durable enough to prevent churn.": "Free trials, discounts, and private servers show how renewed subscribers first entered the product. Comparing select-all motivations with the single primary reason separates broad awareness hooks from the benefit that mattered most.",
    "The structured churn data points to a product-value mismatch: monthly Robux is the most common renewlation reason across all return-intent groups. Open-end keywords show financial pressure is even more common than Robux language among Low Intent respondents.": "The structured renewal data shows which parts of Plus subscribers say are most valuable, then splits that value mix by likelihood to stay subscribed. Open-end keywords summarize the language renewed subscribers use when explaining their stay intent.",
    "product-specific churn concerns": "product-specific retention concerns",
    "renewlation": "renewal",
    "churn reason": "value driver",
    "structured churn": "structured renewal",
    "prevent churn": "support retention",
    'data-tab="churn"': 'data-tab="renewal"',
    'id="churn"': 'id="renewal"',
    "Analysis generated from outputs/analysis_ready_respondent_level.csv": "Analysis generated from renewed_outputs/analysis_ready_respondent_level.csv",
}


def main() -> None:
    source = SOURCE_REPORT_GENERATOR.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS.items():
        source = source.replace(old, new)

    namespace = {"__name__": "__main__", "__file__": str(SOURCE_REPORT_GENERATOR)}
    exec(compile(source, str(SOURCE_REPORT_GENERATOR), "exec"), namespace)

    report_path = Path("roblox_plus_renewed_subscriber_report.html")
    html = report_path.read_text(encoding="utf-8")
    for old, new in POST_REPLACEMENTS.items():
        html = html.replace(old, new)
    report_path.write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
