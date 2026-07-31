"""
modules/decision_engine.py
============================
Decision Analytics Engine: goes beyond descriptive reporting by
automatically detecting business issues and generating concrete,
prioritized recommendations.

Design decisions:
- Each detector is a pure function returning a list of "Issue" dicts
  with a consistent schema (category, severity, finding, recommendation,
  business_impact, estimated_benefit, priority). That consistent schema
  is what lets the UI render them uniformly AND lets modules/ai.py feed
  them to Gemini as structured context instead of raw dataframes.
- Thresholds are defined as named constants at the top, not buried in
  logic, so they're easy to defend/explain in an interview ("why 15%?").
- Severity is derived from how far a metric deviates from the peer
  average, not a fixed magic number — this generalizes to any dataset,
  not just the sample data.

v2 changes (enterprise polish pass):
- Every issue now also carries:
    * business_impact   — a quantified, dollarized statement of what the
                           finding is costing (distinct from `finding`,
                           which is the descriptive detection itself)
    * estimated_benefit  — a quantified, dollarized statement of the
                           upside if the recommendation is acted on
    * priority           — P1..Pn, assigned in run_decision_engine() by
                           ranking severity first, then estimated dollar
                           benefit within each severity tier
  These are deliberately conservative, back-of-envelope estimates built
  only from data already in the dataframe (peer averages, revenue
  gaps, prior-month deltas) — not model output — so every number is
  explainable in one sentence if asked in an interview.
- app.py already reads issue.get("business_impact"/"estimated_benefit"/
  "priority") with graceful fallbacks, so no app.py changes are needed
  to pick these up.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Thresholds (tune these; documented so they're defensible in interviews)
# ---------------------------------------------------------------------------
DISCOUNT_ALERT_MULTIPLIER = 1.5      # flag if a region's discount rate > 1.5x the average of others
LOW_INVENTORY_THRESHOLD = 50         # units — below this is a stockout risk
REVENUE_DECLINE_THRESHOLD = -5.0     # % month-over-month decline that triggers an alert
UNDERPERFORMER_PERCENTILE = 0.25     # bottom 25% of revenue = "poor-performing"


def _make_issue(category, severity, finding, recommendation,
                 business_impact=None, estimated_benefit=None, benefit_value=0.0):
    """
    benefit_value: a plain float (dollars) used only internally by
    run_decision_engine() to rank findings within the same severity tier.
    It's harmless to leave on the returned dict (useful for the PDF
    report or future sorting), but app.py doesn't need to read it.
    """
    return {
        "category": category,
        "severity": severity,  # "High" | "Medium" | "Low"
        "finding": finding,
        "recommendation": recommendation,
        "business_impact": business_impact or finding,
        "estimated_benefit": estimated_benefit or "Not yet quantified",
        "benefit_value": benefit_value,
    }


def detect_discount_leakage(df: pd.DataFrame) -> list:
    """Flag regions discounting well above the peer average -> margin leakage."""
    issues = []
    region_stats = df.groupby("Region").apply(
        lambda g: pd.Series({
            "discount_rate": g["Discount"].sum() / g["Revenue"].sum() * 100 if g["Revenue"].sum() else 0,
            "revenue": g["Revenue"].sum(),
        }), include_groups=False
    )
    if region_stats.empty:
        return issues

    avg_rate = region_stats["discount_rate"].mean()

    for region, row in region_stats.iterrows():
        rate = row["discount_rate"]
        region_revenue = row["revenue"]
        others_avg = region_stats.drop(index=region)["discount_rate"].mean() if len(region_stats) > 1 else avg_rate
        if others_avg > 0 and rate > others_avg * DISCOUNT_ALERT_MULTIPLIER:
            excess_pct_points = rate - others_avg
            # Recoverable margin ≈ the excess discount percentage applied to
            # that region's revenue base — i.e. what would be saved if the
            # region's discount rate matched the peer average.
            recoverable_margin = (excess_pct_points / 100) * region_revenue

            issues.append(_make_issue(
                category="Margin Leakage",
                severity="High" if rate > others_avg * 2 else "Medium",
                finding=f"{region} region has a discount rate of {rate:.1f}%, "
                        f"vs. {others_avg:.1f}% average across other regions.",
                recommendation=f"Reduce discount campaigns in {region} by 5-8% and audit "
                                f"rep-level discount approvals to recover margin.",
                business_impact=f"{region}'s discounting runs {excess_pct_points:.1f} percentage "
                                 f"points above peer regions on a ${region_revenue:,.0f} revenue base, "
                                 f"directly eroding margin on every sale in the region.",
                estimated_benefit=f"~${recoverable_margin:,.0f} in recoverable margin if {region}'s "
                                   f"discount rate is brought in line with the peer average.",
                benefit_value=recoverable_margin,
            ))
    return issues


def detect_low_inventory(df: pd.DataFrame) -> list:
    """Flag drugs with average inventory below the safety threshold."""
    issues = []
    inv = df.groupby("Drug_Name")["Inventory"].mean()
    drug_revenue = df.groupby("Drug_Name")["Revenue"].sum()

    for drug, level in inv.items():
        if level < LOW_INVENTORY_THRESHOLD:
            # Conservative stockout-risk estimate: treat one month of this
            # drug's average historical revenue as the amount at risk if a
            # stockout interrupts fulfillment — a standard back-of-envelope
            # framing for inventory risk, not a demand forecast.
            months_in_data = max(df["Date"].nunique() // 30, 1) if "Date" in df.columns else 1
            monthly_revenue_at_risk = drug_revenue.get(drug, 0) / max(months_in_data, 1)

            issues.append(_make_issue(
                category="Inventory Risk",
                severity="High" if level < LOW_INVENTORY_THRESHOLD * 0.5 else "Medium",
                finding=f"Average inventory for {drug} is {level:.0f} units, "
                        f"below the {LOW_INVENTORY_THRESHOLD}-unit safety threshold.",
                recommendation=f"Increase replenishment priority for {drug} to avoid stockouts "
                                f"in high-demand regions.",
                business_impact=f"{drug} is running at {level:.0f} units of average inventory, "
                                 f"putting an estimated ~${monthly_revenue_at_risk:,.0f} of monthly "
                                 f"revenue at risk of a stockout-driven disruption.",
                estimated_benefit=f"~${monthly_revenue_at_risk:,.0f} in protected monthly revenue "
                                   f"by restoring {drug} above the safety threshold.",
                benefit_value=monthly_revenue_at_risk,
            ))
    return issues


def detect_revenue_decline(df: pd.DataFrame) -> list:
    """Flag an overall month-over-month revenue decline."""
    issues = []
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    monthly = d.set_index("Date").resample("ME")["Revenue"].sum()

    if len(monthly) >= 2 and monthly.iloc[-2] != 0:
        pct_change = (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100
        if pct_change < REVENUE_DECLINE_THRESHOLD:
            dollar_decline = monthly.iloc[-2] - monthly.iloc[-1]
            issues.append(_make_issue(
                category="Revenue Decline",
                severity="High" if pct_change < -15 else "Medium",
                finding=f"Revenue declined {pct_change:.1f}% in the most recent month "
                        f"compared to the prior month.",
                recommendation="Investigate regional and therapy-area breakdowns for the decline; "
                                "consider a targeted promotional push in the underperforming segments.",
                business_impact=f"The most recent month brought in ${dollar_decline:,.0f} less "
                                 f"revenue than the prior month ({pct_change:.1f}% decline).",
                estimated_benefit=f"~${dollar_decline:,.0f} in monthly revenue recovered if the "
                                   f"decline is reversed back to the prior month's run rate.",
                benefit_value=dollar_decline,
            ))
    return issues


def detect_underperforming_regions(df: pd.DataFrame) -> list:
    """Flag regions in the bottom quartile of revenue contribution."""
    issues = []
    region_rev = df.groupby("Region")["Revenue"].sum().sort_values()
    if len(region_rev) < 3:
        return issues

    cutoff = region_rev.quantile(UNDERPERFORMER_PERCENTILE)
    peer_avg = region_rev.mean()

    for region, rev in region_rev.items():
        if rev <= cutoff:
            share = rev / region_rev.sum() * 100
            gap_to_avg = max(peer_avg - rev, 0)
            issues.append(_make_issue(
                category="Underperforming Region",
                severity="Medium",
                finding=f"{region} contributes only {share:.1f}% of total revenue, "
                        f"the lowest among all regions.",
                recommendation=f"Evaluate root cause in {region} — sales rep coverage, "
                                f"hospital contracts, or marketing spend allocation.",
                business_impact=f"{region} generates ${rev:,.0f}, ${gap_to_avg:,.0f} below the "
                                 f"${peer_avg:,.0f} average across all regions.",
                estimated_benefit=f"~${gap_to_avg:,.0f} in additional revenue if {region} were "
                                   f"brought up to the all-region average.",
                benefit_value=gap_to_avg,
            ))
    return issues


def detect_underperforming_therapy_areas(df: pd.DataFrame) -> list:
    """Flag therapy areas in the bottom quartile of revenue contribution."""
    issues = []
    therapy_rev = df.groupby("Therapy_Area")["Revenue"].sum().sort_values()
    if len(therapy_rev) < 3:
        return issues

    cutoff = therapy_rev.quantile(UNDERPERFORMER_PERCENTILE)
    peer_avg = therapy_rev.mean()

    for area, rev in therapy_rev.items():
        if rev <= cutoff:
            share = rev / therapy_rev.sum() * 100
            gap_to_avg = max(peer_avg - rev, 0)
            issues.append(_make_issue(
                category="Underperforming Therapy Area",
                severity="Low",
                finding=f"{area} contributes only {share:.1f}% of total revenue.",
                recommendation=f"Assess whether {area} warrants increased marketing investment "
                                f"or is a strategic candidate for portfolio deprioritization.",
                business_impact=f"{area} generates ${rev:,.0f}, ${gap_to_avg:,.0f} below the "
                                 f"${peer_avg:,.0f} average across all therapy areas.",
                estimated_benefit=f"~${gap_to_avg:,.0f} in additional revenue if {area} were "
                                   f"brought up to the all-therapy-area average.",
                benefit_value=gap_to_avg,
            ))
    return issues


def run_decision_engine(df: pd.DataFrame) -> list:
    """
    Run all detectors and return a combined, severity-sorted list of issues.
    This is the single function app.py calls for the Decision Engine page.

    Priority (P1, P2, ...) is assigned after sorting: first by severity
    (High > Medium > Low), then by estimated dollar benefit (largest
    first) within the same severity tier. This gives a defensible,
    fully-transparent ranking — no black-box scoring.
    """
    if df.empty:
        return []

    all_issues = []
    all_issues += detect_discount_leakage(df)
    all_issues += detect_low_inventory(df)
    all_issues += detect_revenue_decline(df)
    all_issues += detect_underperforming_regions(df)
    all_issues += detect_underperforming_therapy_areas(df)

    severity_order = {"High": 0, "Medium": 1, "Low": 2}
    all_issues.sort(key=lambda x: (severity_order.get(x["severity"], 3), -x.get("benefit_value", 0)))

    for idx, issue in enumerate(all_issues, start=1):
        issue["priority"] = f"P{idx}"

    return all_issues
