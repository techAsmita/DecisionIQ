"""
modules/analytics.py
=====================
Business Intelligence layer: computes KPIs and builds Plotly charts
from the sales dataframe.

Design decision: this module returns plain numbers/dicts and Plotly
figure objects — it does NOT call st.* directly. That keeps analytics
logic testable and reusable (e.g. the same compute_kpis() function
feeds both the dashboard AND the PDF report).

v2 changes (enterprise polish pass):
- Every chart now runs through apply_enterprise_layout() so title font,
  margins, gridlines, and hover behavior are identical across the app
  (matches the palette defined in app.py: blue primary, green success,
  orange warning, red critical).
- discount_chart() simplified — the previous groupby(...).apply(...)
  round-trip recomputed the same grouping twice just to reattach labels.
- compute_kpis() unchanged in signature/return shape, so app.py's KPI
  cards keep working without modification.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Enterprise theme constants (kept in sync with the CSS palette in app.py)
# ---------------------------------------------------------------------------
PRIMARY = "#2563EB"
SUCCESS = "#16A34A"
WARNING = "#F59E0B"
CRITICAL = "#DC2626"
INK = "#0F172A"
MUTED = "#64748B"

# Sequential/discrete palettes reused across bar & pie charts so every
# categorical chart in the app draws from the same visual language.
SEQUENTIAL_BLUES = ["#DBEAFE", "#93C5FD", "#3B82F6", "#1D4ED8", "#1E3A8A"]
CATEGORY_PALETTE = [PRIMARY, SUCCESS, WARNING, CRITICAL, "#7C3AED", "#0891B2"]


def apply_enterprise_layout(fig: go.Figure, height: int = 380) -> go.Figure:
    """Apply the shared enterprise look to any Plotly figure in the app.

    Centralizing this means every chart — dashboard, decision engine,
    forecasting, PDF report — gets identical typography, spacing, and
    gridline treatment with a single call.
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Segoe UI, system-ui, sans-serif", size=13, color=INK),
        title_font=dict(size=16, color=INK),
        margin=dict(l=40, r=30, t=60, b=40),
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E2E8F0")
    fig.update_yaxes(showgrid=True, gridcolor="#F1F5F9", zeroline=False)
    return fig


def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute headline KPIs used across the dashboard and PDF report."""
    if df.empty:
        return {}

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    total_revenue = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    profit_margin = (total_profit / total_revenue * 100) if total_revenue else 0
    total_orders = df["Transaction_ID"].nunique()
    avg_order_value = total_revenue / total_orders if total_orders else 0

    region_revenue = df.groupby("Region")["Revenue"].sum()
    best_region = region_revenue.idxmax() if not region_revenue.empty else "N/A"
    worst_region = region_revenue.idxmin() if not region_revenue.empty else "N/A"

    therapy_revenue = df.groupby("Therapy_Area")["Revenue"].sum()
    top_therapy = therapy_revenue.idxmax() if not therapy_revenue.empty else "N/A"

    drug_revenue = df.groupby("Drug_Name")["Revenue"].sum()
    top_drug = drug_revenue.idxmax() if not drug_revenue.empty else "N/A"

    # Monthly growth: compare latest full month vs the one before it
    monthly = df.set_index("Date").resample("ME")["Revenue"].sum()
    monthly_growth = 0.0
    if len(monthly) >= 2 and monthly.iloc[-2] != 0:
        monthly_growth = (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100

    # Quarterly growth: compare latest quarter vs the one before it
    quarterly = df.set_index("Date").resample("QE")["Revenue"].sum()
    quarterly_growth = 0.0
    if len(quarterly) >= 2 and quarterly.iloc[-2] != 0:
        quarterly_growth = (quarterly.iloc[-1] - quarterly.iloc[-2]) / quarterly.iloc[-2] * 100

    return {
        "total_revenue": total_revenue,
        "total_profit": total_profit,
        "profit_margin": profit_margin,
        "total_orders": total_orders,
        "avg_order_value": avg_order_value,
        "best_region": best_region,
        "worst_region": worst_region,
        "top_therapy": top_therapy,
        "top_drug": top_drug,
        "monthly_growth": monthly_growth,
        "quarterly_growth": quarterly_growth,
    }


def revenue_trend_chart(df: pd.DataFrame) -> go.Figure:
    monthly = df.copy()
    monthly["Date"] = pd.to_datetime(monthly["Date"])
    monthly = monthly.set_index("Date").resample("ME")["Revenue"].sum().reset_index()
    fig = px.line(monthly, x="Date", y="Revenue", markers=True,
                  title="Revenue Trend (Monthly)",
                  color_discrete_sequence=[PRIMARY])
    return apply_enterprise_layout(fig)


def profit_trend_chart(df: pd.DataFrame) -> go.Figure:
    monthly = df.copy()
    monthly["Date"] = pd.to_datetime(monthly["Date"])
    monthly = monthly.set_index("Date").resample("ME")["Profit"].sum().reset_index()
    fig = px.line(monthly, x="Date", y="Profit", markers=True,
                  title="Profit Trend (Monthly)",
                  color_discrete_sequence=[SUCCESS])
    return apply_enterprise_layout(fig)


def regional_revenue_chart(df: pd.DataFrame) -> go.Figure:
    region_rev = df.groupby("Region", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    fig = px.bar(region_rev, x="Region", y="Revenue", title="Revenue by Region",
                 color="Revenue", color_continuous_scale=SEQUENTIAL_BLUES)
    return apply_enterprise_layout(fig)


def therapy_revenue_chart(df: pd.DataFrame) -> go.Figure:
    therapy_rev = df.groupby("Therapy_Area", as_index=False)["Revenue"].sum().sort_values("Revenue", ascending=False)
    fig = px.bar(therapy_rev, x="Therapy_Area", y="Revenue", title="Revenue by Therapy Area",
                 color="Revenue", color_continuous_scale=SEQUENTIAL_BLUES)
    return apply_enterprise_layout(fig)


def top_drugs_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    drug_rev = df.groupby("Drug_Name", as_index=False)["Revenue"].sum().sort_values(
        "Revenue", ascending=False).head(top_n)
    fig = px.bar(drug_rev, x="Revenue", y="Drug_Name", orientation="h",
                 title=f"Top {top_n} Drugs by Revenue",
                 color_discrete_sequence=[PRIMARY])
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return apply_enterprise_layout(fig)


def inventory_chart(df: pd.DataFrame) -> go.Figure:
    inv = df.groupby("Drug_Name", as_index=False)["Inventory"].mean().sort_values("Inventory")
    fig = px.bar(inv, x="Drug_Name", y="Inventory", title="Average Inventory by Drug",
                 color="Inventory", color_continuous_scale=["#FCA5A5", CRITICAL])
    return apply_enterprise_layout(fig)


def discount_chart(df: pd.DataFrame) -> go.Figure:
    # Discount rate per region = total discount / total revenue for that
    # region. A single groupby with .agg() avoids the double-grouping
    # round-trip the previous version used just to reattach region labels.
    grouped = df.groupby("Region").agg(
        Discount_Sum=("Discount", "sum"),
        Revenue_Sum=("Revenue", "sum"),
    ).reset_index()
    grouped["Discount_Rate"] = grouped.apply(
        lambda row: (row["Discount_Sum"] / row["Revenue_Sum"] * 100) if row["Revenue_Sum"] else 0,
        axis=1,
    )

    fig = px.bar(grouped, x="Region", y="Discount_Rate", title="Discount Rate by Region (%)",
                 color="Discount_Rate", color_continuous_scale=["#FED7AA", WARNING])
    return apply_enterprise_layout(fig)


def customer_segmentation_chart(df: pd.DataFrame) -> go.Figure:
    seg = df.groupby("Customer_Type", as_index=False)["Revenue"].sum()
    fig = px.pie(seg, names="Customer_Type", values="Revenue", title="Revenue by Customer Type",
                 hole=0.4, color_discrete_sequence=CATEGORY_PALETTE)
    return apply_enterprise_layout(fig)


def profit_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(index="Region", columns="Therapy_Area", values="Profit", aggfunc="sum", fill_value=0)
    fig = px.imshow(pivot, text_auto=".2s", aspect="auto",
                     title="Profit Heatmap: Region x Therapy Area",
                     color_continuous_scale="RdYlGn")
    return apply_enterprise_layout(fig, height=420)
