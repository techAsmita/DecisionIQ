"""
data/generate_sample_data.py
=============================
One-off script to generate a realistic synthetic pharma sales dataset
for development/testing/demo purposes. Run once:

    python data/generate_sample_data.py

Produces data/sample_sales.csv (~5000 rows) with believable patterns:
- Oncology & Cardiology dominate revenue (realistic for pharma)
- North region has higher discounting (feeds Module 3's recommendations)
- Revenue trends upward with seasonality (feeds Module 4's forecasting)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

REGIONS = ["North", "South", "East", "West", "Central"]
COUNTRIES = {"North": "USA", "South": "Brazil", "East": "India",
             "West": "Germany", "Central": "UAE"}

DRUGS = {
    "Oncology": ["Keytruda", "Opdivo", "Herceptin", "Avastin"],
    "Cardiology": ["Eliquis", "Entresto", "Brilinta"],
    "Neurology": ["Aduhelm", "Vyepti", "Nurtec"],
    "Diabetes": ["Ozempic", "Jardiance", "Trulicity"],
    "Vaccines": ["Comirnaty", "Shingrix", "Prevnar"],
}

DOCTOR_SEGMENTS = ["High Prescriber", "Medium Prescriber", "Low Prescriber", "Non-Prescriber"]
CUSTOMER_TYPES = ["Hospital", "Retail Pharmacy", "Government", "Online Pharmacy"]
HOSPITALS = ["Mercy General", "St. Luke's", "City Medical Center", "Apollo Hospital",
             "Cleveland Partners", "Metro Health", "Sunrise Clinic"]
REPS = [f"Rep_{i:03d}" for i in range(1, 41)]

n_rows = 5000
start_date = datetime(2023, 1, 1)
end_date = datetime(2025, 12, 31)
date_range_days = (end_date - start_date).days

rows = []
for i in range(1, n_rows + 1):
    region = np.random.choice(REGIONS, p=[0.30, 0.20, 0.20, 0.15, 0.15])
    therapy_area = np.random.choice(list(DRUGS.keys()), p=[0.35, 0.25, 0.15, 0.15, 0.10])
    drug = np.random.choice(DRUGS[therapy_area])

    day_offset = np.random.randint(0, date_range_days)
    date = start_date + timedelta(days=day_offset)
    # mild upward trend + seasonality
    trend_factor = 1 + (day_offset / date_range_days) * 0.4
    seasonal_factor = 1 + 0.15 * np.sin(2 * np.pi * date.month / 12)

    units_sold = int(np.random.gamma(shape=5, scale=40) * trend_factor * seasonal_factor)
    unit_price = np.random.uniform(50, 800)
    revenue = round(units_sold * unit_price, 2)

    # North region deliberately over-discounts (fuels Module 3 recommendation demo)
    base_discount_rate = 0.18 if region == "North" else np.random.uniform(0.03, 0.10)
    discount = round(revenue * base_discount_rate, 2)

    manufacturing_cost = round(revenue * np.random.uniform(0.25, 0.45), 2)
    profit = round(revenue - manufacturing_cost - discount, 2)
    marketing_spend = round(revenue * np.random.uniform(0.02, 0.08), 2)

    # Ozempic deliberately runs low inventory (fuels Module 3 recommendation demo)
    if drug == "Ozempic":
        inventory = int(np.random.uniform(5, 40))
    else:
        inventory = int(np.random.uniform(50, 500))

    rows.append({
        "Transaction_ID": f"TXN{i:06d}",
        "Date": date.strftime("%Y-%m-%d"),
        "Region": region,
        "Country": COUNTRIES[region],
        "Drug_Name": drug,
        "Therapy_Area": therapy_area,
        "Hospital": np.random.choice(HOSPITALS),
        "Doctor_Segment": np.random.choice(DOCTOR_SEGMENTS),
        "Sales_Representative": np.random.choice(REPS),
        "Units_Sold": units_sold,
        "Revenue": revenue,
        "Discount": discount,
        "Manufacturing_Cost": manufacturing_cost,
        "Profit": profit,
        "Marketing_Spend": marketing_spend,
        "Inventory": inventory,
        "Customer_Type": np.random.choice(CUSTOMER_TYPES),
        "Quarter": f"Q{(date.month - 1) // 3 + 1}",
        "Month": date.strftime("%B"),
    })

df = pd.DataFrame(rows)
df.to_csv("data/sample_sales.csv", index=False)
print(f"Generated {len(df)} rows -> data/sample_sales.csv")
print(df.head())
