"""Exploratory Data Analysis (EDA) for the provided traffic metadata CSV.

Run from the RoadPulse-AI folder:
    python eda.py

Outputs are saved in eda_output/ (CSV summaries, text report, and PNG charts).
This explores metadata only; it does not analyze CCTV video pixels.
"""
from pathlib import Path
import re
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CANDIDATES = [
    ROOT / "data" / "traffic_dataset.csv",
    ROOT / "traffic_dataset.csv",
    ROOT.parent / "traffic_inspect" / "traffic_dataset.csv",
]
CSV_PATH = next((p for p in CANDIDATES if p.exists()), None)
OUT = ROOT / "eda_output"
OUT.mkdir(exist_ok=True)

if CSV_PATH is None:
    print("ERROR: traffic_dataset.csv was not found.")
    print("Place it in the project's data folder: RoadPulse-AI/data/traffic_dataset.csv")
    sys.exit(1)

# Load the dataset
df = pd.read_csv(CSV_PATH)
print(f"Loaded: {CSV_PATH}")
print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

# Clean column names and make a working copy.
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Parse YYYYMMDD values as dates while retaining the original date column.
if "date" in df.columns:
    df["date_parsed"] = pd.to_datetime(df["date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce")

# Extract CCTV ID from filenames when present.
if "filename" in df.columns:
    df["cctv_id"] = df["filename"].astype(str).str.extract(r"(cctv\d+)", flags=re.IGNORECASE, expand=False)

# 1) General overview and data quality
with (OUT / "eda_report.txt").open("w", encoding="utf-8") as f:
    f.write("ROADPULSE AI - EXPLORATORY DATA ANALYSIS\n")
    f.write("=" * 48 + "\n\n")
    f.write(f"Source file: {CSV_PATH}\n")
    f.write(f"Rows: {len(df)}\nColumns: {len(df.columns)}\n\n")
    f.write("COLUMNS AND DATA TYPES\n")
    f.write(df.dtypes.astype(str).to_string() + "\n\n")
    f.write("MISSING VALUES (count)\n")
    f.write(df.isna().sum().to_string() + "\n\n")
    f.write("DUPLICATE ROWS\n")
    f.write(f"{int(df.duplicated().sum())}\n\n")
    f.write("NUMERIC SUMMARY\n")
    f.write(df.select_dtypes(include="number").describe().round(3).to_string() + "\n\n")
    for col in ["traffic_class", "weather", "direction", "day_night", "date", "number_of_frames", "cctv_id", "notes"]:
        if col in df.columns:
            f.write(f"VALUE COUNTS: {col}\n")
            f.write(df[col].value_counts(dropna=False).to_string() + "\n\n")

# Save reusable summaries.
df.isna().sum().rename("missing_count").to_csv(OUT / "missing_values.csv")
df.describe(include="all").transpose().to_csv(OUT / "descriptive_summary.csv")
for col in ["traffic_class", "weather", "direction", "day_night", "date", "number_of_frames", "cctv_id"]:
    if col in df.columns:
        df[col].value_counts(dropna=False).rename_axis(col).reset_index(name="count").to_csv(OUT / f"{col}_counts.csv", index=False)

# Helper: save a readable bar chart.
def bar_chart(series, title, xlabel, ylabel, filename, rotate=0, color="#3478c8"):
    counts = series.value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.index = counts.index.map(lambda x: "Missing" if pd.isna(x) else str(x))
    ax.bar(counts.index, counts.values, color=color)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=rotate, ha="right" if rotate else "center")
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=160)
    plt.close(fig)

if "traffic_class" in df.columns:
    bar_chart(df["traffic_class"], "Traffic Class Distribution", "Traffic class", "Number of records", "traffic_class_distribution.png")
if "weather" in df.columns:
    bar_chart(df["weather"], "Weather Conditions in Dataset", "Weather", "Number of records", "weather_distribution.png")
if "date" in df.columns:
    bar_chart(df["date"].astype(str), "Records per Date", "Date (YYYYMMDD)", "Number of records", "records_by_date.png")
if "number_of_frames" in df.columns:
    bar_chart(df["number_of_frames"], "Number of Frames per Observation", "Number of frames", "Number of records", "number_of_frames_distribution.png")
if "cctv_id" in df.columns and df["cctv_id"].notna().any():
    bar_chart(df["cctv_id"], "CCTV IDs Represented", "CCTV ID", "Number of records", "cctv_id_distribution.png")

# Traffic class by weather stacked chart.
if {"traffic_class", "weather"}.issubset(df.columns):
    cross = pd.crosstab(df["weather"].fillna("Missing"), df["traffic_class"].fillna("Missing"))
    cross.to_csv(OUT / "traffic_class_by_weather.csv")
    ax = cross.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="tab20")
    ax.set_title("Traffic Class by Weather")
    ax.set_xlabel("Weather")
    ax.set_ylabel("Number of records")
    ax.legend(title="Traffic class")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "traffic_class_by_weather.png", dpi=160)
    plt.close()

# Date vs traffic class counts (if multiple dates exist).
if {"date", "traffic_class"}.issubset(df.columns):
    cross_date = pd.crosstab(df["date"].astype(str), df["traffic_class"].fillna("Missing"))
    cross_date.to_csv(OUT / "traffic_class_by_date.csv")
    ax = cross_date.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Traffic Class by Date")
    ax.set_xlabel("Date (YYYYMMDD)")
    ax.set_ylabel("Number of records")
    ax.legend(title="Traffic class")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUT / "traffic_class_by_date.png", dpi=160)
    plt.close()

# Print compact findings in terminal.
print("\n--- EDA highlights ---")
print(f"Rows/columns: {df.shape[0]}/{df.shape[1]}")
if "traffic_class" in df.columns:
    print("Traffic class counts:\n", df["traffic_class"].value_counts(dropna=False).to_string())
if "weather" in df.columns:
    print("\nWeather counts:\n", df["weather"].value_counts(dropna=False).to_string())
if "cctv_id" in df.columns:
    print("\nCCTV IDs found:", df["cctv_id"].dropna().unique().tolist())
print("\nEDA complete. Check the 'eda_output' folder for the report, CSV summaries, and charts.")
print("Note: this analyzes the CSV metadata only; it does not measure vehicles from video.")
