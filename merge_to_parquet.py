# merge_to_parquet.py
import pandas as pd

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
SOURCE_FILE = "BusinessOverview.xlsx"
TARGET_FILE = "secondary_sales.parquet"

SHEETS = [
    "Amazon Secondary",
    "Blinkit Secondary",
    "Instamart Secondary",
    "Zepto Secondary"
]

# --------------------------------------------------
# MONTH HELPERS
# --------------------------------------------------
MONTH_ORDER = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]

def normalize_month(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    s = s[:3]   # jan, feb, mar...
    if s == "sep":
        return "sep"
    return s if s in MONTH_ORDER else None

# --------------------------------------------------
# MERGE LOGIC
# --------------------------------------------------
frames = []
xls = pd.ExcelFile(SOURCE_FILE)

for sheet in SHEETS:
    if sheet not in xls.sheet_names:
        print(f"⚠️ Sheet '{sheet}' not found, skipping.")
        continue

    df = pd.read_excel(xls, sheet_name=sheet)

    # Keep only required columns if present
    keep_cols = [
        "Item Name",
        "City Name",
        "State Name",
        "Region Name",
        "Qty Sold",
        "Cat 1",
        "Month",
        "Year",
        "Day",
        "Revenue",
        "GMV",
        "Platform",
        "Primary Cat"
    ]

    df = df[[c for c in keep_cols if c in df.columns]].copy()

    # Platform from sheet name (single source of truth)
    df["Platform"] = sheet.replace(" Secondary", "").strip()

    # Normalize month
    df["MonthKey"] = df["Month"].apply(normalize_month)

    # Drop rows with invalid month
    df = df.dropna(subset=["MonthKey"])

    # Month number (1–12)
    df["MonthNum"] = df["MonthKey"].apply(lambda m: MONTH_ORDER.index(m) + 1)

    frames.append(df)

# --------------------------------------------------
# FINALIZE
# --------------------------------------------------
if not frames:
    raise ValueError("❌ No valid sheets found in Excel!")

merged = (
    pd.concat(frames, ignore_index=True)
    .sort_values(["Year", "MonthNum"])
    .reset_index(drop=True)
)

# --------------------------------------------------
# SAVE
# --------------------------------------------------
merged.to_parquet(TARGET_FILE, index=False)
print(f"✅ Saved merged parquet → {TARGET_FILE} | rows={len(merged)}")
