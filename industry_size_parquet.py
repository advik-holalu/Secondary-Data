import pandas as pd

# ---------------------------
# CONFIG
# ---------------------------
SOURCE_FILE = "BusinessOverview.xlsx"
TARGET_FILE = "industry_size.parquet"
SHEET_NAME = "Category Size"

# ---------------------------
# MONTH NORMALIZATION
# ---------------------------
month_map = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12
}

month_label_map = {
    1: "Jan", 2: "Feb", 3: "Mar",
    4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep",
    10: "Oct", 11: "Nov", 12: "Dec"
}

# ---------------------------
# LOAD
# ---------------------------
xls = pd.ExcelFile(SOURCE_FILE)
if SHEET_NAME not in xls.sheet_names:
    raise ValueError(f"❌ Sheet '{SHEET_NAME}' not found!")

df = pd.read_excel(xls, sheet_name=SHEET_NAME)

# ---------------------------
# CLEANING & VALIDATION
# ---------------------------
required_cols = ["Primary Cat", "City Name", "Month", "Industry Size", "Platform"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"❌ Missing columns in Category Size sheet: {missing}")

df = df[required_cols].copy()

# Normalize text columns
for col in ["Primary Cat", "City Name", "Platform", "Month"]:
    df[col] = df[col].astype(str).str.strip()

# ---------------------------
# MONTH PROCESSING
# ---------------------------
df["MonthKey"] = df["Month"].str.lower().str[:3]
df["MonthNum"] = df["MonthKey"].map(month_map)

# Drop rows where month could not be parsed
df = df.dropna(subset=["MonthNum"])

df["MonthNum"] = df["MonthNum"].astype(int)
df["MonthLabel"] = df["MonthNum"].map(month_label_map)

# ---------------------------
# FINAL SORT
# ---------------------------
df = (
    df.sort_values(
        ["Primary Cat", "Platform", "City Name", "MonthNum"],
        ascending=[True, True, True, True]
    )
    .reset_index(drop=True)
)

# ---------------------------
# SAVE
# ---------------------------
df.to_parquet(TARGET_FILE, index=False)
print(f"✅ Saved industry size parquet → {TARGET_FILE} | rows={len(df)}")
