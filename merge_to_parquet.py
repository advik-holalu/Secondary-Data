# merge_to_parquet.py
import pandas as pd

# ---------- CONFIG ----------
SOURCE_FILE = "BusinessOverview.xlsx"
TARGET_FILE = "secondary_sales.parquet"
EXCLUDE_MONTHS = ["jan", "feb", "mar", "nov"]  # ignore these
SHEETS = ["Amazon Secondary", "Blinkit Secondary", "Instamart Secondary", "Zepto Secondary"]

# ---------- HELPERS ----------
MONTH_ORDER = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
def normalize_month(val):
    if pd.isna(val):
        return None
    s = str(val).strip().lower()[:3]
    if s.startswith("sep"):
        s = "sep"
    return s if s in MONTH_ORDER else None

# ---------- MERGE ----------
frames = []
xls = pd.ExcelFile(SOURCE_FILE)
for s in SHEETS:
    if s not in xls.sheet_names:
        print(f"⚠️ Sheet '{s}' not found, skipping.")
        continue
    df = pd.read_excel(xls, sheet_name=s)
    keep = ["Item Name","City Name","State Name","Region Name","Qty Sold","Cat 1",
            "Month","Year","Day","Revenue","GMV","Platform","Primary Cat"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["Platform"] = s.replace(" Secondary","")
    df["MonthKey"] = df["Month"].apply(normalize_month)
    df = df[~df["MonthKey"].isna()]
    df["MonthNum"] = df["MonthKey"].apply(lambda m: MONTH_ORDER.index(m)+1)
    df = df[~df["MonthKey"].isin(EXCLUDE_MONTHS)]
    frames.append(df)

if not frames:
    raise ValueError("No valid sheets found!")

merged = pd.concat(frames, ignore_index=True)
merged = merged.sort_values(by=["Year","MonthNum"]).reset_index(drop=True)
merged.to_parquet(TARGET_FILE, index=False)

print(f"✅ Saved merged parquet file '{TARGET_FILE}' with shape {merged.shape}")
