import pandas as pd

df = pd.read_excel("BusinessOverview.xlsx", sheet_name="P Type")
df.to_parquet("ptype.parquet", index=False)
