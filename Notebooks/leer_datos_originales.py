import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import re


def clean(n):
    return re.sub(r"[^a-zA-Z0-9_]", "_", n.strip().lower()).strip("_")


df = pd.read_csv(
    "cfn_volumencreditocontingente_2025_enero-septiembre.csv",
    sep=";",
    encoding="latin-1",
)

df = df[df["SECTOR"] != "ACTIVIDADES FINANCIERAS Y DE SEGUROS"].copy()

df["SECTOR"] = df["SECTOR"].str.replace(
    "AGRICULTURA GANADERÍA  SILVICULTURA Y PESCA",
    "AGRICULTURA, GANADERÍA, SILVICULTURA Y PESCA",
)
df["SECTOR"] = df["SECTOR"].str.replace(
    "AGRICULTURA, GANADERÍA,  SILVICULTURA Y PESCA",
    "AGRICULTURA, GANADERÍA, SILVICULTURA Y PESCA",
)

df.columns = [clean(c) for c in df.columns]

# Drop unused
drop_cols = [
    "canton",
    "subsector",
    "actividad",
    "subsistema",
    "entidad",
    "ruc",
    "nombre",
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Simplify Categories
for col in ["provincia", "sector"]:
    top = df[col].value_counts().nlargest(6).index
    df[col] = df[col].apply(lambda x: x if x in top else "OTROS")

date_col = [c for c in df.columns if "fecha" in c][0]
df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
df.dropna(subset=[date_col], inplace=True)
df["quarter"] = df[date_col].dt.to_period("Q").astype(str)

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].astype("category")
        print(f"{col}: {np.sort(df[col].unique()).tolist()}")


# We log the target immediately.
target = "monto_otorgado"
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df[df[target] > 0].copy()  # Log requires > 0
df["log_monto"] = np.log(df[target])  # Y -> log(Y)

# Drop unused
drop_cols = [
    date_col,
    target,
]
df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

# Dummies
df_final = pd.get_dummies(df, drop_first=True, dtype=int)
df_final.columns = [clean(c) for c in df_final.columns]

# Remove Duplicates & NaT
df_final = df_final.loc[:, ~df_final.columns.duplicated()]
df_final = df_final.loc[:, [c for c in df_final.columns if "nat" not in c]]

# Save Cleaned Data
df_final.to_csv("cfn_volumencreditocontingente_cleaned.csv", index=False)
