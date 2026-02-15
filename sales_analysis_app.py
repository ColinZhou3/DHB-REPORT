import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

# =========================
# Page config
# =========================
st.set_page_config(
    page_title="Sales & Price Trend (Keyword Select)",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Sales & Price Trend Analysis")
st.caption("Upload raw wide-format data (date_SALES + date_VOLUME columns). Then search products by keyword, pick SKUs, and compare trends.")
st.markdown("---")

# =========================
# Helpers
# =========================
def _safe_str(s):
    if pd.isna(s):
        return ""
    return str(s)

def keyword_filter(df: pd.DataFrame, keyword_text: str, search_cols: list[str]) -> pd.DataFrame:
    kw = (keyword_text or "").strip()
    if kw == "":
        return df

    # Supports:
    # "grape seedless" (AND)
    # "grape, seedless" (AND across all tokens anyway)
    # "grape -raisin" (exclude)
    parts = re.split(r"[,\n]+", kw)
    include_terms = []
    exclude_terms = []

    for p in parts:
        p = p.strip()
        if not p:
            continue
        for t in p.split():
            if t.startswith("-") and len(t) > 1:
                exclude_terms.append(t[1:].lower())
            else:
                include_terms.append(t.lower())

    # Build single search text per row
    txt = None
    for c in search_cols:
        if c in df.columns:
            col_txt = df[c].fillna("").astype(str)
            txt = col_txt if txt is None else (txt + " " + col_txt)

    if txt is None:
        return df

    txt = txt.str.lower()

    mask = pd.Series(True, index=df.index)
    for t in include_terms:
        mask &= txt.str.contains(re.escape(t), na=False)
    for t in exclude_terms:
        mask &= ~txt.str.contains(re.escape(t), na=False)

    return df[mask]

def detect_date_cols(columns: list[str]):
    sales_cols = [c for c in columns if str(c).endswith("_SALES")]
    vol_cols   = [c for c in columns if str(c).endswith("_VOLUME")]
    return sales_cols, vol_cols

def parse_date_from_col(col_name: str, suffix: str):
    # remove suffix, parse with dayfirst + flexible formats
    base = str(col_name).replace(suffix, "")
    dt = pd.to_datetime(base, dayfirst=True, errors="coerce")
    return dt

@st.cache_data(show_spinner=False)
def load_data(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded, low_memory=False)
    else:
        df = pd.read_excel(uploaded)
    return df

@st.cache_data(show_spinner=True)
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    # detect columns
    sales_cols, vol_cols = detect_date_cols(list(df.columns))
    if len(sales_cols) == 0 or len(vol_cols) == 0:
        raise ValueError("Cannot find columns ending with _SALES and _VOLUME. Please check your file format.")

    # id columns: keep whatever exists
    base_id = [
        "PRODUCT_NAME", "PRODUCT_CODE", "PRODUCT_GROUP",
        "FORMAT_NAME", "STORE_NAME", "STORE_REGION", "STORE_CODE"
    ]
    id_vars = [c for c in base_id if c in df.columns]
    if "PRODUCT_NAME" not in id_vars:
        raise ValueError("Missing required column: PRODUCT_NAME")

    # melt
    sales_long = df.melt(id_vars=id_vars, value_vars=sales_cols, var_name="DateCol", value_name="Sales")
    vol_long   = df.melt(id_vars=id_vars, value_vars=vol_cols,   var_name="DateCol", value_name="Volume")

    # parse date
    sales_long["Date"] = sales_long["DateCol"].map(lambda x: parse_date_from_col(x, "_SALES"))
    vol_long["Date"]   = vol_long["DateCol"].map(lambda x: parse_date_from_col(x, "_VOLUME"))

    sales_long = sales_long.drop(columns=["DateCol"])
    vol_long   = vol_long.drop(columns=["DateCol"])

    # drop bad date rows
    sales_long = sales_long.dropna(subset=["Date"])
    vol_long   = vol_long.dropna(subset=["Date"])

    # numeric
    sales_long["Sales"] = pd.to_numeric(sales_long["Sales"], errors="coerce")
    vol_long["Volume"]  = pd.to_numeric(vol_long["Volume"], errors="coerce")

    # merge with full keys
    merge_keys = id_vars + ["Date"]
    data_long = sales_long.merge(
        vol_long[merge_keys + ["Volume"]],
        on=merge_keys,
        how="left"
    )

    return data_long

def build_label_table(data_long: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["PRODUCT_NAME","PRODUCT_CODE","PRODUCT_GROUP","FORMAT_NAME"] if c in data_long.columns]
    prod = data_long[cols].drop_duplicates()

    # label
    if "PRODUCT_CODE" in prod.columns:
        prod["label"] = (
            prod["PRODUCT_NAME"].astype(str)
            + " | " + prod["PRODUCT_CODE"].astype(str)
            + ((" | " + prod["PRODUCT_GROUP"].astype(str)) if "PRODUCT_GROUP" in prod.columns else "")
        )
    else:
        prod["label"] = prod["PRODUCT_NAME"].astype(str)

    return prod.sort_values("label")

def kpi_summary(filtered_long: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # Aggregate weekly/daily as-is, then compute change from first to last date
    if filtered_long.empty:
        return pd.DataFrame()

    agg = (
        filtered_long
        .groupby(group_cols + ["Date"], dropna=False, as_index=False)
        .agg(Sales=("Sales","sum"), Volume=("Volume","sum"))
    )
    agg["Avg_Price"] = agg["Sales"] / agg["Volume"]
    agg["Avg_Price"] = agg["Avg_Price"].replace([np.inf, -np.inf], np.nan)

    first_date = agg["Date"].min()
    last_date = agg["Date"].max()

    first = agg[agg["Date"] == first_date].copy()
    last  = agg[agg["Date"] == last_date].copy()

    key = group_cols
    out = first.merge(last, on=key, suffixes=("_first","_last"), how="outer")

    for col in ["Sales","Volume","Avg_Price"]:
        out[f"{col}_chg"] = out[f"{col}_last"] - out[f"{col}_first"]
        out[f"{col}_chg_pct"] = np.where(
            out[f"{col}_first"].isna() | (out[f"{col}_first"] == 0),
            np.nan,
            out[f"{col}_chg"] / out[f"{col}_first"]
        )

    # sort by sales change
    out = out.sort_values("Sales_chg", ascending=False)
    return out

# =========================
# UI: Upload
# =========================
uploaded_file = st.file_uploader("Upload file (xlsx / xls / csv)", type=["xlsx","xls","csv"])

if uploaded_file is None:
    st.info("Upload your raw data file to start.")
    st.stop()

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Failed to read file: {e}")
    st.stop()

with st.spinner("Processing raw data (wide → long). For large files, this can take a while..."):
    try:
        data_long = process_data(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

# =========================
# Sidebar filters
# =========================
st.sidebar.header("Filters")

# Keyword search → product selector
prod_table = build_label_table(data_long)

kw = st.sidebar.text_input("Product keyword search", value="", placeholder="example: grape seedless  |  grape -raisin")
prod_filtered = keyword_filter(prod_table, kw, ["PRODUCT_NAME","PRODUCT_GROUP","label"])

max_show = st.sidebar.slider("Max search results", 20, 300, 80)
prod_filtered = prod_filtered.head(max_show)

default_n = st.sidebar.slider("Default select N", 1, 10, 3)
default_labels = prod_filtered["label"].tolist()[:default_n]

selected_labels = st.sidebar.multiselect(
    "Select products to compare",
    options=prod_filtered["label"].tolist(),
    default=default_labels
)

# Convert selection to a filter
filtered = data_long.copy()
if selected_labels:
    chosen = prod_filtered[prod_filtered["label"].isin(selected_labels)]
    if "PRODUCT_CODE" in chosen.columns:
        codes = chosen["PRODUCT_CODE"].astype(str).unique().tolist()
        filtered = filtered[filtered["PRODUCT_CODE"].astype(str).isin(codes)]
    else:
        names = chosen["PRODUCT_NAME"].astype(str).unique().tolist()
        filtered = filtered[filtered["PRODUCT_NAME"].astype(str).isin(names)]

# Banner filter
if "FORMAT_NAME" in filtered.columns:
    banners = sorted(filtered["FORMAT_NAME"].dropna().unique().tolist())
    selected_banners = st.sidebar.multiselect("Banners (FORMAT_NAME)", options=banners, default=banners)
    if selected_banners:
        filtered = filtered[filtered["FORMAT_NAME"].isin(selected_banners)]

# Store filter (optional)
if "STORE_NAME" in filtered.columns:
    stores = sorted(filtered["STORE_NAME"].dropna().unique().tolist())
    store_mode = st.sidebar.radio("Store filter", ["All stores", "Pick stores"], index=0)
    if store_mode == "Pick stores":
        selected_stores = st.sidebar.multiselect("Stores", options=stores, default=stores[:10])
        if selected_stores:
            filtered = filtered[filtered["STORE_NAME"].isin(selected_stores)]

# Date range
min_date = filtered["Date"].min()
max_date = filtered["Date"].max()

if pd.isna(min_date) or pd.isna(max_date):
    st.warning("No valid dates found after filters.")
    st.stop()

start_date, end_date = st.sidebar.date_input(
    "Date range",
    value=(min_date.date(), max_date.date()),
    min_value=min_date.date(),
    max_value=max_date.date()
)

filtered = filtered[(filtered["Date"].dt.date >= start_date) & (filtered["Date"].dt.date <= end_date)]

# Grouping control
group_by = st.sidebar.selectbox(
    "Compare by",
    options=[
        "Product only",
        "Product + Banner",
        "Product + Store"
    ],
    index=1
)

if group_by == "Product only":
    group_cols = [c for c in ["PRODUCT_NAME","PRODUCT_CODE"] if c in filtered.columns]
elif group_by == "Product + Banner":
    group_cols = [c for c in ["PRODUCT_NAME","PRODUCT_CODE","FORMAT_NAME"] if c in filtered.columns]
else:
    group_cols = [c for c in ["PRODUCT_NAME","PRODUCT_CODE","STORE_NAME"] if c in filtered.columns]

# =========================
# Aggregations
# =========================
agg = (
    filtered
    .groupby(group_cols + ["Date"], dropna=False, as_index=False)
    .agg(Sales=("Sales","sum"), Volume=("Volume","sum"))
)
agg["Avg_Price"] = agg["Sales"] / agg["Volume"]
agg["Avg_Price"] = agg["Avg_Price"].replace([np.inf, -np.inf], np.nan)

# Label for charts
def _make_series_name(row):
    parts = []
    for c in group_cols:
        if c in row and not pd.isna(row[c]):
            parts.append(str(row[c]))
    return " | ".join(parts) if parts else "Series"

if len(group_cols) == 1:
    agg["Series"] = agg[group_cols[0]].astype(str)
else:
    agg["Series"] = agg.apply(_make_series_name, axis=1)

# =========================
# Main layout
# =========================
colA, colB, colC = st.columns(3)
colA.metric("Rows (filtered)", f"{len(filtered):,}")
colB.metric("Total Sales", f"{agg['Sales'].sum():,.2f}")
colC.metric("Avg Price (weighted)", f"{(agg['Sales'].sum() / agg['Volume'].sum()) if agg['Volume'].sum() else np.nan:.3f}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["Sales Trend", "Avg Price Trend", "Volume Trend", "Key Findings"])

with tab1:
    fig = px.line(agg, x="Date", y="Sales", color="Series", markers=True)
    fig.update_layout(height=520, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.line(agg, x="Date", y="Avg_Price", color="Series", markers=True)
    fig.update_layout(height=520, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = px.bar(agg, x="Date", y="Volume", color="Series", barmode="group")
    fig.update_layout(height=520, legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    kpi = kpi_summary(filtered, group_cols=group_cols)
    if kpi.empty:
        st.info("No data for KPI after filters.")
    else:
        show_cols = group_cols + [
            "Sales_first","Sales_last","Sales_chg","Sales_chg_pct",
            "Avg_Price_first","Avg_Price_last","Avg_Price_chg","Avg_Price_chg_pct",
            "Volume_first","Volume_last","Volume_chg","Volume_chg_pct"
        ]
        keep = [c for c in show_cols if c in kpi.columns]
        st.dataframe(kpi[keep], use_container_width=True, height=520)

# =========================
# Downloads
# =========================
st.markdown("---")
st.subheader("Download")

# aggregated data CSV
csv_bytes = agg.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download aggregated data (CSV)",
    data=csv_bytes,
    file_name="trend_aggregated.csv",
    mime="text/csv"
)

# raw filtered CSV (careful size)
raw_limit = st.checkbox("I understand: raw filtered file can be large", value=False)
if raw_limit:
    raw_csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download raw filtered data (CSV)",
        data=raw_csv,
        file_name="raw_filtered.csv",
        mime="text/csv"
    )
