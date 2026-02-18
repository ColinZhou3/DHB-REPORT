import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px

st.set_page_config(page_title="Sales & Price Trend (Fast)", page_icon="📈", layout="wide")
st.title("📈 Sales & Price Trend Analysis")
st.caption("Filter first, then reshape only selected products + selected date range (much faster on Streamlit Cloud).")

BASE_ID_COLS = ["PRODUCT_NAME","PRODUCT_CODE","PRODUCT_GROUP","FORMAT_NAME","STORE_NAME","STORE_REGION","STORE_CODE"]

def detect_date_cols(columns):
    sales_cols = [c for c in columns if str(c).endswith("_SALES")]
    vol_cols   = [c for c in columns if str(c).endswith("_VOLUME")]
    return sales_cols, vol_cols

def parse_date_from_col(col_name: str, suffix: str):
    base = str(col_name).replace(suffix, "")
    return pd.to_datetime(base, dayfirst=True, errors="coerce")

def keyword_filter(df: pd.DataFrame, keyword_text: str, search_cols):
    """
    Friendly matching:
    - Space = AND inside a group
      example: "grape seedless" -> must contain grape AND seedless
    - Comma = OR between groups
      example: "grape, seedless" -> contains grape OR seedless
    - Exclude: "-xxx"
      example: "grape -raisin"
    """
    kw = (keyword_text or "").strip()
    if kw == "":
        return df

    groups = [g.strip() for g in re.split(r"[,\n]+", kw) if g.strip()]
    include_groups = []
    exclude_terms = []

    for g in groups:
        tokens = []
        for t in g.split():
            if t.startswith("-") and len(t) > 1:
                exclude_terms.append(t[1:].lower())
            else:
                tokens.append(t.lower())
        if tokens:
            include_groups.append(tokens)

    # Build single search text per row
    txt = None
    for c in search_cols:
        if c in df.columns:
            col_txt = df[c].fillna("").astype(str)
            txt = col_txt if txt is None else (txt + " " + col_txt)

    if txt is None:
        return df

    txt = txt.str.lower()

    # OR across groups, AND inside each group
    if include_groups:
        mask_any = pd.Series(False, index=df.index)
        for tokens in include_groups:
            mask_g = pd.Series(True, index=df.index)
            for t in tokens:
                mask_g &= txt.str.contains(re.escape(t), na=False)
            mask_any |= mask_g
        mask = mask_any
    else:
        mask = pd.Series(True, index=df.index)

    for t in exclude_terms:
        mask &= ~txt.str.contains(re.escape(t), na=False)

    return df[mask]

@st.cache_data(show_spinner=False)
def load_data(uploaded):
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        try:
            return pd.read_csv(uploaded, low_memory=False, engine="pyarrow")
        except Exception:
            return pd.read_csv(uploaded, low_memory=False)
    else:
        return pd.read_excel(uploaded)

@st.cache_data(show_spinner=False)
def build_date_map(columns):
    sales_cols, vol_cols = detect_date_cols(columns)

    sales_map = {}
    for c in sales_cols:
        d = parse_date_from_col(c, "_SALES")
        if not pd.isna(d):
            sales_map[d.normalize()] = c

    vol_map = {}
    for c in vol_cols:
        d = parse_date_from_col(c, "_VOLUME")
        if not pd.isna(d):
            vol_map[d.normalize()] = c

    common_dates = sorted(set(sales_map.keys()) & set(vol_map.keys()))
    return common_dates, sales_map, vol_map

@st.cache_data(show_spinner=True)
def process_filtered(df_wide: pd.DataFrame,
                     id_cols: list[str],
                     selected_sales_cols: list[str],
                     selected_vol_cols: list[str]) -> pd.DataFrame:

    keep_cols = id_cols + selected_sales_cols + selected_vol_cols
    df = df_wide[keep_cols].copy()

    # downcast to reduce memory
    for c in selected_sales_cols + selected_vol_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    sales_long = df.melt(id_vars=id_cols, value_vars=selected_sales_cols, var_name="DateCol", value_name="Sales")
    vol_long   = df.melt(id_vars=id_cols, value_vars=selected_vol_cols,   var_name="DateCol", value_name="Volume")

    sales_long["Date"] = sales_long["DateCol"].map(lambda x: parse_date_from_col(x, "_SALES"))
    vol_long["Date"]   = vol_long["DateCol"].map(lambda x: parse_date_from_col(x, "_VOLUME"))

    sales_long = sales_long.drop(columns=["DateCol"]).dropna(subset=["Date"])
    vol_long   = vol_long.drop(columns=["DateCol"]).dropna(subset=["Date"])

    merge_keys = id_cols + ["Date"]
    out = sales_long.merge(vol_long[merge_keys + ["Volume"]], on=merge_keys, how="left")
    return out

def make_series_name(df: pd.DataFrame, group_cols: list[str]) -> pd.Series:
    if len(group_cols) == 1:
        return df[group_cols[0]].astype(str)
    return df[group_cols].astype(str).agg(" | ".join, axis=1)

# -------------------------
# Upload
# -------------------------
uploaded_file = st.file_uploader("Upload file (xlsx/xls/csv)", type=["xlsx","xls","csv"])
if uploaded_file is None:
    st.info("Upload a file to start.")
    st.stop()

df_wide = load_data(uploaded_file)

sales_cols, vol_cols = detect_date_cols(list(df_wide.columns))
if len(sales_cols) == 0 or len(vol_cols) == 0:
    st.error("Cannot find *_SALES and *_VOLUME columns. Please check your format.")
    st.stop()

id_cols = [c for c in BASE_ID_COLS if c in df_wide.columns]
if "PRODUCT_NAME" not in id_cols:
    st.error("Missing required column: PRODUCT_NAME")
    st.stop()

# product table (no melt)
prod_cols = [c for c in ["PRODUCT_NAME","PRODUCT_CODE","PRODUCT_GROUP","FORMAT_NAME"] if c in df_wide.columns]
prod_table = df_wide[prod_cols].drop_duplicates().copy()

if "PRODUCT_CODE" in prod_table.columns:
    prod_table["label"] = (
        prod_table["PRODUCT_NAME"].astype(str) + " | " +
        prod_table["PRODUCT_CODE"].astype(str) +
        ((" | " + prod_table["PRODUCT_GROUP"].astype(str)) if "PRODUCT_GROUP" in prod_table.columns else "")
    )
else:
    prod_table["label"] = prod_table["PRODUCT_NAME"].astype(str)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Filters (filter first, then process)")

kw = st.sidebar.text_input("Product keyword search", value="", placeholder="example: grape seedless  |  grape, seedless  |  grape -raisin")
prod_filtered = keyword_filter(prod_table, kw, ["PRODUCT_NAME","PRODUCT_GROUP","label"])

st.sidebar.caption(f"Matched products: {len(prod_filtered):,}")
if len(prod_filtered) == 0:
    sample_vals = prod_table["PRODUCT_NAME"].astype(str).head(6).tolist()
    st.sidebar.caption("No match. Example product names in your file:")
    st.sidebar.caption(" / ".join(sample_vals))

max_show = st.sidebar.slider("Max search results", 20, 300, 80)
prod_filtered = prod_filtered.head(max_show)

default_n = st.sidebar.slider("Default select N", 1, 10, 3)
default_labels = prod_filtered["label"].tolist()[:default_n]

selected_labels = st.sidebar.multiselect(
    "Select products to compare",
    options=prod_filtered["label"].tolist(),
    default=default_labels
)

selected_banners = None
if "FORMAT_NAME" in df_wide.columns:
    banners = sorted(df_wide["FORMAT_NAME"].dropna().unique().tolist())
    selected_banners = st.sidebar.multiselect("Banners (FORMAT_NAME)", options=banners, default=banners)

selected_stores = None
if "STORE_NAME" in df_wide.columns:
    store_mode = st.sidebar.radio("Store filter", ["All stores", "Pick stores"], index=0)
    if store_mode == "Pick stores":
        stores = sorted(df_wide["STORE_NAME"].dropna().unique().tolist())
        selected_stores = st.sidebar.multiselect("Stores", options=stores, default=stores[:10])

common_dates, sales_map, vol_map = build_date_map(list(df_wide.columns))
if len(common_dates) == 0:
    st.error("Could not parse any valid dates from column names.")
    st.stop()

min_d, max_d = common_dates[0], common_dates[-1]
start_date, end_date = st.sidebar.date_input(
    "Date range (controls which columns to melt)",
    value=(min_d.date(), max_d.date()),
    min_value=min_d.date(),
    max_value=max_d.date()
)

group_by = st.sidebar.selectbox(
    "Compare by",
    options=["Product only", "Product + Banner", "Product + Store"],
    index=1
)

if group_by == "Product only":
    group_cols = [c for c in ["PRODUCT_NAME","PRODUCT_CODE"] if c in df_wide.columns]
elif group_by == "Product + Banner":
    group_cols = [c for c in ["PRODUCT_NAME","PRODUCT_CODE","FORMAT_NAME"] if c in df_wide.columns]
else:
    group_cols = [c for c in ["PRODUCT_NAME","PRODUCT_CODE","STORE_NAME"] if c in df_wide.columns]

run = st.sidebar.button("✅ Run analysis", type="primary")

# -------------------------
# Filter wide rows first
# -------------------------
df_f = df_wide.copy()

if selected_labels:
    chosen = prod_filtered[prod_filtered["label"].isin(selected_labels)]
    if "PRODUCT_CODE" in chosen.columns and "PRODUCT_CODE" in df_f.columns:
        codes = chosen["PRODUCT_CODE"].astype(str).unique().tolist()
        df_f = df_f[df_f["PRODUCT_CODE"].astype(str).isin(codes)]
    else:
        names = chosen["PRODUCT_NAME"].astype(str).unique().tolist()
        df_f = df_f[df_f["PRODUCT_NAME"].astype(str).isin(names)]

if selected_banners is not None and "FORMAT_NAME" in df_f.columns:
    if len(selected_banners) > 0:
        df_f = df_f[df_f["FORMAT_NAME"].isin(selected_banners)]

if selected_stores is not None and "STORE_NAME" in df_f.columns:
    if len(selected_stores) > 0:
        df_f = df_f[df_f["STORE_NAME"].isin(selected_stores)]

st.subheader("Step 1: Quick preview (no heavy processing yet)")
st.write(f"Rows after row-filters (before melting): **{len(df_f):,}**")
st.dataframe(df_f[id_cols].head(20), use_container_width=True)

if not run:
    st.info("Set filters on the left, then click **Run analysis**.")
    st.stop()

# -------------------------
# Pick date columns to melt
# -------------------------
start_dt = pd.to_datetime(start_date)
end_dt   = pd.to_datetime(end_date)
chosen_dates = [d for d in common_dates if (d >= start_dt) and (d <= end_dt)]
if len(chosen_dates) == 0:
    st.warning("No dates selected. Please expand date range.")
    st.stop()

selected_sales_cols = [sales_map[d] for d in chosen_dates]
selected_vol_cols   = [vol_map[d] for d in chosen_dates]

with st.spinner("Processing selected rows + selected date columns (fast mode)..."):
    long_df = process_filtered(df_f, id_cols=id_cols, selected_sales_cols=selected_sales_cols, selected_vol_cols=selected_vol_cols)

if long_df.empty:
    st.warning("No data after processing. Try a wider date range or more products.")
    st.stop()

agg = (
    long_df
    .groupby(group_cols + ["Date"], dropna=False, as_index=False)
    .agg(Sales=("Sales","sum"), Volume=("Volume","sum"))
)
agg["Avg_Price"] = agg["Sales"] / agg["Volume"]
agg["Avg_Price"] = agg["Avg_Price"].replace([np.inf, -np.inf], np.nan)

agg["Series"] = make_series_name(agg, group_cols)

c1, c2, c3 = st.columns(3)
c1.metric("Rows (long)", f"{len(long_df):,}")
c2.metric("Total Sales", f"{agg['Sales'].sum():,.2f}")
c3.metric("Avg Price (weighted)", f"{(agg['Sales'].sum()/agg['Volume'].sum()) if agg['Volume'].sum() else np.nan:.3f}")

st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Sales Trend", "Avg Price Trend", "Volume Trend"])

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

st.markdown("---")
st.subheader("Download")
st.download_button(
    "Download aggregated data (CSV)",
    data=agg.to_csv(index=False).encode("utf-8"),
    file_name="trend_aggregated.csv",
    mime="text/csv"
)
