import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ------------------------------------------------------------
st.set_page_config(page_title="Secondary Sales Dashboard", layout="wide")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data(_v="v4"):
    # Load both parquet files
    df = pd.read_parquet("secondary_sales.parquet")
    df_ind = pd.read_parquet("industry_size.parquet")

    # ---------- BASIC NORMALIZATION ----------
    df["Year"] = df["Year"].astype(int)
    df["Month"] = df["Month"].astype(str).str.strip()
    df["Primary Cat"] = df["Primary Cat"].astype(str).str.strip()
    df["Platform"] = df["Platform"].astype(str).str.strip()

    # ---------- MONTH NORMALIZATION ----------
    month_map = {
        "jan": 1, "feb": 2, "mar": 3,
        "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9,
        "oct": 10, "nov": 11, "dec": 12
    }

    df["MonthKey"] = df["Month"].str.lower().str[:3]
    df["MonthNum"] = df["MonthKey"].map(month_map)

    # Drop rows with invalid month parsing
    df = df.dropna(subset=["MonthNum"])
    df["MonthNum"] = df["MonthNum"].astype(int)

    # Month label (dynamic, no restriction)
    month_label_map = {
        1: "Jan", 2: "Feb", 3: "Mar",
        4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep",
        10: "Oct", 11: "Nov", 12: "Dec"
    }
    df["MonthLabel"] = df["MonthNum"].map(month_label_map)

    return df, df_ind

# ---------- LOAD ----------
try:
    df, df_ind = load_data()
except FileNotFoundError:
    st.error("❌ Missing parquet file(s). Ensure secondary_sales.parquet and industry_size.parquet exist.")
    st.stop()

# ------------------------------------------------------------
# LOAD P-TYPE DATA
# ------------------------------------------------------------
@st.cache_data
def load_ptype_data():
    return pd.read_parquet("ptype.parquet")

pt_df = load_ptype_data()
pt_df.columns = pt_df.columns.str.strip()

# ------------------------------------------------------------
# INDIAN NUMBER FORMATTING
# ------------------------------------------------------------
def format_indian(value: float) -> str:
    if pd.isna(value):
        return "-"
    if value >= 10_000_000:
        return f"{value / 10_000_000:.2f} Cr"
    return f"{value / 100_000:.2f} L"

def indian_ticktexts(max_val: float, ticks: int = 6):
    if max_val <= 0 or np.isnan(max_val):
        return [0], ["0"]
    step = max_val / (ticks - 1)
    vals = [i * step for i in range(ticks)]
    labels = [format_indian(v) for v in vals]
    return vals, labels

# ------------------------------------------------------------
# FILTER FUNCTION (USED ACROSS TABS)
# ------------------------------------------------------------
def apply_filters(df, regions, states, categories, platforms):
    out = df.copy()
    if regions:
        out = out[out["Region Name"].isin(regions)]
    if states:
        out = out[out["State Name"].isin(states)]
    if categories:
        out = out[out["Primary Cat"].isin(categories)]
    if platforms:
        out = out[out["Platform"].isin(platforms)]
    return out

# ------------------------------------------------------------
# PERIOD DEFINITIONS (NO HARD MONTH DROPS)
# ------------------------------------------------------------
Q1_KEYS = {"apr", "may", "jun"}
Q2_KEYS = {"jul", "aug", "sep"}
Q3_KEYS = {"oct", "nov", "dec"}

def compute_q1_benchmark(df, metric):
    q1 = df[df["MonthKey"].isin(Q1_KEYS)]
    if q1.empty:
        return np.nan
    monthly = (
        q1.groupby(["Year", "MonthKey"], as_index=False)[metric]
        .sum()
    )
    return monthly[metric].mean()

# ------------------------------------------------------------
# TAB 5 — P-TYPE SECTION RENDERER (UNCHANGED LOGIC)
# ------------------------------------------------------------
def render_ptype_section(pt_df, ptype, selected_platforms, selected_cities, key_suffix=""):

    subset = pt_df.copy()

    if selected_platforms:
        subset = subset[subset["Platform"].isin(selected_platforms)]
    if selected_cities:
        subset = subset[subset["City"].isin(selected_cities)]

    subset = subset[subset["P Type"] == ptype]

    if subset.empty:
        st.info(f"No data for {ptype} with current filters.")
        return

    variants = sorted(subset["Variant"].dropna().unique().tolist())
    variant_key = f"variants_{ptype}_{key_suffix}"

    if variants:
        selected_variants = st.multiselect(
            f"Variants for {ptype}",
            options=variants,
            default=variants,
            key=variant_key
        )
        if selected_variants:
            subset = subset[subset["Variant"].isin(selected_variants)]

    if subset.empty:
        st.info(f"No data for {ptype} after variant filter.")
        return

    subset["Date"] = pd.to_datetime(subset["Date"])
    subset["MonthNum"] = subset["Date"].dt.month
    subset["MonthLabel"] = subset["Date"].dt.strftime("%b")

    industry = subset.groupby("MonthNum")["Absolute size"].sum()
    godesi = (
        subset[subset["Brand"].astype(str).str.upper().str.strip() == "GO DESI"]
        .groupby("MonthNum")["Absolute size"]
        .sum()
    )

    if industry.empty:
        st.info(f"No monthly data for {ptype}.")
        return

    months = sorted(industry.index)
    industry = industry.reindex(months)
    godesi = godesi.reindex(months, fill_value=0)

    month_labels = (
        subset.groupby("MonthNum")["MonthLabel"]
        .first()
        .reindex(months)
        .tolist()
    )

    industry_crore = industry.values / 1e7
    share_pct = np.where(
        industry.values > 0,
        (godesi.values / industry.values) * 100,
        np.nan
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=industry_crore,
            name="Industry Absolute Size (₹ crore)",
            mode="lines+markers"
        ),
        secondary_y=True
    )

    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=share_pct,
            name="GO DESI Share (%)",
            mode="lines+markers",
            line=dict(dash="dot")
        ),
        secondary_y=False
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right")
    )

    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="GO DESI Share (%)", secondary_y=False)
    fig.update_yaxes(title_text="Industry Size (₹ crore)", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)
    
# ------------------------------------------------------------
# DEFINE TABS
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Sales Overview",
    "Top Markets",
    "Growth vs Laggard Markets",
    "Metro Industry View",
    "Product type / Variant Deep Dive"
])


# ============================================================
# TAB 1 – OVERVIEW DASHBOARD
# ============================================================
with tab1:
    st.title("Secondary Sales Overview")

    # ------------------------------------------------------------
    # Filters (left sidebar)
    # ------------------------------------------------------------
    with st.sidebar:
        st.header("Sales Overview Filters")
        metric = st.radio("Metric", ["Revenue", "GMV"], index=0, key="metric_tab1")

        all_regions = sorted(df["Region Name"].dropna().unique().tolist())
        region_sel = st.multiselect("Region", options=all_regions, default=[])

        if region_sel:
            all_states = sorted(
                df[df["Region Name"].isin(region_sel)]["State Name"].dropna().unique().tolist()
            )
        else:
            all_states = sorted(df["State Name"].dropna().unique().tolist())

        state_sel = st.multiselect("State", options=all_states, default=[])

        all_cats = sorted(df["Primary Cat"].dropna().unique().tolist())
        cat_sel = st.multiselect("Category", options=all_cats, default=[])

        platforms = sorted(df["Platform"].dropna().unique().tolist())
        platform_sel = st.multiselect("Platform", options=platforms, default=[])

    df_filt = apply_filters(df, region_sel, state_sel, cat_sel, platform_sel)

    if df_filt.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # ------------------------------------------------------------
    # DYNAMIC MONTH ORDER (ONLY MONTHS THAT EXIST)
    # ------------------------------------------------------------
    month_order = (
        df_filt[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    # ============================================================
    # SECTION 1 — Combined Category Trend (Dynamic Months) + Q1 Benchmark
    # ============================================================
    st.subheader("Category-wise Trend (Dynamic Months) with Q1 Benchmark")

    cat_timeline = (
        df_filt
        .groupby(["Year", "MonthNum", "MonthLabel", "Primary Cat"], as_index=False)[metric]
        .sum()
        .sort_values(["Year", "MonthNum"])
    )

    q1_bench = compute_q1_benchmark(df_filt, metric)

    fig_timeline = px.line(
        cat_timeline,
        x="MonthLabel",
        y=metric,
        color="Primary Cat",
        markers=True,
        category_orders={"MonthLabel": month_order},
    )

    ymax = float(cat_timeline[metric].max()) if not cat_timeline.empty else 0.0
    tvals, ttext = indian_ticktexts(ymax)

    fig_timeline.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(
            tickmode="array",
            tickvals=tvals,
            ticktext=ttext,
            title=metric
        ),
        xaxis_title="Month",
        legend_title_text="Primary Category"
    )

    if not np.isnan(q1_bench):
        fig_timeline.add_hline(
            y=q1_bench,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Q1 Avg: {format_indian(q1_bench)}",
            annotation_position="top left"
        )

    st.plotly_chart(fig_timeline, use_container_width=True)

    # ============================================================
    # SECTION 2 — Donut Charts (Q1, Q2, Q3)
    # ============================================================
    st.subheader("Sales Distribution — Q1, Q2, Q3")

    def donut_pair(df_range, title_suffix):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"**Region-wise Share — {title_suffix}**")
            reg_agg = df_range.groupby("Region Name", as_index=False)[metric].sum()
            if not reg_agg.empty:
                fig = px.pie(reg_agg, names="Region Name", values=metric, hole=0.45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No region data.")

        with c2:
            st.markdown(f"**State-wise Share — {title_suffix}**")
            state_agg = df_range.groupby("State Name", as_index=False)[metric].sum()
            if not state_agg.empty:
                fig = px.pie(state_agg, names="State Name", values=metric, hole=0.45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No state data.")

    df_q1 = df_filt[df_filt["MonthKey"].isin(Q1_KEYS)]
    df_q2 = df_filt[df_filt["MonthKey"].isin(Q2_KEYS)]
    df_q3 = df_filt[df_filt["MonthKey"].isin(Q3_KEYS)]

    donut_pair(df_q1, "Q1 (Apr–Jun)")
    donut_pair(df_q2, "Q2 (Jul–Sep)")
    donut_pair(df_q3, "Q3 (Oct–Dec)")

    # ============================================================
    # SECTION 3 — Top 20 SKUs (Q1, Q2, Q3)
    # ============================================================
    st.subheader(f"Top 20 SKUs (by {metric}) — Q1, Q2, Q3")

    def render_top_table(df_in, title):
        st.markdown(f"**{title}**")
        if df_in.empty:
            st.info("No SKUs match the filters.")
            return

        group_cols = ["Item Name", "Primary Cat", "Region Name", "State Name", "Platform"]

        sku_agg = (
            df_in.groupby(group_cols, as_index=False)[metric]
            .sum()
            .sort_values(by=metric, ascending=False)
        )

        total_metric = sku_agg[metric].sum()
        sku_agg["% of Total"] = (sku_agg[metric] / total_metric * 100).round(2)
        sku_agg[f"{metric} (₹ in L/Cr)"] = sku_agg[metric].apply(format_indian)

        st.dataframe(
            sku_agg[[*group_cols, f"{metric} (₹ in L/Cr)", "% of Total"]].head(20),
            use_container_width=True
        )

    render_top_table(df_q1, "Q1 (Apr–Jun)")
    render_top_table(df_q2, "Q2 (Jul–Sep)")
    render_top_table(df_q3, "Q3 (Oct–Dec)")

    # ============================================================
    # SECTION 4 — State Performance TABLE (Q1 vs Q2 + Q3)
    # ============================================================
    st.subheader("State Performance — Q1 vs Q2 + Q3")

    q1_state = df_q1.groupby("State Name", as_index=False)[metric].sum().rename(columns={metric: "Q1"})
    q2_state = df_q2.groupby("State Name", as_index=False)[metric].sum().rename(columns={metric: "Q2"})
    q3_state = df_q3.groupby("State Name", as_index=False)[metric].sum().rename(columns={metric: "Q3"})

    merged = (
        q1_state
        .merge(q2_state, on="State Name", how="outer")
        .merge(q3_state, on="State Name", how="outer")
        .fillna(0)
    )

    merged["Q2 Δ% vs Q1"] = (
        (merged["Q2"] - merged["Q1"]) / merged["Q1"].replace(0, np.nan) * 100
    )

    total_sales = merged["Q2"].sum()
    merged["Share %"] = (merged["Q2"] / total_sales * 100).round(2)

    merged["Q1 (₹ in L/Cr)"] = merged["Q1"].apply(format_indian)
    merged["Q2 (₹ in L/Cr)"] = merged["Q2"].apply(format_indian)
    merged["Q3 (₹ in L/Cr)"] = merged["Q3"].apply(format_indian)

    st.dataframe(
        merged[[
            "State Name",
            "Q1 (₹ in L/Cr)",
            "Q2 (₹ in L/Cr)",
            "Q3 (₹ in L/Cr)",
            "Q2 Δ% vs Q1",
            "Share %"
        ]],
        use_container_width=True
    )

# ============================================================
# TAB 2 — PRIMARY MARKETS (Top 70% State Trends)
# ============================================================
with tab2:
    st.title("Top Markets - State Trends (Top 70% Contribution)")

    # ----------------------------
    # TAB 2 FILTERS (LEFT SIDEBAR)
    # ----------------------------
    with st.sidebar:
        st.header("Top Markets Filter")

        metric_tab2 = st.radio("Metric (Tab 2)", ["Revenue", "GMV"], index=0, key="metric_tab2_sidebar")

        cat_tab2 = st.multiselect(
            "Primary Category (Tab 2)",
            sorted(df["Primary Cat"].dropna().unique().tolist()),
            default=[],
            key="cat_tab2_sidebar"
        )

        platform_tab2 = st.multiselect(
            "Platform (Tab 2)",
            sorted(df["Platform"].dropna().unique().tolist()),
            default=[],
            key="platform_tab2_sidebar"
        )

    df2 = apply_filters(df, regions=[], states=[], categories=cat_tab2, platforms=platform_tab2)

    if df2.empty:
        st.warning("No data available for selected Tab 2 filters.")
        st.stop()

    # ----------------------------
    # Helper → Top 70% states
    # ----------------------------
    def get_top_states(dfin, metric):
        state_tot = (
            dfin.groupby("State Name", as_index=False)[metric]
            .sum()
            .sort_values(metric, ascending=False)
        )
        total = state_tot[metric].sum()
        state_tot["CumShare%"] = state_tot[metric].cumsum() / total * 100
        top_states = state_tot[state_tot["CumShare%"] <= 70]["State Name"].tolist()
        return top_states or state_tot.head(1)["State Name"].tolist()

    # ----------------------------
    # PERIOD DEFINITIONS
    # ----------------------------
    Q1_KEYS = {"apr","may","jun"}
    Q2_KEYS = {"jul","aug","sep"}
    Q3_KEYS = {"oct","nov","dec"}

    q1 = df2[df2["MonthKey"].isin(Q1_KEYS)]
    q2 = df2[df2["MonthKey"].isin(Q2_KEYS)]
    q3 = df2[df2["MonthKey"].isin(Q3_KEYS)]

    top_q1 = get_top_states(q1, metric_tab2)
    top_q2 = get_top_states(q2, metric_tab2)
    top_q3 = get_top_states(q3, metric_tab2)

    # ============================================================
    # OVERALL TREND — Based on Q1 Top 70% States (Dynamic Months)
    # ============================================================
    st.subheader("Overall State Trend — Based on Q1 Top 70% States")

    q1_only = df2[df2["MonthKey"].isin(Q1_KEYS)]
    top_q1_states = get_top_states(q1_only, metric_tab2)

    full_df_q1_based = df2[df2["State Name"].isin(top_q1_states)]

    # Dynamic month order (only months present)
    month_order_full = (
        full_df_q1_based[["MonthNum","MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    full_timeline_q1 = (
        full_df_q1_based
        .groupby(["State Name","MonthNum","MonthLabel"], as_index=False)[metric_tab2]
        .sum()
        .sort_values("MonthNum")
    )

    full_timeline_q1["MonthLabel"] = pd.Categorical(
        full_timeline_q1["MonthLabel"],
        month_order_full,
        ordered=True
    )

    fig_full_q1 = px.line(
        full_timeline_q1,
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True
    )

    ymax = full_timeline_q1[metric_tab2].max()
    tv, tt = indian_ticktexts(ymax)
    fig_full_q1.update_layout(
        yaxis=dict(tickmode="array", tickvals=tv, ticktext=tt),
        height=420
    )

    st.plotly_chart(fig_full_q1, use_container_width=True)

    # ============================================================
    # Q1 CHART
    # ============================================================
    st.subheader("Q1 State Trend (Apr–Jun)")

    q1_plot = q1[q1["State Name"].isin(top_q1)]
    fig_q1 = px.line(
        q1_plot.groupby(["State Name","MonthNum","MonthLabel"], as_index=False)[metric_tab2].sum(),
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True
    )
    st.plotly_chart(fig_q1, use_container_width=True)

    # ============================================================
    # Q2 CHART
    # ============================================================
    st.subheader("Q2 State Trend (Jul–Sep)")

    q2_plot = q2[q2["State Name"].isin(top_q2)]
    fig_q2 = px.line(
        q2_plot.groupby(["State Name","MonthNum","MonthLabel"], as_index=False)[metric_tab2].sum(),
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True
    )
    st.plotly_chart(fig_q2, use_container_width=True)

    # ============================================================
# Q3 STATE TREND (Oct–Dec) — LINE CHART
# ============================================================
st.subheader("Q3 State Trend (Oct–Dec)")

Q3_KEYS = {"oct", "nov", "dec"}

q3 = df2[df2["MonthKey"].isin(Q3_KEYS)].copy()

if q3.empty:
    st.info("No Q3 data available.")
else:
    top_q3 = get_top_states(q3, metric_tab2)

    q3_plot = q3[q3["State Name"].isin(top_q3)]

    q3_timeline = (
        q3_plot
        .groupby(["State Name", "MonthNum", "MonthLabel"], as_index=False)[metric_tab2]
        .sum()
        .sort_values("MonthNum")
    )

    # Enforce correct month order dynamically
    month_order_q3 = (
        q3_timeline[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    q3_timeline["MonthLabel"] = pd.Categorical(
        q3_timeline["MonthLabel"],
        month_order_q3,
        ordered=True
    )

    fig_q3 = px.line(
        q3_timeline,
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True
    )

    ymax_q3 = q3_timeline[metric_tab2].max()
    tv3, tt3 = indian_ticktexts(ymax_q3)

    fig_q3.update_layout(
        yaxis=dict(tickmode="array", tickvals=tv3, ticktext=tt3),
        height=420,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_q3, use_container_width=True)

# ============================================================
# TAB 3 — GROWTH vs LAGGARD MARKETS (TOP 70% STATES ONLY)
# ============================================================
with tab3:
    st.title("Growth vs Laggard Markets (Top 70% Contribution Only)")

    # -----------------------------------------------------------
    # QUARTER DEFINITIONS (FUTURE-PROOF)
    # -----------------------------------------------------------
    QUARTER_MAP = {
        "Q1 (Apr–Jun)": {"apr", "may", "jun"},
        "Q2 (Jul–Sep)": {"jul", "aug", "sep"},
        "Q3 (Oct–Dec)": {"oct", "nov", "dec"},
    }

    # ----------------------------
    # TAB 3 FILTERS (LEFT SIDEBAR)
    # ----------------------------
    with st.sidebar:
        st.header("Growth and Laggard Markets Filter")

        metric_tab3 = st.radio(
            "Metric (Tab 3)",
            ["Revenue", "GMV"],
            index=0,
            key="metric_tab3_sidebar"
        )

        compare_quarter = st.selectbox(
            "Compare Quarter",
            options=list(QUARTER_MAP.keys()),
            index=2,
            key="compare_quarter_tab3"
        )

        baseline_quarter = st.selectbox(
            "Baseline Quarter",
            options=[q for q in QUARTER_MAP.keys() if q != compare_quarter],
            index=0,
            key="baseline_quarter_tab3"
        )

        cat_tab3 = st.multiselect(
            "Primary Category (Tab 3)",
            sorted(df["Primary Cat"].dropna().unique().tolist()),
            default=[],
            key="cat_tab3_sidebar"
        )

        platform_tab3 = st.multiselect(
            "Platform (Tab 3)",
            sorted(df["Platform"].dropna().unique().tolist()),
            default=[],
            key="platform_tab3_sidebar"
        )

    # -----------------------------------------------------------
    # APPLY FILTERS
    # -----------------------------------------------------------
    df3 = apply_filters(
        df,
        regions=[],
        states=[],
        categories=cat_tab3,
        platforms=platform_tab3
    )

    if df3.empty:
        st.warning("No data available for selected Tab 3 filters.")
        st.stop()

    # -----------------------------------------------------------
    # STEP 1 — IDENTIFY TOP 70% STATES (FULL PERIOD)
    # -----------------------------------------------------------
    state_totals = (
        df3.groupby("State Name")[metric_tab3]
        .sum()
        .reset_index()
        .sort_values(metric_tab3, ascending=False)
    )

    total_sum = state_totals[metric_tab3].sum()
    state_totals["CumPct"] = state_totals[metric_tab3].cumsum() / total_sum

    top70_states = state_totals[state_totals["CumPct"] <= 0.70]["State Name"].tolist()
    df3 = df3[df3["State Name"].isin(top70_states)]

    if df3.empty:
        st.warning("No data after applying top 70% state logic.")
        st.stop()

    # -----------------------------------------------------------
    # STEP 2 — BASELINE vs COMPARISON QUARTER AVG
    # -----------------------------------------------------------
    baseline_months = QUARTER_MAP[baseline_quarter]
    compare_months = QUARTER_MAP[compare_quarter]

    baseline_state = (
        df3[df3["MonthKey"].isin(baseline_months)]
        .groupby(["State Name", "MonthKey"])[metric_tab3]
        .sum()
        .groupby(level=0)
        .mean()
        .rename("Baseline_avg")
    )

    compare_state = (
        df3[df3["MonthKey"].isin(compare_months)]
        .groupby(["State Name", "MonthKey"])[metric_tab3]
        .sum()
        .groupby(level=0)
        .mean()
        .rename("Compare_avg")
    )

    growth_df = (
        pd.concat([baseline_state, compare_state], axis=1)
        .reset_index()
        .fillna(0.0)
    )

    growth_df["Growth %"] = (
        (growth_df["Compare_avg"] - growth_df["Baseline_avg"])
        / growth_df["Baseline_avg"].replace(0, np.nan)
        * 100
    )

    # -----------------------------------------------------------
    # STEP 3 — SPLIT GROWTH & LAGGARDS
    # -----------------------------------------------------------
    growth_only = growth_df[growth_df["Growth %"] > 0].sort_values("Growth %", ascending=False)
    laggard_only = growth_df[growth_df["Growth %"] < 0].sort_values("Growth %")

    top5_growth = growth_only.head(5)
    top5_laggards = laggard_only.head(5)

    # ----------------------------
    # A. Δ GROWTH % BAR CHARTS
    # ----------------------------
    c1, c2 = st.columns(2)

    with c1:
        st.subheader(f"Top Growth Markets — {compare_quarter} vs {baseline_quarter}")

        if top5_growth.empty:
            st.info("No growth markets for selected filters.")
        else:
            gbar = top5_growth.copy()
            gbar["Label"] = gbar["Growth %"].map(lambda x: f"{x:.1f}%")

            fig = px.bar(
                gbar,
                x="Growth %",
                y="State Name",
                orientation="h",
                text="Label",
                color_discrete_sequence=["#2e7d32"]
            )

            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis_title="Growth %",
                height=420,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader(f"Top Laggard Markets — {compare_quarter} vs {baseline_quarter}")

        if top5_laggards.empty:
            st.info("No laggard markets for selected filters.")
        else:
            lbar = top5_laggards.copy()
            lbar["Label"] = lbar["Growth %"].map(lambda x: f"{x:.1f}%")

            fig = px.bar(
                lbar,
                x="Growth %",
                y="State Name",
                orientation="h",
                text="Label",
                color_discrete_sequence=["#c62828"]
            )

            fig.update_traces(textposition="outside")
            fig.update_layout(
                xaxis_title="Growth %",
                height=420,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # B. DUMBBELL CHART — BENCHMARK SHIFT
    # ----------------------------
    st.subheader(f"Benchmark Shift — {baseline_quarter} ● vs {compare_quarter} ○")

    shown_states = pd.concat([top5_growth, top5_laggards]).sort_values("Growth %", ascending=False)

    if shown_states.empty:
        st.info("No states to display.")
    else:
        fig = go.Figure()

        for _, r in shown_states.iterrows():
            fig.add_trace(go.Scatter(
                x=[r["Baseline_avg"], r["Compare_avg"]],
                y=[r["State Name"], r["State Name"]],
                mode="lines",
                line=dict(color="#9e9e9e", width=2),
                showlegend=False
            ))

        fig.add_trace(go.Scatter(
            x=shown_states["Baseline_avg"],
            y=shown_states["State Name"],
            mode="markers",
            name=f"{baseline_quarter} Avg",
            marker=dict(size=10, color="#616161")
        ))

        fig.add_trace(go.Scatter(
            x=shown_states["Compare_avg"],
            y=shown_states["State Name"],
            mode="markers",
            name=f"{compare_quarter} Avg",
            marker=dict(size=10, symbol="circle-open", color="#9575cd")
        ))

        xmax = max(shown_states["Baseline_avg"].max(), shown_states["Compare_avg"].max())
        tv, tt = indian_ticktexts(xmax)

        fig.update_layout(
            height=520,
            xaxis=dict(title=metric_tab3, tickmode="array", tickvals=tv, ticktext=tt),
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # C. DRILL-DOWN — MULTI-LINE MONTHLY TRENDS
    # ============================================================
    st.subheader("Drill-down Trends (Dynamic Months)")

    month_order = (
        df3[["MonthNum", "MonthLabel"]]
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    def plot_multiline(states, title):
        if not states:
            return

        temp_df = df3[df3["State Name"].isin(states)]

        timeline = (
            temp_df.groupby(["State Name", "MonthNum", "MonthLabel"], as_index=False)[metric_tab3]
            .sum()
            .sort_values("MonthNum")
        )

        timeline["MonthLabel"] = pd.Categorical(
            timeline["MonthLabel"], month_order, ordered=True
        )

        fig = px.line(
            timeline,
            x="MonthLabel",
            y=metric_tab3,
            color="State Name",
            markers=True
        )

        ymax = timeline[metric_tab3].max()
        tv, tt = indian_ticktexts(ymax)

        fig.update_layout(
            yaxis=dict(tickmode="array", tickvals=tv, ticktext=tt),
            height=420
        )

        st.markdown(f"### {title}")
        st.plotly_chart(fig, use_container_width=True)

    if not top5_growth.empty:
        selected_growth = st.multiselect(
            "Select Growth States",
            sorted(top5_growth["State Name"]),
            default=sorted(top5_growth["State Name"]),
            key="growth_states_tab3"
        )
        plot_multiline(selected_growth, "Top Growth States — Monthly Trend")

    if not top5_laggards.empty:
        selected_laggards = st.multiselect(
            "Select Laggard States",
            sorted(top5_laggards["State Name"]),
            default=sorted(top5_laggards["State Name"]),
            key="laggard_states_tab3"
        )
        plot_multiline(selected_laggards, "Top Laggard States — Monthly Trend")


# ============================================================
# TAB 4 — METRO INDUSTRY VIEW (Grouped Bar: Industry vs GO DESi)
# ============================================================
with tab4:
    st.title("Metro Industry View — GO DESi vs Industry Size")

    # ----------------------------
    # TAB 4 FILTERS
    # ----------------------------
    with st.sidebar:
        st.header("Industry Size Filters")

        cat4 = st.multiselect(
            "Primary Category (Tab 4)",
            sorted(df["Primary Cat"].dropna().unique().tolist()),
            default=[],
            key="cat_tab4"
        )

        plat4 = st.multiselect(
            "Platform (Tab 4)",
            sorted(df["Platform"].dropna().unique().tolist()),
            default=[],
            key="plat_tab4"
        )

        metro_cities = ["Bengaluru", "Mumbai", "Kolkata", "Delhi"]
        city4 = st.multiselect(
            "City (Tab 4)",
            metro_cities,
            default=["Bengaluru"],
            key="city_tab4"
        )

    # ----------------------------
    # APPLY FILTERS
    # ----------------------------
    df_rev = df.copy()
    df_ind4 = df_ind.copy()

    if cat4:
        df_rev = df_rev[df_rev["Primary Cat"].isin(cat4)]
        df_ind4 = df_ind4[df_ind4["Primary Cat"].isin(cat4)]

    if plat4:
        df_rev = df_rev[df_rev["Platform"].isin(plat4)]
        df_ind4 = df_ind4[df_ind4["Platform"].isin(plat4)]

    if city4:
        df_rev = df_rev[df_rev["City Name"].isin(city4)]
        df_ind4 = df_ind4[df_ind4["City Name"].isin(city4)]

    if df_ind4.empty:
        st.warning("No industry size data available for these filters.")
        st.stop()

    # ----------------------------
    # PREPARE MONTH-WISE DATA (DYNAMIC)
    # ----------------------------
    month_order = (
        pd.concat([
            df_ind4[["MonthNum", "MonthLabel"]],
            df_rev[["MonthNum", "MonthLabel"]]
        ])
        .drop_duplicates()
        .sort_values("MonthNum")["MonthLabel"]
        .tolist()
    )

    # Industry
    ind = (
        df_ind4.groupby(["MonthNum", "MonthLabel"], as_index=False)["Industry Size"]
        .sum()
        .sort_values("MonthNum")
    )

    # Revenue
    rev = (
        df_rev.groupby(["MonthNum", "MonthLabel"], as_index=False)["Revenue"]
        .sum()
        .sort_values("MonthNum")
    )

    # Merge for table
    merged = pd.merge(
        ind,
        rev,
        on=["MonthNum", "MonthLabel"],
        how="outer"
    )

    merged["Industry Size"] = merged["Industry Size"].fillna(0)
    merged["Revenue"] = merged["Revenue"].fillna(0)

    merged["Share %"] = (
        merged["Revenue"] /
        merged["Industry Size"].replace(0, np.nan) * 100
    ).fillna(0).round(2)

    # Enforce dynamic month order
    merged["MonthLabel"] = pd.Categorical(
        merged["MonthLabel"], month_order, ordered=True
    )
    merged = merged.sort_values("MonthLabel")

    # ----------------------------
    # GROUPED BAR CHART
    # ----------------------------
    st.subheader("Industry Size vs GO DESi Revenue (Grouped Bars) — Dynamic Months")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=merged["MonthLabel"],
        y=merged["Industry Size"],
        name="Industry Size",
        marker_color="#4A90E2",
        hovertemplate="Industry Size: %{y:,.0f}<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        x=merged["MonthLabel"],
        y=merged["Revenue"],
        name="GO DESi Revenue",
        marker_color="#FF8C00",
        hovertemplate="GO DESi Revenue: %{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        barmode="group",
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        title=dict(
            text="Industry Size (₹) vs GO DESi Revenue (₹) — Grouped Bars",
            x=0.01,
            y=0.93
        ),
        yaxis=dict(title="₹ (Raw Values)", tickformat=",")
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # SHARE TABLE
    # ----------------------------
    st.subheader("GO DESi Share of Industry (%)")

    st.dataframe(
        merged[["MonthLabel", "Industry Size", "Revenue", "Share %"]]
            .rename(columns={"MonthLabel": "Month"}),
        use_container_width=True
    )

# ------------------------------
# TAB 5 — P-Type Deep Dive
# ------------------------------
with tab5:
    st.title("P-Type Deep Dive — Industry vs GO DESi")

    # --------------------------
    # GLOBAL FILTERS (SIDEBAR)
    # --------------------------
    with st.sidebar:
        st.header("Deep Dive Filters")

        platforms_tab5 = st.multiselect(
            "Platform (Tab 5)",
            sorted(pt_df["Platform"].dropna().unique().tolist()),
            default=[],
            key="platform_tab5_sidebar",
        )

        cities_tab5 = st.multiselect(
            "City (Tab 5)",
            sorted(pt_df["City"].dropna().unique().tolist()),
            default=[],
            key="city_tab5_sidebar",
        )

    # --------------------------
    # SUBTABS: Sweets vs Candy
    # --------------------------
    sweets_tab, candy_tab = st.tabs(["Indian Sweets", "Candies & Gum"])

    # ---------- Indian Sweets ----------
    with sweets_tab:
        st.subheader("Indian Sweets — P Type Trends")

        sweets_ptypes = ["Barfi", "Katli", "Laddu", "Peda", "Chikki", "Gajak", "Mysore Pak"]

        for ptype in sweets_ptypes:
            with st.container():
                st.markdown(f"### {ptype}")
                render_ptype_section(
                    pt_df,
                    ptype=ptype,
                    selected_platforms=platforms_tab5,
                    selected_cities=cities_tab5,
                    key_suffix="sweets",
                )
                st.markdown("---")

    # ---------- Candies & Gum ----------
    with candy_tab:
        st.subheader("Candies & Gum — P Type Trends")

        candy_ptypes = ["Candy", "Gum", "Mint"]

        for ptype in candy_ptypes:
            with st.container():
                st.markdown(f"### {ptype}")
                render_ptype_section(
                    pt_df,
                    ptype=ptype,
                    selected_platforms=platforms_tab5,
                    selected_cities=cities_tab5,
                    key_suffix="candy",
                )
                st.markdown("---")
