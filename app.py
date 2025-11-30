import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ------------------------------------------------------------
# PAGE CONFIG — must be the very first Streamlit command
# ------------------------------------------------------------
st.set_page_config(page_title="Secondary Sales Dashboard", layout="wide")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    # Load both parquet files
    df = pd.read_parquet("secondary_sales.parquet")
    df_ind = pd.read_parquet("industry_size.parquet")

    # Standardize main sales dataframe
    df["Month"] = df["Month"].astype(str)
    df["Year"] = df["Year"].astype(int)
    df["Primary Cat"] = df["Primary Cat"].astype(str)
    df["Platform"] = df["Platform"].astype(str)

    # Month maps
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

    # Normalize month
    df["MonthKey"] = df["Month"].str[:3].str.lower()
    df["MonthNum"] = df["Month"].str.lower().map(month_map).astype(int)

    # Only keep Apr → Oct
    valid_months = [4, 5, 6, 7, 8, 9, 10]
    df = df[df["MonthNum"].isin(valid_months)].copy()

    # Month label for charts
    mlabel = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
              8: "Aug", 9: "Sep", 10: "Oct"}
    df["MonthLabel"] = df["MonthNum"].map(mlabel)

    return df, df_ind


# Try loading both datasets
try:
    df, df_ind = load_data()
except FileNotFoundError:
    st.error("Missing parquet file(s). Ensure both secondary_sales.parquet and industry_size.parquet are present.")
    st.stop()

# ---------------------------------
# LOAD P TYPE DATA (excel sheet)
# ---------------------------------
@st.cache_data
def load_ptype_data():
    # change file name if needed
    return pd.read_excel("BusinessOverview.xlsx", sheet_name="P Type")

pt_df = load_ptype_data()

# Clean up column names just in case
pt_df.columns = pt_df.columns.str.strip()

# ------------------------------------------------------------
# INDIAN NUMBER FORMATTING
# ------------------------------------------------------------
def format_indian(value: float) -> str:
    """< 1 Cr shown in Lakhs (L); >= 1 Cr shown in Crores (Cr)."""
    if pd.isna(value):
        return "-"
    if value >= 10_000_000:  # 1 Cr
        return f"{value/10_000_000:.2f} Cr"
    else:
        return f"{value/100_000:.2f} L"

def indian_ticktexts(max_val: float, ticks: int = 6):
    """Generate tick positions and Indian-formatted tick labels for y-axes."""
    if max_val <= 0 or np.isnan(max_val):
        return [0], ["0"]
    step = max_val / (ticks - 1)
    vals = [i * step for i in range(ticks)]
    labels = [format_indian(v) for v in vals]
    return vals, labels

# ------------------------------------------------------------
# FILTER FUNCTION
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
# BENCHMARK CALCULATIONS
# ------------------------------------------------------------
Q1_KEYS = {"apr", "may", "jun"}
Q2_KEYS = {"jul", "aug", "sep"}

def compute_q1_benchmark(df, metric):
    q1 = df[df["MonthKey"].isin(Q1_KEYS)]
    if q1.empty:
        return np.nan
    monthly = q1.groupby(["Year","MonthKey"], as_index=False)[metric].sum()
    return monthly[metric].mean()

# ------------------------------------------------------------
# TAB 5 — P TYPE SECTION RENDERER
# ------------------------------------------------------------

def render_ptype_section(pt_df, ptype, selected_platforms, selected_cities, key_suffix=""):
    """
    Renders:
      - Variant multiselect
      - Dual-axis line chart (GO DESI share % vs Industry Abs Size in crores)
      - GO DESI = dotted line
      - Industry Size = solid line
      - Indian numbering format (crores)
    """

    subset = pt_df.copy()

    # Apply global filters
    if selected_platforms:
        subset = subset[subset["Platform"].isin(selected_platforms)]

    if selected_cities:
        subset = subset[subset["City"].isin(selected_cities)]

    # Filter by P Type
    subset = subset[subset["P Type"] == ptype]

    if subset.empty:
        st.info(f"No data for {ptype} with current filters.")
        return

    # Local Variant filter
    variants = sorted(subset["Variant"].dropna().unique().tolist())
    variant_key = f"variants_{ptype}_{key_suffix}"

    if variants:
        selected_variants = st.multiselect(
            f"Variants for {ptype}",
            options=variants,
            default=variants,
            key=variant_key,
        )
        if selected_variants:
            subset = subset[subset["Variant"].isin(selected_variants)]

    if subset.empty:
        st.info(f"No data for {ptype} after variant filter.")
        return

    # Ensure Date is datetime
    subset["Date"] = pd.to_datetime(subset["Date"])
    subset["MonthNum"] = subset["Date"].dt.month
    subset["MonthLabel"] = subset["Date"].dt.strftime("%b")

    # Industry Absolute Size (solid line)
    industry = subset.groupby("MonthNum")["Absolute size"].sum()

    # GO DESI rows only
    godesi_mask = subset["Brand"].astype(str).str.strip().str.upper() == "GO DESI"
    godesi = subset[godesi_mask].groupby("MonthNum")["Absolute size"].sum()

    if industry.empty:
        st.info(f"No monthly data for {ptype}.")
        return

    # Align months
    months = sorted(industry.index.tolist())
    industry = industry.reindex(months)
    godesi = godesi.reindex(months, fill_value=0.0)

    month_labels = (
        subset.groupby("MonthNum")["MonthLabel"]
        .first()
        .reindex(months)
        .tolist()
    )

    # Convert absolute size to CRORES
    industry_crore = industry.values.astype(float) / 1e7
    godesi_crore = godesi.values.astype(float) / 1e7

    # GO DESI share % 
    share_pct = np.where(
        industry.values > 0,
        (godesi.values / industry.values) * 100.0,
        np.nan,
    )

    # -----------------------------
    # Build dual-axis Plotly chart
    # -----------------------------
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 🔵 Blue solid line = Industry Absolute Size
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=industry_crore,
            name="Industry Absolute Size (₹ crore)",
            mode="lines+markers",
            line=dict(color="#1f77b4", dash="solid")   # Blue
        ),
        secondary_y=True,
    )

    # 🔴 Orange dotted line = GO DESI Share (%)
    fig.add_trace(
        go.Scatter(
            x=month_labels,
            y=share_pct,
            name="GO DESI Share (%)",
            mode="lines+markers",
            line=dict(color="#FF7F32", dash="dot")    # Red dotted
        ),
        secondary_y=False,
    )

    # Layout
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    # Axis labels
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
            all_states = sorted(df[df["Region Name"].isin(region_sel)]["State Name"].dropna().unique().tolist())
        else:
            all_states = sorted(df["State Name"].dropna().unique().tolist())
        state_sel = st.multiselect("State", options=all_states, default=[])

        all_cats = sorted(df["Primary Cat"].dropna().unique().tolist())
        cat_sel = st.multiselect("Category", options=all_cats, default=[])

        platforms = sorted(df["Platform"].dropna().unique().tolist())
        platform_sel = st.multiselect("Platform", options=platforms, default=[])

    df_filt = apply_filters(df, region_sel, state_sel, cat_sel, platform_sel)
    
    # Safety
    if df_filt.empty:
        st.warning("No data available for the selected filters.")
        st.stop()

    # ============================================================
    # SECTION 1 — Combined Category Trend (Apr–Oct) + Q1 Benchmark
    # ============================================================
    st.subheader("Category-wise Trend (Apr–Oct) with Q1 Benchmark")

    # Group for line chart (Apr..Oct)
    cat_timeline = (
        df_filt.groupby(["Year","MonthNum","MonthLabel","Primary Cat"], as_index=False)[metric]
        .sum()
        .sort_values(["Year","MonthNum"])
    )

    # Compute Q1 benchmark on filtered df
    q1_bench = compute_q1_benchmark(df_filt, metric)
    bench_label = "Q1 Average " + metric

    # Build figure
    fig_timeline = px.line(
        cat_timeline,
        x="MonthLabel",
        y=metric,
        color="Primary Cat",
        markers=True,
        category_orders={"MonthLabel": ["Apr","May","Jun","Jul","Aug","Sep","Oct"]},
        title="Apr–Oct Category-wise Performance"
    )
    fig_timeline.update_traces(mode="lines+markers")

    # Axis ticks in Indian format
    ymax = float(cat_timeline[metric].max()) if not cat_timeline.empty else 0.0
    tvals, ttext = indian_ticktexts(ymax)
    fig_timeline.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        legend_title_text="Primary Category",
        yaxis=dict(tickmode="array", tickvals=tvals, ticktext=ttext, title=metric),
        xaxis_title="Month"
    )

    # Q1 benchmark line + annotation (still absolute values)
    if not np.isnan(q1_bench):
        fig_timeline.add_hline(
            y=q1_bench,
            line_dash="dot",
            line_color="red",
            line_width=2.5,
            annotation_text=f"Q1 Avg: {format_indian(q1_bench)}",
            annotation_position="top left",
            annotation_font=dict(color="red", size=12)
        )

    # Hover shows Indian units
    fig_timeline.update_traces(
        hovertemplate="<b>%{legendgroup}</b><br>Month=%{x}<br>" + metric + "=%{y:.0f}<extra></extra>"
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ============================================================
    # SECTION 2 — Donut Charts (Q1, Q2, October)
    # ============================================================
    st.subheader("Sales Distribution — Q1, Q2, and October")

    def donut_pair(df_range, title_suffix):
        c1, c2 = st.columns(2)
        # Region share
        with c1:
            st.markdown(f"**Region-wise Share — {title_suffix}**")
            reg_agg = df_range.groupby("Region Name", as_index=False)[metric].sum()
            if not reg_agg.empty:
                fig = px.pie(reg_agg, names="Region Name", values=metric, hole=0.45)
                fig.update_traces(textposition="inside")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=330)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No region data.")
        # State share
        with c2:
            st.markdown(f"**State-wise Share — {title_suffix}**")
            state_agg = df_range.groupby("State Name", as_index=False)[metric].sum()
            if not state_agg.empty:
                fig = px.pie(state_agg, names="State Name", values=metric, hole=0.45)
                fig.update_traces(textposition="inside")
                fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=330)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No state data.")

    df_q1 = df_filt[df_filt["MonthKey"].isin(Q1_KEYS)]
    df_q2 = df_filt[df_filt["MonthKey"].isin(Q2_KEYS)]
    df_oct = df_filt[df_filt["MonthKey"].isin({"oct"})]

    donut_pair(df_q1, "Q1 (Apr–Jun)")
    donut_pair(df_q2, "Q2 (Jul–Sep)")
    donut_pair(df_oct, "October")

    # ============================================================
    # SECTION 3 — Top 20 SKU Tables (Q1, Q2, Oct)
    # ============================================================
    st.subheader(f"Top 20 SKUs (by {metric}) — Q1, Q2 and Oct")

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

        # Indian unit display column (human-friendly)
        sku_agg[f"{metric} (₹ in L/Cr)"] = sku_agg[metric].apply(format_indian)

        st.dataframe(
            sku_agg[[*group_cols, f"{metric} (₹ in L/Cr)", "% of Total"]].head(20),
            use_container_width=True
        )

    render_top_table(df_q1, "Q1 (Apr–Jun)")
    render_top_table(df_q2, "Q2 (Jul–Sep)")
    render_top_table(df_oct, "October")

    # ============================================================
    # SECTION 4 — State Performance TABLE (Q1 vs Q2 + October)
    # ============================================================
    st.subheader("State Performance — Q1 vs Q2 + October")

    # Q1 totals
    q1_state = df_q1.groupby("State Name", as_index=False)[metric].sum().rename(columns={metric: "Q1"})

    # Q2 totals
    q2_state = df_q2.groupby("State Name", as_index=False)[metric].sum().rename(columns={metric: "Q2"})

    # October totals
    oct_state = df_oct.groupby("State Name", as_index=False)[metric].sum().rename(columns={metric: "October"})

    # Merge all
    merged = (
        q1_state
        .merge(q2_state, on="State Name", how="outer")
        .merge(oct_state, on="State Name", how="outer")
        .fillna(0)
    )

    # Growth %
    merged["Q2 Δ% vs Q1"] = ((merged["Q2"] - merged["Q1"]) / merged["Q1"].replace(0, np.nan) * 100)

    # Share % based on Q2 (unchanged logic)
    total_sales = merged["Q2"].sum()
    merged["Share %"] = (merged["Q2"] / total_sales * 100)

    # Round & sort
    merged = merged.sort_values(by="Share %", ascending=False).reset_index(drop=True)
    merged["Q2 Δ% vs Q1"] = merged["Q2 Δ% vs Q1"].round(2)
    merged["Share %"] = merged["Share %"].round(2)

    # Apply Indian number formatting
    merged["Q1 (₹ in L/Cr)"] = merged["Q1"].apply(format_indian)
    merged["Q2 (₹ in L/Cr)"] = merged["Q2"].apply(format_indian)
    merged["October (₹ in L/Cr)"] = merged["October"].apply(format_indian)

    # Display clean formatted table
    st.dataframe(
        merged[[
            "State Name",
            "Q1 (₹ in L/Cr)",
            "Q2 (₹ in L/Cr)",
            "October (₹ in L/Cr)",
            "Q2 Δ% vs Q1",
            "Share %"
        ]],
        use_container_width=True
    )


    # ============================================================
    # SECTION 5 — (REMOVED: separate Jul–Oct graph) COMBINED ABOVE
    # ============================================================

    # ============================================================
    # SECTION 6 — State Trend (Two-point line: Q1 vs Q2) with markers + labels
    # ============================================================
    st.subheader("State Trend — Q1 vs Q2 (per selected State)")

    # If user has not selected any state, use all states
    selected_states = state_sel if state_sel else sorted(df_filt["State Name"].dropna().unique().tolist())

    # Compute Q1/Q2 averages per selected state (per your design)
    def period_avg(df_in, months_set):
        mask = df_in["MonthKey"].isin(months_set)
        return (df_in[mask].groupby("State Name")[metric].sum() /
                df_in[mask].groupby("State Name")["MonthKey"].count()).rename("avg")

    # A safer version: true average over 3 months for states present; if any month missing, we average by count available
    avg_q1 = (
        df_filt[df_filt["MonthKey"].isin(Q1_KEYS)]
        .groupby(["State Name","MonthKey"])[metric]
        .sum()
        .groupby(level=0).mean()
        .rename("Q1_avg")
    )
    avg_q2 = (
        df_filt[df_filt["MonthKey"].isin(Q2_KEYS)]
        .groupby(["State Name","MonthKey"])[metric]
        .sum()
        .groupby(level=0).mean()
        .rename("Q2_avg")
    )

    state_curve = pd.concat([avg_q1, avg_q2], axis=1).reset_index()
    state_curve = state_curve[state_curve["State Name"].isin(selected_states)].fillna(0.0)

    # Build long format Q1/Q2 rows for each state
    long_rows = []
    for _, r in state_curve.iterrows():
        long_rows.append({"State Name": r["State Name"], "Period": "Q1", "Value": r["Q1_avg"]})
        long_rows.append({"State Name": r["State Name"], "Period": "Q2", "Value": r["Q2_avg"]})
    long_df = pd.DataFrame(long_rows)

    if long_df.empty:
        st.info("No data for the selected states.")
    else:
        # Plot with markers and text labels "Q1"/"Q2" on each point
        fig_state = px.line(
            long_df,
            x="Period",
            y="Value",
            color="State Name",
            markers=True,
            title="Q1 vs Q2 by State",
        )
        # Indian y-axis ticks
        ymax2 = float(long_df["Value"].max())
        tvals2, ttext2 = indian_ticktexts(ymax2)
        fig_state.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            yaxis=dict(tickmode="array", tickvals=tvals2, ticktext=ttext2, title=metric),
            xaxis_title="",
            legend_title_text="State"
        )

        # Add text labels on markers
        fig_state.update_traces(mode="lines+markers+text", text=long_df["Period"], textposition="top center")

        # Hover as Indian units
        fig_state.update_traces(hovertemplate="State=%{legendgroup}<br>%{x}: %{y:.0f}<extra></extra>")

        st.plotly_chart(fig_state, use_container_width=True)

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

    # Filter dataset for tab 2
    df2 = apply_filters(df, regions=[], states=[], categories=cat_tab2, platforms=platform_tab2)

    if df2.empty:
        st.warning("No data available for selected Tab 2 filters.")
        st.stop()

    # Helper → get Top 70% states by contribution in that period
    def get_top_states(dfin, metric):
        state_tot = (
            dfin.groupby("State Name", as_index=False)[metric]
            .sum()
            .sort_values(metric, ascending=False)
        )
        total = state_tot[metric].sum()
        state_tot["CumShare%"] = state_tot[metric].cumsum() / total * 100
        top_states = state_tot[state_tot["CumShare%"] <= 70]["State Name"].tolist()

        # Ensure at least one state
        if not top_states and not state_tot.empty:
            top_states = [state_tot.iloc[0]["State Name"]]
        return top_states

    # Define periods
    Q1_KEYS = {"apr","may","jun"}
    Q2_KEYS = {"jul","aug","sep"}
    OCT_KEY = {"oct"}

    q1 = df2[df2["MonthKey"].isin(Q1_KEYS)].copy()
    q2 = df2[df2["MonthKey"].isin(Q2_KEYS)].copy()
    oct_df = df2[df2["MonthKey"].isin(OCT_KEY)].copy()

    top_q1 = get_top_states(q1, metric_tab2)
    top_q2 = get_top_states(q2, metric_tab2)
    top_oct = get_top_states(oct_df, metric_tab2)

    # ============================================================
    # OVERALL TREND (Apr–Oct) — Based Only on Top 70% Q1 States
    # ============================================================
    st.subheader("Overall State Trend (Apr–Oct) — Based on Q1 Top 70% States")

    # Step 1: Identify Top 70% States *using only Q1 data*
    q1_only = df2[df2["MonthKey"].isin(Q1_KEYS)].copy()
    top_q1_states = get_top_states(q1_only, metric_tab2)

    # Step 2: Filter entire Apr–Oct dataset to only these Q1-top states
    full_df_q1_based = df2[df2["State Name"].isin(top_q1_states)].copy()

    # Month order Apr→Oct
    month_order_full = ["Apr","May","Jun","Jul","Aug","Sep","Oct"]

    # Step 3: Build timeline Apr→Oct
    full_timeline_q1 = (
        full_df_q1_based.groupby(["State Name","MonthNum","MonthLabel"], as_index=False)[metric_tab2]
        .sum()
        .sort_values("MonthNum")
    )

    # Apply correct month label ordering
    full_timeline_q1["MonthLabel"] = pd.Categorical(
        full_timeline_q1["MonthLabel"],
        month_order_full,
        ordered=True
    )

    # Step 4: Plot
    fig_full_q1 = px.line(
        full_timeline_q1,
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True
    )

    # Add Indian formatting
    ymax_full_q1 = full_timeline_q1[metric_tab2].max()
    tv_full_q1, tt_full_q1 = indian_ticktexts(ymax_full_q1)
    fig_full_q1.update_layout(
        yaxis=dict(tickmode="array", tickvals=tv_full_q1, ticktext=tt_full_q1),
        height=420,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    st.plotly_chart(fig_full_q1, use_container_width=True)


    # ============================================================
    # Q1 CHART (Apr–Jun)
    # ============================================================
    st.subheader("Q1 State Trend (Apr–Jun)")

    q1_plot = q1[q1["State Name"].isin(top_q1)]
    q1_timeline = (
        q1_plot.groupby(["State Name","MonthNum","MonthLabel"], as_index=False)[metric_tab2]
        .sum()
        .sort_values("MonthNum")
    )
    # Correct month order
    q1_timeline["MonthLabel"] = pd.Categorical(q1_timeline["MonthLabel"], ["Apr","May","Jun"], ordered=True)

    fig_q1 = px.line(
        q1_timeline,
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True
    )
    ymax_q1 = q1_timeline[metric_tab2].max()
    tv1, tt1 = indian_ticktexts(ymax_q1)
    fig_q1.update_layout(yaxis=dict(tickmode="array", tickvals=tv1, ticktext=tt1))
    st.plotly_chart(fig_q1, use_container_width=True)

    # ============================================================
    # Q2 CHART (Jul–Sep)
    # ============================================================
    st.subheader("Q2 State Trend (Jul–Sep)")

    q2_plot = q2[q2["State Name"].isin(top_q2)]
    q2_timeline = (
        q2_plot.groupby(["State Name","MonthNum","MonthLabel"], as_index=False)[metric_tab2]
        .sum()
        .sort_values("MonthNum")
    )
    # Correct month order
    q2_timeline["MonthLabel"] = pd.Categorical(q2_timeline["MonthLabel"], ["Jul","Aug","Sep"], ordered=True)

    fig_q2 = px.line(
        q2_timeline,
        x="MonthLabel",
        y=metric_tab2,
        color="State Name",
        markers=True
    )
    ymax_q2 = q2_timeline[metric_tab2].max()
    tv2, tt2 = indian_ticktexts(ymax_q2)
    fig_q2.update_layout(yaxis=dict(tickmode="array", tickvals=tv2, ticktext=tt2))
    st.plotly_chart(fig_q2, use_container_width=True)

    # ============================================================
    # OCTOBER CHART (Single Month Snapshot)
    # ============================================================
    st.subheader("October State Performance")

    oct_plot = oct_df[oct_df["State Name"].isin(top_oct)]
    oct_agg = (
        oct_plot.groupby("State Name", as_index=False)[metric_tab2]
        .sum()
        .sort_values(metric_tab2, ascending=False)
    )

    fig_oct = px.bar(
        oct_agg,
        x="State Name",
        y=metric_tab2,
        color="State Name",
        text=metric_tab2
    )
    ymax_oct = oct_agg[metric_tab2].max()
    tv3, tt3 = indian_ticktexts(ymax_oct)
    fig_oct.update_traces(texttemplate="%{text:.0f}", textposition="outside")
    fig_oct.update_layout(yaxis=dict(tickmode="array", tickvals=tv3, ticktext=tt3))
    st.plotly_chart(fig_oct, use_container_width=True)

# ============================================================
# TAB 3 — GROWTH vs LAGGARD MARKETS (TOP 70% STATES ONLY)
# ============================================================
with tab3:
    st.title("Growth vs Laggard Markets (Top 70% Contribution Only)")

    # ----------------------------
    # TAB 3 FILTERS (LEFT SIDEBAR)
    # ----------------------------
    with st.sidebar:
        st.header("Growth and Laggard Markets Filter")
        metric_tab3 = st.radio("Metric (Tab 3)", ["Revenue", "GMV"], index=0, key="metric_tab3_sidebar")
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

    # Apply filters (no region/state filters here)
    df3 = apply_filters(df, regions=[], states=[], categories=cat_tab3, platforms=platform_tab3)
    if df3.empty:
        st.warning("No data available for selected Tab 3 filters.")
        st.stop()

    # -----------------------------------------------------------
    # STEP 1 — IDENTIFY TOP 70% STATES (using full-period totals)
    # -----------------------------------------------------------
    state_totals = (
        df3.groupby("State Name")[metric_tab3]
        .sum()
        .reset_index()
        .sort_values(metric_tab3, ascending=False)
    )

    state_totals["CumSum"] = state_totals[metric_tab3].cumsum()
    total_sum = state_totals[metric_tab3].sum()
    state_totals["CumPct"] = state_totals["CumSum"] / total_sum

    # Top 70% states only
    top70_states = state_totals[state_totals["CumPct"] <= 0.70]["State Name"].tolist()

    # Filter dataset to only these states
    df3 = df3[df3["State Name"].isin(top70_states)]

    if df3.empty:
        st.warning("No data after applying top 70% state logic.")
        st.stop()

    # -----------------------------------------------------------
    # STEP 2 — Q1 AVG vs Q2 AVG (GROWTH CALC)
    # -----------------------------------------------------------
    Q1_SET = {"apr","may","jun"}
    JUL_OCT_SET = {"jul","aug","sep","oct"}

    # Q1 avg per state
    q1_state = (
        df3[df3["MonthKey"].isin(Q1_SET)]
        .groupby(["State Name","MonthKey"])[metric_tab3]
        .sum()
        .groupby(level=0)
        .mean()
        .rename("Q1_avg")
    )

    # Jul–Oct avg per state
    jo_state = (
        df3[df3["MonthKey"].isin(JUL_OCT_SET)]
        .groupby(["State Name","MonthKey"])[metric_tab3]
        .sum()
        .groupby(level=0)
        .mean()
        .rename("JulOct_avg")
    )

    # Merge & growth calculation
    growth_df = pd.concat([q1_state, jo_state], axis=1).reset_index().fillna(0.0)
    growth_df["Growth %"] = ((growth_df["JulOct_avg"] - growth_df["Q1_avg"]) /
                             growth_df["Q1_avg"].replace(0, np.nan) * 100)

    # -----------------------------------------------------------
    # STEP 3 — SPLIT INTO TOP GROWTH & TOP LAGGARD (ONLY INSIDE TOP 70% STATES)
    # -----------------------------------------------------------
    growth_only = growth_df[growth_df["Growth %"] > 0].sort_values("Growth %", ascending=False)
    laggard_only = growth_df[growth_df["Growth %"] < 0].sort_values("Growth %", ascending=True)

    top5_growth = growth_only.head(5)
    top_laggards = laggard_only.head(5)

    # ----------------------------
    # A. Δ GROWTH % BAR CHARTS
    # ----------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Growth Markets — Q1 vs Q2 (in %)")
        if top5_growth.empty:
            st.info("No growth markets for selected filters.")
        else:
            gbar = top5_growth.copy()
            gbar["Label"] = gbar["Growth %"].map(lambda x: f"{x:.1f}%")
            fig_gbar = px.bar(
                gbar,
                x="Growth %",
                y="State Name",
                orientation="h",
                text="Label",
                color_discrete_sequence=["#2e7d32"]
            )
            fig_gbar.update_traces(textposition="outside")
            fig_gbar.update_layout(
                xaxis_title="Growth vs Q1 (%)",
                yaxis_title="",
                height=420,
                xaxis=dict(zeroline=True, zerolinecolor="#999"),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_gbar, use_container_width=True)

    with c2:
        st.subheader("Top Laggard Markets — Q1 vs Q2 (in %)")
        if top_laggards.empty:
            st.info("No laggard markets for selected filters.")
        else:
            lbar = top_laggards.copy()
            lbar["Label"] = lbar["Growth %"].map(lambda x: f"{x:.1f}%")
            fig_lbar = px.bar(
                lbar,
                x="Growth %",
                y="State Name",
                orientation="h",
                text="Label",
                color_discrete_sequence=["#c62828"]
            )
            fig_lbar.update_traces(textposition="outside")
            fig_lbar.update_layout(
                xaxis_title="Growth vs Q1 (%)",
                yaxis_title="",
                height=420,
                xaxis=dict(zeroline=True, zerolinecolor="#999"),
                margin=dict(l=10, r=10, t=40, b=10)
            )
            st.plotly_chart(fig_lbar, use_container_width=True)

    # ----------------------------
    # B. DUMBBELL CHART — Q1 ● vs Q2 ○
    # ----------------------------
    st.subheader("Benchmark Shift — Q1 Avg ● vs Q2 Avg ○")

    JUL_SEP_SET = {"jul","aug","sep"}

    # Recompute Q1 vs Jul–Sep avg
    q1_state = (
        df3[df3["MonthKey"].isin(Q1_SET)]
        .groupby(["State Name","MonthKey"])[metric_tab3]
        .sum()
        .groupby(level=0).mean().rename("Q1_avg")
    )

    js_state = (
        df3[df3["MonthKey"].isin(JUL_SEP_SET)]
        .groupby(["State Name","MonthKey"])[metric_tab3]
        .sum()
        .groupby(level=0).mean().rename("JulSep_avg")
    )

    growth_df = pd.concat([q1_state, js_state], axis=1).reset_index().fillna(0.0)
    growth_df["Growth %"] = ((growth_df["JulSep_avg"] - growth_df["Q1_avg"]) /
                             growth_df["Q1_avg"].replace(0, np.nan) * 100)

    growth_only = growth_df[growth_df["Growth %"] > 0].sort_values("Growth %", ascending=False)
    laggard_only = growth_df[growth_df["Growth %"] < 0].sort_values("Growth %", ascending=True)

    top5_growth = growth_only.head(5)
    top_laggards = laggard_only.head(5)

    shown_states = pd.concat([top5_growth, top_laggards])
    shown_states = shown_states.sort_values("Growth %", ascending=False)

    if shown_states.empty:
        st.info("No states to display.")
    else:
        y_states = shown_states["State Name"].tolist()
        q1_vals = shown_states["Q1_avg"].tolist()
        js_vals = shown_states["JulSep_avg"].tolist()

        fig_dumb = go.Figure()

        # Connecting lines
        for i, state in enumerate(y_states):
            fig_dumb.add_trace(go.Scatter(
                x=[q1_vals[i], js_vals[i]],
                y=[state, state],
                mode="lines",
                line=dict(color="#9e9e9e", width=2),
                showlegend=False,
                hoverinfo="skip"
            ))

        # Q1 points
        fig_dumb.add_trace(go.Scatter(
            x=q1_vals,
            y=y_states,
            mode="markers",
            name="Q1 Avg",
            marker=dict(size=10, color="#616161")
        ))

        # Jul–Sep points
        fig_dumb.add_trace(go.Scatter(
            x=js_vals,
            y=y_states,
            mode="markers",
            name="Jul–Sep Avg",
            marker=dict(size=10, symbol="circle-open", color="#9575cd")
        ))

        # Formatting
        xmax = max(js_vals + q1_vals)
        tvals, ttext = indian_ticktexts(xmax)

        fig_dumb.update_layout(
            height=520,
            xaxis=dict(title=metric_tab3, tickmode="array", tickvals=tvals, ticktext=ttext),
            yaxis=dict(title=""),
            legend_title_text="",
            margin=dict(l=10, r=10, t=30, b=10)
        )

        st.plotly_chart(fig_dumb, use_container_width=True)

    # ============================================================
    # C. DRILL-DOWN — MULTI-LINE TRENDS FOR GROWTH & LAGGARD STATES
    # ============================================================

    st.subheader("Drill-down Trends (Apr → Oct)")

    MONTH_SET_FULL = {"apr", "may", "jun", "jul", "aug", "sep", "oct"}
    month_order_drill = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]

    # ------------------------------------------
    # 1️⃣ DRILL-DOWN FOR TOP GROWTH STATES
    # ------------------------------------------
    if top5_growth.empty:
        st.info("No growth states available for drill-down based on current filters.")
    else:
        st.markdown("### Top Growth States — Multi-line Trend (Apr → Oct)")

        growth_state_options = sorted(top5_growth["State Name"].tolist())
        selected_growth_states = st.multiselect(
            "Select Growth States to Display",
            growth_state_options,
            default=growth_state_options,
            key="growth_drilldown_states"
        )

        if selected_growth_states:
            g_df = df3[
                (df3["State Name"].isin(selected_growth_states)) &
                (df3["MonthKey"].isin(MONTH_SET_FULL))
            ].copy()

            g_timeline = (
                g_df.groupby(["State Name", "MonthNum", "MonthLabel"], as_index=False)[metric_tab3]
                .sum()
                .sort_values("MonthNum")
            )

            g_timeline["MonthLabel"] = pd.Categorical(
                g_timeline["MonthLabel"], month_order_drill, ordered=True
            )

            fig_growth_drill = px.line(
                g_timeline,
                x="MonthLabel",
                y=metric_tab3,
                color="State Name",
                markers=True
            )

            ymax_g = g_timeline[metric_tab3].max()
            tvg, ttg = indian_ticktexts(ymax_g)

            fig_growth_drill.update_layout(
                yaxis=dict(tickmode="array", tickvals=tvg, ticktext=ttg),
                height=420,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig_growth_drill, use_container_width=True)

        else:
            st.info("Select at least one growth state to view the graph.")

    # ------------------------------------------
    # 2️⃣ DRILL-DOWN FOR LAGGARD STATES
    # ------------------------------------------
    if top_laggards.empty:
        st.info("No laggard states available for drill-down based on current filters.")
    else:
        st.markdown("### Top Laggard States — Multi-line Trend (Apr → Oct)")

        laggard_state_options = sorted(top_laggards["State Name"].tolist())
        selected_laggard_states = st.multiselect(
            "Select Laggard States to Display",
            laggard_state_options,
            default=laggard_state_options,
            key="laggard_drilldown_states"
        )

        if selected_laggard_states:
            l_df = df3[
                (df3["State Name"].isin(selected_laggard_states)) &
                (df3["MonthKey"].isin(MONTH_SET_FULL))
            ].copy()

            l_timeline = (
                l_df.groupby(["State Name", "MonthNum", "MonthLabel"], as_index=False)[metric_tab3]
                .sum()
                .sort_values("MonthNum")
            )

            l_timeline["MonthLabel"] = pd.Categorical(
                l_timeline["MonthLabel"], month_order_drill, ordered=True
            )

            fig_laggard_drill = px.line(
                l_timeline,
                x="MonthLabel",
                y=metric_tab3,
                color="State Name",
                markers=True
            )

            ymax_l = l_timeline[metric_tab3].max()
            tvl, ttl = indian_ticktexts(ymax_l)

            fig_laggard_drill.update_layout(
                yaxis=dict(tickmode="array", tickvals=tvl, ticktext=ttl),
                height=420,
                margin=dict(l=10, r=10, t=40, b=10)
            )

            st.plotly_chart(fig_laggard_drill, use_container_width=True)

        else:
            st.info("Select at least one laggard state to view the graph.")


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
    # PREPARE MONTH-WISE DATA
    # ----------------------------
    month_order = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]

    # Industry
    ind = (
        df_ind4.groupby("MonthLabel", as_index=False)["Industry Size"]
        .sum()
    )
    ind["MonthLabel"] = pd.Categorical(ind["MonthLabel"], month_order, ordered=True)
    ind = ind.sort_values("MonthLabel")

    # Revenue
    rev = (
        df_rev.groupby("MonthLabel", as_index=False)["Revenue"]
        .sum()
    )
    rev["MonthLabel"] = pd.Categorical(rev["MonthLabel"], month_order, ordered=True)
    rev = rev.sort_values("MonthLabel")

    # Merge for table
    merged = pd.merge(ind, rev, on="MonthLabel", how="outer")

    # Only fill NA for numeric columns (not MonthLabel categorical)
    merged["Industry Size"] = merged["Industry Size"].fillna(0)
    merged["Revenue"] = merged["Revenue"].fillna(0)

    merged["Share %"] = (
        merged["Revenue"] / merged["Industry Size"].replace(0, float("nan")) * 100
    ).fillna(0).round(2)

    # ----------------------------
    # GROUPED BAR CHART
    # ----------------------------
    st.subheader("Industry Size vs GO DESi Revenue (Grouped Bars) — Apr to Oct")

    fig = go.Figure()

    # Industry Bars
    fig.add_trace(go.Bar(
        x=ind["MonthLabel"],
        y=ind["Industry Size"],
        name="Industry Size",
        marker_color="#4A90E2",
        hovertemplate="Industry Size: %{y:,.0f}<extra></extra>"
    ))

    # Revenue Bars
    fig.add_trace(go.Bar(
        x=rev["MonthLabel"],
        y=rev["Revenue"],
        name="GO DESi Revenue",
        marker_color="#FF8C00",
        hovertemplate="GO DESi Revenue: %{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        barmode="group",       # <-- GROUPED SIDE-BY-SIDE
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        title=dict(
            text="Industry Size (₹) vs GO DESi Revenue (₹) — Grouped Bars",
            x=0.01,
            y=0.93
        ),
        yaxis=dict(
            title="₹ (Raw Values)",
            tickformat=",",
        )
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
