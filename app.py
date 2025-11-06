import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
st.cache_data.clear()

# ------------------------------------------------------------
# PAGE CONFIG — must be the very first Streamlit command
# ------------------------------------------------------------
st.set_page_config(page_title="Secondary Sales Dashboard", layout="wide")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_parquet("secondary_sales.parquet")

    # Standardize types
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
    df["MonthKey"] = df["Month"].str[:3].str.lower()
    df["MonthNum"] = df["Month"].str.lower().map(month_map).astype(int)

    # Universal rule: ignore Jan–Mar and November across the app
    valid_months = [4,5,6,7,8,9,10]  # Apr..Oct
    df = df[df["MonthNum"].isin(valid_months)].copy()

    # Nice month label for x-axis
    mlabel = {4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct"}
    df["MonthLabel"] = df["MonthNum"].map(mlabel)

    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Missing 'secondary_sales.parquet'. Run merge_to_parquet.py first.")
    st.stop()

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
# DEFINE TABS
# ------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "Sales Overview",
    "Top Markets",
    "Growth vs Laggard Markets"
])

# ============================================================
# TAB 1 – OVERVIEW DASHBOARD
# ============================================================
with tab1:
    st.title("Secondary Sales — Overview (Tab 1)")

    # ------------------------------------------------------------
    # Filters (left sidebar)
    # ------------------------------------------------------------
    with st.sidebar:
        st.header("Tab 1 Filters")
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
    st.title("Top Markets — State Trends (Top 70%)")

    # ----------------------------
    # TAB 2 FILTERS (LEFT SIDEBAR)
    # ----------------------------
    with st.sidebar:
        st.header("Tab 2 Filters")

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
# TAB 3 — GROWTH vs LAGGARD MARKETS (Q1 Benchmark)
# ============================================================
with tab3:
    st.title("Growth vs Laggard Markets (Q1 Benchmark)")

    # ----------------------------
    # TAB 3 FILTERS (LEFT SIDEBAR)
    # ----------------------------
    with st.sidebar:
        st.header("Tab 3 Filters")
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

    # Filtered dataset for Tab 3 (we intentionally do NOT filter Regions/States here)
    df3 = apply_filters(df, regions=[], states=[], categories=cat_tab3, platforms=platform_tab3)
    if df3.empty:
        st.warning("No data available for selected Tab 3 filters.")
        st.stop()

    # Periods
    Q1_SET = {"apr","may","jun"}
    JUL_OCT_SET = {"jul","aug","sep","oct"}
    month_order_jo = ["Jul","Aug","Sep","Oct"]

    # --- Monthly averages per state ---
    # Q1 average (per state)
    q1_state = (
        df3[df3["MonthKey"].isin(Q1_SET)]
        .groupby(["State Name","MonthKey"])[metric_tab3]
        .sum()
        .groupby(level=0).mean()   # average across 3 months
        .rename("Q1_avg")
    )
    # Jul–Oct average (per state)
    jo_state = (
        df3[df3["MonthKey"].isin(JUL_OCT_SET)]
        .groupby(["State Name","MonthKey"])[metric_tab3]
        .sum()
        .groupby(level=0).mean()   # average across 4 months
        .rename("JulOct_avg")
    )

    # Merge & growth %
    growth_df = pd.concat([q1_state, jo_state], axis=1).reset_index().fillna(0.0)
    growth_df["Growth %"] = ((growth_df["JulOct_avg"] - growth_df["Q1_avg"]) /
                             growth_df["Q1_avg"].replace(0, np.nan) * 100)

    # Split into growth & laggard sets (limit to top/bottom 5, but allow fewer laggards)
    growth_only = growth_df[growth_df["Growth %"] > 0].sort_values("Growth %", ascending=False)
    laggard_only = growth_df[growth_df["Growth %"] < 0].sort_values("Growth %", ascending=True)

    top5_growth = growth_only.head(5)
    top_laggards = laggard_only.head(5)   # may have < 5 rows; that's fine

    # ----------------------------
    # A. Δ Growth % BAR CHARTS (style C)
    # ----------------------------
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Growth Markets — Q1 vs Q2 (in %)")
        if top5_growth.empty:
            st.info("No growth markets for the selected filters.")
        else:
            gbar = top5_growth.copy()
            gbar["Label"] = gbar["Growth %"].map(lambda x: f"{x:.1f}%")
            fig_gbar = px.bar(
                gbar,
                x="Growth %",
                y="State Name",
                orientation="h",
                text="Label",
                color_discrete_sequence=["#2e7d32"]  # green
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
            st.info("No laggard markets — fewer than 1 state dipped vs Q1.")
        else:
            lbar = top_laggards.copy()
            lbar["Label"] = lbar["Growth %"].map(lambda x: f"{x:.1f}%")
            fig_lbar = px.bar(
                lbar,
                x="Growth %",
                y="State Name",
                orientation="h",
                text="Label",
                color_discrete_sequence=["#c62828"]  # red
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
# B. DUMBBELL (Q1 avg ● vs Jul–Sep avg ○)
# ----------------------------
st.subheader("Benchmark Shift — Q1 Avg ● vs Q2 Avg ○")

import plotly.graph_objects as go

# Recompute Jul–Sep Avg (instead of Jul–Oct)
JUL_SEP_SET = {"jul", "aug", "sep"}

q1_state = (
    df3[df3["MonthKey"].isin(Q1_SET)]
    .groupby(["State Name","MonthKey"])[metric_tab3]
    .sum()
    .groupby(level=0).mean()
    .rename("Q1_avg")
)

js_state = (
    df3[df3["MonthKey"].isin(JUL_SEP_SET)]
    .groupby(["State Name","MonthKey"])[metric_tab3]
    .sum()
    .groupby(level=0).mean()
    .rename("JulSep_avg")
)

# Merge and recalc growth %
growth_df = pd.concat([q1_state, js_state], axis=1).reset_index().fillna(0.0)
growth_df["Growth %"] = ((growth_df["JulSep_avg"] - growth_df["Q1_avg"]) /
                         growth_df["Q1_avg"].replace(0, np.nan) * 100)

# Same top growth + laggards selection logic
growth_only = growth_df[growth_df["Growth %"] > 0].sort_values("Growth %", ascending=False)
laggard_only = growth_df[growth_df["Growth %"] < 0].sort_values("Growth %", ascending=True)

top5_growth = growth_only.head(5)
top_laggards = laggard_only.head(5)

# Combine for chart
shown_states = pd.concat([top5_growth, top_laggards], ignore_index=True)
if shown_states.empty:
    st.info("No states to display for benchmark comparison.")
else:
    shown_states = shown_states.sort_values("Growth %", ascending=False)

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
        marker=dict(size=10, symbol="circle", color="#616161"),
        hovertemplate="State=%{y}<br>Q1 Avg="+metric_tab3+"=%{x:.0f}<extra></extra>"
    ))

    # Jul–Sep points
    fig_dumb.add_trace(go.Scatter(
        x=js_vals,
        y=y_states,
        mode="markers",
        name="Jul–Sep Avg",
        marker=dict(size=10, symbol="circle-open", color="#9575cd"),
        hovertemplate="State=%{y}<br>Jul–Sep Avg="+metric_tab3+"=%{x:.0f}<extra></extra>"
    ))

    # Apply Indian number formatting to axis ticks
    xmax = max(js_vals + q1_vals) if (js_vals or q1_vals) else 0.0
    tvals, ttext = indian_ticktexts(xmax)

    fig_dumb.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title=metric_tab3, tickmode="array", tickvals=tvals, ticktext=ttext),
        yaxis=dict(title=""),
        legend_title_text=""
    )

    st.plotly_chart(fig_dumb, use_container_width=True)

    # ----------------------------
    # C. DRILL-DOWN — Monthly Trend for Selected State (Jul→Oct)
    # ----------------------------
    if not top5_growth.empty:
        default_state = top5_growth.iloc[0]["State Name"]
    elif not shown_states.empty:
        default_state = shown_states.iloc[0]["State Name"]
    else:
        default_state = None

    if default_state is None:
        st.info("No state available for drill-down trend.")
    else:
        st.subheader("Drill-down: Monthly Trend (Jul→Oct)")
        state_choice = st.selectbox(
            "Pick a state",
            options=sorted(df3["State Name"].dropna().unique().tolist()),
            index=sorted(df3["State Name"].dropna().unique().tolist()).index(default_state)
            if default_state in df3["State Name"].values else 0
        )

        # Monthly series for chosen state (Jul–Oct)
        s_df = df3[(df3["State Name"] == state_choice) & (df3["MonthKey"].isin(JUL_OCT_SET))].copy()
        s_line = (
            s_df.groupby(["MonthNum","MonthLabel"], as_index=False)[metric_tab3]
            .sum()
            .sort_values("MonthNum")
        )
        s_line["MonthLabel"] = pd.Categorical(s_line["MonthLabel"], month_order_jo, ordered=True)

        fig_trend = px.line(
            s_line,
            x="MonthLabel",
            y=metric_tab3,
            markers=True
        )
        # Add the state's Q1 average as benchmark hline
        s_q1 = growth_df.set_index("State Name").get("Q1_avg")
        s_q1_val = float(s_q1.get(state_choice, np.nan)) if s_q1 is not None else np.nan
        if not np.isnan(s_q1_val):
            fig_trend.add_hline(
                y=s_q1_val,
                line_dash="dot",
                line_color="#9e9e9e",
                annotation_text=f"Q1 Avg: {format_indian(s_q1_val)}",
                annotation_position="top left",
                annotation_font=dict(size=11, color="#616161")
            )

        ymax_tr = float(s_line[metric_tab3].max()) if not s_line.empty else 0.0
        tv_tr, tt_tr = indian_ticktexts(ymax_tr)
        fig_trend.update_layout(
            yaxis=dict(tickmode="array", tickvals=tv_tr, ticktext=tt_tr, title=metric_tab3),
            xaxis_title="Month",
            height=420,
            margin=dict(l=10, r=10, t=40, b=10)
        )
        fig_trend.update_traces(hovertemplate="Month=%{x}<br>"+metric_tab3+"=%{y:.0f}<extra></extra>")
        st.plotly_chart(fig_trend, use_container_width=True)
