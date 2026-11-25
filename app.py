import streamlit as st
import duckdb
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import time
import skill_position_nn as spnn

st.set_page_config(
    page_title="NFL Big Data Bowl - Database Explorer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_FULL_DB = os.path.join(_ROOT_DIR, "nfl_data.duckdb")
_LITE_DB = os.path.join(_ROOT_DIR, "nfl_data_lite.duckdb")
DB_PATH = _FULL_DB if os.path.exists(_FULL_DB) else _LITE_DB

@st.cache_resource
def get_db_connection():
    if not os.path.exists(DB_PATH):
        st.error(f"Database file '{DB_PATH}' not found. Please run `python build_database.py` first.")
        st.stop()
    # Connect read-only so multiple queries/tabs can run concurrently safely
    return duckdb.connect(DB_PATH, read_only=True)

con = get_db_connection()

# Sidebar: Database Overview
st.sidebar.title("🏈 NFL BDB Query Engine")
st.sidebar.caption("Powered by DuckDB Columnar Database")

# Tables and Counts
@st.cache_data(ttl=3600)
def get_tables_metadata():
    tables = con.execute("""
        SELECT table_name, table_type 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_type, table_name;
    """).df()
    counts = {}
    for t in tables["table_name"]:
        c = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
        counts[t] = c
    tables["row_count"] = tables["table_name"].map(counts)
    return tables

@st.cache_data(ttl=3600)
def get_columns_metadata():
    return con.execute("""
        SELECT table_name, column_name, data_type, ordinal_position
        FROM information_schema.columns 
        WHERE table_schema = 'main'
        ORDER BY table_name, ordinal_position;
    """).df()

tables_meta = get_tables_metadata()
columns_meta = get_columns_metadata()

total_records = tables_meta[tables_meta["table_type"] == "BASE TABLE"]["row_count"].sum()
db_size_mb = os.path.getsize(DB_PATH) / (1024 * 1024)

st.sidebar.metric("Total Indexed Records", f"{total_records:,}")
st.sidebar.metric("Database File Size", f"{db_size_mb:.1f} MB")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "🧠 Skill Position Neural Net & Clutch Projector",
        "⚡ SQL Query Lab",
        "📋 Database Schema & Preview",
        "📊 Instant Visualizations",
        "💡 Quick Query Guide",
    ],
)

@st.cache_resource
def load_skill_nn_bundle(schema_version: str = "v3_end_of_combine_only"):
    df_players, df_games, df_tracking = spnn.build_skill_position_dataset(con)
    tier_results = spnn.train_and_evaluate_tiers(df_players, k=5, seed=42)
    return df_players, df_games, df_tracking, tier_results

# -------------------------------------------------------------
# SIDEBAR: TABLE & COLUMN EXPLORER
# -------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🗂️ Tables & Columns")

search_term = st.sidebar.text_input(
    "🔍 Filter tables & columns",
    value="",
    placeholder="e.g. speed, yards, nfl_id..."
).strip().lower()

# Group columns by table
cols_by_table = {}
for _, row in columns_meta.iterrows():
    tbl = row["table_name"]
    cols_by_table.setdefault(tbl, []).append((row["column_name"], row["data_type"]))

# Display tables and columns
matching_tables_count = 0
for _, row in tables_meta.iterrows():
    tbl_name = row["table_name"]
    tbl_type = row["table_type"]
    row_count = row["row_count"]
    cols = cols_by_table.get(tbl_name, [])
    
    # Filter logic
    if search_term:
        tbl_matches = search_term in tbl_name.lower()
        matching_cols = [c for c in cols if search_term in c[0].lower() or search_term in c[1].lower()]
        if not tbl_matches and not matching_cols:
            continue
        display_cols = matching_cols if not tbl_matches else cols
        is_expanded = True
    else:
        display_cols = cols
        is_expanded = False
        
    matching_tables_count += 1
    icon = "👁️" if tbl_type == "VIEW" else "📁"
    expander_title = f"{icon} {tbl_name} ({len(cols)} cols)"
    
    with st.sidebar.expander(expander_title, expanded=is_expanded):
        st.caption(f"**Type:** `{tbl_type}` &nbsp;|&nbsp; **Rows:** `{row_count:,}`")
        
        # Click-to-copy column list
        col_names_str = ", ".join([c[0] for c in cols])
        st.code(col_names_str, language="sql")
        st.caption("☝️ *Hover & click copy icon to copy columns for SELECT*")
        
        st.markdown("**Columns:**")
        for col_name, dtype in display_cols:
            if search_term and search_term in col_name.lower():
                st.markdown(f"- **`{col_name}`** &nbsp; <span style='color:#ff4b4b; font-size:0.8em; font-weight:bold;'>{dtype}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"- `{col_name}` &nbsp; <span style='color:#888; font-size:0.8em;'>{dtype}</span>", unsafe_allow_html=True)

if search_term and matching_tables_count == 0:
    st.sidebar.info(f"No tables or columns matching '{search_term}'")


# -------------------------------------------------------------
# TAB 1: SQL QUERY LAB
# -------------------------------------------------------------
if nav_choice == "⚡ SQL Query Lab":
    st.header("⚡ SQL Query Lab")
    st.markdown("Execute analytical SQL queries directly against **23.4 million records** in milliseconds.")

    SAMPLE_QUERIES = {
        "Custom Query": "",
        "Top 10 Fastest Players Recorded in 2024": """SELECT 
    p.display_name,
    p.nfl_position,
    p.college_name,
    MAX(t.s) AS max_speed_mps,
    ROUND(MAX(t.s) * 2.04545, 2) AS max_speed_mph
FROM game_tracking_2024 t
JOIN players p ON t.nfl_id = p.nfl_id
GROUP BY p.display_name, p.nfl_position, p.college_name
ORDER BY max_speed_mps DESC
LIMIT 10;""",
        
        "Combine 40-yd Dash vs Top Recorded Game Speed": """SELECT 
    p.display_name, 
    p.nfl_position, 
    c.forty, 
    c.vertical,
    c.broad_jump,
    ROUND(MAX(t.s) * 2.04545, 2) AS top_game_speed_mph
FROM players p
JOIN combine_results c ON p.nfl_id = c.nfl_id
JOIN game_tracking t ON p.nfl_id = t.nfl_id
WHERE c.forty IS NOT NULL
GROUP BY p.display_name, p.nfl_position, c.forty, c.vertical, c.broad_jump
ORDER BY top_game_speed_mph DESC
LIMIT 15;""",

        "Offensive Formations: Usage & Average Yards Gained": """SELECT 
    offense_formation, 
    count(*) AS total_plays, 
    ROUND(AVG(yards_gained), 2) AS avg_yards_gained,
    ROUND(AVG(expected_points_added), 3) AS avg_epa
FROM player_play
WHERE offense_formation IS NOT NULL
GROUP BY offense_formation
ORDER BY total_plays DESC;""",

        "Unified Tracking: Events Breakdown Across All 3 Seasons": """SELECT 
    season,
    event,
    count(*) AS event_count
FROM game_tracking
WHERE event IS NOT NULL
GROUP BY season, event
ORDER BY season DESC, event_count DESC
LIMIT 20;""",

        "Top College Programs by Drafted Players in Dataset": """SELECT 
    college_name, 
    college_conference,
    count(*) AS player_count,
    ROUND(AVG(draft_overall_pick), 1) AS avg_overall_pick
FROM players
WHERE college_name IS NOT NULL
GROUP BY college_name, college_conference
ORDER BY player_count DESC
LIMIT 10;"""
    }

    selected_sample = st.selectbox("Load Example Query:", list(SAMPLE_QUERIES.keys()))
    default_sql = SAMPLE_QUERIES[selected_sample] if selected_sample != "Custom Query" else "SELECT * FROM players LIMIT 10;"

    sql_input = st.text_area("SQL Statement:", value=default_sql, height=160)

    col_btn1, col_btn2, col_info = st.columns([1, 1, 3])
    run_pressed = col_btn1.button("▶ Run Query", type="primary")

    if run_pressed or sql_input:
        if sql_input.strip():
            t0 = time.time()
            try:
                result_df = con.execute(sql_input).df()
                elapsed = time.time() - t0
                
                st.success(f"Query returned **{len(result_df):,} rows** in **{elapsed*1000:.1f} ms**")
                
                # Result table
                st.dataframe(result_df, width="stretch")
                
                # Download button
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="query_results.csv",
                    mime="text/csv"
                )
            except Exception as e:
                st.error(f"SQL Execution Error:\n{e}")

# -------------------------------------------------------------
# TAB 2: DATABASE SCHEMA & PREVIEW
# -------------------------------------------------------------
elif nav_choice == "📋 Database Schema & Preview":
    st.header("📋 Database Schema & Data Dictionary")
    st.markdown("Inspect all tables, column definitions, data types, and preview raw data.")
    
    st.dataframe(tables_meta, width="stretch")
    
    selected_table = st.selectbox("Select Table or View to Inspect:", tables_meta["table_name"].tolist())
    
    schema_df = con.execute(f"DESCRIBE {selected_table}").df()
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Columns & Data Types")
        st.dataframe(schema_df[["column_name", "column_type", "null"]], width="stretch")
    
    with col2:
        st.subheader(f"Data Preview for `{selected_table}` (First 50 Rows)")
        preview_df = con.execute(f"SELECT * FROM {selected_table} LIMIT 50").df()
        st.dataframe(preview_df, width="stretch")

# -------------------------------------------------------------
# TAB 3: INSTANT VISUALIZATIONS
# -------------------------------------------------------------
elif nav_choice == "📊 Instant Visualizations":
    st.header("📊 Interactive Visual Analytics")
    st.markdown("Pre-aggregated visual explorations running live DuckDB analytical queries.")
    
    chart_choice = st.radio("Choose Dashboard View:", [
        "Combine 40-yd Dash vs Top Recorded Game Speed (mph)",
        "Yards Gained Distribution by Offensive Formation",
        "Top 15 Colleges by NFL Players Drafted"
    ], horizontal=True)
    
    if chart_choice == "Combine 40-yd Dash vs Top Recorded Game Speed (mph)":
        query = """
        SELECT 
            p.display_name, 
            p.nfl_position, 
            c.forty, 
            ROUND(MAX(t.s) * 2.04545, 2) AS top_game_speed_mph
        FROM players p
        JOIN combine_results c ON p.nfl_id = c.nfl_id
        JOIN game_tracking t ON p.nfl_id = t.nfl_id
        WHERE c.forty IS NOT NULL
        GROUP BY p.display_name, p.nfl_position, c.forty
        ORDER BY top_game_speed_mph DESC;
        """
        df = con.execute(query).df()
        fig = px.scatter(
            df, 
            x="forty", 
            y="top_game_speed_mph", 
            color="nfl_position", 
            hover_name="display_name",
            title="Combine 40-Yard Dash (s) vs Max On-Field Tracking Speed (mph)",
            labels={"forty": "Combine 40-Yard Dash (seconds)", "top_game_speed_mph": "Top In-Game Speed (mph)"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    elif chart_choice == "Yards Gained Distribution by Offensive Formation":
        query = """
        SELECT 
            offense_formation, 
            ROUND(AVG(yards_gained), 2) AS avg_yards_gained,
            count(*) as play_count
        FROM player_play
        WHERE offense_formation IS NOT NULL
        GROUP BY offense_formation
        ORDER BY avg_yards_gained DESC;
        """
        df = con.execute(query).df()
        fig = px.bar(
            df,
            x="offense_formation",
            y="avg_yards_gained",
            text="avg_yards_gained",
            color="play_count",
            title="Average Yards Gained by Offensive Formation",
            labels={"offense_formation": "Offensive Formation", "avg_yards_gained": "Avg Yards Gained"}
        )
        st.plotly_chart(fig, use_container_width=True)

    elif chart_choice == "Top 15 Colleges by NFL Players Drafted":
        query = """
        SELECT 
            college_name, 
            count(*) AS player_count
        FROM players
        WHERE college_name IS NOT NULL
        GROUP BY college_name
        ORDER BY player_count DESC
        LIMIT 15;
        """
        df = con.execute(query).df()
        fig = px.bar(
            df,
            x="college_name",
            y="player_count",
            text="player_count",
            color="player_count",
            title="Top Colleges by Total Players Drafted in Dataset",
            labels={"college_name": "College", "player_count": "Players"}
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: QUICK QUERY GUIDE
# -------------------------------------------------------------
elif nav_choice == "💡 Quick Query Guide":
    st.header("💡 How to Query This Database")
    st.markdown("""
    ### 1. In Python
    ```python
    import duckdb

    # Connect to the persistent database file
    con = duckdb.connect("nfl_data.duckdb", read_only=True)

    # Query directly to a pandas DataFrame
    df = con.execute("SELECT * FROM players LIMIT 5").df()
    print(df)
    ```

    ### 2. From the Terminal (CLI)
    - **Interactive Shell:**
      ```bash
      python query.py
      ```
    - **Single Query:**
      ```bash
      python query.py "SELECT display_name, nfl_position FROM players LIMIT 5;"
      ```
    - **Export Query to CSV:**
      ```bash
      python query.py --csv output.csv "SELECT * FROM combine_results;"
      ```

    ### 3. In External GUI Tools (DBeaver, VS Code, JetBrains)
    You can connect to `nfl_data.duckdb` directly using:
    - **DBeaver**: New Connection -> DuckDB -> select `nfl_data.duckdb`
    - **VS Code**: Install the "DuckDB" or "SQLTools" extension
    - **MotherDuck / DuckDB CLI**: `duckdb nfl_data.duckdb`
    """)

# -------------------------------------------------------------
# TAB 5: SKILL POSITION MULTI-MODEL, CLUTCH & TEAM IMPACT SUITE
# -------------------------------------------------------------
elif nav_choice == "🧠 Skill Position Neural Net & Clutch Projector":
    st.header("🧠 Skill Position Multi-Model, Clutch & Team-Impact Projector")
    st.markdown(
        "Compare **4 Model Families** (**Hybrid Stack Ensemble**, **Dual-Branch Multi-Task Neural Net**, "
        "**Bayesian Hierarchical Partial-Pooling Model**, and **XGBoost / GBDT**) across **11 Targets**: "
        "Rookie Box-Score Production, **4th-Quarter Clutch Scoring**, and **Beyond-Box-Score Team Impact & Offensive Gravity** "
        "(Chain-Moving 1st Downs, 3rd/4th Down Conversions, Target EPA, Target-Earning Rate `TPRR%`, and Decoy Non-Target Team EPA)."
    )

    with st.spinner("Training 5-Fold Ensembles (Dual-Branch NN + Bayesian Hierarchical + XGBoost/GBDT)..."):
        df_players, df_games, df_tracking, tier_results = load_skill_nn_bundle("v3_end_of_combine_only")

    top_c1, top_c2 = st.columns([1.15, 1])
    with top_c1:
        selected_model_family = st.selectbox(
            "🤖 Select Active Model Architecture:",
            spnn.MODEL_FAMILIES,
            index=0,
            help=(
                "Switch live predictions between the Hybrid Stack Ensemble, Dual-Branch Neural Network, "
                "Bayesian Hierarchical Model (with analytical 10th/90th percentile Floor & Ceiling credible intervals), "
                "and XGBoost/GBDT (with native missing-value split routing)."
            ),
        )
    with top_c2:
        tier_names = list(spnn.FEATURE_TIERS.keys())
        selected_tier = st.selectbox(
            "🎚️ Select End-of-Combine Feature Tier:",
            tier_names,
            index=1,
            help="All models strictly use information available at the conclusion of the NFL Combine (no draft round or draft pick used).",
        )

    active_res = tier_results[selected_tier]
    oof_preds = active_res["oof_by_family"][selected_model_family]
    fold_stds = active_res["ensemble_std"]
    bayes_std = active_res["bayes_std"]
    bayes_floor = active_res["bayes_floor_10"]
    bayes_ceil = active_res["bayes_ceil_90"]

    view_df = df_players.copy()
    for t_i, t_col in enumerate(spnn.TARGET_COLUMNS):
        view_df[f"pred_{t_col}"] = oof_preds[:, t_i]
        view_df[f"std_{t_col}"] = fold_stds[:, t_i]
        view_df[f"bstd_{t_col}"] = bayes_std[:, t_i]
        view_df[f"floor10_{t_col}"] = bayes_floor[:, t_i]
        view_df[f"ceil90_{t_col}"] = bayes_ceil[:, t_i]

    # Composite 0-100 Late-Game Clutch Athlete Percentile
    clutch_raw = (
        0.35 * view_df["pred_q4_tds_per_game"].rank(pct=True)
        + 0.35 * view_df["pred_q4_explosive_per_game"].rank(pct=True)
        + 0.30 * view_df["pred_clutch_yards_per_game"].rank(pct=True)
    )
    view_df["clutch_percentile"] = (clutch_raw * 100.0).round(1)

    # Composite 0-100 Beyond-Box-Score Team Impact & Offensive Gravity Percentile
    impact_raw = (
        0.25 * view_df["pred_first_downs_per_100_routes"].rank(pct=True)
        + 0.25 * view_df["pred_third_fourth_conv_per_season"].rank(pct=True)
        + 0.20 * view_df["pred_target_epa_per_season"].rank(pct=True)
        + 0.15 * view_df["pred_tprr_pct"].rank(pct=True)
        + 0.15 * view_df["pred_gravity_non_target_epa_100"].rank(pct=True)
    )
    view_df["team_impact_percentile"] = (impact_raw * 100.0).round(1)

    # Actual observed Team Impact composite percentile for comparison
    actual_impact_raw = (
        0.25 * view_df["first_downs_per_100_routes"].rank(pct=True)
        + 0.25 * view_df["third_fourth_conv_per_season"].rank(pct=True)
        + 0.20 * view_df["target_epa_per_season"].rank(pct=True)
        + 0.15 * view_df["tprr_pct"].rank(pct=True)
        + 0.15 * view_df["gravity_non_target_epa_100"].rank(pct=True)
    )
    view_df["actual_impact_percentile"] = (actual_impact_raw * 100.0).round(1)

    sub_tab1, sub_tab2, sub_tab3, sub_tab4, sub_tab5 = st.tabs([
        "🎯 Player Game, Bayesian & Clutch Projector",
        "🌐 Team Impact & Offensive Gravity Lab",
        "🏆 Leaderboards & Model Predictions",
        "🧪 Custom Prospect 'What-If' Lab",
        "🔬 Multi-Model Benchmark & Signal Proof",
    ])

    # =========================================================
    # SUB-TAB 1: PLAYER GAME, BAYESIAN & CLUTCH PROJECTOR
    # =========================================================
    with sub_tab1:
        col_sel1, col_sel2, col_sel3 = st.columns([2, 1, 1])
        with col_sel2:
            pos_filter = st.selectbox("Filter Position:", ["All (WR/TE/RB/FB)", "WR", "TE", "RB", "FB"])
        with col_sel3:
            yr_filter = st.selectbox("Filter Draft Class:", ["All (2023-2025)", 2023, 2024, 2025])

        filtered_p = view_df.copy()
        if pos_filter != "All (WR/TE/RB/FB)":
            filtered_p = filtered_p[filtered_p["nfl_position"] == pos_filter]
        if yr_filter != "All (2023-2025)":
            filtered_p = filtered_p[filtered_p["draft_year"] == int(yr_filter)]

        filtered_p = filtered_p.sort_values("yards_per_game", ascending=False)
        player_options = [
            f"{r['display_name']} ({r['nfl_position']} - {int(r['draft_year'])}, {r['college_name']})"
            for _, r in filtered_p.iterrows()
        ]
        id_map = {opt: r["nfl_id"] for opt, (_, r) in zip(player_options, filtered_p.iterrows())}

        with col_sel1:
            selected_opt = st.selectbox("🏈 Select Skill Position Player:", player_options)

        sel_id = id_map[selected_opt]
        p_row = view_df[view_df["nfl_id"] == sel_id].iloc[0]

        forty_str = f"{p_row['forty']:.2f}s" if pd.notna(p_row["forty"]) else "Skipped"
        go_a_str = f"{p_row['drill_go_route_max_a']:.2f} yd/s²" if pd.notna(p_row["drill_go_route_max_a"]) else "N/A"
        tr40_a_str = f"{p_row['tr40_p90_a']:.2f} yd/s²" if pd.notna(p_row["tr40_p90_a"]) else "N/A"
        slant_s_str = (
            f"{p_row['drill_slant_ro_max_s'] * 2.04545:.1f} mph"
            if pd.notna(p_row["drill_slant_ro_max_s"])
            else "N/A"
        )

        st.info(
            f"**{p_row['display_name']}** (`nfl_id={int(sel_id)}`) &nbsp;|&nbsp; "
            f"**Pos:** `{p_row['nfl_position']}` &nbsp;|&nbsp; **College:** `{p_row['college_name']}` &nbsp;|&nbsp; "
            f"**Combine Year:** `{int(p_row['draft_year'])}` &nbsp;|&nbsp; "
            f"**40-Yd:** `{forty_str}` &nbsp;|&nbsp; **40-Yd P90 Accel:** `{tr40_a_str}` &nbsp;|&nbsp; "
            f"**Go-Route Peak Accel:** `{go_a_str}` &nbsp;|&nbsp; **Slant Speed:** `{slant_s_str}` &nbsp;|&nbsp; "
            f"**Clutch Score:** `{p_row['clutch_percentile']:.1f}th pct` &nbsp;|&nbsp; "
            f"**Team Impact & Gravity Score:** `{p_row['team_impact_percentile']:.1f}th pct`"
        )

        st.subheader(f"📊 Box-Score & 4th-Quarter Clutch Projections (`{selected_model_family}`)")
        st.caption(
            "Includes **Bayesian 80% Credible Intervals** (`10th-pct Floor` to `90th-pct Ceiling`) alongside out-of-sample predictions."
        )
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            "Proj Rookie Yds/G",
            f"{p_row['pred_yards_per_game']:.1f} yd/g",
            delta=f"Actual: {p_row['yards_per_game']:.1f} (80% CI: {p_row['floor10_yards_per_game']:.0f}–{p_row['ceil90_yards_per_game']:.0f})",
            delta_color="off",
        )
        m2.metric(
            "Proj Rookie TDs/G",
            f"{p_row['pred_tds_per_game']:.2f} td/g",
            delta=f"Actual: {p_row['tds_per_game']:.2f} (80% CI: {p_row['floor10_tds_per_game']:.2f}–{p_row['ceil90_tds_per_game']:.2f})",
            delta_color="off",
        )
        m3.metric(
            "Proj Q4 Clutch Yds/Yr",
            f"{p_row['pred_clutch_yards_per_game']:.1f} yds",
            delta=f"Actual: {p_row['clutch_yards_per_game']:.1f} yds",
            delta_color="off",
        )
        m4.metric(
            "Proj 4th-Quarter TDs/Yr",
            f"{p_row['pred_q4_tds_per_game']:.2f} TDs",
            delta=f"Actual: {p_row['q4_tds_per_game']:.2f} TDs",
            delta_color="off",
        )
        m5.metric(
            "Proj Q4 Explosive (15+ yd)/Yr",
            f"{p_row['pred_q4_explosive_per_game']:.2f}",
            delta=f"Actual: {p_row['q4_explosive_per_game']:.2f}",
            delta_color="off",
        )
        q4_step_actual = p_row["q4_vs_q13_ypr_diff"]
        q4_step_str = f"{q4_step_actual:+.2f} yd/rt" if pd.notna(q4_step_actual) else "N/A"
        m6.metric(
            "Proj Q4 Clutch WPA/Yr",
            f"+{p_row['pred_clutch_wpa_per_season']*100:.1f}% WPA",
            delta=f"Actual: +{p_row['clutch_wpa_per_season']*100:.1f}% (ΔYPRR {q4_step_str})",
            delta_color="off",
        )

        st.subheader("🌐 Beyond-Box-Score Team Impact & Offensive Gravity Scorecard")
        ti1, ti2, ti3, ti4, ti5, ti6 = st.columns(6)
        ti1.metric(
            "Chain-Moving: 1st Dns/100 Rts",
            f"{p_row['pred_first_downs_per_100_routes']:.1f}",
            delta=f"Actual: {p_row['first_downs_per_100_routes']:.1f} / 100 rts",
            delta_color="off",
        )
        ti2.metric(
            "Money Downs: 3rd/4th Conv/Yr",
            f"{p_row['pred_third_fourth_conv_per_season']:.1f}",
            delta=f"Actual: {p_row['third_fourth_conv_per_season']:.1f} conv/yr",
            delta_color="off",
        )
        ti3.metric(
            "Target EPA / Season",
            f"{p_row['pred_target_epa_per_season']:+.1f} EPA",
            delta=f"Actual: {p_row['target_epa_per_season']:+.1f} EPA/yr",
            delta_color="off",
        )
        ti4.metric(
            "Target-Earning Rate (TPRR)",
            f"{p_row['pred_tprr_pct']:.1f}%",
            delta=f"Actual: {p_row['tprr_pct']:.1f}% of routes",
            delta_color="off",
        )
        ti5.metric(
            "Gravity: Decoy EPA/100 Snaps",
            f"{p_row['pred_gravity_non_target_epa_100']:+.2f} EPA",
            delta=f"Actual: {p_row['gravity_non_target_epa_100']:+.2f} EPA/100",
            delta_color="off",
        )
        ti6.metric(
            "YAC Over Expected (YACOE)",
            f"{p_row['yacoe_per_catch']:+.2f} yd/rec",
            delta=f"Motion Rate: {p_row['motion_rate_pct']:.1f}%",
            delta_color="off",
        )

        st.markdown("---")
        # Bayesian In-Season Belief Updating Chart (Week 0 Combine Prior -> Week K Posterior)
        p_games = df_games[df_games["nfl_id"] == sel_id].sort_values(["season", "week"]).copy()
        st.subheader(f"📈 Bayesian Hierarchical Belief Updating: {p_row['display_name']} (Week 0 Combine Prior → NFL Posterior)")
        st.caption(
            "Solves the small-sample rookie problem: at Week 0, our estimate is 100% anchored to the player's **Combine & Movement Prior** "
            "with its 80% credible interval (`10th–90th percentile`). As the player logs NFL routes each week, Bayes' rule partially pools "
            "the Combine Prior with cumulative NFL evidence."
        )
        bayes_traj_df = spnn.compute_bayesian_season_trajectory(
            prior_ypg=p_row["pred_yards_per_game"],
            prior_std=p_row["bstd_yards_per_game"],
            player_games_df=p_games,
            tau_prior_games=6.5,
        )
        fig_bay = go.Figure()
        fig_bay.add_trace(
            go.Scatter(
                x=bayes_traj_df["game_label"],
                y=bayes_traj_df["ceil_90"],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig_bay.add_trace(
            go.Scatter(
                x=bayes_traj_df["game_label"],
                y=bayes_traj_df["floor_10"],
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(31, 119, 180, 0.18)",
                name="Bayesian 80% Credible Band (10th–90th Pct)",
            )
        )
        fig_bay.add_trace(
            go.Scatter(
                x=bayes_traj_df["game_label"],
                y=bayes_traj_df["posterior_mean"],
                mode="lines+markers",
                line=dict(color="#1f77b4", width=3),
                name="Bayesian Posterior True Talent (Yds/G)",
            )
        )
        if len(p_games) > 0:
            fig_bay.add_trace(
                go.Scatter(
                    x=bayes_traj_df["game_label"],
                    y=bayes_traj_df["cum_observed_avg"],
                    mode="lines+markers",
                    line=dict(color="#2ca02c", width=2, dash="dot"),
                    name="Unshrunk Cumulative NFL Average",
                )
            )
        fig_bay.add_hline(
            y=float(p_row["pred_yards_per_game"]),
            line_dash="dash",
            line_color="#ff7f0e",
            annotation_text="Pre-Draft Combine Prior",
        )
        fig_bay.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=35, b=20),
            xaxis_title="Career Progression (Pre-Draft Combine Prior → Game-by-Game Posterior)",
            yaxis_title="Estimated True Yards / Game",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bay, use_container_width=True)

        st.markdown("---")
        # Interactive Game Situation Simulator + Actual Game Log
        sim_col1, sim_col2 = st.columns([1, 1.35])
        with sim_col1:
            st.subheader("🕹️ Single-Game & Late-Game Clutch Simulator")
            st.caption(
                f"Condition this player's `{selected_model_family}` projection on game route volume and 4th-quarter leverage."
            )
            default_routes = int(np.clip(round(p_row["routes_per_game"]) if p_row["routes_per_game"] > 0 else 18, 4, 45))
            sim_situation = st.radio(
                "Game State / Window:",
                [
                    "Standard Full Game (Q1-Q4)",
                    "4th Quarter / Overtime Only",
                    "Clutch Q4 (One-Score Game / 2-Min Drill)",
                ],
            )
            max_rt = 45 if sim_situation == "Standard Full Game (Q1-Q4)" else 18
            init_rt = default_routes if sim_situation == "Standard Full Game (Q1-Q4)" else max(4, int(round(default_routes / 3.5)))
            sim_routes = st.slider("Projected Routes Run in Window:", min_value=1, max_value=max_rt, value=min(init_rt, max_rt))
            sim_close = st.checkbox("One-Score Game / High Win-Probability Leverage (|Score Diff| ≤ 8)", value=True)

            oof_dict = {t_col: p_row[f"pred_{t_col}"] for t_col in spnn.TARGET_COLUMNS}
            game_proj = spnn.project_player_game(
                p_row,
                oof_dict,
                routes_in_game=sim_routes,
                game_situation=sim_situation,
                is_clutch_close=sim_close,
            )

            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Expected Yards", f"{game_proj['projected_yards']:.1f} yds")
            sc2.metric("Expected Touchdowns", f"{game_proj['projected_tds']:.2f} TDs")
            sc3.metric("1+ Touchdown Prob", f"{game_proj['prob_1plus_td']*100:.1f}%")

            sc4, sc5, sc6 = st.columns(3)
            sc4.metric("Expected 1st Downs", f"{game_proj['projected_first_downs']:.2f} 1D")
            sc5.metric("15+ Yd Explosive Prob", f"{game_proj['prob_explosive_15yd']*100:.1f}%")
            sc6.metric("Expected WPA", f"+{game_proj['expected_wpa']*100:.1f}%")

        with sim_col2:
            st.subheader(f"📅 {p_row['display_name']} — Game-by-Game Projected vs. Actual Log")
            if len(p_games) == 0:
                st.warning("No offensive game snaps recorded in `player_play` for this player.")
            else:
                proj_g_yds = []
                proj_g_tds = []
                for _, gr in p_games.iterrows():
                    gp = spnn.project_player_game(
                        p_row,
                        oof_dict,
                        routes_in_game=max(1, int(gr["routes"])),
                        game_situation="Standard Full Game (Q1-Q4)",
                        is_clutch_close=(gr["clutch_routes"] > 0),
                    )
                    proj_g_yds.append(round(gp["projected_yards"], 1))
                    proj_g_tds.append(round(gp["projected_tds"], 2))
                p_games["model_proj_yards"] = proj_g_yds
                p_games["model_proj_tds"] = proj_g_tds
                p_games["game_label"] = (
                    p_games["season"].astype(str)
                    + " W"
                    + p_games["week"].astype(str)
                    + " vs "
                    + p_games["opponent_abbr"]
                )

                fig_g = go.Figure()
                fig_g.add_trace(
                    go.Bar(
                        x=p_games["game_label"],
                        y=p_games["total_yards"],
                        name="Actual Game Yards",
                        marker_color="#1f77b4",
                    )
                )
                fig_g.add_trace(
                    go.Scatter(
                        x=p_games["game_label"],
                        y=p_games["model_proj_yards"],
                        name="Model Projected Yards (given game routes)",
                        mode="lines+markers",
                        line=dict(color="#ff7f0e", width=3),
                    )
                )
                fig_g.update_layout(
                    title=f"{p_row['display_name']}: Game-by-Game Actual vs. Projected Yards",
                    xaxis_title="Game",
                    yaxis_title="Scrimmage Yards",
                    height=310,
                    margin=dict(l=20, r=20, t=40, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_g, use_container_width=True)

                display_g_cols = [
                    "season",
                    "week",
                    "opponent_abbr",
                    "routes",
                    "targets",
                    "catches",
                    "total_yards",
                    "model_proj_yards",
                    "first_downs",
                    "money_down_conv",
                    "target_epa",
                    "q4_yards",
                    "q4_tds",
                ]
                st.dataframe(p_games[display_g_cols].round(2), width="stretch", height=210)

        st.markdown("---")
        st.subheader(f"🏃‍♂️ {p_row['display_name']} — Combine Movement Tracking Trajectory (`combine_tracking`)")
        p_tr = df_tracking[df_tracking["nfl_id"] == sel_id].copy()
        if len(p_tr) == 0:
            st.warning("No Combine tracking frames available for this player.")
        else:
            tr_col1, tr_col2 = st.columns([1.2, 1])
            avail_drills = sorted(p_tr["drill_name"].unique().tolist())
            default_drill_idx = (
                avail_drills.index("GO_ROUTE_RIGHT")
                if "GO_ROUTE_RIGHT" in avail_drills
                else (avail_drills.index("GAUNTLET_DRILL") if "GAUNTLET_DRILL" in avail_drills else 0)
            )
            with tr_col1:
                sel_drill = st.selectbox("Select Combine Drill Trajectory:", avail_drills, index=default_drill_idx)
                drill_sub = p_tr[p_tr["drill_name"] == sel_drill].copy()
                drill_sub["speed_mph"] = drill_sub["s"] * 2.04545
                drill_sub["frame_time_s"] = drill_sub.groupby("attempt").cumcount() * 0.1

                fig_traj = px.scatter(
                    drill_sub,
                    x="x",
                    y="y",
                    color="speed_mph",
                    size=np.maximum(1.0, drill_sub["a"].fillna(1.0)),
                    hover_data=["attempt", "frame_time_s", "speed_mph", "a", "dir"],
                    color_continuous_scale="Turbo",
                    title=f"{p_row['display_name']} — {sel_drill} Field Trajectory (Color = Speed mph, Size = Accel yd/s²)",
                    labels={"x": "Field X (yards)", "y": "Field Y (yards)", "speed_mph": "Speed (mph)"},
                )
                fig_traj.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_traj, use_container_width=True)

            with tr_col2:
                color_metric = st.radio(
                    "Kinematic Curve Metric:", ["Speed (mph)", "Acceleration (yds/s²)", "Centripetal Cut Load (v·ω)"], horizontal=True
                )
                y_col = (
                    "speed_mph"
                    if color_metric == "Speed (mph)"
                    else ("a" if color_metric == "Acceleration (yds/s²)" else "centripetal_a")
                )
                fig_kin = px.line(
                    drill_sub,
                    x="frame_time_s",
                    y=y_col,
                    color=drill_sub["attempt"].astype(str),
                    markers=True,
                    title=f"{sel_drill} — {color_metric} vs. Time (0.1s frames)",
                    labels={"frame_time_s": "Elapsed Drill Time (s)", y_col: color_metric, "color": "Attempt"},
                )
                fig_kin.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_kin, use_container_width=True)

    # =========================================================
    # SUB-TAB 2: TEAM IMPACT & OFFENSIVE GRAVITY LAB
    # =========================================================
    with sub_tab2:
        st.subheader("🌐 Beyond Box-Score: Team Impact & Offensive Gravity Lab")
        st.markdown(
            "Raw yardage and touchdowns miss how a skill player impacts drives and teammates. "
            "This lab models **5 true team-impact dimensions** directly from `player_play` and `combine_tracking`:\n"
            "1. **Chain-Moving Efficiency (`1st Downs / 100 Routes`)** — predicted by *Play Speed Ratio (`route_to_40_speed_ratio`, `r = +0.339`)* & *Dagger-Route Accel (`r = +0.314`)*.\n"
            "2. **Money-Down Conversions (`3rd & 4th Down Conversions / Season`)** — predicted by *Combine Agility (`3-Cone + Short Shuttle`, `r = -0.509`)*.\n"
            "3. **Value Creation (`Target EPA / Season`)** — Expected Points Added on targets, weighting 3rd-down & red-zone conversions over empty yardage.\n"
            "4. **Target-Earning Power (`TPRR %`)** — Targets earned per 100 routes run.\n"
            "5. **Offensive Gravity (`Decoy Non-Target Team EPA / 100 Snaps`)** — How much the team's offense gains on plays where this player is on the field running a route **even when the ball goes to a teammate** (predicted by *Deep Over-Shoulder & Slant Route Speed, `r = +0.243`*)."
        )

        g_c1, g_c2, g_c3 = st.columns(3)
        with g_c1:
            x_imp = st.selectbox(
                "X-Axis Metric:",
                ["yards_per_game", "pred_team_impact_Or_gravity"] + spnn.TEAM_IMPACT_TARGETS,
                index=0,
                format_func=lambda c: (
                    "Rookie Yards / Game (Box Score)"
                    if c == "yards_per_game"
                    else ("Projected Team Impact Percentile" if c == "pred_team_impact_Or_gravity" else spnn.TARGET_LABELS.get(c, c))
                ),
            )
        with g_c2:
            y_imp = st.selectbox(
                "Y-Axis Team-Impact Metric:",
                spnn.TEAM_IMPACT_TARGETS + ["yacoe_per_catch", "actual_impact_percentile"],
                index=0,
                format_func=lambda c: (
                    "YAC Over Expected / Catch (YACOE)"
                    if c == "yacoe_per_catch"
                    else ("Actual Composite Team Impact Percentile (0-100)" if c == "actual_impact_percentile" else spnn.TARGET_LABELS.get(c, c))
                ),
            )
        with g_c3:
            min_rts = st.slider("Minimum Career Routes Run Filter:", 0, 200, 40, 10)

        imp_sub = view_df[view_df["career_total_routes"] >= min_rts].copy()
        x_col_actual = "team_impact_percentile" if x_imp == "pred_team_impact_Or_gravity" else x_imp

        fig_grav = px.scatter(
            imp_sub,
            x=x_col_actual,
            y=y_imp,
            color="nfl_position",
            size=np.maximum(3.0, imp_sub["team_impact_percentile"]),
            hover_name="display_name",
            hover_data=[
                "draft_year",
                "college_name",
                "yards_per_game",
                "first_downs_per_100_routes",
                "third_fourth_conv_per_season",
                "target_epa_per_season",
                "gravity_non_target_epa_100",
                "yacoe_per_catch",
                "team_impact_percentile",
            ],
            title=f"Skill Player Team Impact: {x_col_actual} vs. {spnn.TARGET_LABELS.get(y_imp, y_imp)} (Size = Projected Team Impact Score)",
        )
        fig_grav.update_layout(height=440)
        st.plotly_chart(fig_grav, use_container_width=True)

        grav_cols = [
            "display_name",
            "nfl_position",
            "draft_year",
            "career_total_routes",
            "team_impact_percentile",
            "actual_impact_percentile",
            "pred_first_downs_per_100_routes",
            "first_downs_per_100_routes",
            "pred_third_fourth_conv_per_season",
            "third_fourth_conv_per_season",
            "pred_target_epa_per_season",
            "target_epa_per_season",
            "pred_tprr_pct",
            "tprr_pct",
            "pred_gravity_non_target_epa_100",
            "gravity_non_target_epa_100",
            "yacoe_per_catch",
            "motion_rate_pct",
            "yards_per_game",
        ]
        st.dataframe(
            imp_sub[grav_cols].sort_values("team_impact_percentile", ascending=False).round(2),
            width="stretch",
            height=380,
        )

    # =========================================================
    # SUB-TAB 3: LEADERBOARDS & MODEL PREDICTIONS
    # =========================================================
    with sub_tab3:
        st.subheader(f"🏆 All 11 Targets Leaderboard (`{selected_model_family}`)")
        lb_c1, lb_c2, lb_c3 = st.columns(3)
        with lb_c1:
            target_plot_col = st.selectbox(
                "Select Target Variable to Compare:",
                spnn.TARGET_COLUMNS,
                format_func=lambda c: spnn.TARGET_LABELS[c],
            )
        with lb_c2:
            lb_pos = st.selectbox("Position Filter:", ["All", "WR", "TE", "RB", "FB"], key="lb_pos")
        with lb_c3:
            lb_year = st.selectbox("Draft Year Filter:", ["All", 2023, 2024, 2025], key="lb_yr")

        lb_df = view_df.copy()
        if lb_pos != "All":
            lb_df = lb_df[lb_df["nfl_position"] == lb_pos]
        if lb_year != "All":
            lb_df = lb_df[lb_df["draft_year"] == int(lb_year)]

        r_val, p_val = spnn.pearson_r_pval(lb_df[f"pred_{target_plot_col}"], lb_df[target_plot_col])
        rho_val, _ = spnn.spearman_rho_pval(lb_df[f"pred_{target_plot_col}"], lb_df[target_plot_col])

        fig_sc = px.scatter(
            lb_df,
            x=f"pred_{target_plot_col}",
            y=target_plot_col,
            color="nfl_position",
            size=np.maximum(2.0, lb_df["games_played"]),
            hover_name="display_name",
            hover_data=[
                "draft_year",
                "college_name",
                f"floor10_{target_plot_col}",
                f"ceil90_{target_plot_col}",
                "clutch_percentile",
                "team_impact_percentile",
            ],
            title=(
                f"Out-of-Sample Projected vs. Actual {spnn.TARGET_LABELS[target_plot_col]} "
                f"(Pearson r = {r_val:.3f}, p = {p_val:.4f} | Spearman ρ = {rho_val:.3f})"
            ),
            labels={
                f"pred_{target_plot_col}": f"Out-of-Sample Projected {spnn.TARGET_LABELS[target_plot_col]}",
                target_plot_col: f"Actual {spnn.TARGET_LABELS[target_plot_col]}",
            },
        )
        min_v = float(min(lb_df[f"pred_{target_plot_col}"].min(), lb_df[target_plot_col].min()))
        max_v = float(max(lb_df[f"pred_{target_plot_col}"].max(), lb_df[target_plot_col].max()))
        fig_sc.add_trace(
            go.Scatter(
                x=[min_v, max_v],
                y=[min_v, max_v],
                mode="lines",
                name="1:1 Diagonal",
                line=dict(color="gray", dash="dash"),
            )
        )
        fig_sc.update_layout(height=440)
        st.plotly_chart(fig_sc, use_container_width=True)

        lb_cols = [
            "display_name",
            "nfl_position",
            "draft_year",
            "college_name",
            "clutch_percentile",
            "team_impact_percentile",
            f"pred_{target_plot_col}",
            f"floor10_{target_plot_col}",
            f"ceil90_{target_plot_col}",
            target_plot_col,
            "pred_yards_per_game",
            "yards_per_game",
            "pred_tds_per_game",
            "tds_per_game",
            "pred_q4_explosive_per_game",
            "q4_explosive_per_game",
            "pred_first_downs_per_100_routes",
            "first_downs_per_100_routes",
            "pred_third_fourth_conv_per_season",
            "third_fourth_conv_per_season",
            "pred_target_epa_per_season",
            "target_epa_per_season",
            "pred_tprr_pct",
            "tprr_pct",
            "pred_gravity_non_target_epa_100",
            "gravity_non_target_epa_100",
        ]
        # Deduplicate columns while preserving order
        lb_cols = list(dict.fromkeys(lb_cols))
        lb_sorted = lb_df[lb_cols].sort_values(f"pred_{target_plot_col}", ascending=False).round(3)
        st.dataframe(lb_sorted, width="stretch", height=400)

        csv_lb = lb_sorted.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Full Multi-Model & Team Impact Projections (CSV)",
            data=csv_lb,
            file_name="skill_position_multimodel_projections.csv",
            mime="text/csv",
        )

    # =========================================================
    # SUB-TAB 4: CUSTOM PROSPECT "WHAT-IF" LAB (ALL 4 MODELS)
    # =========================================================
    with sub_tab4:
        st.subheader("🧪 Custom Prospect 'What-If' Combine & Movement Simulator (All 4 Model Families)")
        st.markdown(
            "Load any player's Combine profile as a template, then adjust **Combine Physical Measurables** or "
            "**Combine Drill Tracking Kinematics** to see live forward-pass predictions across **all 4 model families** "
            "(**Hybrid Ensemble**, **Dual-Branch NN**, **Bayesian Hierarchical + 80% Credible Interval**, and **XGBoost/GBDT**)."
        )

        template_opt = st.selectbox("Load Template Player Profile:", player_options, key="whatif_template")
        t_row = view_df[view_df["nfl_id"] == id_map[template_opt]].iloc[0]

        def _val(col, fallback):
            v = t_row.get(col, np.nan)
            return float(v) if pd.notna(v) else float(fallback)

        w_c1, w_c2, w_c3 = st.columns(3)
        with w_c1:
            st.markdown("#### 📏 Physical & Stopwatch")
            w_pos = st.selectbox("Position:", ["WR", "TE", "RB"], index=0 if t_row["nfl_position"] == "WR" else 1)
            w_height = st.slider("Height (inches):", 67.0, 80.0, _val("combine_height", 73.0), 0.25)
            w_weight = st.slider("Weight (lbs):", 165.0, 270.0, _val("combine_weight", 205.0), 1.0)
            w_hand = st.slider("Hand Size (inches — Q4 Step-Up Trait):", 8.25, 11.0, _val("hand_size", 9.5), 0.125)
            w_arm = st.slider("Arm Length (inches):", 29.0, 35.5, _val("arm_length", 32.0), 0.25)
            w_wing = st.slider("Wingspan (inches):", 70.0, 85.0, _val("wing_span", 77.0), 0.25)
            w_forty = st.slider("40-Yard Dash (s):", 4.21, 4.95, _val("forty", 4.46), 0.01)
            w_3cone = st.slider("3-Cone Drill (s — Money-Down Trait):", 6.40, 7.60, _val("three_cone", 6.95), 0.01)
            w_shuttle = st.slider("Short Shuttle (s):", 3.90, 4.65, _val("short_shuttle", 4.25), 0.01)

        with w_c2:
            st.markdown("#### 📡 Combine Movement Tracking")
            w_go_a = st.slider("Go-Route Peak Accel (yd/s² — #1 Q4 TD Trait):", 3.5, 9.5, _val("drill_go_route_max_a", 6.2), 0.1)
            w_tr40_p90_a = st.slider("40-Yd Tracking P90 Accel (yd/s²):", 2.5, 6.5, _val("tr40_p90_a", 4.4), 0.05)
            w_tr40_tang_a = st.slider("40-Yd Peak Tangential Accel (yd/s²):", 3.5, 9.5, _val("tr40_max_tang_a", 6.3), 0.1)
            w_tr40_mean_s = st.slider("40-Yd Tracking Mean Speed (yd/s):", 4.5, 8.2, _val("tr40_mean_s", 6.4), 0.05)
            w_gauntlet_a = st.slider("Gauntlet Drill Peak Accel (yd/s²):", 3.0, 9.0, _val("drill_gauntlet_max_a", 5.5), 0.1)
            w_slant_s = st.slider("Slant Route Peak Speed (yd/s — Q4 & Gravity):", 5.5, 9.8, _val("drill_slant_ro_max_s", 7.8), 0.05)
            w_route_max_s = st.slider("All-Route Peak Speed (yd/s):", 6.5, 10.8, _val("trroute_max_s", 8.8), 0.05)
            w_centrip = st.slider("Route Centripetal Cut Accel (yd/s²):", 5.0, 22.0, _val("trroute_max_centrip", 12.5), 0.25)

        with w_c3:
            st.markdown("#### 📊 Explosiveness & Combine NGS Scores")
            w_vert = st.slider("Vertical Jump (inches):", 28.0, 44.0, _val("vertical", 36.0), 0.5)
            w_broad = st.slider("Broad Jump (inches):", 108.0, 140.0, _val("broad_jump", 123.0), 1.0)
            w_ten = st.slider("10-Yard Split (s):", 1.44, 1.75, _val("ten_yd_split", 1.54), 0.01)
            w_ngs_prod = st.slider("NGS Combine College Production Score:", 50.0, 99.0, _val("ngs_college_production_score", 76.0), 1.0)
            w_ngs_ath = st.slider("NGS Combine Athleticism Score:", 50.0, 99.0, _val("ngs_athleticism_score", 78.0), 1.0)
            w_ngs_fin = st.slider("NGS Combine Final Score:", 50.0, 99.0, _val("ngs_final_score", 77.0), 1.0)

        custom_dict = t_row.to_dict()
        custom_dict.update({
            "is_wr": 1.0 if w_pos == "WR" else 0.0,
            "is_te": 1.0 if w_pos == "TE" else 0.0,
            "combine_height": w_height,
            "combine_weight": w_weight,
            "bmi": (w_weight * 703.0) / (w_height ** 2),
            "hand_size": w_hand,
            "arm_length": w_arm,
            "wing_span": w_wing,
            "catch_radius": w_arm + w_wing + w_hand,
            "ten_yd_split": w_ten,
            "forty": w_forty,
            "speed_score": (w_weight * 200.0) / (w_forty ** 4),
            "three_cone": w_3cone,
            "short_shuttle": w_shuttle,
            "agility_score": w_3cone + w_shuttle,
            "vertical": w_vert,
            "broad_jump": w_broad,
            "burst_score": w_vert + w_broad,
            "drill_go_route_max_a": w_go_a,
            "tr40_p90_a": w_tr40_p90_a,
            "tr40_max_tang_a": w_tr40_tang_a,
            "tr40_mean_s": w_tr40_mean_s,
            "drill_gauntlet_max_a": w_gauntlet_a,
            "drill_slant_ro_max_s": w_slant_s,
            "trroute_max_s": w_route_max_s,
            "trroute_max_centrip": w_centrip,
            "route_to_40_speed_ratio": w_route_max_s / max(7.0, _val("tr40_max_s", 9.8)),
            "ngs_college_production_score": w_ngs_prod,
            "ngs_athleticism_score": w_ngs_ath,
            "ngs_final_score": w_ngs_fin,
        })

        base_pred = spnn.predict_custom_prospect(active_res, active_res["feature_cols"], t_row.to_dict())
        cust_pred = spnn.predict_custom_prospect(active_res, active_res["feature_cols"], custom_dict)

        st.markdown("#### 🚀 Multi-Model Consensus Projection (Custom Prospect vs. Template Baseline)")
        whatif_rows = []
        for t_col in spnn.TARGET_COLUMNS:
            c_info = cust_pred[t_col]
            b_info = base_pred[t_col]
            whatif_rows.append({
                "Target Dimension": spnn.TARGET_LABELS[t_col],
                "Hybrid Ensemble": round(c_info["hybrid"], 2),
                "Δ vs Template": round(c_info["hybrid"] - b_info["hybrid"], 2),
                "Dual-Branch NN": round(c_info["nn"], 2),
                "Bayesian Hierarchical": round(c_info["bayesian"], 2),
                "Bayesian 10th-Pct Floor": round(c_info["floor_10"], 2),
                "Bayesian 90th-Pct Ceiling": round(c_info["ceil_90"], 2),
                "XGBoost / GBDT": round(c_info["xgb"], 2),
            })
        st.dataframe(pd.DataFrame(whatif_rows), width="stretch", height=430)

    # =========================================================
    # SUB-TAB 5: MULTI-MODEL BENCHMARK & SIGNAL PROOF
    # =========================================================
    with sub_tab5:
        st.subheader(f"🔬 Head-to-Head 5-Fold Cross-Validation Benchmark (`{selected_tier}`)")
        st.markdown(
            "Out-of-sample 5-Fold Cross-Validation accuracy comparing **Hybrid Stack Ensemble**, "
            "**Dual-Branch Multi-Task Neural Network**, **Bayesian Hierarchical Model**, and **XGBoost / GBDT** "
            "across all 11 Box-Score, 4th-Quarter Clutch, and Team-Impact targets:"
        )

        bench_df = active_res["benchmark_df"].copy()
        fig_bm = px.bar(
            bench_df,
            x="target_label",
            y="r",
            color="model_family",
            barmode="group",
            title=f"5-Fold Out-of-Sample Correlation (Pearson r) by Model Family — {selected_tier}",
            labels={"target_label": "Target Dimension", "r": "Out-of-Sample Pearson r", "model_family": "Model Architecture"},
        )
        fig_bm.update_layout(height=440, xaxis_tickangle=-25)
        st.plotly_chart(fig_bm, use_container_width=True)

        pivot_r = bench_df.pivot(index="target_label", columns="model_family", values="r").reindex(
            [spnn.TARGET_LABELS[c] for c in spnn.TARGET_COLUMNS]
        )
        st.markdown("**Out-of-Sample Pearson `r` Matrix Across Model Families:**")
        st.dataframe(pivot_r.round(3), width="stretch")

        st.markdown("---")
        st.subheader(f"📈 Top Combine & Movement Drivers in `{selected_tier}`")
        imp_target = st.selectbox(
            "Inspect Standardized Feature Sensitivity For:",
            spnn.TARGET_COLUMNS,
            format_func=lambda c: spnn.TARGET_LABELS[c],
            key="imp_target",
        )
        imp_df = active_res["importances_df"].sort_values(f"abs_effect_{imp_target}", ascending=False).head(18)
        fig_imp = px.bar(
            imp_df.iloc[::-1],
            x=f"effect_{imp_target}",
            y="feature",
            color="category",
            orientation="h",
            title=f"Top 18 Combine & Movement Features Driving {spnn.TARGET_LABELS[imp_target]} (Directional Standardized Effect)",
            labels={
                f"effect_{imp_target}": f"Directional Effect on {spnn.TARGET_LABELS[imp_target]} (+0.5 SD vs -0.5 SD)",
                "feature": "Combine / Movement Feature",
            },
        )
        fig_imp.update_layout(height=500)
        st.plotly_chart(fig_imp, use_container_width=True)


