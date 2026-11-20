import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import os
import time

st.set_page_config(
    page_title="NFL Big Data Bowl - Database Explorer",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_data.duckdb")

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
    ["⚡ SQL Query Lab", "📋 Database Schema & Preview", "📊 Instant Visualizations", "💡 Quick Query Guide"]
)

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
