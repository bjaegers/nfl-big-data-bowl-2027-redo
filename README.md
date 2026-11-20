# 🏈 NFL Big Data Bowl - Database & Query Engine

All CSV datasets from the `data/` folder have been indexed and loaded into a high-performance, columnar **DuckDB** database: **`nfl_data.duckdb`**.

---

## 🚀 Why DuckDB?

The dataset contains **23.4+ million records** (~2.36 GB of raw CSVs). 
- **Lightning Fast**: Sub-second (often 10–50 ms) query execution on 22+ million rows.
- **Columnar & Compressed**: 2.36 GB of CSV data compressed down to **533 MB** (4.4x compression) with zero loss of fidelity.
- **Zero Configuration**: Single self-contained database file `nfl_data.duckdb` requiring no background server or daemon.
- **Standard SQL**: Full support for standard ANSI SQL, window functions, CTEs, joins, and aggregations.

---

## 📊 Database Catalog

| Table / View | Type | Rows | Description |
| :--- | :--- | :--- | :--- |
| **`game_tracking`** | `VIEW` | **22,654,290** | **Unified tracking view** combining 2023, 2024, and 2025 seasons with a `season` column. |
| **`game_tracking_2025`** | `BASE TABLE` | 11,160,903 | Player tracking frames for the 2025 season (x, y, speed, acceleration, orientation, event). |
| **`game_tracking_2024`** | `BASE TABLE` | 7,827,944 | Player tracking frames for the 2024 season. |
| **`game_tracking_2023`** | `BASE TABLE` | 3,665,443 | Player tracking frames for the 2023 season. |
| **`player_play`** | `BASE TABLE` | 314,197 | Play-level player metrics (formations, coverage, EPA, pass rush, yards gained, etc.). |
| **`combine_tracking`** | `BASE TABLE` | 463,189 | Frame-by-frame player tracking during NFL Scouting Combine drills. |
| **`combine_results`** | `BASE TABLE` | 510 | Combine physical measurements & workout drill scores (40-yd dash, vertical, etc.). |
| **`players`** | `BASE TABLE` | 510 | Player roster details (name, position, college, draft round, pick, birth date). |
| **`player_career_successes`** | `BASE TABLE` | 510 | Career snaps, games started, All-Pro honors, and Pro Bowl ballots. |
| **`games`** | `BASE TABLE` | 1,002 | Game metadata (season, week, home team, visitor team). |

---

## 🛠️ How to Query the Data

### 1. Interactive Command Line Shell (CLI)
Run the interactive SQL prompt:
```bash
python query.py
```
Inside the interactive shell:
- Type any SQL query ending with `;` and press **Enter**.
- `.tables` — List all tables, views, and row counts.
- `.schema <table_name>` — View column names and types for a table.
- `.examples` — Print ready-to-run sample queries.
- `.exit` — Quit the shell.

**One-line queries directly from terminal:**
```bash
python query.py "SELECT display_name, nfl_position, college_name FROM players LIMIT 5;"
```

**Export results directly to CSV:**
```bash
python query.py --csv results.csv "SELECT * FROM combine_results WHERE forty < 4.45;"
```

---

### 2. Streamlit Web App (Interactive Dashboard)
Launch the graphical web interface with SQL console, schema explorer, and visual charts:
```bash
streamlit run app.py
```
*Features:*
- **SQL Query Lab**: Run any SQL query with instant tabular preview and CSV export.
- **Schema Explorer**: Inspect column types and view sample records for any table.
- **Visual Analytics**: Interactive charts (e.g. Combine 40-yd speed vs in-game top speed).

---

### 3. In Python Scripts or Jupyter Notebooks
Query the database directly using DuckDB's Python API:

```python
import duckdb

# Connect to the database (read-only mode allows multiple concurrent readers)
con = duckdb.connect("nfl_data.duckdb", read_only=True)

# 1. Fetch as Pandas DataFrame
df = con.execute("""
    SELECT p.display_name, p.nfl_position, MAX(t.s) * 2.04545 AS max_mph
    FROM game_tracking_2024 t
    JOIN players p ON t.nfl_id = p.nfl_id
    GROUP BY p.display_name, p.nfl_position
    ORDER BY max_mph DESC
    LIMIT 10;
""").df()

print(df)

# 2. Fetch as Arrow Table or Python Tuples
tuples = con.execute("SELECT count(*) FROM game_tracking").fetchall()
print("Total tracking rows:", tuples[0][0])
```

---

### 4. External GUI Clients (DBeaver, VS Code, JetBrains DataGrip)
You can connect to `nfl_data.duckdb` using any database viewer with DuckDB support:
- **DBeaver**: Create New Connection -> Select **DuckDB** -> Point to `c:\nfl_big_data_bowl_2027_redo\nfl_data.duckdb`.
- **VS Code**: Install the **DuckDB** extension (by MotherDuck) or **SQLTools**.

---

## 💡 Example Queries

### Top 5 Fastest Recorded In-Game Speeds (2024 Season)
```sql
SELECT 
    p.display_name, 
    p.nfl_position, 
    MAX(t.s) AS max_speed_mps, 
    ROUND(MAX(t.s) * 2.04545, 2) AS max_speed_mph
FROM game_tracking_2024 t
JOIN players p ON t.nfl_id = p.nfl_id
GROUP BY p.display_name, p.nfl_position
ORDER BY max_speed_mps DESC
LIMIT 5;
```

### Combine 40-Yard Dash vs In-Game Top Speed
```sql
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
ORDER BY top_game_speed_mph DESC
LIMIT 10;
```

### Play Count & Average Yards Gained by Offensive Formation
```sql
SELECT 
    offense_formation, 
    count(*) AS plays, 
    ROUND(AVG(yards_gained), 2) AS avg_yards_gained,
    ROUND(AVG(expected_points_added), 3) AS avg_epa
FROM player_play
WHERE offense_formation IS NOT NULL
GROUP BY offense_formation
ORDER BY plays DESC;
```

---

## 🔄 Rebuilding the Database
If new CSV files are placed in `data/`, simply re-run the build script:
```bash
python build_database.py
```
This script reads all CSVs, handles `NA`/null values, creates the tables and views, and rebuilds `nfl_data.duckdb`.
