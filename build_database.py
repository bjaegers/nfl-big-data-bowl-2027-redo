import duckdb
import time
import os

db_path = "nfl_data.duckdb"

print(f"Connecting to {db_path}...")
con = duckdb.connect(db_path)

tables_to_load = [
    ("players", "data/players.csv"),
    ("games", "data/games.csv"),
    ("player_career_successes", "data/player_career_successes.csv"),
    ("combine_results", "data/combine_results.csv"),
    ("combine_tracking", "data/combine_tracking.csv"),
    ("player_play", "data/player_play.csv"),
    ("game_tracking_2023", "data/game_tracking_2023.csv"),
    ("game_tracking_2024", "data/game_tracking_2024.csv"),
    ("game_tracking_2025", "data/game_tracking_2025.csv"),
]

for table_name, csv_path in tables_to_load:
    t0 = time.time()
    print(f"Loading {table_name} from {csv_path}...", end=" ", flush=True)
    con.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT * FROM read_csv('{csv_path}', auto_detect=true, nullstr=['NA', 'N/A', 'null', 'NULL', '']);
    """)
    count = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]
    elapsed = time.time() - t0
    print(f"Done! {count:,} rows in {elapsed:.2f}s")

# Create a unified view for all game tracking with season column
print("Creating unified view 'game_tracking'...")
con.execute("""
CREATE OR REPLACE VIEW game_tracking AS
SELECT 2023 AS season, * FROM game_tracking_2023
UNION ALL
SELECT 2024 AS season, * FROM game_tracking_2024
UNION ALL
SELECT 2025 AS season, * FROM game_tracking_2025;
""")

con.close()

db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
print(f"\nAll done! Database saved to '{db_path}' ({db_size_mb:.2f} MB)")
