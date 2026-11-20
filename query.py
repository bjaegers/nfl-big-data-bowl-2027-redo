"""
NFL Big Data Bowl - Interactive SQL Query CLI
Allows fast SQL queries against nfl_data.duckdb directly from the command line.
"""

import sys
import os
import time
import argparse
import duckdb

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nfl_data.duckdb")

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

TABLE_FORMAT = "psql" if HAS_TABULATE else None


def get_connection(read_only=True):
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at '{DB_PATH}'. Run 'python build_database.py' first.")
        sys.exit(1)
    return duckdb.connect(DB_PATH, read_only=read_only)


def format_results(rows, col_names, total_rows, max_rows=50):
    if total_rows == 0:
        return "(0 rows returned)"
    display_rows = rows[:max_rows]
    if HAS_TABULATE:
        rendered = tabulate(display_rows, headers=col_names, tablefmt=TABLE_FORMAT)
    else:
        # Simple text formatting
        widths = [len(str(c)) for c in col_names]
        for row in display_rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))
        header_line = " | ".join(str(col).ljust(widths[i]) for i, col in enumerate(col_names))
        sep_line = "-+-".join("-" * widths[i] for i in range(len(col_names)))
        row_lines = [
            " | ".join(str(val if val is not None else "").ljust(widths[i]) for i, val in enumerate(row))
            for row in display_rows
        ]
        rendered = "\n".join([header_line, sep_line] + row_lines)
    
    if total_rows > max_rows:
        rendered += f"\n... showing first {max_rows} of {total_rows:,} rows. (Use LIMIT in query to control size)"
    else:
        rendered += f"\n({total_rows:,} rows)"
    return rendered


def show_tables(con):
    tables = con.execute("""
        SELECT table_name, table_type 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_type, table_name;
    """).fetchall()
    
    table_stats = []
    for name, ttype in tables:
        count = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        table_stats.append((name, ttype, f"{count:,}"))
    
    if HAS_TABULATE:
        print(tabulate(table_stats, headers=["Table / View", "Type", "Row Count"], tablefmt=TABLE_FORMAT))
    else:
        for name, ttype, count in table_stats:
            print(f"  {name:25} [{ttype:4}] {count:>12} rows")


def show_schema(con, table_name):
    try:
        cols = con.execute(f"DESCRIBE {table_name}").fetchall()
        col_data = [(col[0], col[1], col[2], col[3]) for col in cols]
        if HAS_TABULATE:
            print(f"\nSchema for '{table_name}':")
            print(tabulate(col_data, headers=["Column", "Type", "Null", "Key"], tablefmt=TABLE_FORMAT))
        else:
            print(f"\nSchema for '{table_name}':")
            for c in col_data:
                print(f"  {c[0]:35} {c[1]}")
    except Exception as e:
        print(f"Error describing '{table_name}': {e}")


def show_examples():
    examples = [
        ("1. View players with combine vertical & 40-yard times",
         "SELECT p.display_name, p.nfl_position, c.forty, c.vertical, c.broad_jump\nFROM players p\nJOIN combine_results c ON p.nfl_id = c.nfl_id\nWHERE c.forty IS NOT NULL\nORDER BY c.forty ASC\nLIMIT 10;"),
        
        ("2. Top 5 fastest ball carrier speeds recorded in 2024",
         "SELECT p.display_name, p.nfl_position, MAX(t.s) AS max_mps, ROUND(MAX(t.s) * 2.04545, 2) AS max_mph\nFROM game_tracking_2024 t\nJOIN players p ON t.nfl_id = p.nfl_id\nGROUP BY p.display_name, p.nfl_position\nORDER BY max_mps DESC\nLIMIT 5;"),
        
        ("3. Event frequency across all seasons (unified game_tracking)",
         "SELECT event, count(*) as event_count\nFROM game_tracking\nWHERE event IS NOT NULL\nGROUP BY event\nORDER BY event_count DESC\nLIMIT 10;"),
         
        ("4. Play results by offensive formation",
         "SELECT offense_formation, count(*) as plays, ROUND(AVG(yards_gained), 2) as avg_yards_gained\nFROM player_play\nWHERE offense_formation IS NOT NULL\nGROUP BY offense_formation\nORDER BY plays DESC;")
    ]
    print("\n" + "="*70)
    print(" SAMPLE QUERIES ")
    print("="*70)
    for title, query in examples:
        print(f"\n-- {title}:")
        print(query)
    print("="*70 + "\n")


def execute_sql(con, sql, max_rows=50, export_csv=None):
    t0 = time.time()
    try:
        rel = con.execute(sql)
        elapsed = time.time() - t0
        
        col_names = [desc[0] for desc in rel.description] if rel.description else []
        rows = rel.fetchall()
        
        if export_csv:
            import csv
            with open(export_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if col_names:
                    writer.writerow(col_names)
                writer.writerows(rows)
            print(f"Successfully exported {len(rows):,} rows to '{export_csv}' in {elapsed:.3f}s")
            return
            
        print(format_results(rows, col_names, total_rows=len(rows), max_rows=max_rows))
        print(f"Query executed in {elapsed*1000:.1f} ms\n")
    except Exception as e:
        print(f"SQL Error: {e}\n")


def repl(con):
    print("="*70)
    print(" NFL Big Data Bowl - Interactive SQL Shell (DuckDB)")
    print(" Database: nfl_data.duckdb (~23.4 million records)")
    print(" Commands: .tables | .schema <table> | .examples | .exit | .help")
    print("="*70)
    
    show_tables(con)
    print("\nType your SQL query and press Enter. End with ';' (or multi-line queries until ';').\n")
    
    buffer = []
    while True:
        try:
            prompt = "sql> " if not buffer else " ... "
            line = input(prompt).strip()
            
            if not buffer and line.startswith("."):
                cmd = line.split()
                c = cmd[0].lower()
                if c in (".exit", ".quit", ".q"):
                    print("Goodbye!")
                    break
                elif c == ".tables":
                    show_tables(con)
                elif c == ".schema":
                    if len(cmd) > 1:
                        show_schema(con, cmd[1])
                    else:
                        print("Usage: .schema <table_name>")
                elif c == ".examples":
                    show_examples()
                elif c == ".help":
                    print("Commands:")
                    print("  .tables          List all tables, views, and row counts")
                    print("  .schema <name>   Show columns and types for a table")
                    print("  .examples        Display sample queries to copy/paste")
                    print("  .exit            Exit the shell")
                else:
                    print(f"Unknown command '{c}'. Type .help for assistance.")
                continue
            
            if not line:
                continue
                
            buffer.append(line)
            if line.endswith(";"):
                query = " ".join(buffer)
                buffer = []
                execute_sql(con, query)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break


def main():
    parser = argparse.ArgumentParser(description="Query the NFL Big Data Bowl database.")
    parser.add_argument("query", nargs="?", help="SQL query to execute directly")
    parser.add_argument("-f", "--file", help="Path to .sql file to execute")
    parser.add_argument("--csv", help="Export query result to specified CSV filename")
    parser.add_argument("--tables", action="store_true", help="List database tables and row counts")
    parser.add_argument("--schema", help="Describe schema for specified table")
    parser.add_argument("--examples", action="store_true", help="Show example queries")
    parser.add_argument("--max-rows", type=int, default=50, help="Maximum rows to display in terminal (default: 50)")

    args = parser.parse_args()
    con = get_connection(read_only=True)

    if args.tables:
        show_tables(con)
    elif args.schema:
        show_schema(con, args.schema)
    elif args.examples:
        show_examples()
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            sql = f.read()
        execute_sql(con, sql, max_rows=args.max_rows, export_csv=args.csv)
    elif args.query:
        execute_sql(con, args.query, max_rows=args.max_rows, export_csv=args.csv)
    else:
        repl(con)


if __name__ == "__main__":
    main()
