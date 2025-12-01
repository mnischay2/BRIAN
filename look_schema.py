import psycopg2
import json
import textwrap
import scripts.postgres_config as cfg

def get_db_connection():
    return psycopg2.connect(dbname=cfg.TARGET_DB_NAME, **cfg.PG_CREDENTIALS)

def get_tables(conn):
    """Fetch list of all public tables."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        return [row[0] for row in cur.fetchall()]

def print_schema(conn, table_name):
    """Prints column details for a specific table."""
    print(f"\n📘 Schema for table: {table_name}")
    print("-" * 60)
    print(f"{'Column Name':<25} | {'Type':<15} | {'Nullable'}")
    print("-" * 60)
    
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns 
            WHERE table_name = '{table_name}'
            ORDER BY ordinal_position;
        """)
        for col in cur.fetchall():
            c_name, c_type, c_null = col
            print(f"{c_name:<25} | {c_type:<15} | {c_null}")
    print("-" * 60)

def truncate_value(val, limit=50):
    """Helper to truncate long text or hide vectors for display."""
    if val is None:
        return "NULL"
    
    # Handle Pgvector objects or lists (Embeddings)
    str_val = str(val)
    if "Vector" in str(type(val).__name__) or (str_val.startswith("[") and len(str_val) > 100):
        return "[VECTOR DATA HIDDEN]"
    
    # Handle chunks of text
    if len(str_val) > limit:
        return str_val[:limit] + "..."
    return str_val

def print_table_data(conn, table_name, limit=5):
    """Fetches and displays actual rows from the table."""
    print(f"\n📄 Contents of '{table_name}' (First {limit} rows)")
    
    try:
        with conn.cursor() as cur:
            # 1. Get Column Names
            cur.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            col_names = [desc[0] for desc in cur.description]
            rows = cur.fetchall()

            if not rows:
                print("   [Table is empty]")
                return

            # 2. Calculate dynamic column widths (max 30 chars)
            col_widths = [len(c) for c in col_names]
            formatted_rows = []
            
            for row in rows:
                new_row = []
                for i, val in enumerate(row):
                    display_val = truncate_value(val, limit=40)
                    new_row.append(display_val)
                    # Update width if data is longer than header
                    col_widths[i] = max(col_widths[i], len(display_val))
                formatted_rows.append(new_row)

            # Cap widths to prevent messy wrapping
            col_widths = [min(w, 40) for w in col_widths]

            # 3. Print Header
            header = " | ".join(f"{name:<{w}}" for name, w in zip(col_names, col_widths))
            print("-" * len(header))
            print(header)
            print("-" * len(header))

            # 4. Print Rows
            for row in formatted_rows:
                print(" | ".join(f"{val:<{w}}" for val, w in zip(row, col_widths)))
            print("-" * len(header))
            print(f"   (Showing {len(rows)} rows. Text truncated for readability.)")

    except Exception as e:
        print(f"❌ Error fetching data: {e}")

def main():
    try:
        conn = get_db_connection()
        tables = get_tables(conn)
        
        if not tables:
            print("❌ No tables found in database 'BRIAN'.")
            return

        while True:
            print("\n" + "="*40)
            print(f" Database: {cfg.TARGET_DB_NAME} Explorer")
            print("="*40)
            print(" 0. Exit")
            for idx, tbl in enumerate(tables):
                print(f" {idx+1}. {tbl}")
            
            choice = input("\nSelect a table to view (Number): ").strip()
            
            if choice == '0':
                print("Bye!")
                break
                
            if choice.isdigit() and 1 <= int(choice) <= len(tables):
                selected_table = tables[int(choice) - 1]
                
                action = input(f"View (S)chema or (D)ata for '{selected_table}'? [s/d]: ").lower()
                
                if action == 's':
                    print_schema(conn, selected_table)
                elif action == 'd':
                    print_table_data(conn, selected_table, limit=10)
                else:
                    print("Invalid option.")
                
                input("\nPress Enter to continue...")
            else:
                print("Invalid selection.")

    except Exception as e:
        print(f"❌ Fatal Error: {e}")
    finally:
        if 'conn' in locals() and conn:
            conn.close()

if __name__ == "__main__":
    main()