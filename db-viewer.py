import sqlite3
import pandas as pd
import os

# --- Configuration ---
# This MUST match the DB_FILE name used in the Streamlit application (SpamDetection.py)
DB_FILE = 'analysis_history.db'
TABLE_NAME = 'analysis_results'

def view_database_contents():
    """
    Connects to the SQLite database and prints the contents of the 
    'analysis_results' table using a Pandas DataFrame for clear formatting.
    The columns are explicitly ordered for better readability, 
    with 'status' displayed prominently.
    """
    
    if not os.path.exists(DB_FILE):
        print(f"Error: Database file '{DB_FILE}' not found.")
        print("Please run the Streamlit application ('SpamDetection.py') at least once to create the database file.")
        return

    try:
        # 1. Connect to the SQLite database file
        conn = sqlite3.connect(DB_FILE)
        
        # 2. Explicitly select and order columns, placing 'status' early for visibility.
        query = f"""
        SELECT 
            id, 
            timestamp, 
            status, 
            sender, 
            subject, 
            prediction, 
            confidence, 
            full_email 
        FROM {TABLE_NAME} 
        ORDER BY id DESC
        """
        
        # 3. Read the results directly into a Pandas DataFrame
        df = pd.read_sql_query(query, conn)
        
        # 4. Close the connection
        conn.close()
        
        print(f"--- SecureScan Analysis History ({DB_FILE}) ---")
        print(f"Total Records in {TABLE_NAME}: {len(df)}")
        print("-" * 50)
        
        # 5. Print the DataFrame with formatting
        if not df.empty:
            # Configure pandas to display more of the columns/rows
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', 1000)
            
            # Truncate the 'full_email' column for display purposes 
            # so the output is readable in the terminal/console.
            df['full_email'] = df['full_email'].fillna('').str[:75] + '...'
            
            # Rename columns for cleaner display (e.g., 'status' becomes 'Status')
            df.columns = [col.replace('_', ' ').title() for col in df.columns]
            
            print(df)
        else:
            print("The 'analysis_results' table is currently empty. Run some scans in the app!")

    except Exception as e:
        print(f"An error occurred while reading the database:")
        print(f"Error: {e}")

if __name__ == "__main__":
    view_database_contents()
