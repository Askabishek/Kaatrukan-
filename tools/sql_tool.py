"""
SQL Tool — Natural language to SQL queries for pollution database.
"""

import os
import sqlite3
import pandas as pd
from groq import Groq

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pollution_db.sqlite")


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def get_db_connection():
    """Get SQLite database connection."""
    return sqlite3.connect(DB_PATH)


def get_schema_info() -> str:
    """Get database schema for context."""
    return """
Database Schema:

TABLE: pollution_reports
- id (INTEGER, PRIMARY KEY)
- location_name (TEXT) — name of the location
- latitude (REAL)
- longitude (REAL)
- city (TEXT) — city name (Hyderabad, Delhi, Mumbai)
- pollution_type (TEXT) — type of pollution (Garbage Burning, Industrial Smoke, Vehicle Exhaust, Construction Dust, Factory Emissions, Open Waste Dump, Chemical Fumes, Crop Residue Burning, Road Dust, Smoke from Street Food Stalls)
- severity (TEXT) — Low, Moderate, High, Severe, Hazardous
- description (TEXT)
- aqi_reading (INTEGER) — Air Quality Index value
- reported_at (TIMESTAMP)
- status (TEXT) — Reported, Under Review, Action Taken, Resolved, Escalated
- reporter_name (TEXT)
- contact_info (TEXT)
- image_analysis (TEXT)
- recommended_action (TEXT)

TABLE: sensor_readings
- id (INTEGER, PRIMARY KEY)
- location_name (TEXT)
- latitude (REAL)
- longitude (REAL)
- city (TEXT)
- pm25 (REAL) — PM2.5 reading
- pm10 (REAL) — PM10 reading
- no2 (REAL) — NO2 reading
- so2 (REAL) — SO2 reading
- co (REAL) — CO reading
- aqi (INTEGER) — calculated AQI
- recorded_at (TIMESTAMP)

TABLE: alerts
- id (INTEGER, PRIMARY KEY)
- location_name (TEXT)
- city (TEXT)
- alert_type (TEXT) — Spike Predicted, Hotspot Detected, Threshold Exceeded, Recurring Issue
- severity (TEXT)
- message (TEXT)
- recommended_action (TEXT)
- created_at (TIMESTAMP)
- is_active (INTEGER) — 1 for active, 0 for resolved
"""


def nl_to_sql(question: str) -> str:
    """Convert natural language question to SQL query using Groq LLM."""
    client = get_client()
    
    schema = get_schema_info()
    
    prompt = f"""You are a SQL expert. Convert the following natural language question into a SQLite SQL query.

{schema}

Rules:
- Return ONLY the SQL query, nothing else
- Use proper SQLite syntax
- Limit results to 50 rows max
- Use appropriate JOINs if needed
- For date filtering, use datetime functions

Question: {question}

SQL Query:"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.1
        )
        
        sql = response.choices[0].message.content.strip()
        # Clean up any markdown formatting
        sql = sql.replace("```sql", "").replace("```", "").strip()
        return sql
        
    except Exception as e:
        return f"Error: {str(e)}"


def execute_query(sql: str) -> pd.DataFrame:
    """Execute SQL query and return results as DataFrame."""
    try:
        conn = get_db_connection()
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})


def query_database(question: str) -> tuple:
    """
    Full pipeline: NL question → SQL → Results.
    Returns (sql_query, results_dataframe)
    """
    sql = nl_to_sql(question)
    if sql.startswith("Error:"):
        return sql, pd.DataFrame()
    
    results = execute_query(sql)
    return sql, results


def get_pollution_stats() -> dict:
    """Get overall pollution statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    stats = {}
    
    cursor.execute("SELECT COUNT(*) FROM pollution_reports")
    stats["total_reports"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM pollution_reports WHERE status = 'Reported'")
    stats["pending_reports"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM pollution_reports WHERE status = 'Resolved'")
    stats["resolved_reports"] = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(aqi_reading) FROM pollution_reports")
    stats["avg_aqi"] = round(cursor.fetchone()[0] or 0, 1)
    
    cursor.execute("SELECT COUNT(*) FROM alerts WHERE is_active = 1")
    stats["active_alerts"] = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT city, COUNT(*) as count 
        FROM pollution_reports 
        GROUP BY city 
        ORDER BY count DESC
    """)
    stats["reports_by_city"] = cursor.fetchall()
    
    cursor.execute("""
        SELECT pollution_type, COUNT(*) as count 
        FROM pollution_reports 
        GROUP BY pollution_type 
        ORDER BY count DESC 
        LIMIT 5
    """)
    stats["top_pollution_types"] = cursor.fetchall()
    
    cursor.execute("""
        SELECT severity, COUNT(*) as count 
        FROM pollution_reports 
        GROUP BY severity
    """)
    stats["severity_distribution"] = cursor.fetchall()
    
    conn.close()
    return stats


def get_hotspot_data() -> pd.DataFrame:
    """Get data for hotspot map visualization."""
    conn = get_db_connection()
    df = pd.read_sql_query("""
        SELECT location_name, latitude, longitude, city, 
               pollution_type, severity, aqi_reading, 
               reported_at, status, description, recommended_action
        FROM pollution_reports 
        ORDER BY reported_at DESC
        LIMIT 200
    """, conn)
    conn.close()
    return df


def get_recent_reports(limit: int = 20) -> pd.DataFrame:
    """Get most recent pollution reports."""
    conn = get_db_connection()
    df = pd.read_sql_query(f"""
        SELECT id, location_name, city, pollution_type, severity, 
               aqi_reading, reported_at, status, recommended_action
        FROM pollution_reports 
        ORDER BY reported_at DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    return df


def insert_report(location_name: str, lat: float, lon: float, city: str,
                  pollution_type: str, severity: str, description: str,
                  aqi: int, status: str = "Reported", 
                  image_analysis: str = None, recommended_action: str = None) -> int:
    """Insert a new pollution report into the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    from datetime import datetime
    
    cursor.execute("""
        INSERT INTO pollution_reports 
        (location_name, latitude, longitude, city, pollution_type, severity,
         description, aqi_reading, reported_at, status, image_analysis, recommended_action)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (location_name, lat, lon, city, pollution_type, severity,
          description, aqi, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
          status, image_analysis, recommended_action))
    
    report_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return report_id
