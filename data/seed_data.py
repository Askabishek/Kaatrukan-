"""
Synthetic data generator for CleanAir & Clear Streets.
Generates realistic pollution reports for Indian cities.
"""

import sqlite3
import random
import os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(__file__), "pollution_db.sqlite")

# Realistic locations across Indian cities
LOCATIONS = [
    # Hyderabad
    {"name": "Jubilee Hills Check Post", "lat": 17.4319, "lon": 78.4071, "city": "Hyderabad"},
    {"name": "HITEC City Junction", "lat": 17.4435, "lon": 78.3772, "city": "Hyderabad"},
    {"name": "Charminar Area", "lat": 17.3616, "lon": 78.4747, "city": "Hyderabad"},
    {"name": "Kukatpally Housing Board", "lat": 17.4947, "lon": 78.3996, "city": "Hyderabad"},
    {"name": "Secunderabad Railway Station", "lat": 17.4344, "lon": 78.5013, "city": "Hyderabad"},
    {"name": "Miyapur X Roads", "lat": 17.4969, "lon": 78.3548, "city": "Hyderabad"},
    {"name": "LB Nagar Ring Road", "lat": 17.3457, "lon": 78.5522, "city": "Hyderabad"},
    {"name": "Begumpet Flyover", "lat": 17.4437, "lon": 78.4672, "city": "Hyderabad"},
    # Delhi
    {"name": "Anand Vihar ISBT", "lat": 28.6469, "lon": 77.3164, "city": "Delhi"},
    {"name": "ITO Junction", "lat": 28.6289, "lon": 77.2405, "city": "Delhi"},
    {"name": "Chandni Chowk Market", "lat": 28.6506, "lon": 77.2301, "city": "Delhi"},
    {"name": "Dwarka Sector 21", "lat": 28.5523, "lon": 77.0586, "city": "Delhi"},
    {"name": "Rohini Sector 3", "lat": 28.7155, "lon": 77.1143, "city": "Delhi"},
    {"name": "Okhla Industrial Area", "lat": 28.5308, "lon": 77.2716, "city": "Delhi"},
    {"name": "Mundka Industrial Zone", "lat": 28.6839, "lon": 77.0295, "city": "Delhi"},
    {"name": "Sarai Kale Khan", "lat": 28.5894, "lon": 77.2567, "city": "Delhi"},
    # Mumbai
    {"name": "Dharavi Junction", "lat": 19.0424, "lon": 72.8548, "city": "Mumbai"},
    {"name": "Bandra-Worli Sea Link Entry", "lat": 19.0380, "lon": 72.8198, "city": "Mumbai"},
    {"name": "Chembur Industrial Belt", "lat": 19.0522, "lon": 72.8994, "city": "Mumbai"},
    {"name": "Andheri East MIDC", "lat": 19.1136, "lon": 72.8697, "city": "Mumbai"},
    {"name": "Sion-Panvel Highway", "lat": 19.0437, "lon": 72.8627, "city": "Mumbai"},
    {"name": "Mahim Causeway", "lat": 19.0404, "lon": 72.8399, "city": "Mumbai"},
]

POLLUTION_TYPES = [
    "Garbage Burning", "Industrial Smoke", "Vehicle Exhaust", 
    "Construction Dust", "Factory Emissions", "Open Waste Dump",
    "Chemical Fumes", "Crop Residue Burning", "Road Dust",
    "Smoke from Street Food Stalls"
]

DESCRIPTIONS = [
    "Heavy black smoke rising from garbage dump, affecting nearby residential area",
    "Thick dust cloud from construction site without water sprinklers",
    "Industrial chimney releasing dark fumes continuously since morning",
    "Burning of plastic waste in open ground near school",
    "Dense vehicle exhaust at traffic signal during peak hours",
    "Chemical smell from factory making residents sick",
    "Road dust due to unpaved stretch causing visibility issues",
    "Open burning of municipal waste by roadside",
    "Smoke from multiple tandoor/food stalls in market area",
    "Dust from demolition activity without proper barriers",
    "Crop stubble burning visible from highway, smoke drifting into city",
    "Illegal waste incineration behind commercial complex",
    "Cement factory dust covering entire neighbourhood",
    "Diesel generator smoke in residential colony during power cut",
    "Paint factory fumes causing breathing difficulty in nearby houses",
]

SEVERITY_LEVELS = ["Low", "Moderate", "High", "Severe", "Hazardous"]
STATUS_OPTIONS = ["Reported", "Under Review", "Action Taken", "Resolved", "Escalated"]


def create_database():
    """Create the SQLite database and tables."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pollution_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            city TEXT NOT NULL,
            pollution_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            description TEXT NOT NULL,
            aqi_reading INTEGER,
            reported_at TIMESTAMP NOT NULL,
            status TEXT DEFAULT 'Reported',
            reporter_name TEXT,
            contact_info TEXT,
            image_analysis TEXT,
            recommended_action TEXT
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            city TEXT NOT NULL,
            pm25 REAL,
            pm10 REAL,
            no2 REAL,
            so2 REAL,
            co REAL,
            aqi INTEGER,
            recorded_at TIMESTAMP NOT NULL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location_name TEXT NOT NULL,
            city TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            recommended_action TEXT,
            created_at TIMESTAMP NOT NULL,
            is_active INTEGER DEFAULT 1
        )
    """)
    
    conn.commit()
    conn.close()


def seed_pollution_reports(num_reports=150):
    """Generate synthetic pollution reports."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now()
    reports = []
    
    for i in range(num_reports):
        loc = random.choice(LOCATIONS)
        # Add slight randomness to coordinates
        lat = loc["lat"] + random.uniform(-0.005, 0.005)
        lon = loc["lon"] + random.uniform(-0.005, 0.005)
        
        pollution_type = random.choice(POLLUTION_TYPES)
        severity = random.choices(
            SEVERITY_LEVELS, 
            weights=[15, 30, 30, 15, 10], 
            k=1
        )[0]
        
        description = random.choice(DESCRIPTIONS)
        
        # AQI based on severity
        aqi_ranges = {
            "Low": (50, 100),
            "Moderate": (101, 200),
            "High": (201, 300),
            "Severe": (301, 400),
            "Hazardous": (401, 500)
        }
        aqi_range = aqi_ranges[severity]
        aqi = random.randint(aqi_range[0], aqi_range[1])
        
        # Random time in last 30 days
        hours_ago = random.randint(1, 720)
        reported_at = now - timedelta(hours=hours_ago)
        
        status = random.choices(
            STATUS_OPTIONS,
            weights=[30, 25, 20, 15, 10],
            k=1
        )[0]
        
        actions = [
            "Deploy water-mist cannon to suppress dust",
            "Send cleanup crew immediately",
            "Issue notice to factory/construction site",
            "Deploy air purification unit",
            "Coordinate with traffic police for diversion",
            "Alert fire department for waste burning",
            "Schedule road sweeping and water sprinkling",
            "Notify pollution control board for inspection"
        ]
        
        reports.append((
            loc["name"], lat, lon, loc["city"],
            pollution_type, severity, description, aqi,
            reported_at.strftime("%Y-%m-%d %H:%M:%S"),
            status, f"Citizen_{random.randint(1000, 9999)}",
            f"+91-{random.randint(7000000000, 9999999999)}",
            None, random.choice(actions)
        ))
    
    cursor.executemany("""
        INSERT INTO pollution_reports 
        (location_name, latitude, longitude, city, pollution_type, severity, 
         description, aqi_reading, reported_at, status, reporter_name, 
         contact_info, image_analysis, recommended_action)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, reports)
    
    conn.commit()
    conn.close()


def seed_sensor_readings(num_readings=500):
    """Generate synthetic sensor readings."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now()
    readings = []
    
    for i in range(num_readings):
        loc = random.choice(LOCATIONS)
        
        # Realistic sensor values
        pm25 = random.uniform(20, 350)
        pm10 = pm25 * random.uniform(1.2, 2.5)
        no2 = random.uniform(10, 120)
        so2 = random.uniform(5, 80)
        co = random.uniform(0.5, 5.0)
        
        # Calculate AQI (simplified)
        aqi = int(max(pm25 * 1.2, pm10 * 0.8, no2 * 2, so2 * 1.5))
        aqi = min(aqi, 500)
        
        # Random time in last 7 days (more recent data)
        hours_ago = random.randint(0, 168)
        recorded_at = now - timedelta(hours=hours_ago)
        
        readings.append((
            loc["name"], loc["lat"], loc["lon"], loc["city"],
            round(pm25, 1), round(pm10, 1), round(no2, 1),
            round(so2, 1), round(co, 2), aqi,
            recorded_at.strftime("%Y-%m-%d %H:%M:%S")
        ))
    
    cursor.executemany("""
        INSERT INTO sensor_readings 
        (location_name, latitude, longitude, city, pm25, pm10, no2, so2, co, aqi, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, readings)
    
    conn.commit()
    conn.close()


def seed_alerts(num_alerts=20):
    """Generate synthetic alerts."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    now = datetime.now()
    alerts_data = []
    
    alert_types = ["Spike Predicted", "Hotspot Detected", "Threshold Exceeded", "Recurring Issue"]
    alert_messages = [
        "AQI predicted to exceed 300 in next 6 hours based on wind patterns and current emissions",
        "Multiple reports from same area indicate persistent pollution source",
        "PM2.5 levels crossed safe threshold — immediate action recommended",
        "Recurring garbage burning detected — enforcement team needed",
        "Industrial emissions spike detected — notify pollution control board",
        "Construction dust levels critical — water sprinkling required",
        "Traffic congestion causing localized air quality deterioration",
        "Predicted weather inversion may trap pollutants — preventive measures needed"
    ]
    
    for i in range(num_alerts):
        loc = random.choice(LOCATIONS)
        alert_type = random.choice(alert_types)
        severity = random.choice(["High", "Severe", "Hazardous"])
        message = random.choice(alert_messages)
        
        actions = [
            "Deploy water-mist cannons at identified locations",
            "Send cleanup crew within 30 minutes",
            "Issue immediate shutdown notice to factory",
            "Activate emergency air quality protocol",
            "Coordinate multi-department rapid response"
        ]
        
        hours_ago = random.randint(0, 48)
        created_at = now - timedelta(hours=hours_ago)
        
        alerts_data.append((
            loc["name"], loc["city"], alert_type, severity,
            message, random.choice(actions),
            created_at.strftime("%Y-%m-%d %H:%M:%S"),
            1 if hours_ago < 24 else 0
        ))
    
    cursor.executemany("""
        INSERT INTO alerts 
        (location_name, city, alert_type, severity, message, recommended_action, created_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, alerts_data)
    
    conn.commit()
    conn.close()


def initialize_database():
    """Initialize database with seed data if empty."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM pollution_reports")
    count = cursor.fetchone()[0]
    conn.close()
    
    if count == 0:
        print("🌱 Seeding database with synthetic data...")
        seed_pollution_reports()
        seed_sensor_readings()
        seed_alerts()
        print("✅ Database seeded successfully!")
    else:
        print(f"📊 Database already has {count} reports.")


if __name__ == "__main__":
    create_database()
    initialize_database()
