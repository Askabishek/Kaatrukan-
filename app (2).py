"""
CleanAir & Clear Streets — Streamlit UI
Neighbourhood-level pollution hotspot detection and alert system.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from data.seed_data import create_database, initialize_database
from tools.sql_tool import (
    get_pollution_stats, get_hotspot_data, get_recent_reports,
    query_database, insert_report
)
from tools.rag_tool import index_reports, semantic_search
from pipeline import (
    process_image_report, process_voice_report, process_text_report,
    generate_prediction, chat_query
)
from tools.voice_tool import text_to_speech, speech_to_text
from tools.vision_tool import analyze_pollution_image

# --- Page Config ---
st.set_page_config(
    page_title="CleanAir & Clear Streets",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1B5E20;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #558B2F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #2E7D32;
    }
    .alert-card {
        background: linear-gradient(135deg, #FFF3E0, #FFE0B2);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #E65100;
        margin-bottom: 0.8rem;
    }
    .severity-hazardous { color: #B71C1C; font-weight: bold; }
    .severity-severe { color: #E65100; font-weight: bold; }
    .severity-high { color: #F57F17; font-weight: bold; }
    .severity-moderate { color: #FF8F00; }
    .severity-low { color: #2E7D32; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Initialize Database ---
@st.cache_resource
def init_db():
    create_database()
    initialize_database()
    count = index_reports()
    return count

init_db()


# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🌬️ CleanAir & Clear Streets")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "📸 Report Pollution", "🗺️ Pollution Map", 
         "⚠️ Predictions & Alerts", "🔍 Search Reports", "💬 Ask AI"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 🏙️ City Filter")
    city_filter = st.selectbox("Select City", ["All", "Hyderabad", "Delhi", "Mumbai"])
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.85rem;'>
        <b>Built with</b><br>
        Groq Llama 4 Scout • Whisper<br>
        Chroma • Streamlit<br><br>
        <i>Hackathon 2026</i>
    </div>
    """, unsafe_allow_html=True)


# --- Helper Functions ---
def get_severity_color(severity):
    colors = {
        "Low": "#4CAF50",
        "Moderate": "#FFC107",
        "High": "#FF9800",
        "Severe": "#FF5722",
        "Hazardous": "#B71C1C"
    }
    return colors.get(severity, "#9E9E9E")


def get_aqi_category(aqi):
    if aqi <= 50: return "Good", "#4CAF50"
    elif aqi <= 100: return "Satisfactory", "#8BC34A"
    elif aqi <= 200: return "Moderate", "#FFC107"
    elif aqi <= 300: return "Poor", "#FF9800"
    elif aqi <= 400: return "Very Poor", "#FF5722"
    else: return "Severe", "#B71C1C"


# ==================== PAGES ====================

# --- Dashboard ---
if page == "🏠 Dashboard":
    st.markdown('<p class="main-header">🌬️ CleanAir & Clear Streets</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Spotting and Fixing Local Pollution Hotspots</p>', unsafe_allow_html=True)
    
    stats = get_pollution_stats()
    
    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Reports", stats["total_reports"])
    with col2:
        st.metric("⏳ Pending", stats["pending_reports"])
    with col3:
        st.metric("✅ Resolved", stats["resolved_reports"])
    with col4:
        st.metric("💨 Avg AQI", stats["avg_aqi"])
    with col5:
        st.metric("🚨 Active Alerts", stats["active_alerts"])
    
    st.markdown("---")
    
    # Charts Row
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Reports by City")
        if stats["reports_by_city"]:
            city_df = pd.DataFrame(stats["reports_by_city"], columns=["City", "Count"])
            fig = px.bar(city_df, x="City", y="Count", color="City",
                        color_discrete_map={"Hyderabad": "#1565C0", "Delhi": "#C62828", "Mumbai": "#2E7D32"})
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right:
        st.subheader("🏭 Top Pollution Types")
        if stats["top_pollution_types"]:
            type_df = pd.DataFrame(stats["top_pollution_types"], columns=["Type", "Count"])
            fig = px.pie(type_df, values="Count", names="Type", hole=0.4)
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Severity Distribution
    st.subheader("⚡ Severity Distribution")
    if stats["severity_distribution"]:
        sev_df = pd.DataFrame(stats["severity_distribution"], columns=["Severity", "Count"])
        sev_order = ["Low", "Moderate", "High", "Severe", "Hazardous"]
        sev_df["Severity"] = pd.Categorical(sev_df["Severity"], categories=sev_order, ordered=True)
        sev_df = sev_df.sort_values("Severity")
        
        fig = px.bar(sev_df, x="Severity", y="Count", 
                    color="Severity",
                    color_discrete_map={
                        "Low": "#4CAF50", "Moderate": "#FFC107",
                        "High": "#FF9800", "Severe": "#FF5722", "Hazardous": "#B71C1C"
                    })
        fig.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent Reports Table
    st.subheader("📋 Recent Reports")
    recent = get_recent_reports(10)
    if not recent.empty:
        st.dataframe(recent, use_container_width=True, hide_index=True)


# --- Report Pollution ---
elif page == "📸 Report Pollution":
    st.markdown('<p class="main-header">📸 Report Pollution</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a photo, record your voice, or type a report</p>', unsafe_allow_html=True)
    
    # Location Input
    st.subheader("📍 Location Details")
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        location_name = st.text_input("Location Name", placeholder="e.g., Jubilee Hills Check Post")
        city = st.selectbox("City", ["Hyderabad", "Delhi", "Mumbai"])
    with loc_col2:
        latitude = st.number_input("Latitude", value=17.4319, format="%.4f")
        longitude = st.number_input("Longitude", value=78.4071, format="%.4f")
    
    st.markdown("---")
    
    # Report Methods
    tab1, tab2, tab3 = st.tabs(["📷 Photo Report", "🎤 Voice Report", "✍️ Text Report"])
    
    with tab1:
        st.markdown("**Upload a photo of the pollution source**")
        uploaded_image = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png", "webp"],
            help="Upload a clear photo showing smoke, dust, or pollution"
        )
        
        additional_text = st.text_area(
            "Additional details (optional)", 
            placeholder="Any extra context about what you're seeing..."
        )
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", width=400)
            
            if st.button("🔍 Analyze & Submit Report", key="img_submit"):
                if not location_name:
                    st.error("Please enter a location name!")
                elif not os.getenv("GROQ_API_KEY"):
                    st.error("GROQ_API_KEY not set! Please set it in your .env file.")
                else:
                    with st.spinner("🤖 Analyzing image with AI..."):
                        result = process_image_report(
                            uploaded_image.getvalue(),
                            location_name, latitude, longitude, city,
                            additional_text
                        )
                    
                    st.success(f"✅ Report #{result['report_id']} submitted successfully!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pollution Type", result["pollution_type"])
                    with col2:
                        st.metric("Severity", result["severity"])
                    with col3:
                        st.metric("Est. AQI Impact", result["aqi"])
                    
                    st.info(f"**Recommended Action:** {result['recommended_action']}")
                    
                    with st.expander("🔎 Detailed AI Analysis"):
                        st.write(result["vision_analysis"].get("raw_response", ""))
                    
                    if result.get("similar_reports"):
                        with st.expander("📂 Similar Past Reports"):
                            for sr in result["similar_reports"][:3]:
                                st.write(f"- {sr['document'][:200]}...")
    
    with tab2:
        st.markdown("**Record or upload a voice report**")
        
        audio_file = st.file_uploader(
            "Upload audio file", type=["wav", "mp3", "m4a", "ogg", "webm"],
            help="Record a voice message describing the pollution"
        )
        
        if audio_file:
            st.audio(audio_file)
            
            if st.button("🎙️ Transcribe & Submit", key="voice_submit"):
                if not location_name:
                    st.error("Please enter a location name!")
                elif not os.getenv("GROQ_API_KEY"):
                    st.error("GROQ_API_KEY not set!")
                else:
                    with st.spinner("🎙️ Transcribing voice report..."):
                        result = process_voice_report(
                            audio_file.getvalue(),
                            location_name, latitude, longitude, city,
                            audio_file.name
                        )
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"✅ Report #{result['report_id']} submitted!")
                        st.info(f"**Transcription:** {result['transcription']}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Type", result["analysis"]["pollution_type"])
                        with col2:
                            st.metric("Severity", result["analysis"]["severity"])
                        with col3:
                            st.metric("AQI", result["analysis"]["estimated_aqi"])
    
    with tab3:
        st.markdown("**Describe the pollution in text**")
        
        text_report = st.text_area(
            "Describe what you're observing",
            placeholder="e.g., Heavy black smoke from garbage burning near the school. Children are having difficulty breathing. This has been going on since morning.",
            height=150
        )
        
        if st.button("📝 Submit Text Report", key="text_submit"):
            if not location_name:
                st.error("Please enter a location name!")
            elif not text_report:
                st.error("Please describe the pollution!")
            elif not os.getenv("GROQ_API_KEY"):
                st.error("GROQ_API_KEY not set!")
            else:
                with st.spinner("🤖 Analyzing report..."):
                    result = process_text_report(
                        text_report, location_name, latitude, longitude, city
                    )
                
                st.success(f"✅ Report #{result['report_id']} submitted!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Type", result["analysis"]["pollution_type"])
                with col2:
                    st.metric("Severity", result["analysis"]["severity"])
                with col3:
                    st.metric("AQI", result["analysis"]["estimated_aqi"])
                
                st.info(f"**Recommended Action:** {result['analysis']['recommended_action']}")


# --- Pollution Map ---
elif page == "🗺️ Pollution Map":
    st.markdown('<p class="main-header">🗺️ Pollution Hotspot Map</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive map showing pollution severity across neighbourhoods</p>', unsafe_allow_html=True)
    
    hotspot_data = get_hotspot_data()
    
    if city_filter != "All":
        hotspot_data = hotspot_data[hotspot_data["city"] == city_filter]
    
    if hotspot_data.empty:
        st.warning("No pollution data available for the selected filter.")
    else:
        # Center map based on data
        center_lat = hotspot_data["latitude"].mean()
        center_lon = hotspot_data["longitude"].mean()
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=11, 
                      tiles="CartoDB positron")
        
        # Add markers with severity-based colors
        for _, row in hotspot_data.iterrows():
            color = get_severity_color(row["severity"])
            
            # Radius based on AQI
            radius = max(5, min(20, (row["aqi_reading"] or 100) / 25))
            
            popup_html = f"""
            <div style='width:250px'>
                <b>{row['location_name']}</b><br>
                <b>Type:</b> {row['pollution_type']}<br>
                <b>Severity:</b> <span style='color:{color}'>{row['severity']}</span><br>
                <b>AQI:</b> {row['aqi_reading']}<br>
                <b>Status:</b> {row['status']}<br>
                <b>Action:</b> {row['recommended_action']}<br>
                <small>{row['reported_at']}</small>
            </div>
            """
            
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{row['location_name']} | AQI: {row['aqi_reading']}"
            ).add_to(m)
        
        # Legend
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                    background: white; padding: 10px; border-radius: 8px; 
                    box-shadow: 0 2px 6px rgba(0,0,0,0.3);">
            <b>Severity Legend</b><br>
            <span style="color:#4CAF50">●</span> Low<br>
            <span style="color:#FFC107">●</span> Moderate<br>
            <span style="color:#FF9800">●</span> High<br>
            <span style="color:#FF5722">●</span> Severe<br>
            <span style="color:#B71C1C">●</span> Hazardous
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        st_folium(m, width=None, height=550, use_container_width=True)
        
        # Summary stats below map
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📍 Hotspots Shown", len(hotspot_data))
        with col2:
            st.metric("💨 Max AQI", hotspot_data["aqi_reading"].max())
        with col3:
            st.metric("📊 Avg AQI", round(hotspot_data["aqi_reading"].mean(), 1))
        with col4:
            severe_count = len(hotspot_data[hotspot_data["severity"].isin(["Severe", "Hazardous"])])
            st.metric("🚨 Critical Spots", severe_count)


# --- Predictions & Alerts ---
elif page == "⚠️ Predictions & Alerts":
    st.markdown('<p class="main-header">⚠️ Predictions & Alerts</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered 24-hour forecasts and active alerts for municipal teams</p>', unsafe_allow_html=True)
    
    # Active Alerts Section
    st.subheader("🚨 Active Alerts")
    
    import sqlite3
    db_path = os.path.join(os.path.dirname(__file__), "data", "pollution_db.sqlite")
    conn = sqlite3.connect(db_path)
    alerts_df = pd.read_sql_query(
        "SELECT * FROM alerts WHERE is_active = 1 ORDER BY created_at DESC", conn
    )
    conn.close()
    
    if alerts_df.empty:
        st.info("No active alerts at the moment. ✅")
    else:
        for _, alert in alerts_df.iterrows():
            severity_emoji = {"High": "🟠", "Severe": "🔴", "Hazardous": "⚫"}.get(alert["severity"], "🟡")
            st.markdown(f"""
            <div class="alert-card">
                <b>{severity_emoji} {alert['alert_type']}</b> — {alert['location_name']}, {alert['city']}<br>
                <span style='color:#555'>{alert['message']}</span><br>
                <b>Action:</b> {alert['recommended_action']}<br>
                <small>Created: {alert['created_at']}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Audio alert option
        if st.button("🔊 Generate Audio Alert for Latest"):
            latest_alert = alerts_df.iloc[0]
            audio_msg = f"{latest_alert['alert_type']} at {latest_alert['location_name']}. {latest_alert['message']}. Recommended action: {latest_alert['recommended_action']}"
            try:
                audio_bytes = text_to_speech(audio_msg)
                st.audio(audio_bytes, format="audio/mp3")
            except Exception as e:
                st.error(f"Error generating audio: {e}")
    
    st.markdown("---")
    
    # AI Prediction Section
    st.subheader("🔮 24-Hour Air Quality Prediction")
    
    pred_city = st.selectbox("Generate prediction for:", ["All Cities", "Hyderabad", "Delhi", "Mumbai"])
    
    if st.button("🤖 Generate AI Prediction", type="primary"):
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set!")
        else:
            with st.spinner("🔮 Analyzing patterns and generating prediction..."):
                city_param = None if pred_city == "All Cities" else pred_city
                prediction = generate_prediction(city_param)
            
            if prediction["status"] == "success":
                st.markdown("### 📊 Prediction Results")
                st.markdown(prediction["prediction"])
                
                # TTS option
                if st.button("🔊 Read Prediction Aloud"):
                    try:
                        audio = text_to_speech(prediction["prediction"][:500])
                        st.audio(audio, format="audio/mp3")
                    except:
                        pass
            else:
                st.error("Failed to generate prediction. Check your API key.")


# --- Search Reports ---
elif page == "🔍 Search Reports":
    st.markdown('<p class="main-header">🔍 Search Reports</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Semantic search through pollution reports using AI</p>', unsafe_allow_html=True)
    
    # Search interface
    search_query = st.text_input(
        "🔎 Search pollution reports",
        placeholder="e.g., garbage burning near schools, industrial smoke in Hyderabad, dust from construction"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        num_results = st.slider("Number of results", 3, 20, 10)
    with col2:
        search_city = st.selectbox("Filter by city", ["All", "Hyderabad", "Delhi", "Mumbai"], key="search_city")
    
    if st.button("🔍 Search", type="primary") and search_query:
        with st.spinner("Searching..."):
            results = semantic_search(search_query, n_results=num_results, 
                                    city_filter=search_city if search_city != "All" else None)
        
        if results:
            st.success(f"Found {len(results)} relevant reports")
            
            for i, result in enumerate(results, 1):
                similarity = result.get("similarity", 0)
                meta = result.get("metadata", {})
                
                severity_color = get_severity_color(meta.get("severity", ""))
                
                with st.expander(
                    f"#{i} | {meta.get('location', 'Unknown')} | "
                    f"{meta.get('pollution_type', 'Unknown')} | "
                    f"Similarity: {similarity:.2%}"
                ):
                    st.markdown(f"**Location:** {meta.get('location', 'N/A')}")
                    st.markdown(f"**City:** {meta.get('city', 'N/A')}")
                    st.markdown(f"**Type:** {meta.get('pollution_type', 'N/A')}")
                    st.markdown(f"**Severity:** :{severity_color}[{meta.get('severity', 'N/A')}]")
                    st.markdown(f"**AQI:** {meta.get('aqi', 'N/A')}")
                    st.markdown(f"**Details:** {result['document']}")
        else:
            st.warning("No matching reports found. Try a different search term.")
    
    st.markdown("---")
    
    # NL-to-SQL Query
    st.subheader("💾 Database Query (Natural Language)")
    nl_query = st.text_input(
        "Ask a question about the data",
        placeholder="e.g., How many severe pollution reports are there in Delhi this week?"
    )
    
    if st.button("🗄️ Query Database") and nl_query:
        if not os.getenv("GROQ_API_KEY"):
            st.error("GROQ_API_KEY not set!")
        else:
            with st.spinner("Converting to SQL and querying..."):
                sql, results = query_database(nl_query)
            
            st.code(sql, language="sql")
            
            if not results.empty:
                st.dataframe(results, use_container_width=True, hide_index=True)
            else:
                st.info("No results found for this query.")


# --- Ask AI ---
elif page == "💬 Ask AI":
    st.markdown('<p class="main-header">💬 Ask CleanAir AI</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Chat with AI about pollution data, get insights and recommendations</p>', unsafe_allow_html=True)
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about pollution trends, hotspots, recommendations...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        if not os.getenv("GROQ_API_KEY"):
            response = "⚠️ GROQ_API_KEY not set. Please configure your API key in the .env file."
        else:
            with st.spinner("Thinking..."):
                response = chat_query(user_input)
        
        # Add assistant response
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # TTS option
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔊", key=f"tts_{len(st.session_state.chat_history)}"):
                try:
                    audio = text_to_speech(response[:500])
                    st.audio(audio, format="audio/mp3")
                except:
                    pass
    
    # Quick prompts
    st.markdown("---")
    st.markdown("**Quick Questions:**")
    quick_cols = st.columns(3)
    quick_prompts = [
        "What are the worst pollution hotspots right now?",
        "Which areas need immediate cleanup crews?",
        "What's the trend in garbage burning reports?",
        "Recommend preventive actions for next week",
        "Compare pollution levels across cities",
        "Which pollution type is most common?"
    ]
    
    for i, prompt in enumerate(quick_prompts):
        col_idx = i % 3
        with quick_cols[col_idx]:
            if st.button(prompt, key=f"quick_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                if os.getenv("GROQ_API_KEY"):
                    with st.spinner("Thinking..."):
                        response = chat_query(prompt)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
