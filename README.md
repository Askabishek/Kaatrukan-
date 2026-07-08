# 🌬️ CleanAir & Clear Streets

**Spotting and Fixing Local Pollution Hotspots**

An AI-powered neighbourhood-level pollution monitoring and alert system that combines citizen-uploaded photos, voice reports, sensor data, and satellite imagery to detect hidden pollution hotspots, predict air quality spikes, and alert municipal teams for rapid response.

---

## 🎯 Problem Statement

City-level air quality apps miss hyper-local events — a garbage dump fire, an industrial cluster, a smog trap at a busy junction — because local authorities can't have eyes on every street. These pockets go unnoticed while directly harming nearby residents.

## 💡 Solution

A multimodal AI platform where citizens can report pollution via **photos**, **voice**, or **text**. The system:
- Analyzes images using **Groq Llama 4 Scout** (vision)
- Transcribes voice reports using **Groq Whisper**
- Generates audio alerts using **gTTS**
- Performs semantic search over reports using **Chroma + Sentence-Transformers**
- Stores structured data in **SQLite**
- Visualizes hotspots on interactive maps
- Predicts 24-hour air quality spikes
- Recommends specific actions for municipal teams

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Main LLM (Text + Vision) | Groq Llama 4 Scout 17B |
| Speech-to-Text | Groq Whisper Large V3 |
| Text-to-Speech | gTTS |
| RAG / Semantic Search | Chroma + Sentence-Transformers |
| Database | SQLite |
| Frontend | Streamlit |
| Backend | Python |
| Maps | Folium |
| Charts | Plotly |

---

## 📁 Project Structure

```
cleanair/
├── app.py              # Streamlit UI (main entry point)
├── pipeline.py         # Core single multimodal pipeline
├── tools/
│   ├── sql_tool.py     # NL-to-SQL for querying pollution DB
│   ├── rag_tool.py     # Semantic search over reports
│   ├── vision_tool.py  # Image analysis via Groq Llama 4 Vision
│   └── voice_tool.py   # Whisper STT + gTTS TTS
├── data/
│   ├── pollution_db.sqlite  # (auto-generated at runtime)
│   └── seed_data.py    # Synthetic data generator
├── embeddings/         # Chroma vector store (auto-generated)
├── .env.example        # Template for environment variables
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd cleanair
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

### 5. Run the application
```bash
streamlit run app.py
```

The app will automatically:
- Create the SQLite database
- Seed it with synthetic data (150 reports, 500 sensor readings, 20 alerts)
- Index reports in the Chroma vector store
- Launch the Streamlit UI

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Your Groq API key for Llama 4 and Whisper |

Get your free API key at: https://console.groq.com/

---

## 📱 Features

### 1. 📸 Report Pollution
- Upload photos → AI vision analysis detects pollution type and severity
- Record voice → Whisper transcription + AI analysis
- Type text → NLP-based classification

### 2. 🗺️ Pollution Map
- Interactive map with colour-coded severity markers
- Clickable hotspots with full report details
- City-wise filtering

### 3. ⚠️ Predictions & Alerts
- AI-generated 24-hour air quality forecasts
- Active alert dashboard for municipal teams
- Audio alerts via text-to-speech

### 4. 🔍 Search Reports
- Semantic search using natural language
- NL-to-SQL database queries
- City and severity filtering

### 5. 💬 Ask AI
- Chat interface for pollution insights
- Context-aware responses using RAG
- Quick-prompt suggestions

---

## 🏙️ Demo Data

The system comes pre-loaded with synthetic data covering:
- **Hyderabad**: Jubilee Hills, HITEC City, Charminar, Kukatpally, Secunderabad, Miyapur, LB Nagar, Begumpet
- **Delhi**: Anand Vihar, ITO, Chandni Chowk, Dwarka, Rohini, Okhla, Mundka, Sarai Kale Khan
- **Mumbai**: Dharavi, Bandra-Worli, Chembur, Andheri East, Sion-Panvel, Mahim

---

## 🏆 Built for Build with AI Hackathon 2026

**Challenge**: CleanAir & Clear Streets - Spotting and Fixing Local Pollution Hotspots

---

## 📄 License

MIT License
