"""
Core Pipeline — Single multimodal pipeline for CleanAir & Clear Streets.
Orchestrates vision, voice, RAG, and SQL tools into a unified flow.
"""

import os
from groq import Groq
from tools.vision_tool import analyze_pollution_image
from tools.voice_tool import speech_to_text, text_to_speech
from tools.sql_tool import (
    insert_report, get_pollution_stats, get_hotspot_data,
    query_database, get_recent_reports
)
from tools.rag_tool import semantic_search, add_report_to_index, index_reports


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def process_image_report(image_bytes: bytes, location_name: str, lat: float, 
                         lon: float, city: str, additional_text: str = "") -> dict:
    """
    Process a pollution report submitted with an image.
    
    Pipeline: Image → Vision Analysis → Store in DB → Index in RAG → Generate Response
    """
    # Step 1: Analyze image with Groq Llama 4 Vision
    vision_result = analyze_pollution_image(image_bytes)
    
    # Step 2: Combine vision analysis with user text
    description = vision_result.get("description", "")
    if additional_text:
        description = f"{additional_text}. AI Analysis: {description}"
    
    pollution_type = vision_result.get("pollution_type", "Unknown")
    severity = vision_result.get("severity", "Moderate")
    aqi = vision_result.get("estimated_aqi_impact", 150)
    recommended_action = vision_result.get("recommended_action", "Manual inspection required")
    
    # Step 3: Store in database
    report_id = insert_report(
        location_name=location_name,
        lat=lat, lon=lon, city=city,
        pollution_type=pollution_type,
        severity=severity,
        description=description,
        aqi=aqi,
        image_analysis=vision_result.get("raw_response", ""),
        recommended_action=recommended_action
    )
    
    # Step 4: Add to RAG index
    doc_text = (
        f"Location: {location_name}, City: {city}. "
        f"Pollution Type: {pollution_type}. Severity: {severity}. "
        f"Description: {description}. AQI: {aqi}. "
        f"Action: {recommended_action}"
    )
    add_report_to_index(report_id, doc_text, {
        "location": location_name,
        "city": city,
        "pollution_type": pollution_type,
        "severity": severity,
        "aqi": aqi
    })
    
    # Step 5: Find similar past reports
    similar = semantic_search(
        f"{pollution_type} at {location_name}", n_results=3
    )
    
    return {
        "report_id": report_id,
        "vision_analysis": vision_result,
        "pollution_type": pollution_type,
        "severity": severity,
        "aqi": aqi,
        "recommended_action": recommended_action,
        "similar_reports": similar,
        "status": "Report submitted successfully"
    }


def process_voice_report(audio_bytes: bytes, location_name: str, lat: float,
                         lon: float, city: str, filename: str = "recording.wav") -> dict:
    """
    Process a pollution report submitted via voice.
    
    Pipeline: Audio → Whisper STT → LLM Analysis → Store in DB → Index in RAG
    """
    # Step 1: Transcribe audio
    transcription = speech_to_text(audio_bytes, filename)
    
    if transcription.startswith("Error"):
        return {"error": transcription, "status": "Failed to transcribe audio"}
    
    # Step 2: Analyze transcription with LLM
    analysis = analyze_text_report(transcription)
    
    # Step 3: Store in database
    report_id = insert_report(
        location_name=location_name,
        lat=lat, lon=lon, city=city,
        pollution_type=analysis["pollution_type"],
        severity=analysis["severity"],
        description=f"Voice Report: {transcription}",
        aqi=analysis["estimated_aqi"],
        recommended_action=analysis["recommended_action"]
    )
    
    # Step 4: Add to RAG index
    doc_text = (
        f"Location: {location_name}, City: {city}. "
        f"Pollution Type: {analysis['pollution_type']}. Severity: {analysis['severity']}. "
        f"Description: {transcription}. AQI: {analysis['estimated_aqi']}. "
        f"Action: {analysis['recommended_action']}"
    )
    add_report_to_index(report_id, doc_text, {
        "location": location_name,
        "city": city,
        "pollution_type": analysis["pollution_type"],
        "severity": analysis["severity"],
        "aqi": analysis["estimated_aqi"]
    })
    
    return {
        "report_id": report_id,
        "transcription": transcription,
        "analysis": analysis,
        "status": "Voice report submitted successfully"
    }


def process_text_report(text: str, location_name: str, lat: float,
                        lon: float, city: str) -> dict:
    """
    Process a text-based pollution report.
    
    Pipeline: Text → LLM Analysis → Store in DB → Index in RAG
    """
    # Step 1: Analyze text
    analysis = analyze_text_report(text)
    
    # Step 2: Store in database
    report_id = insert_report(
        location_name=location_name,
        lat=lat, lon=lon, city=city,
        pollution_type=analysis["pollution_type"],
        severity=analysis["severity"],
        description=text,
        aqi=analysis["estimated_aqi"],
        recommended_action=analysis["recommended_action"]
    )
    
    # Step 3: Add to RAG index
    doc_text = (
        f"Location: {location_name}, City: {city}. "
        f"Pollution Type: {analysis['pollution_type']}. Severity: {analysis['severity']}. "
        f"Description: {text}. AQI: {analysis['estimated_aqi']}. "
        f"Action: {analysis['recommended_action']}"
    )
    add_report_to_index(report_id, doc_text, {
        "location": location_name,
        "city": city,
        "pollution_type": analysis["pollution_type"],
        "severity": analysis["severity"],
        "aqi": analysis["estimated_aqi"]
    })
    
    return {
        "report_id": report_id,
        "analysis": analysis,
        "status": "Text report submitted successfully"
    }


def analyze_text_report(text: str) -> dict:
    """Use LLM to analyze a text pollution report."""
    client = get_client()
    
    prompt = f"""Analyze this pollution report and extract key information.

Report: {text}

Respond in this exact format:
POLLUTION_TYPE: (e.g., Garbage Burning, Industrial Smoke, Vehicle Exhaust, Construction Dust, Factory Emissions, Open Waste Dump, Chemical Fumes, Road Dust)
SEVERITY: (Low/Moderate/High/Severe/Hazardous)
ESTIMATED_AQI: (number between 50-500)
RECOMMENDED_ACTION: (specific action for municipal teams)
SUMMARY: (one-line summary)"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.2
        )
        
        result_text = response.choices[0].message.content
        return parse_analysis_response(result_text)
        
    except Exception as e:
        return {
            "pollution_type": "Unknown",
            "severity": "Moderate",
            "estimated_aqi": 150,
            "recommended_action": "Manual inspection required",
            "summary": text[:100]
        }


def parse_analysis_response(text: str) -> dict:
    """Parse LLM analysis response."""
    result = {
        "pollution_type": "Unknown",
        "severity": "Moderate",
        "estimated_aqi": 150,
        "recommended_action": "Manual inspection required",
        "summary": ""
    }
    
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("POLLUTION_TYPE:"):
            result["pollution_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("SEVERITY:"):
            result["severity"] = line.split(":", 1)[1].strip()
        elif line.startswith("ESTIMATED_AQI:"):
            try:
                result["estimated_aqi"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("RECOMMENDED_ACTION:"):
            result["recommended_action"] = line.split(":", 1)[1].strip()
        elif line.startswith("SUMMARY:"):
            result["summary"] = line.split(":", 1)[1].strip()
    
    return result


def generate_prediction(city: str = None) -> dict:
    """
    Generate 24-hour air quality prediction based on historical data and patterns.
    """
    client = get_client()
    
    # Get recent data for context
    stats = get_pollution_stats()
    recent = get_recent_reports(20)
    
    recent_summary = ""
    if not recent.empty:
        recent_summary = recent.to_string(index=False, max_rows=10)
    
    prompt = f"""Based on the following pollution data, predict air quality for the next 24 hours.

Current Statistics:
- Total Reports: {stats['total_reports']}
- Average AQI: {stats['avg_aqi']}
- Active Alerts: {stats['active_alerts']}
- Reports by City: {stats['reports_by_city']}
- Top Pollution Types: {stats['top_pollution_types']}

Recent Reports:
{recent_summary}

{f'Focus on city: {city}' if city else 'Cover all cities.'}

Provide predictions in this format:
PREDICTION_SUMMARY: (2-3 sentence overview)
RISK_LEVEL: (Low/Moderate/High/Critical)
HOTSPOT_1: location | predicted_aqi | risk_factor
HOTSPOT_2: location | predicted_aqi | risk_factor
HOTSPOT_3: location | predicted_aqi | risk_factor
RECOMMENDED_ACTIONS: (bullet points of preventive actions for municipal teams)
WEATHER_IMPACT: (how weather might affect air quality in next 24h)"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.4
        )
        
        return {
            "prediction": response.choices[0].message.content,
            "stats": stats,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "prediction": f"Error generating prediction: {str(e)}",
            "stats": stats,
            "status": "error"
        }


def chat_query(question: str) -> str:
    """
    Handle a general chat query about pollution data.
    Uses RAG + SQL as needed.
    """
    client = get_client()
    
    # Get relevant context from RAG
    rag_results = semantic_search(question, n_results=5)
    rag_context = "\n".join([r["document"] for r in rag_results])
    
    # Get stats
    stats = get_pollution_stats()
    
    prompt = f"""You are CleanAir AI assistant helping municipal teams and citizens with pollution data.

Context from database:
{rag_context}

Statistics:
- Total Reports: {stats['total_reports']}
- Average AQI: {stats['avg_aqi']}
- Active Alerts: {stats['active_alerts']}

User Question: {question}

Provide a helpful, concise answer based on the available data. If you need to suggest actions, be specific and practical."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Error processing query: {str(e)}"
