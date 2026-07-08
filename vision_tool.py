"""
Vision Tool — Analyzes pollution images using Groq Llama 4 Scout (multimodal).
"""

import os
import base64
from groq import Groq


def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))


def encode_image(image_bytes: bytes) -> str:
    """Encode image bytes to base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


def analyze_pollution_image(image_bytes: bytes) -> dict:
    """
    Analyze an uploaded image for pollution indicators using Groq Llama 4 Scout vision.
    
    Returns a dict with:
        - pollution_detected: bool
        - pollution_type: str
        - severity: str (Low/Moderate/High/Severe/Hazardous)
        - description: str
        - recommended_action: str
        - estimated_aqi_impact: int
    """
    client = get_client()
    
    base64_image = encode_image(image_bytes)
    
    prompt = """Analyze this image for air pollution or environmental pollution indicators.
    
Provide your analysis in the following exact format:
POLLUTION_DETECTED: Yes/No
POLLUTION_TYPE: (e.g., Garbage Burning, Industrial Smoke, Vehicle Exhaust, Construction Dust, Factory Emissions, etc.)
SEVERITY: (Low/Moderate/High/Severe/Hazardous)
DESCRIPTION: (Brief description of what you see and the pollution impact)
RECOMMENDED_ACTION: (What municipal teams should do)
ESTIMATED_AQI_IMPACT: (A number between 50-500 estimating the AQI contribution)

Be specific and practical in your recommendations. If no pollution is visible, still provide your assessment."""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1024,
            temperature=0.3
        )
        
        result_text = response.choices[0].message.content
        return parse_vision_response(result_text)
        
    except Exception as e:
        return {
            "pollution_detected": False,
            "pollution_type": "Unknown",
            "severity": "Unknown",
            "description": f"Error analyzing image: {str(e)}",
            "recommended_action": "Manual inspection required",
            "estimated_aqi_impact": 0,
            "raw_response": str(e)
        }


def parse_vision_response(text: str) -> dict:
    """Parse the structured response from the vision model."""
    result = {
        "pollution_detected": False,
        "pollution_type": "Unknown",
        "severity": "Moderate",
        "description": "",
        "recommended_action": "",
        "estimated_aqi_impact": 150,
        "raw_response": text
    }
    
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("POLLUTION_DETECTED:"):
            val = line.split(":", 1)[1].strip().lower()
            result["pollution_detected"] = val in ["yes", "true", "1"]
        elif line.startswith("POLLUTION_TYPE:"):
            result["pollution_type"] = line.split(":", 1)[1].strip()
        elif line.startswith("SEVERITY:"):
            result["severity"] = line.split(":", 1)[1].strip()
        elif line.startswith("DESCRIPTION:"):
            result["description"] = line.split(":", 1)[1].strip()
        elif line.startswith("RECOMMENDED_ACTION:"):
            result["recommended_action"] = line.split(":", 1)[1].strip()
        elif line.startswith("ESTIMATED_AQI_IMPACT:"):
            try:
                result["estimated_aqi_impact"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                result["estimated_aqi_impact"] = 150
    
    # If parsing didn't get description, use raw response
    if not result["description"]:
        result["description"] = text[:500]
    
    return result
