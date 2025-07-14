import google.generativeai as genai
import json
import re
import os

API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyC2CE4fX4Og7IxpHgIvjX3WbnXc_53kQas")
genai.configure(api_key=API_KEY)

def clean_and_parse_json(response_text):
    cleaned = re.sub(r"```(?:json)?", "", response_text).strip("`\n ")
    try:
        return json.loads(cleaned)
    except Exception as e:
        return [{"error": str(e), "raw": response_text}]

def call_gemini_response(prompt):
    model = genai.GenerativeModel("models/gemini-2.0-flash")
    response = model.generate_content(prompt)
    return clean_and_parse_json(response.text)