import os
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None


def build_fertilizer_advice(form_data, prediction_result):
    """Generate a practical Gemini advisory for the fertilizer recommendation."""
    fallback_advice = _build_fallback_advice(form_data, prediction_result)
    env_path = Path(__file__).resolve().parent.parent / '.env'

    try:
        from dotenv import load_dotenv

        load_dotenv(dotenv_path=env_path, override=True)
    except ImportError:
        pass

    api_key = os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
    model_name = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    if not api_key:
        return {
            'advice': fallback_advice,
            'status': 'disabled',
            'error': 'GEMINI_API_KEY is not configured.',
        }

    if genai is None:
        return {
            'advice': fallback_advice,
            'status': 'unavailable',
            'error': 'google-genai is not installed.',
        }

    prompt = _build_prompt(form_data, prediction_result)

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.4,
                max_output_tokens=700,
                system_instruction=(
                    'You are an agricultural fertilizer assistant. Give clear, '
                    'practical, cautious guidance for farmers. Do not invent exact '
                    'application rates unless they are provided. Recommend local '
                    'soil testing or an agronomist for final dosage decisions.'
                ),
            ),
        )

        advice = (response.text or '').strip()
        if not advice:
            return {
                'advice': fallback_advice,
                'status': 'empty',
                'error': 'Gemini returned an empty response.',
            }

        return {
            'advice': advice,
            'status': 'generated',
            'error': '',
        }
    except Exception as exc:
        return {
            'advice': fallback_advice,
            'status': 'failed',
            'error': str(exc),
        }


def _build_fallback_advice(form_data, prediction_result):
    fertilizer = prediction_result.get('prediction') or 'the recommended'
    crop = form_data.get('crop') or 'your selected crop'
    ph = form_data.get('ph') or 'the provided'

    return (
        f"Recommendation\n"
        f"Apply {fertilizer} fertilizer for {crop} based on the model result.\n\n"
        f"Why this fits\n"
        f"The recommendation uses your submitted nitrogen, phosphorus, potassium, "
        f"rainfall, temperature, and pH values. The submitted pH value is {ph}.\n\n"
        f"Practical caution\n"
        f"Use this as guidance only. Confirm the final dosage with a local soil test "
        f"or an agricultural expert before field application."
    )

def _build_prompt(form_data, prediction_result):
    confidence = prediction_result.get('confidence', 0)
    confidence_percent = round(float(confidence) * 100, 2)

    return f"""
Create a concise fertilizer recommendation note using this model output and soil data.

Model output:
- Recommended fertilizer: {prediction_result.get('prediction', 'Unknown')}
- Model confidence: {confidence_percent}%

Input data:
- Crop: {form_data.get('crop', 'Unknown')}
- Nitrogen: {form_data.get('nitrogen', 'Unknown')} mg/kg
- Phosphorus: {form_data.get('phosphorus', 'Unknown')} mg/kg
- Potassium: {form_data.get('potassium', 'Unknown')} mg/kg
- pH: {form_data.get('ph', 'Unknown')}
- Rainfall: {form_data.get('rainfall', 'Unknown')} mm
- Temperature: {form_data.get('temperature', 'Unknown')} C

Return only these 3 clear sections, with no greeting or preface:
1. Recommendation
2. Why this fits
3. Practical caution
""".strip()
