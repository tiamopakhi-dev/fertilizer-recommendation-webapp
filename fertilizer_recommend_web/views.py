from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .llm import build_fertilizer_advice
from .ml_model.prediction import predict as predict_fertilizer

# Create your views here.
def home(request):
    return render(request, 'index.html')

@require_http_methods(["POST"])
def predict(request):
    try:
        # Create a dictionary from form data
        form_data = {
            'potassium': request.POST.get('potassium'),
            'phosphorus': request.POST.get('phosphorus'),
            'nitrogen': request.POST.get('nitrogen'),
            'rainfall': request.POST.get('rainfall'),
            'ph': request.POST.get('ph'),
            'temperature': request.POST.get('temperature'),
            'crop': request.POST.get('crop'),
        }
        
        # Pass dictionary to predict function; return a dictionary with prediction, confidence, and message
        result = predict_fertilizer(form_data)
        llm_result = build_fertilizer_advice(form_data, result)
        
        # Return JSON response with prediction results
        return JsonResponse({
            'success': True,
            'prediction': result.get('prediction'),
            'confidence': result.get('confidence'),
            'message': result.get('message'),
            'llm_advice': llm_result.get('advice'),
            'llm_status': llm_result.get('status'),
            'llm_error': llm_result.get('error'),
        }, status=200)
    
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=400)
