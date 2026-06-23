import joblib
import pandas as pd
import os
import numpy as np
import warnings
import sys

# Suppress scikit-learn version incompatibility warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Compatibility patch for scikit-learn version mismatch.
# The saved joblib pipeline references this private sklearn class, but it is not
# present in some newer scikit-learn versions.
try:
    from sklearn.compose._column_transformer import _RemainderColsList
except (ImportError, AttributeError):
    import sklearn.compose._column_transformer as ct

    if not hasattr(ct, '_RemainderColsList'):
        class _RemainderColsList(list):
            pass

        ct._RemainderColsList = _RemainderColsList

# Load the model once at module initialization
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'lightgbm_pipeline.joblib')
try:
    # Suppress version incompatibility warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def predict(form_data):
    """
    Args:
        form_data (dict): Dictionary containing form field values
        
    Returns:
        dict: Dictionary with prediction, confidence, and message
    """
    if model is None:
        return {
            'prediction': 'Error',
            'confidence': 0,
            'message': 'Model failed to load. Please try again later.'
        }
    
    try:
        
        df = pd.DataFrame([{
            'Potassium': form_data.get('potassium', 0),
            'Phosphorus': form_data.get('phosphorus', 0),
            'Nitrogen': form_data.get('nitrogen', 0),
            'Rainfall': float(form_data.get('rainfall', 0)),
            'pH': float(form_data.get('ph', 0)),
            'Temperature': float(form_data.get('temperature', 0)),
            'Crop': form_data.get('crop', ''),
        }])
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get prediction probabilities for confidence
        probabilities = model.predict_proba(df)[0]
        confidence = float(np.max(probabilities))
        
        
        # Generate appropriate message
        message = f"Based on your query you should apply {prediction} fertilizer in your field"
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'message': message
        }
    
    except Exception as e:
        return {
            'prediction': 'Error',
            'confidence': 0,
            'message': f'An error occurred during prediction: {str(e)}'
        }
