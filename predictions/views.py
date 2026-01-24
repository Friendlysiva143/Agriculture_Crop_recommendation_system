"""
Views for Crop Recommendation System
Single prediction and batch CSV upload
"""

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from django.views.decorators.http import require_http_methods
import pandas as pd
import json

from .forms import CropPredictionForm, CropCSVUploadForm
from .ml_model import crop_model

@login_required(login_url='users:login')
@require_http_methods(["GET", "POST"])
def dashboard(request):
    """
    Main prediction dashboard for crop recommendation
    """
    prediction_form = CropPredictionForm()
    csv_form = CropCSVUploadForm()
    context = {
        'prediction_form': prediction_form,
        'csv_form': csv_form,
        'prediction_result': None,
        'csv_results': None,
        'total_predictions': 0,
    }
    
    if request.method == 'POST':
        # Handle single crop prediction
        if 'single_predict' in request.POST:
            prediction_form = CropPredictionForm(request.POST)
            if prediction_form.is_valid():
                # Extract cleaned data
                N = prediction_form.cleaned_data['N']
                P = prediction_form.cleaned_data['P']
                K = prediction_form.cleaned_data['K']
                temperature = prediction_form.cleaned_data['temperature']
                humidity = prediction_form.cleaned_data['humidity']
                ph = prediction_form.cleaned_data['ph']
                rainfall = prediction_form.cleaned_data['rainfall']
                
                # Make prediction
                result = crop_model.predict_single(N, P, K, temperature, humidity, ph, rainfall)
                
                if result['success']:
                    context['prediction_result'] = result
                    messages.success(request, f"✓ Predicted Crop: {result['prediction']}")
                else:
                    messages.error(request, f"✗ {result['error']}")
            else:
                messages.error(request, 'Please fill all fields correctly.')
            
            context['prediction_form'] = prediction_form
        
        # Handle CSV batch upload
        elif 'csv_predict' in request.POST:
            csv_form = CropCSVUploadForm(request.POST, request.FILES)
            if csv_form.is_valid():
                try:
                    # Read CSV file
                    csv_file = request.FILES['csv_file']
                    df = pd.read_csv(csv_file)
                    
                    # Validate CSV has required columns
                    required_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        messages.error(
                            request, 
                            f"CSV missing columns: {', '.join(missing_cols)}. Required: {', '.join(required_cols)}"
                        )
                    else:
                        # Make batch predictions
                        results_df, error = crop_model.predict_batch(df)
                        
                        if error:
                            messages.error(request, f"✗ {error}")
                        else:
                            # Store results in session for download
                            request.session['prediction_results'] = results_df.to_json(orient='records')
                            
                            # Show first 10 results in HTML table
                            context['csv_results'] = results_df.head(10).to_html(
                                classes='table table-striped table-hover',
                                index=False
                            )
                            context['total_predictions'] = len(results_df)
                            
                            messages.success(
                                request, 
                                f"✓ Predictions made for {len(results_df)} records!"
                            )
                except pd.errors.ParserError:
                    messages.error(request, "✗ Invalid CSV format. Please check the file.")
                except Exception as e:
                    messages.error(request, f"✗ Error reading CSV: {str(e)}")
            else:
                messages.error(request, '✗ Please upload a valid CSV file.')
            
            context['csv_form'] = csv_form
    
    return render(request, 'predictions/dashboard.html', context)


@login_required(login_url='users:login')
def download_results(request):
    """
    Download prediction results as CSV file
    """
    try:
        # Retrieve results from session
        results_json = request.session.get('prediction_results')
        if not results_json:
            messages.error(request, '✗ No results to download.')
            return redirect('predictions:dashboard')
        
        results_data = json.loads(results_json)
        df = pd.DataFrame(results_data)
        
        # Create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="crop_predictions.csv"'
        
        df.to_csv(path_or_buf=response, index=False)
        return response
    except Exception as e:
        messages.error(request, f"✗ Error downloading results: {str(e)}")
        return redirect('predictions:dashboard')


@login_required(login_url='users:login')
def history(request):
    """
    View prediction history (placeholder for future feature)
    """
    context = {
        'message': 'Prediction history feature coming soon!'
    }
    return render(request, 'predictions/history.html', context)