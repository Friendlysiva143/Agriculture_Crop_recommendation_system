"""
Crop Recommendation ML Model - Django Integration
Based on your Colab project: Agriculture Excellence
Uses ENSEMBLE of multiple models with MAJORITY VOTING
Predicts the best crop based on soil and climate parameters
"""

import pickle
import numpy as np
import pandas as pd
import os
from collections import Counter
from sklearn.preprocessing import LabelEncoder

class CropRecommendationModel:
    def __init__(self):
        """
        Initialize with multiple trained models
        Uses majority voting from ensemble of models
        """
        self.models = {}  # Dictionary to store all models
        self.label_encoder = None
        self.feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        
        # List of model files to load (from your Colab notebook)
        self.model_files = {
            'LogisticRegression': 'logisticregression_model.pkl',
            'Probabilistic LogisticRegression': 'probabilistic_logisticregression_model.pkl',
            'DecisionTree': 'decisiontree_model.pkl',
            'SVC': 'svc_model.pkl',
            'KNeighbors': 'kneighbors_model.pkl',
            'MultinomialNB': 'multinomialnb_model.pkl',
            'VotingClassifier': 'votingclassifier_model.pkl',
            'RandomForest': 'randomforest_model.pkl',
            'AdaBoost': 'adaboost_model.pkl',
            'GradientBoosting': 'gradientboosting_model.pkl',
            'LightGBM': 'lightgbm_model.pkl',
            'XGBoost': 'xgboost_model.pkl',
        }
        
        self.crop_images = {
            'apple': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSRDQPjtXnFzP3gjABS8wPsMwb8tLatM1t0ng&s',
            'banana': 'https://upload.wikimedia.org/wikipedia/commons/4/4c/Bananas.jpg',
            'blackgram': 'https://assets-news.housing.com/news/wp-content/uploads/2022/10/25151945/Black-gram-or-back-mung-bean.jpg',
            'chickpea': 'https://www.universaltradelink.net/images/productimg/11567753486.jpg',
            'coconut': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXprvnQ_ZU8HWZQQMduaR49BuBrZc73XcE0A&s',
            'coffee': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS0a-3uyQ9B4bVyJmbPEeawulA4SLxp7f2cXA&s',
            'cotton': 'https://images.ctfassets.net/3s5io6mnxfqz/4TV7YTCO1DJuMhhn7RD1Ol/b5a6c12340e6529a86bc1b557ed2d8f8/AdobeStock_136921602.jpeg',
            'grapes': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSFFZRn0s7cAU2c_dBoK5iqapNcKqa4OwaC_g&s',
            'jute': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTmHLzZyTRddXjohuuwv2jLUyBJGELS_PkpNA&s',
            'kidneybeans': 'https://www.healthifyme.com/blog/wp-content/uploads/2022/01/807716893sst1641271427-scaled.jpg',
            'lentil': 'https://www.fieldstoneorganics.ca/wp-content/uploads/2023/05/Red-Lentils.jpg',
            'maize': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRfCuZvqpRYJv--ImVOFRbtOXjAIw-ItaYsYA&s',
            'mango': 'https://upload.wikimedia.org/wikipedia/commons/9/90/Hapus_Mango.jpg',
            'mothbeans': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcR6dKkkhwLehX4LG2kxwo8C4p4LhMfGNBnMUA&s',
            'mungbean': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSdncW7dLhnBfXjcY3xWYBuujqfPVG4YmoPpg&s',
            'muskmelon': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSDVSqlPVcWW2CcDGFiVKlaEC1WOR1Sbhm0mA&s',
            'papaya': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRK0m58UoRZCFaiyJ4G9Bm0G0lDF__MH_NOww&s',
            'pigeonpeas': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRQBoUsFpW6bMZP_j6fZh-FEoJXCjU0oSzljFtiDlSG-rzA8Mvtso0pRhuAuvHPXA-I9cY&usqp=CAU',
            'pomegranate': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ9mG93vzJXNT-fUO2wi65Mf4VlSQZ1VTG99g&s',
            'rice': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQGY1YxhUGywhuwLeFswy8BHQimTC7nwECkUg&s',
            'watermelon': 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRocXAegF4BhMZyP8hvIauJpYgHp71RYJ0lcA&s',
            'orange': 'https://upload.wikimedia.org/wikipedia/commons/c/c4/Orange-Fruit-Pieces.jpg'
        }
        
        self.load_models()
    
    def load_models(self):
        """
        Load all trained models from pickle files
        Place all .pkl files in the predictions/ folder
        """
        # Get the directory where this file is located
        model_dir = os.path.dirname(os.path.abspath(__file__))
        
        print(f"\n{'='*60}")
        print(f"🔍 Looking for model files in: {model_dir}")
        print(f"{'='*60}\n")
        
        models_loaded = 0
        models_failed = 0
        
        # Load each model
        for model_name, file_name in self.model_files.items():
            model_path = os.path.join(model_dir, file_name)
            
            try:
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.models[model_name] = pickle.load(f)
                    models_loaded += 1
                    print(f"✓ {model_name:40} loaded from {file_name}")
                else:
                    models_failed += 1
                    print(f"✗ {model_name:40} NOT FOUND at {model_path}")
            except Exception as e:
                models_failed += 1
                print(f"✗ {model_name:40} ERROR: {str(e)}")
        
        # Load label encoder
        encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
        try:
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print(f"✓ {'Label Encoder':40} loaded successfully")
            else:
                print(f"✗ {'Label Encoder':40} NOT FOUND at {encoder_path}")
        except Exception as e:
            print(f"✗ {'Label Encoder':40} ERROR: {str(e)}")
        
        print(f"\n{'='*60}")
        print(f"📊 Models Status: {models_loaded} loaded, {models_failed} failed")
        print(f"{'='*60}\n")
    
    def validate_inputs(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Validate input parameters to ensure they're within reasonable ranges
        """
        errors = []
        
        if ph < 3.5:
            errors.append(f"pH {ph} is too acidic. Crops cannot grow well in highly acidic soil.")
        elif ph > 9.5:
            errors.append(f"pH {ph} is too alkaline. Most crops fail to absorb nutrients properly.")
        
        if humidity < 0 or humidity > 100:
            errors.append(f"Humidity {humidity}% must be between 0% and 100%.")
        
        if temperature < 0 or temperature > 60:
            errors.append(f"Temperature {temperature}°C is extreme. Not suitable for crop growth.")
        
        if rainfall < 0:
            errors.append("Rainfall cannot be negative.")
        elif rainfall > 350:
            errors.append(f"Rainfall {rainfall}mm is too high. May indicate flood conditions.")
        
        if N < 0 or N > 150:
            errors.append(f"Nitrogen level {N} should be between 0-150.")
        if P < 0 or P > 150:
            errors.append(f"Phosphorus level {P} should be between 0-150.")
        if K < 0 or K > 150:
            errors.append(f"Potassium level {K} should be between 0-150.")
        
        return errors
    
    def predict_single(self, N, P, K, temperature, humidity, ph, rainfall):
        """
        Make crop prediction using ENSEMBLE MAJORITY VOTING
        
        Args:
            N: Nitrogen level
            P: Phosphorus level
            K: Potassium level
            temperature: Temperature in Celsius
            humidity: Humidity percentage (0-100)
            ph: Soil pH level (0-14)
            rainfall: Rainfall in mm
        
        Returns:
            Dictionary with prediction, confidence, and crop image
        """
        # Validate inputs
        errors = self.validate_inputs(N, P, K, temperature, humidity, ph, rainfall)
        if errors:
            return {
                'success': False,
                'error': ' | '.join(errors)
            }
        
        try:
            # Check if models are loaded
            if not self.models:
                return {
                    'success': False,
                    'error': 'Models not loaded. Please check predictions/ folder has all 13 .pkl files.'
                }
            
            if self.label_encoder is None:
                return {
                    'success': False,
                    'error': 'Label encoder not loaded. Check label_encoder.pkl exists in predictions/ folder.'
                }
            
            # Create input array in correct order
            input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            
            # Get predictions from all models
            all_predictions = []
            model_predictions = {}
            
            for model_name, model in self.models.items():
                try:
                    encoded_prediction = model.predict(input_data)[0]
                    crop_name = self.label_encoder.inverse_transform([encoded_prediction])[0]
                    all_predictions.append(crop_name)
                    model_predictions[model_name] = crop_name
                except Exception as e:
                    print(f"Prediction error with {model_name}: {str(e)}")
            
            if not all_predictions:
                return {
                    'success': False,
                    'error': 'No predictions could be made. Models may be corrupted.'
                }
            
            # MAJORITY VOTING - Get most frequent crop prediction
            crop_counter = Counter(all_predictions)
            final_crop = crop_counter.most_common(1)[0][0]
            vote_count = crop_counter.most_common(1)[0][1]
            
            # Calculate confidence as percentage of votes received
            confidence = (vote_count / len(self.models)) * 100
            
            # Get crop image URL
            crop_image = self.crop_images.get(final_crop.lower(), '')
            
            return {
                'success': True,
                'prediction': final_crop,
                'confidence': confidence,
                'image_url': crop_image,
                'error': None,
                'votes': vote_count,
                'total_models': len(self.models),
                'model_predictions': model_predictions  # For debugging
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}'
            }
    
    def predict_batch(self, csv_data):
        """
        Make predictions for batch data from CSV using ensemble
        Expected CSV columns: N, P, K, temperature, humidity, ph, rainfall
        
        Args:
            csv_data: pandas DataFrame with feature columns
        
        Returns:
            Tuple (DataFrame with predictions, error message or None)
        """
        try:
            if not self.models or self.label_encoder is None:
                return None, 'Models not loaded. Please ensure all model files are in predictions/ folder.'
            
            # Select only feature columns in correct order
            required_cols = self.feature_names
            
            # Check if all required columns exist
            missing_cols = [col for col in required_cols if col not in csv_data.columns]
            if missing_cols:
                return None, f'Missing columns: {", ".join(missing_cols)}. Required: {", ".join(required_cols)}'
            
            # Extract features in correct order
            X = csv_data[required_cols]
            
            # Make predictions for each row using ensemble
            predictions = []
            confidences = []
            
            for idx, row in X.iterrows():
                input_data = np.array([row.values])
                
                # Get predictions from all models
                row_predictions = []
                
                for model_name, model in self.models.items():
                    try:
                        encoded_pred = model.predict(input_data)[0]
                        crop_name = self.label_encoder.inverse_transform([encoded_pred])[0]
                        row_predictions.append(crop_name)
                    except:
                        pass
                
                if row_predictions:
                    # Majority voting
                    crop_counter = Counter(row_predictions)
                    final_crop = crop_counter.most_common(1)[0][0]
                    vote_count = crop_counter.most_common(1)[0][1]
                    confidence = (vote_count / len(self.models)) * 100
                    
                    predictions.append(final_crop)
                    confidences.append(confidence)
                else:
                    predictions.append('Unknown')
                    confidences.append(0)
            
            # Add predictions and confidence to dataframe
            csv_data['predicted_crop'] = predictions
            csv_data['confidence'] = confidences
            
            return csv_data, None
        
        except Exception as e:
            return None, f'Batch prediction error: {str(e)}'

# Global model instance
print("\n🔄 Initializing Crop Recommendation System...")
crop_model = CropRecommendationModel()
print("✅ System initialized!\n")