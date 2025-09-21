# main.py - Simple University Project Version
"""
Phishing Detection System - University Project
- Load data → Preprocess → Calculate trust scores → Train models → Test
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# Import our modules
from data_preprocessing import DataPreprocessor
from trust_scores import TrustScoreCalculator
from model_training import PhishingModelTrainer
from evaluation import ModelEvaluator

def main():
    print("=== PHISHING DETECTION SYSTEM ===")
    print("University Project - Trust Scores + Smart Ensemble\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Load your CSV file here - UPDATE THIS PATH!
    csv_file = "datasets/Nazario.csv"  # Put your actual CSV filename here
    
    # Check if file exists
    import os
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found!")
        print("📁 Current directory files:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"   - {f}")
        print("\n💡 Update the csv_file variable with your actual CSV filename")
        return
    
    df = preprocessor.load_and_clean_data(csv_file)
    df = preprocessor.clean_text(df)
    df = preprocessor.extract_basic_features(df)
    print(f"Loaded {len(df)} emails\n")
    
    # Step 2: Calculate trust scores (THE INNOVATION!)
    print("Step 2: Calculating trust scores...")
    trust_calc = TrustScoreCalculator()
    
    # Learn weights from data
    learned_weights = trust_calc.learn_optimal_weights(df, 'label')
    
    # Add trust scores to dataframe
    df = trust_calc.add_trust_scores_to_dataframe(df)
    print("Trust scores calculated!\n")
    
    # Step 3: Prepare data for training
    print("Step 3: Preparing features...")
    
    # Create target variable (adjust based on your label column)
    y = df['label'].apply(lambda x: 1 if str(x).lower() in ['phishing', '1', 'spam'] else 0)
    print(f"Labels: {y.sum()} phishing, {len(y)-y.sum()} legitimate")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}\n")
    
    # Step 4: Train models
    print("Step 4: Training models...")
    trainer = PhishingModelTrainer()
    
    # Prepare features
    X_train_features, _, _, _ = trainer.prepare_features(X_train)
    X_test_features, _, _, _ = trainer.prepare_features(X_test)
    
    # Train individual models
    nb_model, lr_model = trainer.train_individual_models(
        X_train_features, y_train, X_test_features, y_test
    )
    
    # Optimize ensemble
    best_threshold = trainer.optimize_ensemble_threshold(X_test_features, y_test)
    print("Models trained!\n")
    
    # Step 5: Final evaluation
    print("Step 5: Final Results")
    print("="*40)
    
    # Get ensemble predictions
    ensemble_pred, ensemble_conf = trainer.smart_ensemble_predict(X_test_features, best_threshold)
    
    # Print results
    f1 = f1_score(y_test, ensemble_pred)
    print(f"Final F1-Score: {f1:.3f}")
    print(f"Average Confidence: {ensemble_conf.mean():.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, ensemble_pred, target_names=['Legitimate', 'Phishing']))
    
    # Show some example predictions
    print("\nExample Predictions:")
    print("-"*50)
    for i in range(min(5, len(X_test))):
        email_text = X_test.iloc[i]['combined_text'][:100] + "..."
        actual = "Phishing" if y_test.iloc[i] == 1 else "Legitimate" 
        predicted = "Phishing" if ensemble_pred[i] == 1 else "Legitimate"
        confidence = ensemble_conf[i]
        
        print(f"Email {i+1}: {email_text}")
        print(f"Actual: {actual} | Predicted: {predicted} | Confidence: {confidence:.2f}")
        print()
    
    # Save models for later use
    trainer.save_models()
    trust_calc.save_weights()
    
    print("Done! Models saved to 'trained_models.pkl' and 'learned_weights.pkl'")

if __name__ == "__main__":
    main()