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
    
    # Create target variable - since all your emails are phishing, create some artificial legitimate ones for training
    print("⚠️  Dataset contains only phishing emails. Creating balanced dataset for training...")
    
    # For demonstration purposes, let's create a balanced dataset
    # In a real scenario, you'd have both phishing and legitimate emails
    phishing_df = df.copy()
    
    # Create some "legitimate" emails by modifying phishing ones (this is just for training)
    # In practice, you'd have real legitimate emails
    legitimate_df = df.sample(n=min(500, len(df)//2), random_state=42).copy()
    
    # Modify trust scores to simulate legitimate emails (lower threat indicators)
    legitimate_df['urgency_index'] = legitimate_df['urgency_index'] * 0.3
    legitimate_df['manipulation_index'] = legitimate_df['manipulation_index'] * 0.2
    legitimate_df['authenticity_score'] = legitimate_df['authenticity_score'] + 0.5
    
    # Create labels
    phishing_df['target'] = 1
    legitimate_df['target'] = 0
    
    # Combine datasets
    balanced_df = pd.concat([phishing_df, legitimate_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    y = balanced_df['target']
    print(f"Balanced dataset: {y.sum()} phishing, {len(y)-y.sum()} legitimate")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"Training: {len(X_train)}, Testing: {len(X_test)}\n")
    
    # Step 4: Train models
    print("Step 4: Training models...")
    trainer = PhishingModelTrainer()
    
    # Prepare features (training data)
    X_train_features, _, _, _ = trainer.prepare_features(X_train, is_training=True)
    
    # Prepare features (test data) - using existing scaler and vectorizer
    X_test_features, _, _, _ = trainer.prepare_features(X_test, is_training=False)
    
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
    print("\n💡 Note: This demo used artificial legitimate emails. In practice, use real legitimate emails for better results.")

if __name__ == "__main__":
    main()