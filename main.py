# main_fixed.py - Corrected Training Script
"""
Fixed Phishing Detection System - University Project
- Properly handles datasets with both phishing (1) and legitimate (0) emails
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
    print("=== PHISHING DETECTION SYSTEM (FIXED) ===")
    print("University Project - Trust Scores + Smart Ensemble\n")
    
    # Step 1: Load and preprocess data
    print("Step 1: Loading and preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Load your CSV file here - UPDATE THIS PATH!
    csv_file = "datasets/phisingdataset.csv"  # Put your actual CSV filename here
    
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
    
    # Step 2: Properly handle labels
    print("Step 2: Processing labels...")
    
    # Convert labels to binary (1 = phishing, 0 = legitimate)
    df['target'] = df['label'].astype(int)  # Since your labels are already 0/1
    
    print(f"Label distribution:")
    print(f"Phishing (1): {(df['target'] == 1).sum()}")
    print(f"Legitimate (0): {(df['target'] == 0).sum()}")
    
    # Check if we have both classes
    if df['target'].nunique() == 1:
        print("❌ Error: Dataset contains only one class!")
        print("Cannot train a classifier with only one class.")
        print("Please use a dataset that contains both phishing and legitimate emails.")
        return
    
    # Balance the dataset if needed (optional)
    phishing_count = (df['target'] == 1).sum()
    legit_count = (df['target'] == 0).sum()
    
    if abs(phishing_count - legit_count) > max(phishing_count, legit_count) * 0.5:
        print(f"Dataset is imbalanced. Balancing...")
        min_count = min(phishing_count, legit_count)
        
        phishing_df = df[df['target'] == 1].sample(n=min_count, random_state=42)
        legit_df = df[df['target'] == 0].sample(n=min_count, random_state=42)
        
        df = pd.concat([phishing_df, legit_df], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced dataset: {(df['target'] == 1).sum()} phishing, {(df['target'] == 0).sum()} legitimate")
    
    # Step 3: Calculate trust scores (THE INNOVATION!)
    print("\nStep 3: Calculating trust scores...")
    trust_calc = TrustScoreCalculator()
    
    # Learn weights from data - this will work properly now with real data
    learned_weights = trust_calc.learn_optimal_weights(df, 'target')
    
    # Add trust scores to dataframe
    df = trust_calc.add_trust_scores_to_dataframe(df)
    print("Trust scores calculated!")
    
    # Show trust score statistics by class
    print("\nTrust Score Statistics:")
    print("Legitimate emails (0):")
    legitimate_stats = df[df['target'] == 0][['urgency_index', 'authenticity_score', 'manipulation_index']].describe()
    print(legitimate_stats)
    
    print("\nPhishing emails (1):")
    phishing_stats = df[df['target'] == 1][['urgency_index', 'authenticity_score', 'manipulation_index']].describe()
    print(phishing_stats)
    
    # Step 4: Split data properly
    print("\nStep 4: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df, df['target'], test_size=0.3, random_state=42, stratify=df['target']
    )
    print(f"Training: {len(X_train)} (Phishing: {(y_train == 1).sum()}, Legitimate: {(y_train == 0).sum()})")
    print(f"Testing: {len(X_test)} (Phishing: {(y_test == 1).sum()}, Legitimate: {(y_test == 0).sum()})")
    
    # Step 5: Train models
    print("\nStep 5: Training models...")
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
    print("Models trained!")
    
    # Step 6: Final evaluation
    print("\nStep 6: Final Results")
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
    print("-"*70)
    for i in range(min(10, len(X_test))):
        email_text = X_test.iloc[i]['combined_text'][:80] + "..."
        actual = "Phishing" if y_test.iloc[i] == 1 else "Legitimate" 
        predicted = "Phishing" if ensemble_pred[i] == 1 else "Legitimate"
        confidence = ensemble_conf[i]
        correct = "✓" if (y_test.iloc[i] == ensemble_pred[i]) else "✗"
        
        print(f"{correct} Email {i+1}: {email_text}")
        print(f"   Actual: {actual} | Predicted: {predicted} | Confidence: {confidence:.2f}")
        print()
    
    # Calculate accuracy
    accuracy = (y_test == ensemble_pred).mean()
    print(f"Overall Accuracy: {accuracy:.1%}")
    
    # Save models for later use
    trainer.save_models()
    trust_calc.save_weights()
    
    print("\n✅ Training complete! Models saved.")
    print("You can now use 'predict_email.py' to test individual emails.")

def quick_data_check():
    """Quick function to check your data format"""
    csv_file = "datasets/Nazario.csv"  # Update this
    
    try:
        df = pd.read_csv(csv_file)
        print("Dataset Summary:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        print(f"\nSample rows:")
        print(df[['subject', 'label']].head())
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df.isnull().sum())
        
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":

    quick_data_check()
    
    main()