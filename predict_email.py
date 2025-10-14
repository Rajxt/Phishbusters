# predict_email.py - DEBUGGING VERSION
"""
Phishing Detection System - Email Predictor with Enhanced Debugging
"""

import pandas as pd
import numpy as np
import pickle
import os
from data_preprocessing import DataPreprocessor
from trust_scores import TrustScoreCalculator

class PhishingPredictor:
    def __init__(self):
        self.models = None
        self.trust_calculator = None
        self.preprocessor = None
        
    def load_models(self):
        """Load trained models and components"""
        try:
            # Load trained models
            with open('trained_models.pkl', 'rb') as f:
                self.models = pickle.load(f)
            print("✓ Models loaded successfully")
            
            # Load trust score calculator with learned weights
            self.trust_calculator = TrustScoreCalculator()
            if os.path.exists('learned_weights.pkl'):
                with open('learned_weights.pkl', 'rb') as f:
                    self.trust_calculator.learned_weights = pickle.load(f)
                print("✓ Trust score weights loaded")
            else:
                print("⚠ No learned weights found, using defaults")
            
            # Initialize preprocessor
            self.preprocessor = DataPreprocessor()
            self.preprocessor.vectorizer = self.models['vectorizer']
            
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Make sure you've run main.py first to train the models!")
            return False
    
    def load_csv_with_encoding(self, csv_file):
        """Load CSV file with automatic encoding detection"""
        print(f"\n🔍 Attempting to load: {csv_file}")
        encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                print(f"✓ Successfully loaded file with {encoding} encoding")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {df.columns.tolist()}")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"   Error with {encoding}: {e}")
                continue
        
        # If all standard encodings fail, try with error handling
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
            print("✓ Loaded file with UTF-8 encoding (ignoring errors)")
            print(f"   Shape: {df.shape}")
            print(f"   Columns: {df.columns.tolist()}")
            return df
        except:
            # Last resort
            try:
                import chardet
                with open(csv_file, 'rb') as f:
                    result = chardet.detect(f.read(100000))
                    detected_encoding = result['encoding']
                    print(f"   Detected encoding: {detected_encoding}")
                    return pd.read_csv(csv_file, encoding=detected_encoding)
            except:
                df = pd.read_csv(csv_file, encoding='latin-1', errors='replace')
                print("✓ Loaded with latin-1 encoding (with replacements)")
                print(f"   Shape: {df.shape}")
                print(f"   Columns: {df.columns.tolist()}")
                return df
    
    def prepare_email(self, sender, subject, body, urls=""):
        """Prepare a single email for prediction"""
        # Create DataFrame with single email
        email_data = pd.DataFrame([{
            'sender': sender,
            'subject': subject if subject else 'no_subject',
            'body': body if body else 'no_body',
            'urls': urls,
            'label': 0  # placeholder, won't be used
        }])
        
        # Preprocess
        email_data = self.preprocessor.clean_text(email_data)
        email_data = self.preprocessor.extract_basic_features(email_data)
        
        # Calculate trust scores
        email_data = self.trust_calculator.add_trust_scores_to_dataframe(email_data)
        
        return email_data
    
    def predict_email(self, email_data):
        """Make prediction on preprocessed email"""
        # Create TF-IDF features using existing vectorizer
        X_text = self.preprocessor.vectorizer.transform(email_data['combined_text'])
        
        # Numerical features
        numerical_features = email_data[['subject_length', 'body_length', 'url_count', 'exclamation_count']]
        
        # Trust scores
        trust_scores = email_data[['urgency_index', 'authenticity_score', 'manipulation_index']]
        
        # Combine features
        combined_dense_features = np.hstack([numerical_features.values, trust_scores.values])
        combined_dense_features_scaled = self.models['scaler'].transform(combined_dense_features)
        
        from scipy.sparse import hstack
        X_combined = hstack([X_text, combined_dense_features_scaled])
        
        # Get predictions from both models
        nb_proba = self.models['nb_model'].predict_proba(X_combined)[:, 1][0]
        lr_proba = self.models['lr_model'].predict_proba(X_combined)[:, 1][0]
        
        # Smart ensemble decision
        nb_confidence = abs(nb_proba - 0.5) * 2
        lr_confidence = abs(lr_proba - 0.5) * 2
        threshold = self.models['best_threshold']
        
        if (nb_confidence > threshold and lr_confidence > threshold and
            (nb_proba > 0.5) == (lr_proba > 0.5)):
            final_pred = 1 if nb_proba > 0.5 else 0
            final_conf = max(nb_confidence, lr_confidence)
        elif nb_confidence > lr_confidence:
            final_pred = 1 if nb_proba > 0.5 else 0
            final_conf = nb_confidence
        else:
            final_pred = 1 if lr_proba > 0.5 else 0
            final_conf = lr_confidence
        
        return {
            'prediction': 'PHISHING' if final_pred == 1 else 'LEGITIMATE',
            'confidence': final_conf,
            'nb_probability': nb_proba,
            'lr_probability': lr_proba,
            'trust_scores': {
                'urgency': email_data['urgency_index'].values[0],
                'authenticity': email_data['authenticity_score'].values[0],
                'manipulation': email_data['manipulation_index'].values[0]
            }
        }


def batch_test():
    """Test multiple emails from a CSV file - ENHANCED DEBUG VERSION"""
    predictor = PhishingPredictor()
    
    if not predictor.load_models():
        return
    
    csv_file = input("Enter CSV filename with emails to test: ").strip()
    
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return
    
    try:
        # Load CSV with encoding detection
        df = predictor.load_csv_with_encoding(csv_file)
        print(f"\n✓ Loaded {len(df)} rows from CSV")
        
        # DEBUG: Show first few rows
        print("\n📊 First 3 rows of data:")
        print(df.head(3))
        
        # DEBUG: Check for required columns
        print("\n🔍 Checking for required columns:")
        required_cols = ['sender', 'subject', 'body']
        for col in required_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"   ✓ '{col}' found ({non_null}/{len(df)} non-null values)")
            else:
                print(f"   ⚠️  '{col}' NOT found")
        
        # Handle missing columns
        if 'sender' not in df.columns:
            print("   → Adding default 'sender' column")
            df['sender'] = 'unknown@example.com'
        if 'subject' not in df.columns:
            print("   → Adding default 'subject' column")
            df['subject'] = ''
        if 'body' not in df.columns:
            print("   → Adding default 'body' column")
            df['body'] = ''
        if 'urls' not in df.columns:
            df['urls'] = ''
        
        # Fill NaN values
        df['sender'] = df['sender'].fillna('unknown@example.com')
        df['subject'] = df['subject'].fillna('')
        df['body'] = df['body'].fillna('')
        df['urls'] = df['urls'].fillna('')
        
        # Clean any encoding issues in the text data
        print("\n🧹 Cleaning text data...")
        for col in ['subject', 'body']:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: str(x).encode('ascii', 'ignore').decode('ascii') if pd.notna(x) else ''
                )
        
        print(f"✓ Data preparation complete. Processing {len(df)} emails...\n")
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        errors = []
        
        for idx, row in df.iterrows():
            print(f"Processing email {idx+1}/{len(df)}...", end=' ')
            
            try:
                email_data = predictor.prepare_email(
                    row['sender'],
                    row['subject'],
                    row['body'],
                    row['urls']
                )
                
                result = predictor.predict_email(email_data)
                
                # Store results
                prediction_result = {
                    'index': idx,
                    'subject': row['subject'][:50] if row['subject'] else 'No subject',
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'nb_prob': result['nb_probability'],
                    'lr_prob': result['lr_probability']
                }
                
                # Check if we have actual labels for accuracy calculation
                if 'label' in row and pd.notna(row['label']):
                    # Handle different label formats
                    label_str = str(row['label']).lower().strip()
                    if label_str in ['1', 'phishing', 'spam', 'true']:
                        actual_label = 1
                    elif label_str in ['0', 'legitimate', 'ham', 'false']:
                        actual_label = 0
                    else:
                        actual_label = int(row['label']) if str(row['label']).isdigit() else None
                    
                    if actual_label is not None:
                        prediction_result['actual'] = 'PHISHING' if actual_label == 1 else 'LEGITIMATE'
                        
                        # Check if prediction is correct
                        predicted_label = 1 if result['prediction'] == 'PHISHING' else 0
                        if predicted_label == actual_label:
                            correct_predictions += 1
                            print("✓")
                        else:
                            print("✗")
                        total_predictions += 1
                    else:
                        prediction_result['actual'] = 'unknown'
                        print("?")
                else:
                    prediction_result['actual'] = 'unknown'
                    print("?")
                
                results.append(prediction_result)
                
            except Exception as e:
                error_msg = f"Email {idx}: {str(e)}"
                errors.append(error_msg)
                print(f"❌ Error: {str(e)[:50]}")
                continue
        
        # Save results
        if len(results) > 0:
            results_df = pd.DataFrame(results)
            output_file = 'prediction_results.csv'
            results_df.to_csv(output_file, index=False)
            print(f"\n✓ Results saved to {output_file}")
        else:
            print("\n❌ No emails were successfully processed!")
            if errors:
                print("\nErrors encountered:")
                for error in errors[:5]:  # Show first 5 errors
                    print(f"  - {error}")
            return
        
        # Show summary
        print("\n" + "="*60)
        print("BATCH TESTING SUMMARY")
        print("="*60)
        print(f"Total emails in CSV: {len(df)}")
        print(f"Successfully processed: {len(results)}")
        print(f"Failed to process: {len(errors)}")
        
        if total_predictions > 0:
            accuracy = (correct_predictions / total_predictions) * 100
            
            # Calculate confusion matrix components
            true_positive = sum(1 for r in results if r.get('actual') == 'PHISHING' and r['prediction'] == 'PHISHING')  # TP
            false_negative = sum(1 for r in results if r.get('actual') == 'PHISHING' and r['prediction'] == 'LEGITIMATE')  # FN
            true_negative = sum(1 for r in results if r.get('actual') == 'LEGITIMATE' and r['prediction'] == 'LEGITIMATE')  # TN
            false_positive = sum(1 for r in results if r.get('actual') == 'LEGITIMATE' and r['prediction'] == 'PHISHING')  # FP
            
            # Calculate metrics
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # False Positive Rate and False Negative Rate
            fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) > 0 else 0
            fnr = false_negative / (false_negative + true_positive) if (false_negative + true_positive) > 0 else 0
            
            # Display Results
            print(f"\n📈 PERFORMANCE METRICS")
            print("-" * 60)
            print(f"Accuracy:        {accuracy:.1f}% ({correct_predictions}/{total_predictions} correct)")
            print(f"Precision:       {precision:.3f} ({precision*100:.1f}%)")
            print(f"Recall:          {recall:.3f} ({recall*100:.1f}%)")
            print(f"F1-Score:        {f1_score:.3f}")
            print(f"FP Rate:         {fpr:.3f} ({fpr*100:.1f}%)")
            print(f"FN Rate:         {fnr:.3f} ({fnr*100:.1f}%)")
            
            print("\n📊 CONFUSION MATRIX")
            print("-" * 60)
            print("                    Predicted")
            print("                 Legit      Phish")
            print(f"Actual Legit  |  {true_negative:4d} (TN)  {false_positive:4d} (FP)")
            print(f"       Phish  |  {false_negative:4d} (FN)  {true_positive:4d} (TP)")
            
            print("\n📉 ERROR ANALYSIS")
            print("-" * 60)
            print(f"True Positives (TP):   {true_positive:3d} - Correctly identified phishing")
            print(f"True Negatives (TN):   {true_negative:3d} - Correctly identified legitimate")
            print(f"False Positives (FP):  {false_positive:3d} - Legitimate marked as phishing")
            print(f"False Negatives (FN):  {false_negative:3d} - Phishing marked as legitimate ⚠️")
        else:
            print("\n⚠️  No labeled data found - cannot calculate accuracy")
        
        # Show confidence statistics
        if len(results) > 0:
            avg_confidence = np.mean([r['confidence'] for r in results])
            print(f"\nAverage confidence: {avg_confidence:.1%}")
            
            # Show prediction distribution
            phishing_pred = sum(1 for r in results if r['prediction'] == 'PHISHING')
            legit_pred = sum(1 for r in results if r['prediction'] == 'LEGITIMATE')
            print(f"\nPredictions: {phishing_pred} Phishing, {legit_pred} Legitimate")
            
            # Show some examples
            print("\nSample predictions (first 5):")
            for i in range(min(5, len(results))):
                r = results[i]
                actual_str = f" | Actual: {r.get('actual', 'unknown'):10s}" if 'actual' in r else ""
                print(f"  {i+1}. {r['subject'][:40]:40s} → {r['prediction']:10s} (Conf: {r['confidence']:.1%}){actual_str}")
        
        # Show errors if any
        if errors:
            print(f"\n⚠️  {len(errors)} errors occurred during processing")
            print("First few errors:")
            for error in errors[:3]:
                print(f"  - {error}")
            
    except Exception as e:
        print(f"\n❌ Error during batch testing: {e}")
        import traceback
        traceback.print_exc()


def interactive_mode():
    """Interactive mode for testing emails"""
    predictor = PhishingPredictor()
    
    if not predictor.load_models():
        return
    
    print("\n" + "="*60)
    print("PHISHING EMAIL DETECTOR - INTERACTIVE MODE")
    print("="*60)
    
    while True:
        print("\nChoose an option:")
        print("1. Test a quick email (just subject and body)")
        print("2. Test with full details (sender, subject, body)")
        print("3. Exit")
        
        choice = input("\nEnter choice (1-3): ")
        
        if choice == '1':
            print("\nQuick Test Mode:")
            subject = input("Subject: ")
            body = input("Body: ")
            
            try:
                email_data = predictor.prepare_email("unknown@example.com", subject, body)
                result = predictor.predict_email(email_data)
                
                print("\n" + "="*60)
                print(f"🔍 VERDICT: {result['prediction']}")
                print(f"📊 Confidence: {result['confidence']:.1%}")
                print(f"   NB Probability: {result['nb_probability']:.1%}")
                print(f"   LR Probability: {result['lr_probability']:.1%}")
                print("="*60)
            except Exception as e:
                print(f"❌ Error: {e}")
            
        elif choice == '2':
            print("\nEnter email details:")
            sender = input("From (email address): ")
            subject = input("Subject: ")
            print("Body (enter 'END' on a new line when done):")
            body_lines = []
            while True:
                line = input()
                if line == 'END':
                    break
                body_lines.append(line)
            body = '\n'.join(body_lines)
            urls = input("URLs in email (comma-separated, or press Enter): ")
            
            try:
                email_data = predictor.prepare_email(sender, subject, body, urls)
                result = predictor.predict_email(email_data)
                
                print("\n" + "="*60)
                print(f"🔍 VERDICT: {result['prediction']}")
                print(f"📊 Confidence: {result['confidence']:.1%}")
                print(f"   NB Probability: {result['nb_probability']:.1%}")
                print(f"   LR Probability: {result['lr_probability']:.1%}")
                print("\n🧠 Trust Scores:")
                print(f"   Urgency: {result['trust_scores']['urgency']:.3f}")
                print(f"   Authenticity: {result['trust_scores']['authenticity']:.3f}")
                print(f"   Manipulation: {result['trust_scores']['manipulation']:.3f}")
                print("="*60)
            except Exception as e:
                print(f"❌ Error: {e}")
            
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    print("="*60)
    print("PHISHBUSTERS - Email Phishing Detection System")
    print("="*60)
    
    print("\nSelect mode:")
    print("1. Interactive mode (test individual emails)")
    print("2. Batch mode (test multiple emails from CSV)")
    
    mode = input("\nEnter choice (1-2): ").strip()
    
    if mode == '1':
        interactive_mode()
    elif mode == '2':
        batch_test()
    else:
        print("Invalid choice!")