# predict_email.py - Test emails against your trained model
"""
Phishing Detection System - Email Predictor
Use this to test individual emails and see if they're phishing or legitimate
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
    
    def prepare_email(self, sender, subject, body, urls=""):
        """Prepare a single email for prediction"""
        # Create DataFrame with single email
        email_data = pd.DataFrame([{
            'sender': sender,
            'subject': subject if subject else 'no_subject',
            'body': body if body else 'no_body',
            'urls': urls,
            'label': 'unknown'  # placeholder
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
    
    def analyze_email_text(self, email_text):
        """Analyze a single email provided as text"""
        # Parse email text (simple parsing - you can enhance this)
        lines = email_text.strip().split('\n')
        
        sender = "unknown@example.com"
        subject = ""
        body = ""
        
        # Simple parsing logic
        in_body = False
        for line in lines:
            if line.lower().startswith('from:'):
                sender = line[5:].strip()
            elif line.lower().startswith('subject:'):
                subject = line[8:].strip()
            elif line.strip() == "" and subject:
                in_body = True
            elif in_body:
                body += line + " "
        
        # If no structured format, treat entire text as body
        if not body:
            body = email_text
        
        return self.analyze_email(sender, subject, body)
    
    def analyze_email(self, sender, subject, body, urls=""):
        """Full analysis of an email"""
        # Prepare email
        email_data = self.prepare_email(sender, subject, body, urls)
        
        # Get prediction
        result = self.predict_email(email_data)
        
        # Display results
        print("\n" + "="*60)
        print("PHISHING DETECTION ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\n📧 EMAIL DETAILS:")
        print(f"From: {sender}")
        print(f"Subject: {subject[:100]}...")
        print(f"Body preview: {body[:200]}...")
        
        print(f"\n🔍 ANALYSIS RESULTS:")
        print(f"Verdict: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        
        print(f"\n📊 MODEL PREDICTIONS:")
        print(f"Naive Bayes: {result['nb_probability']:.1%} phishing probability")
        print(f"Logistic Regression: {result['lr_probability']:.1%} phishing probability")
        
        print(f"\n🧠 PSYCHOLOGICAL TRUST SCORES:")
        print(f"Urgency Index: {result['trust_scores']['urgency']:.3f}")
        print(f"Authenticity Score: {result['trust_scores']['authenticity']:.3f}")
        print(f"Manipulation Index: {result['trust_scores']['manipulation']:.3f}")
        
        print(f"\n🎯 INTERPRETATION:")
        if result['prediction'] == 'PHISHING':
            print("⚠️  This email shows characteristics of a phishing attempt!")
            print("Reasons:")
            if result['trust_scores']['urgency'] > 0.1:
                print("  • High urgency language detected")
            if result['trust_scores']['manipulation'] > 0.1:
                print("  • Manipulative psychological tactics present")
            if result['trust_scores']['authenticity'] < 0:
                print("  • Low authenticity score")
        else:
            print("✅ This email appears to be legitimate.")
            print("However, always verify sender addresses and links before clicking!")
        
        print("="*60)
        
        return result


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
        print("1. Test a sample phishing email")
        print("2. Test a sample legitimate email")
        print("3. Enter your own email")
        print("4. Load email from file")
        print("5. Quick test (just subject and body)")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ")
        
        if choice == '1':
            # Sample phishing email
            sender = "security@amaz0n-verify.com"
            subject = "Urgent: Your Account Will Be Suspended!"
            body = """Dear Customer,

We have detected unusual activity on your account. Your account will be SUSPENDED within 24 hours unless you verify your information immediately.

Click here to verify your account: http://bit.ly/verify-account

This is a time-sensitive matter. Act now to avoid losing access to your account and all your purchases.

If you don't respond within 24 hours, your account will be permanently deleted.

Amazon Security Team
(This is an automated message, do not reply)"""
            
            predictor.analyze_email(sender, subject, body, "http://bit.ly/verify-account")
            
        elif choice == '2':
            # Sample legitimate email
            sender = "noreply@github.com"
            subject = "Your GitHub repository has a new star"
            body = """Hi there,

Someone just starred your repository 'phishing-detector'. Your project now has 5 stars.

Keep up the great work!

Best regards,
The GitHub Team

To manage your notification settings, visit your account preferences."""
            
            predictor.analyze_email(sender, subject, body)
            
        elif choice == '3':
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
            urls = input("URLs in email (comma-separated, or press Enter for none): ")
            
            predictor.analyze_email(sender, subject, body, urls)
            
        elif choice == '4':
            filename = input("Enter filename: ")
            try:
                with open(filename, 'r') as f:
                    email_text = f.read()
                predictor.analyze_email_text(email_text)
            except Exception as e:
                print(f"Error reading file: {e}")
                
        elif choice == '5':
            print("\nQuick Test Mode:")
            subject = input("Subject: ")
            body = input("Body: ")
            predictor.analyze_email("unknown@example.com", subject, body)
            
        elif choice == '6':
            print("Goodbye!")
            break
        else:
            print("Invalid choice!")


def batch_test():
    """Test multiple emails from a CSV file"""
    predictor = PhishingPredictor()
    
    if not predictor.load_models():
        return
    
    csv_file = input("Enter CSV filename with emails to test: ")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} emails")
        
        results = []
        for idx, row in df.iterrows():
            print(f"\nTesting email {idx+1}/{len(df)}...")
            
            email_data = predictor.prepare_email(
                row.get('sender', 'unknown'),
                row.get('subject', ''),
                row.get('body', ''),
                row.get('urls', '')
            )
            
            result = predictor.predict_email(email_data)
            results.append({
                'index': idx,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'actual': row.get('label', 'unknown')
            })
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv('prediction_results.csv', index=False)
        print(f"\nResults saved to prediction_results.csv")
        
        # Show accuracy if labels are available
        if 'actual' in results_df.columns and results_df['actual'].iloc[0] != 'unknown':
            correct = sum(
                (results_df['prediction'] == 'PHISHING') & (results_df['actual'].str.lower().isin(['phishing', 'spam', '1'])) |
                (results_df['prediction'] == 'LEGITIMATE') & (results_df['actual'].str.lower().isin(['legitimate', 'ham', '0']))
            )
            accuracy = correct / len(results_df)
            print(f"Accuracy: {accuracy:.1%}")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("="*60)
    print("PHISHBUSTERS - Email Phishing Detection System")
    print("="*60)
    
    print("\nSelect mode:")
    print("1. Interactive mode (test individual emails)")
    print("2. Batch mode (test multiple emails from CSV)")
    
    mode = input("\nEnter choice (1-2): ")
    
    if mode == '1':
        interactive_mode()
    elif mode == '2':
        batch_test()
    else:
        print("Invalid choice!")