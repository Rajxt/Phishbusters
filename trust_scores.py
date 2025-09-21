#trust_scores.py
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
import pickle

class TrustScoreCalculator:
    def __init__(self):
        self.learned_weights = {}
        self.urgency_words = ['urgent', 'immediate', 'expires', 'deadline', 'act now', 'asap', 'hurry']
        self.time_phrases = ['24 hours', 'limited time', 'expires today', 'today only', 'last chance']
        self.threat_language = ['suspended', 'terminated', 'blocked', 'frozen', 'cancelled', 'close', 'disable']
        self.professional_words = ['sincerely', 'regards', 'thank you', 'best wishes', 'cordially', 'respectfully']
        self.unprofessional_words = ['hey', 'yo', 'sup', 'omg', 'lol', 'wtf']
        self.fear_words = ['security breach', 'virus detected', 'account compromised', 'hacked', 'stolen', 'fraud']
        self.reward_words = ['winner', 'congratulations', 'free money', 'prize', 'lottery', 'reward', 'gift']
        self.authority_words = ['irs', 'government', 'legal action', 'court', 'police', 'fbi', 'federal']
    
    def extract_psychological_features(self, text):
        """Extract all psychological features from text"""
        text_lower = str(text).lower()
        words = text_lower.split()
        total_words = len(words) if len(words) > 0 else 1
        
        features = {
            'urgency_words': sum(1 for word in self.urgency_words if word in text_lower),
            'time_phrases': sum(1 for phrase in self.time_phrases if phrase in text_lower),
            'threat_language': sum(1 for word in self.threat_language if word in text_lower),
            'professional_words': sum(1 for word in self.professional_words if word in text_lower),
            'unprofessional_words': sum(1 for word in self.unprofessional_words if word in text_lower),
            'fear_words': sum(1 for phrase in self.fear_words if phrase in text_lower),
            'reward_words': sum(1 for word in self.reward_words if word in text_lower),
            'authority_words': sum(1 for word in self.authority_words if word in text_lower),
            'total_words': total_words
        }
        
        return features
    
    def learn_optimal_weights(self, df, target_column='label'):
        """
        INNOVATION: Learn optimal weights from correlation analysis
        This replaces your hardcoded weights (×2, ×3, ×4) with data-driven weights!
        """
        print("Learning optimal weights from training data...")
        
        # Extract features for all emails
        all_features = []
        for text in df['combined_text']:
            features = self.extract_psychological_features(text)
            all_features.append(features)
        
        features_df = pd.DataFrame(all_features)
        
        # Convert labels to binary (1 = phishing, 0 = legitimate)
        if target_column in df.columns:
            # Assuming your labels are 'phishing'/'legitimate' or 1/0
            labels = df[target_column].apply(lambda x: 1 if str(x).lower() in ['phishing', '1', 'spam'] else 0)
        else:
            print(f"Warning: Column '{target_column}' not found. Using random labels for demonstration.")
            labels = np.random.choice([0, 1], size=len(df))
        
        # Calculate correlations and learn weights
        learned_weights = {}
        print("\nLearned Feature Correlations:")
        print("-" * 40)
        
        for feature in features_df.columns:
            if feature != 'total_words' and features_df[feature].var() > 0:
                correlation, p_value = pearsonr(features_df[feature], labels)
                if p_value < 0.05:  # Statistically significant
                    learned_weights[feature] = correlation
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
                    print(f"{feature:20s}: {correlation:+.3f} {significance}")
                else:
                    learned_weights[feature] = 0.0  # Not significant
                    print(f"{feature:20s}: {correlation:+.3f} (not significant)")
        
        self.learned_weights = learned_weights
        return learned_weights
    
    def calculate_trust_scores_with_learned_weights(self, text):
        """Calculate trust scores using LEARNED weights instead of hardcoded ones"""
        features = self.extract_psychological_features(text)
        
        # Use learned weights instead of arbitrary multipliers
        urgency_index = (
            features['urgency_words'] * self.learned_weights.get('urgency_words', 0) +
            features['time_phrases'] * self.learned_weights.get('time_phrases', 0) +
            features['threat_language'] * self.learned_weights.get('threat_language', 0)
        ) / features['total_words']
        
        authenticity_score = (
            features['professional_words'] * abs(self.learned_weights.get('professional_words', 0)) -
            features['unprofessional_words'] * abs(self.learned_weights.get('unprofessional_words', 0))
        ) / features['total_words']
        
        manipulation_index = (
            features['fear_words'] * self.learned_weights.get('fear_words', 0) +
            features['reward_words'] * self.learned_weights.get('reward_words', 0) +
            features['authority_words'] * self.learned_weights.get('authority_words', 0)
        ) / features['total_words']
        
        return {
            'urgency_index': urgency_index,
            'authenticity_score': authenticity_score,
            'manipulation_index': manipulation_index
        }
    
    def add_trust_scores_to_dataframe(self, df):
        """Add trust scores to the dataframe"""
        print("Calculating trust scores with learned weights...")
        
        trust_scores = []
        for text in df['combined_text']:
            scores = self.calculate_trust_scores_with_learned_weights(text)
            trust_scores.append(scores)
        
        trust_df = pd.DataFrame(trust_scores)
        
        # Add to original dataframe
        for col in trust_df.columns:
            df[col] = trust_df[col]
        
        return df
    
    def save_weights(self, filename='learned_weights.pkl'):
        """Save learned weights"""
        with open(filename, 'wb') as f:
            pickle.dump(self.learned_weights, f)
        print(f"Learned weights saved to {filename}")

if __name__ == "__main__":
    # Load processed data
    with open('processed_data.pkl', 'rb') as f:
        df = pickle.load(f)
    
    # Initialize trust score calculator
    trust_calculator = TrustScoreCalculator()
    
    # Learn optimal weights (THIS IS YOUR INNOVATION!)
    learned_weights = trust_calculator.learn_optimal_weights(df)
    
    # Add trust scores using learned weights
    df_with_trust = trust_calculator.add_trust_scores_to_dataframe(df)
    
    # Save results
    trust_calculator.save_weights()
    with open('data_with_trust_scores.pkl', 'wb') as f:
        pickle.dump(df_with_trust, f)
    
    print("\n" + "="*50)
    print("TRUST SCORE EXAMPLES:")
    print("="*50)
    print(df_with_trust[['urgency_index', 'authenticity_score', 'manipulation_index']].describe())