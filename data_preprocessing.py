import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = None
    
    def load_and_clean_data(self, csv_file):
        """Load and clean the email dataset"""
        print("Loading and cleaning data...")
        df = pd.read_csv(csv_file)
        
        # Check dataset info
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Handle missing values
        df['subject'] = df['subject'].fillna('').replace('', 'no_subject')
        df['body'] = df['body'].fillna('').replace('', 'no_body')
        df['sender'] = df['sender'].fillna('unknown_sender')
        
        return df
    
    def clean_text(self, df):
        """Clean and preprocess text fields"""
        print("Cleaning text data...")
        
        # Clean subject column
        df['subject_clean'] = (df['subject']
                              .str.lower()
                              .str.replace(r'http\S+', '', regex=True)  # Remove URLs
                              .str.replace(r'\S+@\S+', '', regex=True)  # Remove emails  
                              .str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Keep only letters/spaces
                              .str.replace(r'\s+', ' ', regex=True)     # Multiple spaces to single
                              .str.strip())
        
        # Clean body 
        df['body_clean'] = (df['body']
                           .str.lower()
                           .str.replace(r'http\S+', '', regex=True)
                           .str.replace(r'\S+@\S+', '', regex=True)
                           .str.replace(r'[^a-zA-Z\s]', '', regex=True)
                           .str.replace(r'\s+', ' ', regex=True)
                           .str.strip())
        
        # Combine cleaned text
        df['combined_text'] = df['subject_clean'] + ' ' + df['body_clean']
        
        # Remove very short messages
        min_length = 5
        df = df[df['combined_text'].str.len() >= min_length]
        
        return df
    
    def extract_basic_features(self, df):
        """Extract basic numerical features"""
        print("Extracting basic features...")
        
        # Text length features
        df['subject_length'] = df['subject_clean'].str.len()
        df['body_length'] = df['body_clean'].str.len()
        df['total_length'] = df['subject_length'] + df['body_length']
        
        # Count features
        df['exclamation_count'] = df['combined_text'].str.count('!')
        df['url_count'] = df.get('urls', pd.Series([0]*len(df))).fillna(0)
        
        # Extract sender domain
        df['sender_domain'] = df['sender'].str.extract(r'@([^>]+)')
        
        return df
    
    def create_tfidf_features(self, texts):
        """Create TF-IDF features"""
        print("Creating TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_text = self.vectorizer.fit_transform(texts)
        return X_text
    
    def save_processed_data(self, df, filename='processed_data.pkl'):
        """Save processed data"""
        with open(filename, 'wb') as f:
            pickle.dump(df, f)
        print(f"Processed data saved to {filename}")

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Load and process data
    df = preprocessor.load_and_clean_data("Nazario.csv")
    df = preprocessor.clean_text(df)
    df = preprocessor.extract_basic_features(df)
    
    # Save processed data
    preprocessor.save_processed_data(df)
    
    print("Preprocessing complete!")