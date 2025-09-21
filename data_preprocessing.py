import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = None

    def load_and_clean_data(self, csv_file):
        """Load and clean the email dataset"""
        print("Loading and cleaning data...")
        df = pd.read_csv(csv_file)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Missing values:\n{df.isnull().sum()}")

        # Handle missing values
        df['subject'] = df['subject'].fillna('no_subject').replace('', 'no_subject')
        df['body'] = df['body'].fillna('no_body').replace('', 'no_body')
        df['sender'] = df['sender'].fillna('unknown_sender')
        return df

    def clean_text(self, df):
        """Clean and preprocess text fields"""
        print("Cleaning text data...")
        df['subject_clean'] = (df['subject']
            .str.lower()
            .str.replace(r'http\S+', '', regex=True)
            .str.replace(r'\S+@\S+', '', regex=True)
            .str.replace(r'[^a-zA-Z\s]', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip())
        
        df['body_clean'] = (df['body']
            .str.lower()
            .str.replace(r'http\S+', '', regex=True)
            .str.replace(r'\S+@\S+', '', regex=True)
            .str.replace(r'[^a-zA-Z\s]', '', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip())

        df['combined_text'] = df['subject_clean'] + ' ' + df['body_clean']

        # Remove very short messages
        df = df[df['combined_text'].str.len() >= 5].reset_index(drop=True)
        return df

    def create_tfidf_features(self, text_data):
        """Create TF-IDF features from text data"""
        print("Creating TF-IDF features...")
        
        if self.vectorizer is None:
            # Create and fit new vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2
            )
            X_tfidf = self.vectorizer.fit_transform(text_data)
        else:
            # Use existing vectorizer (for test data)
            X_tfidf = self.vectorizer.transform(text_data)
        
        print(f"TF-IDF matrix shape: {X_tfidf.shape}")
        return X_tfidf

    def extract_basic_features(self, df):
        """Extract basic numerical features"""
        print("Extracting basic features...")
        df['subject_length'] = df['subject_clean'].str.len()
        df['body_length'] = df['body_clean'].str.len()
        df['total_length'] = df['subject_length'] + df['body_length']
        df['exclamation_count'] = df['combined_text'].str.count('!')
        # Robust URL count extraction
        if 'urls' in df.columns:
            df['url_count'] = df['urls'].replace('', 0).fillna(0).astype(str).str.count('http|www')
        else:
            df['url_count'] = 0
        # Sender domain extraction
        df['sender_domain'] = df['sender'].str.extract(r'@([^>]+)')
        return df

    def save_processed_data(self, df, filename='processed_data.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(df, f)
        print(f"Processed data saved to {filename}")

    def inspect_labels(self, df):
        print("Label distribution and unique values:")
        print(df['label'].value_counts())
        print(df['label'].unique())

    def balance_labels(self, df, phishing_labels=['phishing','spam','1'], legit_labels=['legitimate','ham','0']):
        """Ensure both phishing and legitimate emails exist, and balance them if needed."""
        # Normalize labels
        df['label_clean'] = df['label'].astype(str).str.strip().str.lower()
        df['target'] = df['label_clean'].apply(lambda x: 1 if x in phishing_labels else 0)
        phishing_df = df[df['target'] == 1]
        legit_df = df[df['target'] == 0]
        print(f"Phishing samples: {len(phishing_df)}, Legitimate samples: {len(legit_df)}")
        # If classes are imbalanced, balance them (downsample to minority class count)
        min_count = min(len(phishing_df), len(legit_df))
        phishing_df_bal = phishing_df.sample(n=min_count, random_state=42)
        legit_df_bal = legit_df.sample(n=min_count, random_state=42)
        balanced_df = pd.concat([phishing_df_bal, legit_df_bal], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced dataset: {len(balanced_df)} samples ({min_count} phishing, {min_count} legitimate)")
        return balanced_df

if __name__ == "__main__":
    csv_file = "datasets/Nazario.csv" # update this if needed
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        exit()
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_clean_data(csv_file)
    df = preprocessor.clean_text(df)
    df = preprocessor.extract_basic_features(df)
    preprocessor.inspect_labels(df) # Print label info
    df_bal = preprocessor.balance_labels(df)
    preprocessor.save_processed_data(df_bal)
    print("Preprocessing complete!")