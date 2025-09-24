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
        
        # Ensure label column exists and is properly formatted
        if 'label' not in df.columns:
            raise ValueError("Dataset must have a 'label' column")
            
        # Convert labels to integers if they're not already
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])  # Remove rows with invalid labels
        df['label'] = df['label'].astype(int)
        
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

    def check_label_distribution(self, df):
        """Check and report label distribution"""
        print("\n" + "="*40)
        print("LABEL DISTRIBUTION ANALYSIS")
        print("="*40)
        
        if 'label' not in df.columns:
            print("❌ No 'label' column found!")
            return False
            
        label_counts = df['label'].value_counts().sort_index()
        print(f"Label distribution:")
        for label, count in label_counts.items():
            label_name = "Phishing" if label == 1 else "Legitimate"
            percentage = (count / len(df)) * 100
            print(f"  {label} ({label_name}): {count} emails ({percentage:.1f}%)")
        
        total_classes = df['label'].nunique()
        print(f"\nTotal classes: {total_classes}")
        
        if total_classes == 1:
            print("⚠️  WARNING: Only one class found in the dataset!")
            print("   This will prevent proper training. You need both phishing and legitimate emails.")
            return False
        elif total_classes == 2:
            print("✅ Good: Both classes present for training.")
            return True
        else:
            print(f"⚠️  WARNING: {total_classes} classes found. Expected 2 (phishing/legitimate).")
            return False

    def balance_dataset_intelligently(self, df, max_ratio=3.0):
        """
        Intelligently balance the dataset without creating artificial data
        max_ratio: maximum allowed ratio between majority and minority class
        """
        print("\n" + "="*40)
        print("DATASET BALANCING")
        print("="*40)
        
        if not self.check_label_distribution(df):
            return df
        
        class_counts = df['label'].value_counts()
        majority_count = class_counts.max()
        minority_count = class_counts.min()
        current_ratio = majority_count / minority_count
        
        print(f"Current class ratio: {current_ratio:.2f}:1")
        
        if current_ratio <= max_ratio:
            print("✅ Dataset is reasonably balanced. No changes needed.")
            return df
        
        print(f"Dataset is imbalanced (ratio > {max_ratio}:1). Applying intelligent balancing...")
        
        # Find majority and minority classes
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()
        
        # Calculate target counts
        target_majority_count = int(minority_count * max_ratio)
        
        print(f"Downsampling majority class ({majority_class}) from {majority_count} to {target_majority_count}")
        
        # Separate classes
        minority_df = df[df['label'] == minority_class]
        majority_df = df[df['label'] == majority_class]
        
        # Downsample majority class
        majority_df_downsampled = majority_df.sample(n=target_majority_count, random_state=42)
        
        # Combine
        balanced_df = pd.concat([minority_df, majority_df_downsampled], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Balanced dataset created:")
        print(f"   Class {minority_class}: {len(minority_df)} emails")
        print(f"   Class {majority_class}: {len(majority_df_downsampled)} emails")
        print(f"   Total: {len(balanced_df)} emails")
        print(f"   New ratio: {len(majority_df_downsampled)/len(minority_df):.2f}:1")
        
        return balanced_df

    def save_processed_data(self, df, filename='processed_data.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(df, f)
        print(f"Processed data saved to {filename}")

if __name__ == "__main__":
    csv_file = "datasets/phishingdataset.csv"  # update this if needed
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        exit()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.load_and_clean_data(csv_file)
    df = preprocessor.clean_text(df)
    df = preprocessor.extract_basic_features(df)
    
    # Check labels and balance if needed
    df_balanced = preprocessor.balance_dataset_intelligently(df)
    
    preprocessor.save_processed_data(df_balanced)
    print("\n✅ Preprocessing complete!")