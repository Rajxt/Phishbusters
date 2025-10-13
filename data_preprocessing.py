# data_preprocessing.py - FIXED VERSION with encoding handling
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = None

    def load_and_clean_data(self, csv_file):
        """Load and clean the email dataset with automatic encoding detection"""
        print("Loading and cleaning data...")
        
        # Try different encodings
        encodings_to_try = ['utf-8', 'latin-1', 'windows-1252', 'iso-8859-1', 'cp1252']
        
        df = None
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                print(f"✓ Successfully loaded file with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"Error with {encoding}: {e}")
                continue
        
        if df is None:
            # If all standard encodings fail, try with error handling
            try:
                df = pd.read_csv(csv_file, encoding='utf-8', errors='ignore')
                print("Loaded file with UTF-8 encoding (ignoring errors)")
            except Exception as e:
                # Last resort: try to detect encoding
                try:
                    import chardet
                    with open(csv_file, 'rb') as f:
                        result = chardet.detect(f.read(100000))
                        detected_encoding = result['encoding']
                        print(f"Detected encoding: {detected_encoding}")
                        df = pd.read_csv(csv_file, encoding=detected_encoding)
                except ImportError:
                    print("💡 Tip: Install chardet for better encoding detection: pip install chardet")
                    # Final fallback
                    df = pd.read_csv(csv_file, encoding='latin-1', errors='replace')
                    print("Loaded with latin-1 encoding (with replacements)")
                except Exception as e:
                    raise ValueError(f"Failed to read CSV file: {e}")
        
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
        
        # Handle potential encoding issues in text
        for col in ['subject', 'body']:
            if col in df.columns:
                # Clean any remaining encoding issues
                df[col] = df[col].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii') 
                                        if pd.notna(x) else '')
        
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
        df['sender_domain'] = df['sender'].str.extract(r'@([^>]+)', expand=False)
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
        """Save processed data to pickle file"""
        with open(filename, 'wb') as f:
            pickle.dump(df, f)
        print(f"Processed data saved to {filename}")


if __name__ == "__main__":
    # Test with your CSV file
    csv_files_to_try = [
        "sample.csv",
        "datasets/phishingdataset.csv",
        "phishingdataset.csv",
        "datasets/sample.csv"
    ]
    
    csv_file = None
    for file_path in csv_files_to_try:
        if os.path.exists(file_path):
            csv_file = file_path
            print(f"Found CSV file: {csv_file}")
            break
    
    if csv_file is None:
        print("❌ No CSV file found. Please specify the correct path.")
        print("Available CSV files in current directory:")
        for f in os.listdir('.'):
            if f.endswith('.csv'):
                print(f"  - {f}")
        
        # Ask user for input
        csv_file = input("\nEnter the path to your CSV file: ").strip()
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            exit()
    
    # Process the data
    preprocessor = DataPreprocessor()
    
    try:
        df = preprocessor.load_and_clean_data(csv_file)
        df = preprocessor.clean_text(df)
        df = preprocessor.extract_basic_features(df)
        
        # Check labels and balance if needed
        df_balanced = preprocessor.balance_dataset_intelligently(df)
        
        # Save processed data
        preprocessor.save_processed_data(df_balanced)
        print("\n✅ Preprocessing complete!")
        
        # Show sample of processed data
        print("\nSample of processed data:")
        print(df_balanced[['subject_clean', 'label']].head())
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure your CSV has these columns: subject, body, sender, label")
        print("2. Label column should contain 0 (legitimate) and 1 (phishing)")
        print("3. Try installing chardet: pip install chardet")