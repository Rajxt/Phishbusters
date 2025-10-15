# data_preprocessing.py - Multi-Dataset Support
"""
Enhanced Data Preprocessor for Phishing Detection
Supports loading and combining multiple datasets with different formats
"""

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    def __init__(self):
        self.vectorizer = None
        self.dataset_info = {}
    
    def detect_dataset_format(self, df, dataset_name="unknown"):
        """
        Automatically detect the format and columns of a dataset
        """
        print(f"\n📊 Analyzing dataset: {dataset_name}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {df.columns.tolist()}")
        
        # Common column name variations
        subject_cols = ['subject', 'Subject', 'SUBJECT', 'email_subject', 'mail_subject']
        body_cols = ['body', 'Body', 'BODY', 'content', 'text', 'message', 'email_body', 'mail_body', 'text_combined']
        sender_cols = ['sender', 'from', 'From', 'FROM', 'email_from', 'sender_email']
        label_cols = ['label', 'Label', 'LABEL', 'class', 'target', 'spam', 'is_phishing', 'phishing']
        
        detected = {
            'subject': None,
            'body': None,
            'sender': None,
            'label': None
        }
        
        # Detect columns
        for col in df.columns:
            if col in subject_cols:
                detected['subject'] = col
            elif col in body_cols:
                detected['body'] = col
            elif col in sender_cols:
                detected['sender'] = col
            elif col in label_cols:
                detected['label'] = col
        
        # If no exact match, try fuzzy matching
        if detected['subject'] is None:
            for col in df.columns:
                if 'subject' in col.lower():
                    detected['subject'] = col
                    break
        
        if detected['body'] is None:
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['body', 'content', 'text', 'message']):
                    detected['body'] = col
                    break
        
        if detected['sender'] is None:
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['sender', 'from', 'email']):
                    detected['sender'] = col
                    break
        
        if detected['label'] is None:
            for col in df.columns:
                if any(keyword in col.lower() for keyword in ['label', 'class', 'spam', 'phish', 'target']):
                    detected['label'] = col
                    break
        
        print(f"   Detected columns:")
        for key, value in detected.items():
            status = f"✓ {value}" if value else "✗ Not found"
            print(f"      {key}: {status}")
        
        return detected
    
    def standardize_dataset(self, df, dataset_name="unknown"):
        """
        Standardize a dataset to common column names
        """
        detected = self.detect_dataset_format(df, dataset_name)
        
        standardized_df = pd.DataFrame()
        
        # Special handling for datasets with 'text_combined' column (like phish.csv)
        if 'text_combined' in df.columns:
            print("   → Detected 'text_combined' column, splitting into subject and body")
            # Use text_combined for body, leave subject empty
            standardized_df['subject'] = ''
            standardized_df['body'] = df['text_combined'].fillna('')
        else:
            # Map to standard column names
            if detected['subject']:
                standardized_df['subject'] = df[detected['subject']].fillna('')
            else:
                standardized_df['subject'] = ''
            
            if detected['body']:
                standardized_df['body'] = df[detected['body']].fillna('')
            else:
                standardized_df['body'] = ''
        
        if detected['sender']:
            standardized_df['sender'] = df[detected['sender']].fillna('unknown@example.com')
        else:
            standardized_df['sender'] = 'unknown@example.com'
        
        # Handle labels - this is critical
        if detected['label']:
            standardized_df['label'] = df[detected['label']]
        else:
            print(f"   ⚠️ WARNING: No label column found in {dataset_name}")
            standardized_df['label'] = -1  # Mark as unlabeled
        
        # Add dataset source for tracking
        standardized_df['dataset_source'] = dataset_name
        
        # Add URLs column (empty by default)
        standardized_df['urls'] = ''
        
        return standardized_df
    
    def normalize_labels(self, df):
        """
        Normalize labels to binary: 1 = phishing, 0 = legitimate
        Handles various label formats
        """
        print("\n🏷️  Normalizing labels across datasets...")
        
        def convert_label(label):
            """Convert various label formats to binary"""
            label_str = str(label).lower().strip()
            
            # Phishing indicators
            phishing_labels = ['1', 'phishing', 'spam', 'true', 'yes', 'phish', 'malicious']
            if label_str in phishing_labels:
                return 1
            
            # Legitimate indicators
            legitimate_labels = ['0', 'legitimate', 'ham', 'false', 'no', 'legit', 'safe']
            if label_str in legitimate_labels:
                return 0
            
            # Try numeric conversion
            try:
                numeric_val = float(label_str)
                return 1 if numeric_val > 0.5 else 0
            except:
                pass
            
            # Default to -1 for unknown
            return -1
        
        # Convert labels
        df['label'] = df['label'].apply(convert_label)
        
        # Remove unlabeled data
        unlabeled_count = (df['label'] == -1).sum()
        if unlabeled_count > 0:
            print(f"   ⚠️ Removing {unlabeled_count} unlabeled rows")
            df = df[df['label'] != -1].reset_index(drop=True)
        
        # Show label distribution
        label_counts = df['label'].value_counts().sort_index()
        print(f"\n   Label distribution after normalization:")
        for label, count in label_counts.items():
            label_name = "Phishing" if label == 1 else "Legitimate"
            percentage = (count / len(df)) * 100
            print(f"      {label} ({label_name}): {count:,} ({percentage:.1f}%)")
        
        return df
    
    def load_and_combine_datasets(self, dataset_configs):
        """
        Load and combine multiple datasets
        
        Args:
            dataset_configs: List of tuples (file_path, dataset_name)
        """
        print("="*60)
        print("LOADING AND COMBINING DATASETS")
        print("="*60)
        
        all_dataframes = []
        
        for file_path, dataset_name in dataset_configs:
            try:
                print(f"\n📂 Loading: {dataset_name} from {file_path}")
                
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                        print(f"   ✓ Loaded with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception as e:
                        print(f"   Error with {encoding}: {str(e)[:50]}")
                        continue
                
                if df is None:
                    print(f"   ✗ Failed to load {dataset_name}")
                    continue
                
                print(f"   Rows: {len(df):,}")
                
                # Standardize the dataset
                standardized_df = self.standardize_dataset(df, dataset_name)
                
                # Store info
                self.dataset_info[dataset_name] = {
                    'rows': len(standardized_df),
                    'file': file_path
                }
                
                all_dataframes.append(standardized_df)
                print(f"   ✓ {dataset_name} standardized successfully")
                
            except Exception as e:
                print(f"   ✗ Error loading {dataset_name}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("No datasets were successfully loaded!")
        
        # Combine all datasets
        print(f"\n🔗 Combining {len(all_dataframes)} datasets...")
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Normalize labels across all datasets
        combined_df = self.normalize_labels(combined_df)
        
        # Remove duplicates based on subject and body
        print(f"\n🧹 Removing duplicates...")
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['subject', 'body'], keep='first')
        duplicates_removed = original_count - len(combined_df)
        print(f"   Removed {duplicates_removed:,} duplicate emails")
        
        print(f"\n✅ Combined dataset created:")
        print(f"   Total emails: {len(combined_df):,}")
        
        # Show breakdown by dataset source
        print(f"\n   Breakdown by source:")
        for source, count in combined_df['dataset_source'].value_counts().items():
            percentage = (count / len(combined_df)) * 100
            print(f"      {source}: {count:,} ({percentage:.1f}%)")
        
        return combined_df
    
    def load_and_clean_data(self, csv_file):
        """Load and clean a single dataset (backward compatible)"""
        print(f"Loading data from {csv_file}...")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
                break
            except:
                continue
        
        if df is None:
            raise ValueError(f"Could not load {csv_file} with any encoding")
        
        # Standardize
        dataset_name = csv_file.split('/')[-1].replace('.csv', '')
        df = self.standardize_dataset(df, dataset_name)
        df = self.normalize_labels(df)
        
        print(f"✓ Loaded {len(df)} emails")
        return df
    
    def clean_text(self, df):
        """Clean and preprocess text"""
        print("Cleaning text...")
        
        # Clean subject
        df['subject_clean'] = df['subject'].apply(self._clean_text_field)
        
        # Clean body
        df['body_clean'] = df['body'].apply(self._clean_text_field)
        
        # Combine subject and body
        df['combined_text'] = df['subject_clean'] + ' ' + df['body_clean']
        
        print(f"✓ Text cleaned")
        return df
    
    def _clean_text_field(self, text):
        """Clean individual text field"""
        if pd.isna(text):
            return ''
        
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs (but keep their presence for url_count feature)
        text = re.sub(r'http[s]?://\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s!?.,]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def extract_basic_features(self, df):
        """Extract basic numerical features"""
        print("Extracting features...")
        
        df['subject_length'] = df['subject'].fillna('').apply(len)
        df['body_length'] = df['body'].fillna('').apply(len)
        df['url_count'] = df['body'].fillna('').apply(lambda x: len(re.findall(r'http[s]?://', str(x))))
        df['exclamation_count'] = df['body'].fillna('').apply(lambda x: str(x).count('!'))
        
        print(f"✓ Features extracted")
        return df
    
    def create_tfidf_features(self, text_series):
        """Create TF-IDF features"""
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            X = self.vectorizer.fit_transform(text_series)
        else:
            X = self.vectorizer.transform(text_series)
        
        return X
    
    def save_processed_data(self, df, filename='processed_combined_data.pkl'):
        """Save processed data"""
        with open(filename, 'wb') as f:
            pickle.dump(df, f)
        print(f"✓ Processed data saved to {filename}")

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    
    # Example: Load multiple datasets
    dataset_configs = [
        ('datasets/enron.csv', 'Enron'),
        ('datasets/ling.csv', 'Ling'),
        ('datasets/nazario.csv', 'Nazario'),
        ('datasets/phisingdataset.csv', 'Phishing'),
    ]
    
    # Check which exist
    import os
    available = [(f, n) for f, n in dataset_configs if os.path.exists(f)]
    
    if available:
        df = preprocessor.load_and_combine_datasets(available)
        df = preprocessor.clean_text(df)
        df = preprocessor.extract_basic_features(df)
        preprocessor.save_processed_data(df)
        print("\n✅ Preprocessing complete!")
    else:
        print("❌ No datasets found!")