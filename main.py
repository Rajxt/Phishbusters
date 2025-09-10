import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("Nazario.csv")

# Check dataset info
print(df.info())
print(df.head())
print(df.isnull().sum())

#Handling missing values
df['subject'] = df['subject'].fillna('')
df['body'] = df['body'].fillna('')
df['sender'] = df['sender'].fillna('unknown_sender')

# Replacing empty strings with Nan
df['subject'] = df['subject'].replace('', np.nan).fillna('no_subject')
df['body'] = df['body'].replace('', np.nan).fillna('no_body')

# Clean subject column
df['subject_clean'] = (df['subject']
                      .str.lower()                    # Lowercase
                      .str.replace(r'http\S+', '', regex=True)  # Remove URLs
                      .str.replace(r'\S+@\S+', '', regex=True)  # Remove emails  
                      .str.replace(r'[^a-zA-Z\s]', '', regex=True)  # Keep only letters/spaces
                      .str.replace(r'\s+', ' ', regex=True)     # Multiple spaces to single
                      .str.strip())                             # Remove leading/trailing spaces

# Clean body 
df['body_clean'] = (df['body']
                   .str.lower()
                   .str.replace(r'http\S+', '', regex=True)
                   .str.replace(r'\S+@\S+', '', regex=True)
                   .str.replace(r'[^a-zA-Z\s]', '', regex=True)
                   .str.replace(r'\s+', ' ', regex=True)
                   .str.strip())

# Combining cleaned subject and body
df['combined_text'] = df['subject_clean'] + ' ' + df['body_clean']

# Extract sender domain
df['sender_domain'] = df['sender'].str.extract(r'@([^>]+)')

# Get lengths of fields using numpy
df['subject_length'] = df['subject_clean'].str.len()
df['body_length'] = df['body_clean'].str.len()
df['total_length'] = df['subject_length'] + df['body_length']

# Count exclamation marks
df['exclamation_count'] = df['combined_text'].str.count('!')

# URL count feature
df['url_count'] = df['urls'].fillna(0)

#Remove very short messages 
min_length = 5
df = df[df['combined_text'].str.len() >= min_length]

# Select features for modeling
text_features = df['combined_text']
numerical_features = df[['subject_length', 'body_length', 'url_count', 'exclamation_count']]
target = df['label']

# Vectorize text
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_text = vectorizer.fit_transform(text_features)

# Combine with numerical features if desired
from scipy.sparse import hstack
X_combined = hstack([X_text, numerical_features.values])

print(f"Final dataset shape: {X_combined.shape}")
print(f"Target distribution:\n{target.value_counts()}")

