import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# nltk download setup
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initalize lemmatizer and stop word list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Custom function to clean, tokenize, filter, and lemmatize text.
    """
    # Lowercase and tokenize the text
    tokens = word_tokenize(text.lower())
    
    # Filter out punctuation, numbers, and meaningless symbols.
    tokens = [word for word in tokens if word.isalpha()]
    
    # Remove stopwords and apply lemmatization 
    processed_tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words
    ]
    
    return " ".join(processed_tokens)

# Define directory where txt files are located and set up lists for reading
folder_path = './test_files'
documents = []
filenames = []

# Loop through files, opening and reading contents and appending them to lists
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()
            
            # Clean text using nltk function before storing it
            cleaned_text = preprocess_text(raw_text)
            documents.append(cleaned_text)
        
        filenames.append(filename)

# Initialize Vectorizer and fit and transform into matrix
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Create pandas dataframe
df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=vectorizer.get_feature_names_out(), 
    index=filenames # Sets the filenames as the row labels
)

# Create labels list and insert the label column at front of the dataframe
labels = []
for name in filenames:
    uppercase_name = name.upper()

    if 'ALGORITHM' in uppercase_name:
        labels.append('ALGORITHM')
    elif 'SHARING' in uppercase_name:
        labels.append('SHARING')
    else:
        labels.append('UNLABELED')

df.insert(0, 'label', labels)

# Save as CSV
output_csv_path = 'tfidf_output.csv'
df.to_csv(output_csv_path)

print(f"Success! Processed {len(filenames)} files.")
print(f"Data saved to {output_csv_path}")
print("\nPreview of DataFrame:")
print(df.head())