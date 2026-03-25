import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Download necessary NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def process_text(text):
    """
    Process the input text for ARM transaction format:
    - Tokenize
    - Convert to lowercase
    - Remove punctuation from words
    - Remove stopwords
    - Lemmatize
    - Remove duplicates and short words
    - Return comma-separated string
    """
    # Tokenize the text
    tokens = word_tokenize(text.lower())

    # Remove punctuation from each word
    tokens = [''.join(c for c in word if c not in string.punctuation) for word in tokens]
    tokens = [word for word in tokens if word]  # remove empty strings

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Remove duplicates, empty strings, and words shorter than 3 characters
    tokens = list(set(tokens))
    tokens = [word for word in tokens if word and len(word) > 2]

    # Return as comma-separated string
    return ','.join(tokens)

def main():
    # Directory containing the .txt files
    data_dir = 'CleanedTranscription/'

    # List to hold the data
    data = []

    # Process each .txt file
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(data_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Process the text
                transaction = process_text(text)

                # Determine label from filename
                if '_algorithms' in file:
                    label = 'ALGORITHM'
                elif '_music_sharing' in file:
                    label = 'SHARING'
                else:
                    print(f"Warning: Could not determine label for file {file}. Skipping.")
                    continue

                # Skip if transaction is empty
                if not transaction:
                    print(f"Warning: Empty transaction for file {file}. Skipping.")
                    continue

                data.append({'label': label, 'transaction': transaction})

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

    # Create DataFrame
    df = pd.DataFrame(data)

    if df.empty:
        print("No data processed. Exiting.")
        return

    # Create version with labels appended to the beginning of each row
    df_with_labels = pd.DataFrame({
        'transaction': df['label'] + ',' + df['transaction']
    })

    # Create version without labels
    df_without_labels = pd.DataFrame({
        'transaction': df['transaction']
    })

    # Save to CSV files
    df_with_labels.to_csv('cleaned_data_with_labels.csv', index=False)
    df_without_labels.to_csv('cleaned_data_no_labels.csv', index=False)

    print("Data processing complete.")
    print(f"Processed {len(df)} documents.")
    print("Created 'cleaned_data_with_labels.csv' and 'cleaned_data_no_labels.csv'.")

if __name__ == '__main__':
    main()
