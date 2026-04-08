import os
import csv
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
    - Return a list of tokens
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

    # Return as a list instead of a comma-separated string
    return tokens

def main():
    # Directory containing the .txt files (change for your needs)
    data_dir = '/Users/thedrive/Documents/ProgramProjects/algorithm-aversion/Cleaned text data/'

    # Lists to hold the row data
    data_with_labels = []
    data_no_labels = []

    # Process each .txt file
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):
            file_path = os.path.join(data_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Process the text (now returns a list)
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

                # Append to our data lists
                # For labelled data, insert the label at the beginning of the list
                data_with_labels.append([label] + transaction)
                
                # For unlabelled data, just use the transaction list
                data_no_labels.append(transaction)

            except Exception as e:
                print(f"Error processing file {file}: {e}")
                continue

    if not data_no_labels:
        print("No data processed. Exiting.")
        return

    # Save to CSV files using the csv module
    with open('cleaned_data_with_labels.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data_with_labels)

    with open('cleaned_data_no_labels.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data_no_labels)

    print("Data processing complete.")
    print(f"Processed {len(data_no_labels)} documents.")
    print("Created 'cleaned_data_with_labels.csv' and 'cleaned_data_no_labels.csv'.")

if __name__ == '__main__':
    main()