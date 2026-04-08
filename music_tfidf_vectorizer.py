import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Define directory where txt files are located and set up lists for reading
folder_path = 'path/to/files' # TODO: REPLACE!!!
documents = []
filenames = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, 'r', encoding='utf-8') as file:
            documents.append(file.read())
        filenames.append(filename)

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(documents)

df = pd.DataFrame(
    tfidf_matrix.toarray(), 
    columns=vectorizer.get_feature_names_out(), 
    index=filenames # Sets the filenames as the row labels
)