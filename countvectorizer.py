import os 
from string import punctuation
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def clean_doc(document):
    split_doc = document.split()
    split_doc = [word.lower() for word in split_doc]


    return " ".join(split_doc)


def main():
    # read in the data
    folder_path = "./data/Cleaned text data/"
    all_files = [
        os.path.join(folder_path, filename) for filename in os.listdir(folder_path)
    ]

    documents = []

    for file in all_files:
        with open(file, "r") as f:
            documents.append(f.read())

    for i, document in enumerate(documents):
        documents[i] = clean_doc(document)

    
    cv = CountVectorizer(input="content")
    out = cv.fit_transform(documents).toarray()
    print(out)
    # pd.DataFrame(out, columns=cv.get_feature_names_out()).to_csv("count_vec.csv")


if __name__ == "__main__":
    main()