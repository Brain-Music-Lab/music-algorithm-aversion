import os
import re
from string import punctuation

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Constants ────────────────────────────────────────────────────────────────

FOLDER_PATH = "/Users/Hrishikesh/Documents/music-algorithm-aversion/Cleaned_Data"

# Files whose names contain any of these keywords are labelled ALGORITHM;
# everything else is labelled SHARING.
ALGORITHM_KEYWORDS = ["algorithm", "algo", "recommend", "filter", "machine"]

STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


# ── Text cleaning ────────────────────────────────────────────────────────────

def clean_doc(document: str) -> str:
    """
    Lower-case, strip punctuation, remove stop-words, and lemmatize every
    token in *document*.  Returns the cleaned string.
    """
    # Lower-case
    document = document.lower()

    # Remove punctuation (replace with space so words don't accidentally merge)
    document = re.sub(f"[{re.escape(punctuation)}]", " ", document)

    # Remove stray digits / purely numeric tokens
    document = re.sub(r"\b\d+\b", " ", document)

    tokens = document.split()

    cleaned_tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS    # drop stop-words
        and len(token) > 1            # drop single-character noise
    ]

    return " ".join(cleaned_tokens)


# ── Label inference ──────────────────────────────────────────────────────────

def infer_label(filepath: str) -> str:
    """
    Return 'ALGORITHM' if the filename contains an algorithm-related keyword,
    otherwise return 'SHARING'.
    """
    basename = os.path.basename(filepath).lower()
    for kw in ALGORITHM_KEYWORDS:
        if kw in basename:
            return "ALGORITHM"
    return "SHARING"


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Collect file paths ─────────────────────────────────────────────
    if not os.path.isdir(FOLDER_PATH):
        raise FileNotFoundError(
            f"Data folder not found: '{FOLDER_PATH}'\n"
            "Please check the path exists and is correct."
        )

    all_files = sorted(
        os.path.join(FOLDER_PATH, fname)
        for fname in os.listdir(FOLDER_PATH)
        if fname.endswith(".txt")
    )

    if not all_files:
        raise ValueError(f"No .txt files found in '{FOLDER_PATH}'")

    print(f"Found {len(all_files)} file(s).")

    # ── 2. Read & clean documents ─────────────────────────────────────────
    documents: list[str] = []
    labels: list[str] = []

    for filepath in all_files:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
            raw = fh.read()

        documents.append(clean_doc(raw))
        labels.append(infer_label(filepath))

    # ── 3. Vectorize ──────────────────────────────────────────────────────
    cv = CountVectorizer(
        input="content",
        stop_words="english",           # sklearn built-in list as extra safety net
        min_df=1,                       # keep tokens appearing in at least 1 doc
        token_pattern=r"[a-zA-Z]{2,}", # letters only, length >= 2
    )

    matrix = cv.fit_transform(documents).toarray()
    feature_names = cv.get_feature_names_out()

    print(f"Vocabulary size: {len(feature_names)} unique tokens.")

    # ── 4. Build DataFrame ────────────────────────────────────────────────
    df = pd.DataFrame(matrix, columns=feature_names)

    # Guard: the word "label" may appear in the corpus and become a feature
    # column. Drop it first so df.insert doesn't raise a duplicate error.
    if "label" in df.columns:
        df.drop(columns=["label"], inplace=True)

    # Insert the label column at position 0 (front of the DataFrame)
    df.insert(0, "label", labels)

    print(df.head())
    print(f"\nDataFrame shape: {df.shape}")

    # ── 5. Save to CSV ────────────────────────────────────────────────────
    output_path = "count_vec.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to '{output_path}'")


if __name__ == "__main__":
    main()