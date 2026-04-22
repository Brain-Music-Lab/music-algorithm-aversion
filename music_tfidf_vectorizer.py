import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# nltk download setup
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initalize lemmatizer and stop word list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
additional_filler_words = ["also", "call", "called", "could", "else", "got", "gotcha", "hey", "lot", "ohh", "side", "totally", "kind", "would", "yeah", "really", "like", "well", "definitely", "sometimes", "think", "stuff", "could", "know", "pretty"]
stop_words.update(additional_filler_words)
removal_words = ["sharing", "computer", "algorithm"]
stop_words.update(removal_words)

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

df.insert(0, 'file_category_label', labels)

# Save as CSV
output_csv_path = 'tfidf_output.csv'
df.to_csv(output_csv_path, index=False)

print(f"Success! Processed {len(filenames)} files.")
print(f"Data saved to {output_csv_path}")
print("\nPreview of DataFrame:")
print(df.head())

# Create and clean decision tree dataframe
ml_df = df[df['file_category_label'] != 'UNLABELED']

X = ml_df.drop('file_category_label', axis=1)
y = ml_df['file_category_label']

# Train Test split and train classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #random_state=42)

clf = DecisionTreeClassifier() #random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate model accuracy
y_pred = clf.predict(X_test)

print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))


# Plot Tree
plt.figure(figsize=(15, 10))
plot_tree(
    clf, 
    feature_names=X.columns, 
    class_names=clf.classes_, 
    filled=True, 
    rounded=True,
    max_depth=3, 
    fontsize=10
)

plt.title("Decision Tree Visualization")
plt.tight_layout()
plt.show()


# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=clf.classes_, normalize='true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)

fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='.1%')

plt.title("Confusion Matrix")
plt.show()