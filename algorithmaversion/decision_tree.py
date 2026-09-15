import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("tfidf_output.csv")
    labels = df.pop("file_category_label")

    encoder = LabelEncoder()
    enc_labels = encoder.fit_transform(labels)

    x_train, x_test, y_train, y_test = train_test_split(df, enc_labels, train_size=0.7)
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)

    print(dt.pre)
    y_pred = dt.predict(x_test)

    print(encoder.inverse_transform(np.array([1])))
    plot_tree(dt, feature_names=df.columns)
    plt.show()

    cnf = confusion_matrix(y_test, dt.predict(x_test))
    disp = ConfusionMatrixDisplay(cnf)
    disp.plot()
    plt.show()



if __name__ == "__main__":
    main()