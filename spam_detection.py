"""
Email spam detection using Bag-of-Words text features and a
Multinomial Naive Bayes classifier.
"""

import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

nltk.download("stopwords")


def process_text(text: str) -> list[str]:
    """Clean raw email text into a list of tokens: remove punctuation
    and stopwords, ready for use as a CountVectorizer analyzer."""
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = "".join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]
    return clean_words


def main():
    # Load the data
    df = pd.read_csv("emails.csv")
    print(df.head(5))
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    # Remove duplicates
    df.drop_duplicates(inplace=True)
    print(f"Shape after removing duplicates: {df.shape}")

    # Check for missing values
    print(df.isnull().sum())

    # Preview tokenization on the first few rows
    print(df["text"].head().apply(process_text))

    # Convert text into a matrix of token counts
    messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df["text"])
    print(f"Bag-of-words shape: {messages_bow.shape}")

    # Split into training and testing sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        messages_bow, df["spam"], test_size=0.20, random_state=0
    )

    # Train the Multinomial Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Evaluate on the training set
    pred = classifier.predict(X_train)
    print("Training set performance:")
    print(classification_report(y_train, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_train, pred))
    print("Accuracy:", accuracy_score(y_train, pred))

    # Evaluate on the test set
    pred = classifier.predict(X_test)
    print("\nTest set performance:")
    print(classification_report(y_test, pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
    print("Accuracy:", accuracy_score(y_test, pred))


if __name__ == "__main__":
    main()
