# Email Spam Detection

A spam classifier that detects whether an email is spam or not, using a Bag-of-Words text representation and a Multinomial Naive Bayes classifier.

## Overview

Raw email text is cleaned (punctuation and stopwords removed), converted into a Bag-of-Words feature matrix with CountVectorizer, and used to train a Naive Bayes classifier, a standard and effective baseline approach for text classification tasks like spam filtering.

## Results

- **99.71% accuracy** on the training set
- **99.2% accuracy** on the held-out test set

## Tech Stack

- Python
- Pandas, NumPy
- NLTK (stopword removal)
- Scikit-learn (CountVectorizer, Multinomial Naive Bayes)

## Dataset

This expects a CSV file named `emails.csv` with `text` and `spam` columns (5,728 rows in the original dataset used, a common public email-spam dataset with a binary spam label).

## Running the Project

```bash
pip install -r requirements.txt
python spam_detection.py
```

## Author

Aishwarya Ramanath Shanbhag
