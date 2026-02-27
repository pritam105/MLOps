import os
import argparse
import joblib

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    print(f"Timestamp received: {timestamp}")

    # Small subset for CI speed
    categories = [
        "sci.space",
        "rec.sport.baseball",
        "comp.graphics"
    ]

    dataset = fetch_20newsgroups(
        subset="train",
        categories=categories,
        remove=("headers", "footers", "quotes")
    )

    X_text = dataset.data
    y = dataset.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Logistic Regression model
    model = LogisticRegression(
        max_iter=500,
        n_jobs=1
    )

    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Accuracy: {acc}")
    print(f"Macro F1: {f1}")

    # Save artifacts
    os.makedirs("Labs/Github_Labs/Lab2/models", exist_ok=True)

    model_path = f"Labs/Github_Labs/Lab2/models/model_{timestamp}_logreg.joblib"
    vectorizer_path = f"Labs/Github_Labs/Lab2/models/vectorizer_{timestamp}.joblib"

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print("Model and vectorizer saved.")