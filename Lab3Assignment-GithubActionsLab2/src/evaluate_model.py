import os
import argparse
import json
import joblib

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str, required=True)
    args = parser.parse_args()
    timestamp = args.timestamp

    model_path = f"Lab3Assignment-GithubActionsLab2/models/model_{timestamp}_logreg.joblib"
    vectorizer_path = f"Lab3Assignment-GithubActionsLab2/models/vectorizer_{timestamp}.joblib"

    if not os.path.exists(model_path):
        raise ValueError("Model file not found.")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    categories = [
        "sci.space",
        "rec.sport.baseball",
        "comp.graphics"
    ]

    dataset = fetch_20newsgroups(
        subset="test",
        categories=categories,
        remove=("headers", "footers", "quotes")
    )

    X_text = dataset.data
    y = dataset.target

    X_vec = vectorizer.transform(X_text)
    y_pred = model.predict(X_vec)

    f1 = f1_score(y, y_pred, average="macro")

    metrics = {
        "macro_f1": float(f1)
    }

    os.makedirs("Lab3Assignment-GithubActionsLab2/metrics", exist_ok=True)

    metrics_path = f"Lab3Assignment-GithubActionsLab2/metrics/{timestamp}_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("Metrics saved.")