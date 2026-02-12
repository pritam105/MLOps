import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import load_data

MODEL_PATH = "model/news_model.pkl"
LABELS_PATH = "model/news_labels.pkl"

def train():
    X_train, X_test, y_train, y_test, labels = load_data()

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=20000,
            ngram_range=(1,2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            n_jobs=-1
        ))
    ])

    pipeline.fit(X_train, y_train)

    acc = pipeline.score(X_test, y_test)
    print(f"Validation accuracy: {acc:.3f}")

    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(labels, LABELS_PATH)

if __name__ == "__main__":
    train()
