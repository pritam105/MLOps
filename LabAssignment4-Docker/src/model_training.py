import joblib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

if __name__ == '__main__':
    print("Loading 20 Newsgroups dataset...")
    train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    test  = fetch_20newsgroups(subset='test',  remove=('headers', 'footers', 'quotes'))

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, sublinear_tf=True)),
        ('clf',   LogisticRegression(max_iter=1000, C=5, solver='lbfgs', multi_class='auto')),
    ])

    print("Training TF-IDF + Logistic Regression pipeline...")
    pipeline.fit(train.data, train.target)

    preds = pipeline.predict(test.data)
    print(classification_report(test.target, preds, target_names=train.target_names))

    joblib.dump({'pipeline': pipeline, 'target_names': list(train.target_names)}, 'model.joblib')
    print("Model saved to model.joblib")
