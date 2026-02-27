from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

def load_data():
    dataset = fetch_20newsgroups(
        subset="train",
        remove=("headers", "footers", "quotes")
    )

    X_train, X_test, y_train, y_test = train_test_split(
        dataset.data,
        dataset.target,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test, dataset.target_names