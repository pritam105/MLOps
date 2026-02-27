import os
import io
import pickle
import base64
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.datasets import load_iris

def _b64_pickle(obj) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _unb64_pickle(s: str):
    return pickle.loads(base64.b64decode(s))


def load_data():
    """
    Loads Iris dataset directly from sklearn instead of CSV.
    Returns base64 encoded dataframe.
    """
    print("Loading iris dataset from sklearn")

    iris = load_iris(as_frame=True)

    # iris.data already a dataframe
    df = iris.data.copy()
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    return _b64_pickle(df)



def preprocess_iris(df_b64: str) -> str:
    """
    Drops NA, selects numeric features, scales them.
    Returns base64-pickled numpy array (X).
    """
    df = _unb64_pickle(df_b64)
    df = df.dropna()

    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X = df[feature_cols].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # You can optionally save scaler too (often useful)
    return _b64_pickle({"X": X_scaled, "scaler": scaler})


def train_save_models(prepped_b64: str,
                      k_min: int = 2,
                      k_max: int = 10,
                      kmeans_filename: str = "kmeans_iris.sav",
                      agglom_filename: str = "agglom_iris.sav") -> dict:
    """
    Trains:
      - KMeans across k range (stores SSE + silhouette; selects best silhouette)
      - Agglomerative across k range (selects best silhouette)
    Saves best models to dags/model/.
    Returns JSON-safe dict of metrics + chosen k values.
    """
    payload = _unb64_pickle(prepped_b64)
    X = payload["X"]

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(out_dir, exist_ok=True)

    # ---- KMeans sweep ----
    kmeans_sse = {}
    kmeans_sil = {}
    best_km = None
    best_km_sil = -1.0
    best_km_model = None

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(X)
        kmeans_sse[k] = float(km.inertia_)

        sil = silhouette_score(X, labels)
        kmeans_sil[k] = float(sil)

        if sil > best_km_sil:
            best_km_sil = sil
            best_km = k
            best_km_model = km

    kmeans_path = os.path.join(out_dir, kmeans_filename)
    with open(kmeans_path, "wb") as f:
        pickle.dump(best_km_model, f)

    # ---- Agglomerative sweep ----
    ag_sil = {}
    best_ag = None
    best_ag_sil = -1.0
    best_ag_labels = None
    best_ag_model = None

    for k in range(k_min, k_max + 1):
        ag = AgglomerativeClustering(n_clusters=k, linkage="ward")
        labels = ag.fit_predict(X)
        sil = silhouette_score(X, labels)
        ag_sil[k] = float(sil)

        if sil > best_ag_sil:
            best_ag_sil = sil
            best_ag = k
            best_ag_labels = labels
            best_ag_model = ag

    ag_path = os.path.join(out_dir, agglom_filename)
    with open(ag_path, "wb") as f:
        pickle.dump(best_ag_model, f)

    # Additional “fair” comparison metrics using best configs
    km_labels = best_km_model.predict(X)
    km_db = float(davies_bouldin_score(X, km_labels))
    ag_db = float(davies_bouldin_score(X, best_ag_labels))

    return {
        "k_range": [k_min, k_max],
        "kmeans": {
            "best_k": int(best_km),
            "best_silhouette": float(best_km_sil),
            "davies_bouldin": km_db,
            "sse_by_k": kmeans_sse,
            "silhouette_by_k": kmeans_sil,
            "model_file": kmeans_filename,
        },
        "agglomerative": {
            "best_k": int(best_ag),
            "best_silhouette": float(best_ag_sil),
            "davies_bouldin": ag_db,
            "silhouette_by_k": ag_sil,
            "model_file": agglom_filename,
        }
    }


def load_models_and_report(metrics: dict,
                           kmeans_filename: str = "kmeans_iris.sav",
                           agglom_filename: str = "agglom_iris.sav") -> str:
    """
    Loads saved models (sanity), prints a compact comparison summary.
    Returns a short JSON-safe string summary for Airflow logs.
    """
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    km_path = os.path.join(model_dir, kmeans_filename)
    ag_path = os.path.join(model_dir, agglom_filename)

    km = pickle.load(open(km_path, "rb"))
    ag = pickle.load(open(ag_path, "rb"))

    # Just sanity that objects exist
    summary = (
        f"KMeans(best_k={metrics['kmeans']['best_k']}, "
        f"sil={metrics['kmeans']['best_silhouette']:.3f}, "
        f"DB={metrics['kmeans']['davies_bouldin']:.3f}) | "
        f"Agglomerative(best_k={metrics['agglomerative']['best_k']}, "
        f"sil={metrics['agglomerative']['best_silhouette']:.3f}, "
        f"DB={metrics['agglomerative']['davies_bouldin']:.3f})"
    )

    print(summary)
    return summary