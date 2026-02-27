import joblib
import numpy as np

MODEL_PATH = "model/news_model.pkl"
LABELS_PATH = "model/news_labels.pkl"

model = joblib.load(MODEL_PATH)
labels = joblib.load(LABELS_PATH)

def predict_text(text: str):
    probs = model.predict_proba([text])[0]
    pred_idx = int(np.argmax(probs))

    return {
        "label": labels[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top3": [
            {
                "label": labels[i],
                "prob": float(probs[i])
            }
            for i in np.argsort(probs)[-3:][::-1]
        ]
    }
