# FastAPI ML Model Serving — 20 Newsgroups Text Classifier

## Overview

In this Lab, we expose a **text classification ML model** as an API using **FastAPI** and **uvicorn**.  
This version uses the **20 Newsgroups** dataset and a **TF-IDF + Logistic Regression pipeline**.

The API accepts **raw text input** and returns a predicted category with confidence scores.

Technologies used:

1. **FastAPI** — high-performance Python API framework
2. **uvicorn** — ASGI server used to run FastAPI apps
3. **scikit-learn** — dataset, preprocessing, and model pipeline
4. **joblib** — model serialization

---

## Workflow

The workflow involves the following steps:

1. Load the 20 Newsgroups text dataset.
2. Train a text classification pipeline:
   - TF-IDF Vectorizer
   - Logistic Regression classifier
3. Save the trained pipeline and label names.
4. Expose the trained model via FastAPI.
5. Send text to the API → receive predicted class + confidence.
