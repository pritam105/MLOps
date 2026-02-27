# GitHub Actions – Automated Model Retraining (20 Newsgroups)

## Overview

This project demonstrates an automated ML retraining pipeline using **GitHub Actions**.

Whenever code is pushed to the `Github-Actions-lab` branch, the workflow:

1. Generates a timestamp
2. Trains a Logistic Regression model on a subset of the 20 Newsgroups dataset
3. Evaluates the model using Macro F1 score
4. Saves the trained model and TF-IDF vectorizer
5. Saves evaluation metrics as JSON
6. Commits the artifacts back to the repository

---

## How to Trigger the Workflow

Push changes to the branch:

```bash
git checkout Github-Actions-lab
git add .
git commit -m "Trigger workflow"
git push origin Github-Actions-lab
```

The workflow runs automatically on push.

---

## What the Workflow Does

* Loads a 3-category subset of the 20 Newsgroups dataset
* Performs train/test split
* Applies TF-IDF vectorization
* Trains a Logistic Regression classifier
* Evaluates using Macro F1 score
* Saves artifacts with timestamp versioning

---

## Where to See Results

### Workflow Execution Logs

Go to:

```
GitHub Repository → Actions Tab → Latest Run
```

You can view:

* Training logs
* Accuracy & F1 score
* Each pipeline step

---

### Generated Artifacts (Auto-Committed)

After a successful run, new files appear in:

```
Lab3Assignment-GithubActionsLab2/models/
Lab3Assignment-GithubActionsLab2/metrics/
```

Artifacts include:

* `model_<timestamp>_logreg.joblib`
* `vectorizer_<timestamp>.joblib`
* `<timestamp>_metrics.json`

---
