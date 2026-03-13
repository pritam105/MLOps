# Lab Assignment 4 — Docker Lab

Text classification using TF-IDF + Logistic Regression on the [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) dataset, served via a Flask web UI — all containerized with a single multi-stage Dockerfile.

## How to run

```bash
# Build (trains the model inside Docker)
docker build -t newsgroup-classifier .

# Run
docker run -p 8080:80 newsgroup-classifier
```

Open [http://localhost:8080](http://localhost:8080), paste any news-style text, and click **Classify**.

## Result screenshots

Screenshots are in the [`assets/`](assets/) directory.
