# Airflow Iris Clustering DAG - Quick Start Guide

## Overview
This DAG compares KMeans vs Agglomerative clustering on the Iris dataset. Both algorithms found **k=2** as optimal using the Silhouette method, which measures cluster separation and cohesion. Unlike the elbow method (which looks for diminishing returns in variance), Silhouette scoring tends to favor fewer, more distinct clusters when the data has clear separation.

## Prerequisites
- Docker and Docker Compose installed
- The airflow.py and iris_lab.py files in the correct directory structure

## Project Structure
```
project/
├── dags/
│   └── airflow.py          # DAG definition
│   └── model/              # Created automatically for saved models
│   └── src/
│       └── iris_lab.py     # Clustering logic
```

## Running the DAG

### 1. Start Airflow
```bash
# Start all Airflow services
docker-compose up -d

# Wait ~30 seconds for initialization
# Access UI at http://localhost:8080
# Default credentials: airflow / airflow
```

### 2. Trigger the DAG
- Navigate to http://localhost:8080
- Find DAG: `Airflow_Iris_Clustering_Compare`
- Toggle it ON
- Trigger a manual run

### 3. Monitor Execution
```bash
# View logs for all tasks
docker-compose logs -f airflow-scheduler

# View specific task logs
docker logs <container_id> --tail 100 -f
```

### 4. Check Individual Task Logs
In the Airflow UI:
1. Click on the DAG run
2. Click on each task box to view logs:
   - `load_iris` - Loads Iris dataset
   - `preprocess_iris` - Scales features
   - `train_save_models` - Trains both algorithms
   - `load_models_and_report` - **Shows final results**

### 5. Find the Results
The final output appears in the `load_models_and_report` task logs:
```
KMeans(best_k=2, sil=0.582, DB=0.593) | Agglomerative(best_k=2, sil=0.577, DB=0.592)
```

**Metrics:**
- **best_k**: Optimal number of clusters
- **sil** (Silhouette): Higher is better (range: -1 to 1)
- **DB** (Davies-Bouldin): Lower is better

### 6. Stop Airflow
```bashw
# Stop all services
docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

## Understanding k=2 Result
The Silhouette method maximizes cluster separation while minimizing within-cluster spread. For Iris data:
- **k=2** likely separates Iris Setosa (very distinct) from the other two species
- **k=3** would be biologically accurate but Versicolor and Virginica overlap significantly
- Silhouette scoring penalizes this overlap, preferring the cleaner 2-cluster split

To experiment with k=3, modify the DAG's `train_task` parameters or check the `silhouette_by_k` values in the training metrics.