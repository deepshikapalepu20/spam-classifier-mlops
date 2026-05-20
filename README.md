# 🛡️ SpamShield — MLOps Implementation of a Spam Classifier with Full DevOps Pipeline

> **INT377 — Cloud Computing and DevOps Essentials**
> Lovely Professional University | May 2026

A complete, production-grade MLOps pipeline built around a Naive Bayes spam classifier — from model training to cloud deployment, with CI/CD automation and real-time monitoring.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Phases](#pipeline-phases)
  - [Phase 1 — ML Model & Flask API](#phase-1--ml-model--flask-api)
  - [Phase 2 — Docker Containerization](#phase-2--docker-containerization)
  - [Phase 3 — Kubernetes with Minikube](#phase-3--kubernetes-with-minikube)
  - [Phase 4 — Jenkins CI/CD Pipeline](#phase-4--jenkins-cicd-pipeline)
  - [Phase 5 — Terraform & AWS EC2](#phase-5--terraform--aws-ec2)
  - [Phase 6 — AWS Elastic Beanstalk](#phase-6--aws-elastic-beanstalk)
  - [Phase 7 — Prometheus & Grafana Monitoring](#phase-7--prometheus--grafana-monitoring)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Monitoring](#monitoring)
- [Errors Faced & Resolutions](#errors-faced--resolutions)
- [Key Learnings](#key-learnings)

---

## Overview

SpamShield is an end-to-end MLOps project demonstrating the full DevOps lifecycle for a machine learning application. The system classifies SMS/email messages as **SPAM** or **NOT SPAM** using a Naive Bayes classifier trained on 5,574 labelled messages, achieving **~97% accuracy**.

The deployment pipeline spans **7 phases** covering containerization, orchestration, CI/CD automation, Infrastructure as Code, cloud hosting, and observability.

---

## Live Demo

The SpamShield frontend allows users to:
- Input any text message and receive an instant spam/ham prediction with confidence percentage
- View real-time stats (Total Checked, Spam Found, Safe Messages)
- Toggle between light and dark mode
- Try quick example messages (Spam, Phishing, Safe)

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     SpamShield MLOps Pipeline              │
│                                                            │
│ ML Model                                                   │
│ (Naive Bayes + TF-IDF)                                     │
│       │                                                    │
│       ▼                                                    │
│ Flask API + Frontend                                       │
│       │                                                    │
│       ▼                                                    │
│ Docker Image                                               │
│       │                                                    │
│       ▼                                                    │
│ Docker Hub                                                 │
│       │                                                    │
│ ┌──────────────┬──────────────┬────────────────┐           │
│ ▼              ▼              ▼                ▼           │
│ Docker      Kubernetes       AWS EC2      Elastic          │
│ Local       (Minikube)       (Docker      Beanstalk        │
│ Deploy       Deploy)         Image Pull)  (ZIP Upload)    │
│                                                      │     │
│ Jenkins Container                                    │     │
│ (GitHub Webhook Trigger Only)                        │     │
│                                                      │     │
│ Prometheus + Grafana                                 │     │
│ (Monitoring Docker Deployment Only)                  │     │
└────────────────────────────────────────────────────────────┘
```

| Layer | Component | Access |
|-------|-----------|--------|
| ML Model | Naive Bayes + TF-IDF | Internal |
| API | Flask + Gunicorn | `http://localhost:5000` |
| Frontend | SpamShield HTML/CSS/JS | `http://localhost:5000` |
| Container | Docker Image | Docker Hub |
| Orchestration | Kubernetes (Minikube) | `http://127.0.0.1:NodePort` |
| CI/CD | Jenkins Pipeline | `http://localhost:9090` |
| IaC | Terraform → AWS EC2 | `http://EC2_IP:5000` |
| Cloud Hosting | AWS Elastic Beanstalk | `http://ENV.elasticbeanstalk.com` |
| Monitoring | Prometheus | `http://localhost:9091` |
| Dashboards | Grafana | `http://localhost:3000` |

---

## Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| ML Model | Scikit-learn, Naive Bayes, TF-IDF | Spam classification |
| Web Framework | Flask + Gunicorn | REST API and web server |
| Frontend | HTML, CSS, JavaScript | Single-page UI |
| Containerization | Docker | Package app and dependencies |
| Orchestration | Kubernetes + Minikube | Container orchestration |
| CI/CD | Jenkins | Automated build pipeline |
| IaC | Terraform | AWS infrastructure provisioning |
| Cloud Deployment | AWS Elastic Beanstalk | Managed app hosting |
| Monitoring | Prometheus + Grafana | Metrics and dashboards |
| Version Control | Git + GitHub | Source code + webhook trigger |

---

## Project Structure

```
spam-classifier-mlops/
├── app.py                     ← Flask app with Prometheus metrics
├── train_model.py             ← Model training script
├── model.pkl                  ← Trained Naive Bayes model (generated)
├── requirements.txt           ← Python dependencies
├── Dockerfile                 ← Docker image definition
├── Jenkinsfile                ← CI/CD pipeline definition
├── Procfile                   ← Gunicorn startup for Elastic Beanstalk
├── templates/
│   └── index.html             ← SpamShield single-page frontend
├── kubernetes/
│   ├── deployment.yaml        ← K8s deployment config (2 replicas)
│   └── service.yaml           ← K8s NodePort service config
├── terraform/
│   ├── main.tf                ← AWS EC2 and Security Group
│   ├── variables.tf           ← Input variables
│   └── outputs.tf             ← Output values (public IP, URL)
|
└── monitoring/
    └── docker-compose.yml     ← Prometheus + Grafana stack
    └── prometheus
        └── prometheus.yml         ← Prometheus scrape config
```

---

## Pipeline Phases

### Phase 1 — ML Model & Flask API

The spam classifier uses a **Scikit-learn Pipeline** combining TF-IDF vectorization with a Multinomial Naive Bayes classifier, trained on the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) (5,574 messages).

### 1. Install Dependencies

```bash
pip install flask scikit-learn pandas prometheus_client gunicorn
```

### Explanation

| Command | Purpose |
|----------|-----------|
| `flask` | Creates REST API endpoints for serving the spam classification model |
| `scikit-learn` | Builds and trains the spam classification machine learning model |
| `pandas` | Loads, cleans, and processes the dataset |
| `prometheus_client` | Generates monitoring metrics for application performance tracking |
| `gunicorn` | Production-grade WSGI server for deploying the Flask application |

### Dependency Overview

These libraries form the foundation of the Spam Classifier MLOps project. Flask exposes the ML model through REST APIs, Scikit-learn handles model training and prediction, Pandas manages dataset processing, Prometheus Client provides monitoring metrics, and Gunicorn enables production deployment of the Flask server.

## 2. train_model.py

```python
# Import Pandas for handling datasets
import pandas as pd

# Import TF-IDF vectorizer to convert text into numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Naive Bayes classifier for spam prediction
from sklearn.naive_bayes import MultinomialNB

# Import Pipeline to combine preprocessing and model training
from sklearn.pipeline import Pipeline

# Import pickle for saving the trained model
# urllib.request helps in downloading dataset files
import pickle, urllib.request

# SMS Spam dataset URL
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# Read dataset from URL
# sep='\t' means tab-separated file
# header=None because dataset has no column names
# names assigns custom column names
df = pd.read_csv(
    url,
    sep='\t',
    header=None,
    names=['label', 'message']
)

# Store SMS messages as input features
X = df['message']

# Convert labels into numbers
# ham -> 0
# spam -> 1
y = df['label'].map({
    'ham': 0,
    'spam': 1
})

# Create ML pipeline
model = Pipeline([

    # Convert text into TF-IDF vectors
    ('tfidf', TfidfVectorizer(stop_words='english')),

    # Train Naive Bayes classifier
    ('clf', MultinomialNB()),
])

# Train model using dataset
model.fit(X, y)

# Save trained model into a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Display completion message
print("Model trained and saved as model.pkl")

# Display number of training samples
print(f"Training samples: {len(X)}")
```

### Workflow

The script downloads the SMS spam dataset, converts text messages into numerical TF-IDF vectors, trains a Multinomial Naive Bayes model, and saves the trained model as `model.pkl` for later use in the Flask API.

## 3. Run Model Training

Run the training script to train the spam classification model and generate the saved model file.

```bash
python train_model.py
```

### Output

```text
Model trained and saved as model.pkl
Training samples: 5572
```

### Creates

```text
model.pkl
```

### Purpose

The `model.pkl` file contains the trained machine learning model. It is later loaded by the Flask API to perform spam message prediction without retraining the model every time the application starts.

## 4. app.py (Flask API)

```python
# Import Flask framework components
# Flask -> creates web application
# request -> receives incoming API requests
# jsonify -> converts Python data into JSON response
# render_template -> loads HTML frontend pages
from flask import Flask, request, jsonify, render_template

# Import Prometheus monitoring tools
# Counter -> counts API requests
# Histogram -> measures response time
# generate_latest -> generates metrics output
# CONTENT_TYPE_LATEST -> sets Prometheus content type
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

# pickle loads saved ML model
# time measures API execution time
import pickle, time

# Create Flask application instance
app = Flask(__name__)

# Load trained machine learning model
# 'rb' means read binary mode
with open('model.pkl', 'rb') as f:

    # Load model from file
    model = pickle.load(f)

# Create Prometheus counter metric
# Tracks total prediction requests
REQUEST_COUNT = Counter(

    # Metric name
    'spam_prediction_requests_total',

    # Metric description
    'Total prediction requests',

    # Label category
    ['result']
)

# Create histogram metric
# Measures prediction execution time
REQUEST_LATENCY = Histogram(

    # Metric name
    'spam_prediction_latency_seconds',

    # Metric description
    'Prediction latency in seconds'
)

# Homepage route
# Runs when user visits "/"
@app.route('/')
def home():

    # Load frontend page
    return render_template('index.html')

# Health check endpoint
# Used by Kubernetes to verify app health
@app.route('/health')
def health():

    # Return healthy status
    return jsonify({
        "status": "healthy"
    })

# Prediction API endpoint
# Accepts POST requests
@app.route('/predict', methods=['POST'])
def predict():

    # Start timer
    start = time.time()

    # Read incoming JSON request
    data = request.get_json()

    # Validate input
    if not data or 'message' not in data:

        # Return error if message missing
        return jsonify({
            "error":
            "Provide JSON with 'message' field"
        }), 400

    # Extract message text
    msg = data['message']

    # Predict spam or not spam
    pred = model.predict([msg])[0]

    # Get confidence probabilities
    prob = model.predict_proba([msg])[0]

    # Convert prediction number into label
    result = "SPAM" if pred == 1 else "NOT SPAM"

    # Increment request counter
    REQUEST_COUNT.labels(
        result=result
    ).inc()

    # Record execution latency
    REQUEST_LATENCY.observe(
        time.time() - start
    )

    # Return API response
    return jsonify({

        # Original message
        "message": msg,

        # Prediction result
        "prediction": result,

        # Confidence percentage
        "confidence":
        round(float(max(prob)) * 100, 2)

    })

# Metrics endpoint
# Prometheus scrapes metrics from here
@app.route('/metrics')
def metrics():

    # Return monitoring metrics
    return generate_latest(), 200, {
        'Content-Type':
        CONTENT_TYPE_LATEST
    }

# Run application
if __name__ == '__main__':

    # Start Flask server
    # host='0.0.0.0' allows external access
    # port=5000 defines application port
    # debug=False disables debug mode
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False
    )
```

### Workflow

The Flask API loads the trained spam classification model (`model.pkl`) during startup. Users send messages to the `/predict` endpoint, where the model predicts whether the text is spam or not spam. Prometheus metrics track request counts and latency, while `/health` supports Kubernetes health checks and `/metrics` enables monitoring integration.

## 5. Run the Flask App Locally

Run the Flask application locally to start the spam classification API server.

```bash
# Start the Flask application
python app.py

# Application becomes accessible locally
# Default Flask URL
# Open this in browser
http://127.0.0.1:5000
```

### Expected Result

After running the command, Flask starts the web server and loads the trained machine learning model. The application becomes available on:

```text
http://127.0.0.1:5000
```

Open the URL in your browser to access the Spam Classifier frontend interface and test spam predictions.
**Flask API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the SpamShield frontend |
| `/predict` | POST | Returns SPAM or NOT SPAM with confidence |
| `/health` | GET | Returns service health status |
| `/metrics` | GET | Exposes Prometheus metrics |

**Prometheus Metrics Instrumented:**
- `spam_prediction_requests_total` — Counter labelled by result (SPAM / NOT SPAM)
- `spam_prediction_latency_seconds` — Histogram of prediction latency

---

# Phase 2 — Docker Containerization

### Benefits of Docker

- Consistent runtime environment
- Easy deployment across platforms
- Dependency isolation
- Improved portability
- Simplifies Kubernetes deployment
- Reduces environment-related bugs

## 1. requirements.txt

Create a `requirements.txt` file to install all project dependencies.

```txt
# Flask framework for building REST API endpoints
flask==3.0.0

# Machine learning library for training and prediction
scikit-learn==1.4.0

# Data processing and dataset handling
pandas==2.2.0

# Prometheus monitoring and metrics collection
prometheus_client==0.20.0

# Production-grade server for running Flask application
gunicorn==21.2.0
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages with fixed versions to ensure the application behaves consistently across Docker containers, EC2 instances, and Kubernetes deployments.

## 2. Dockerfile

Create a `Dockerfile` to containerize the Flask Spam Classifier application.

```dockerfile
# Use lightweight Python 3.11 image as base container
FROM python:3.11-slim

# Set working directory inside container
# All commands run from /app folder
WORKDIR /app

# Copy dependency file into container
COPY requirements.txt .

# Install Python packages
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir \
-r requirements.txt

# Copy Flask application file
COPY app.py .

# Copy trained ML model file
COPY model.pkl .

# Copy frontend HTML templates
COPY templates/ ./templates/

# Open container port 5000
# Flask/Gunicorn runs on this port
EXPOSE 5000

# Start application using Gunicorn
# --bind connects app to port 5000
# app:app means
# first "app" -> app.py file
# second "app" -> Flask object name
CMD [
"gunicorn",
"--bind",
"0.0.0.0:5000",
"app:app"
]
```

## 3. Build and run:
```bash
docker build -t spam-classifier:v2 .
docker run -d -p 5000:5000 --name spam-app spam-classifier:v2
```

## 4. Push to Docker Hub:
```bash
docker login
docker tag spam-classifier:v2 <YOUR_DOCKERHUB_USERNAME>/spam-classifier:v2
docker push <YOUR_DOCKERHUB_USERNAME>/spam-classifier:v2
```

---

### Phase 3 — Kubernetes with Minikube

Deployed with **2 replicas** and a `NodePort` service exposed on port `30080`. `imagePullPolicy: Always` ensures the latest image is always pulled from Docker Hub.

```bash
minikube start
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
kubectl get pods
minikube service spam-classifier-service --url
```

| Setting | Value |
|---------|-------|
| Replicas | 2 |
| Image | `<YOUR_DOCKERHUB_USERNAME>/spam-classifier:v2` |
| imagePullPolicy | Always |
| Service Type | NodePort |
| NodePort | 30080 |

Both pods reach `Running` status and the SpamShield frontend is accessible via the Minikube-generated URL.

---

### Phase 4 — Jenkins CI/CD Pipeline

Jenkins runs in a **Docker container** on port `9090` (port 8080 was occupied). The pipeline is triggered automatically via a **GitHub webhook** on every push to `main`.

**Webhook setup:**
- ngrok is used to expose Jenkins locally: `ngrok http 9090`
- Webhook URL configured in GitHub: `https://<ngrok-url>/github-webhook/`

**Pipeline Stages:**

| Stage | Description |
|-------|-------------|
| Checkout | Pulls latest code from GitHub |
| Train Model | Executes `train_model.py` to generate `model.pkl` |
| Test Application | Validates spam/ham detection with assertions |
| Build Docker Image | Builds image tagged with `v${BUILD_NUMBER}` |
| Push to Docker Hub | Pushes versioned image and `latest` tag |
| Deploy to Kubernetes | Rolls out updated image via `kubectl set image` |

**Add DockerHub credentials in Jenkins:**
> Manage Jenkins → Credentials → Add → Username/Password → ID: `dockerhub-creds`

The full pipeline runs in under **60 seconds** end-to-end.

---

### Phase 5 — Terraform & AWS EC2

Infrastructure provisioned as code using Terraform. The configuration dynamically fetches the latest Ubuntu 22.04 AMI and default VPC, making it portable across regions and accounts.

```bash
cd terraform/
terraform init
terraform plan
terraform apply -auto-approve
```

**Resources Provisioned:**

| Resource | Details |
|----------|---------|
| AWS Security Group | Ports 22, 5000, 9090 open to `0.0.0.0/0` |
| AWS EC2 Instance | `t3.micro`, Ubuntu 22.04, `us-east-1` |
| AMI | Latest Ubuntu 22.04 via data source |
| VPC | Default VPC via data source |

> **Note:** `key_name` was omitted from the Terraform config intentionally. EC2 Instance Connect is used for browser-based SSH access, eliminating key pair provisioning errors.

**Deploy the app on EC2:**
```bash
docker pull <YOUR_DOCKERHUB_USERNAME>/spam-classifier:v2
docker run -d -p 5000:5000 --name spam-app <YOUR_DOCKERHUB_USERNAME>/spam-classifier:v2
```

**Teardown:**
```bash
terraform destroy -auto-approve
```

---

### Phase 6 — AWS Elastic Beanstalk

The Flask app is deployed to **AWS Elastic Beanstalk** (Python 3.11 on Amazon Linux 2023), which manages load balancing, auto-scaling, health monitoring, and server management automatically.

**Deployment package structure:**
```
spam-classifier.zip
├── app.py
├── model.pkl
├── requirements.txt
├── Procfile              ← web: gunicorn --bind 0.0.0.0:8000 app:app
└── templates/
    └── index.html
```

> **Critical:** Beanstalk's nginx reverse proxy forwards traffic to port **8000**, not 5000. The `Procfile` must bind Gunicorn to `0.0.0.0:8000`.

**Beanstalk Configuration:**

| Setting | Value |
|---------|-------|
| Application Name | `spam-classifier-app` |
| Environment Name | `Spam-classifier-app-env-1` |
| Platform | Python 3.11 on Amazon Linux 2023 |
| WSGI Server | Gunicorn on port 8000 |
| Region | `us-east-1` |

The SpamShield app is publicly accessible at the auto-generated `.elasticbeanstalk.com` URL with no manual server configuration.

---

### Phase 7 — Prometheus & Grafana Monitoring

Prometheus and Grafana are deployed together using **Docker Compose** from the `monitoring/` directory.

```bash
cd monitoring/
docker-compose up -d
```

**Access:**

| Service | URL | Credentials |
|---------|-----|-------------|
| Prometheus | `http://localhost:9091` | None |
| Grafana | `http://localhost:3000` | `admin / admin123` |

> Prometheus is mapped to external port **9091** to avoid conflict with Jenkins on 9090.
> Grafana connects to Prometheus internally using `http://prometheus:9090`.

**Grafana Dashboards Created:**

| Dashboard | PromQL Query |
|-----------|-------------|
| Total Predictions | `spam_prediction_requests_total` |
| Count by Label | `sum by (result) (spam_prediction_requests_total)` |
| Request Rate | `rate(spam_prediction_requests_total[1m])` |
| Avg Latency | `rate(spam_prediction_latency_seconds_sum[1m]) / rate(spam_prediction_latency_seconds_count[1m])` |

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker Desktop
- Minikube
- kubectl
- Jenkins (or Docker-based Jenkins)
- Terraform
- AWS CLI (configured)
- ngrok (for webhook tunneling)

### Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/<YOUR_USERNAME>/spam-classifier-mlops.git
cd spam-classifier-mlops

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_model.py

# 4. Run the Flask app
python app.py
# Visit http://localhost:5000
```

### Run with Docker

```bash
docker build -t spam-classifier:v2 .
docker run -d -p 5000:5000 --name spam-app spam-classifier:v2
# Visit http://localhost:5000
```

### Run Monitoring Stack

```bash
# Make sure the Flask app is running first
cd monitoring/
docker-compose up -d
# Prometheus: http://localhost:9091
# Grafana:    http://localhost:3000
```

---

## API Reference

### `POST /predict`

Classify a message as spam or not spam.

**Request:**
```json
{
  "message": "Congratulations! You won a free iPhone. Click now!"
}
```

**Response:**
```json
{
  "message": "Congratulations! You won a free iPhone. Click now!",
  "prediction": "SPAM",
  "confidence": 99.62
}
```

### `GET /health`

```json
{ "status": "healthy" }
```

### `GET /metrics`

Returns Prometheus metrics in text format (scraped every 5 seconds).

---

## Errors Faced & Resolutions

| Phase | Error | Cause | Resolution |
|-------|-------|-------|------------|
| Docker | Server error 500 | scikit-learn version mismatch (local 1.8 vs container 1.4) | Pinned exact version in `requirements.txt` |
| Docker | `model.pkl not found` | `train_model.py` not run before `docker build` | Run training script first, then rebuild |
| Kubernetes | `ImagePullBackOff` | Wrong Docker Hub image name in `deployment.yaml` | Fixed to exact username/tag |
| Kubernetes | Old frontend after update | K8s cached old image | Built new tag `v2` + set `imagePullPolicy: Always` |
| Terraform | `InvalidKeyPair.NotFound` | Key pair name mismatch | Removed `key_name`, used EC2 Instance Connect |
| Elastic Beanstalk | `502 Bad Gateway` | Gunicorn bound to 5000, Beanstalk expects 8000 | Updated `Procfile` to port 8000 |
| Elastic Beanstalk | Dependency install failed | `scikit-learn==1.8.0` doesn't exist on PyPI | Changed to `scikit-learn==1.3.2` |
| Prometheus | Port 9090 conflict | Jenkins already on 9090 | Mapped Prometheus to external port 9091 |
| Prometheus | `cannot mount — not a directory` | `prometheus.yml` in wrong location | Moved to `monitoring/prometheus/prometheus.yml` |

---

## Key Learnings

- **Version consistency is critical** — the scikit-learn version in `requirements.txt` must exactly match the version used to serialize `model.pkl`, or the model will fail to deserialize inside the container.
- **Kubernetes caches aggressively** — always push a new tag or use `imagePullPolicy: Always` when updating an application.
- **Beanstalk uses port 8000** — nginx in Elastic Beanstalk expects the app on port 8000, not 5000. The `Procfile` must reflect this.
- **Docker networking uses container names** — Grafana must reference Prometheus as `http://prometheus:9090` (not `localhost`) inside the Docker Compose network.
- **Terraform state is authoritative** — resources created outside Terraform's state file will cause `already exists` conflicts on re-apply.
- **ngrok enables local webhook testing** — exposing a local Jenkins port via ngrok is the simplest way to receive GitHub webhook payloads during local development.

---

## Author

**Deepshika Palepu**
Registration Number: 12322963 | Section: 4OM57
Lovely Professional University — INT377, May 2026

---

*SpamShield — From Python script to production MLOps pipeline.*
