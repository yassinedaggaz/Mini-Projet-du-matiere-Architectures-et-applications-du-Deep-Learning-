# Deep Learning Image Classifier — Google Cloud Deployment Guide

## 🗺️ Architecture

```
GitHub (push to main)
   │
   ▼ GitHub Actions CI/CD
   │  ├── 1. pytest (app/tests/)
   │  ├── 2. docker build + push → Artifact Registry
   │  └── 3. kubectl apply → GKE
   ▼
Google Kubernetes Engine (GKE)
   ├── Ingress (NGINX) ← external IP
   ├── Service (ClusterIP)
   └── Deployment (3 pods min, 10 max via HPA)
         └── Container: FastAPI + ResNet18 (CPU)
```

---

## 🔧 Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| `gcloud` CLI | latest | https://cloud.google.com/sdk/docs/install |
| `kubectl` | latest | `gcloud components install kubectl` |
| `docker` | 24+ | https://docs.docker.com/get-docker/ |

---

## 1️⃣ Google Cloud Setup

```bash
# 1. Authenticate
gcloud auth login

# 2. Set your project
export PROJECT_ID="deeplearning-project-495018"
gcloud config set project $PROJECT_ID

# 3. Enable required APIs
gcloud services enable \
  container.googleapis.com \
  artifactregistry.googleapis.com \
  monitoring.googleapis.com \
  logging.googleapis.com

# 4. Create Artifact Registry repository
gcloud artifacts repositories create dl-classifier \
  --repository-format=docker \
  --location=europe-west1 \
  --description="DL image classifier images"

# 5. Configure Docker auth
gcloud auth configure-docker europe-west1-docker.pkg.dev
```

---

## 2️⃣ Create a GKE Cluster

```bash
# Standard cluster (cost-effective for CPU workloads)
gcloud container clusters create dl-classifier-cluster \
  --zone europe-west1-b \
  --num-nodes 3 \
  --machine-type e2-standard-2 \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 5 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials dl-classifier-cluster \
  --zone europe-west1-b
```

---

## 3️⃣ Build & Push Docker Image (manually)

```bash
export PROJECT_ID="deeplearning-project-495018"
export REGION="europe-west1"
export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/dl-classifier/api:1.0.0"

# Build (from project root)
docker build \
  -t $IMAGE \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/dl-classifier/api:latest \
  .

# Push
docker push $IMAGE
docker push ${REGION}-docker.pkg.dev/${PROJECT_ID}/dl-classifier/api:latest
```

---

## 4️⃣ Update Image in Deployment

Edit `k8s/deployment.yaml` and replace:
```yaml
image: europe-west1-docker.pkg.dev/deeplearning-project-495018/dl-classifier/api:1.0.0
```
with your actual image path.

---

## 5️⃣ Deploy to Kubernetes

```bash
# Install NGINX Ingress Controller (one-time)
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml

# Apply all manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
kubectl apply -f k8s/hpa.yaml

# Check status
kubectl get pods -n dl-classifier
kubectl get ingress -n dl-classifier
```

---

## 6️⃣ Test the API

```bash
# Get the external IP
export API_IP=$(kubectl get ingress dl-classifier-ingress \
  -n dl-classifier \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

echo "API available at: http://${API_IP}"

# Health check
curl http://${API_IP}/health | python -m json.tool

# Predict an image
curl -X POST http://${API_IP}/predict \
  -F "file=@/path/to/your/keyboard.jpg" \
  | python -m json.tool

# View metrics
curl http://${API_IP}/metrics
```

### Expected response from /predict:
```json
{
  "prediction": "keyboard",
  "confidence": 0.9345,
  "confidence_pct": "93.45%",
  "top_predictions": [
    {"class_name": "keyboard", "confidence": 0.9345},
    {"class_name": "mouse",    "confidence": 0.0412},
    {"class_name": "laptop",   "confidence": 0.0201}
  ],
  "model_version": "1.0.0",
  "inference_time_ms": 28.5
}
```

---

## 7️⃣ GitHub Actions CI/CD Setup

Add these **Secrets** in `GitHub → Settings → Secrets → Actions`:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | Your GCP project ID |
| `GCP_SA_KEY` | JSON key of the service account (see below) |
| `GKE_CLUSTER_NAME` | `dl-classifier-cluster` |
| `GKE_CLUSTER_ZONE` | `europe-west1-b` |

### Create the Service Account:
```bash
# Create SA
gcloud iam service-accounts create github-actions-sa \
  --display-name="GitHub Actions SA"

# Grant roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/container.developer"

# Export JSON key (paste as GCP_SA_KEY secret)
gcloud iam service-accounts keys create sa-key.json \
  --iam-account="github-actions-sa@${PROJECT_ID}.iam.gserviceaccount.com"
cat sa-key.json
```

---

## 8️⃣ Model Versioning & Updates

To deploy a new model version:

```bash
# 1. Train new model (updates models/best_model.pth)
python src/train.py --epochs 15 --fine_tune

# 2. Bump version in deployment.yaml + requirements-prod.txt
export NEW_VERSION="1.1.0"

# 3. Build & push new image
docker build -t ${IMAGE}:${NEW_VERSION} .
docker push ${IMAGE}:${NEW_VERSION}

# 4. Rolling update (zero downtime)
kubectl set image deployment/dl-classifier \
  dl-classifier=${IMAGE}:${NEW_VERSION} \
  -n dl-classifier

# Monitor rollout
kubectl rollout status deployment/dl-classifier -n dl-classifier
```

---

## 9️⃣ Monitoring

```bash
# Pod logs (live)
kubectl logs -l app=dl-classifier -n dl-classifier -f

# HPA status (scaling)
kubectl get hpa -n dl-classifier

# Resource usage
kubectl top pods -n dl-classifier

# Prometheus metrics (forwarded locally)
kubectl port-forward svc/dl-classifier-svc 8080:80 -n dl-classifier
curl http://localhost:8080/metrics
```

---

## 🗑️ Cleanup (avoid charges)

```bash
# Delete cluster (stops all billing for compute)
gcloud container clusters delete dl-classifier-cluster \
  --zone europe-west1-b --quiet

# Delete Artifact Registry images (optional)
gcloud artifacts repositories delete dl-classifier \
  --location europe-west1 --quiet
```
