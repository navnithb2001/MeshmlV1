#!/bin/bash
set -e

# ==========================================================
# MeshML GKE Deployment Script
# ==========================================================

# Step 2: Script Foundation & Variables
export GCP_PROJECT_ID="${GCP_PROJECT_ID:-my-gcp-project}"
export CLUSTER_NAME="${CLUSTER_NAME:-meshml-cluster}"
export COMPUTE_REGION="${COMPUTE_REGION:-us-central1}"
export ARTIFACT_REGISTRY_URL="${ARTIFACT_REGISTRY_URL:-us-central1-docker.pkg.dev/$GCP_PROJECT_ID/meshml-repo}"
export NAMESPACE="default" # Update if you have a custom namespace like meshml

echo "Have you filled out your production database URLs and GCS Bucket names in k8s/base/secrets.yaml and configmap.yaml? (y/n)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Please fill them out and re-run the script."
    exit 1
fi

echo "[1/5] Authenticating with GKE..."
gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$COMPUTE_REGION" --project "$GCP_PROJECT_ID"

echo "[2/5] Checking for static IP 'meshml-gateway-ip'..."
if ! gcloud compute addresses describe meshml-gateway-ip --region "$COMPUTE_REGION" --project "$GCP_PROJECT_ID" >/dev/null 2>&1; then
    echo "Creating static IP 'meshml-gateway-ip'..."
    gcloud compute addresses create meshml-gateway-ip --region "$COMPUTE_REGION" --project "$GCP_PROJECT_ID"
    # Give it a few seconds to fully initialize
    sleep 5
else
    echo "Static IP 'meshml-gateway-ip' already exists."
fi

# Fetch the actual IPv4 string and store in bash variable
export GATEWAY_IP=$(gcloud compute addresses describe meshml-gateway-ip --region "$COMPUTE_REGION" --project "$GCP_PROJECT_ID" --format="value(address)")
if [ -z "$GATEWAY_IP" ]; then
    echo "Error: Failed to fetch GATEWAY_IP."
    exit 1
fi
echo "==> Gateway IP successfully reserved: $GATEWAY_IP"

# Step 3: Build & Push Images
echo "[3/5] Building and pushing Docker images..."
MICROSERVICES=("api-gateway" "dataset-sharder" "model-registry" "task-orchestrator" "parameter-server" "metrics-service")

for service in "${MICROSERVICES[@]}"; do
    echo "--- Processing $service ---"
    case $service in
      # Add custom paths here if your Dockerfiles are not at the root of the service directly
      *) DOCKER_DIR="services/$service" ;;
    esac

    IMAGE_NAME="$ARTIFACT_REGISTRY_URL/$service:latest"
    docker build -t "$IMAGE_NAME" "$DOCKER_DIR"
    docker push "$IMAGE_NAME"
done

# Step 4: Apply K8s Manifests in Dependency Order
echo "[4/5] Applying Kubernetes Manifests..."

echo "-> Applying foundational manifests..."
kubectl apply -f k8s/base/namespace.yaml || true # Ignore if doesn't exist just in case
kubectl apply -f k8s/base/configmap.yaml
kubectl apply -f k8s/base/secrets.yaml

echo "-> Applying stateful infrastructure..."
# Remove these if you're using managed Cloud SQL / MemoryStore (recommended for prod)
if [ -f "k8s/base/postgres.yaml" ]; then kubectl apply -f k8s/base/postgres.yaml; fi
if [ -f "k8s/base/redis.yaml" ]; then kubectl apply -f k8s/base/redis.yaml; fi

echo "-> Waiting for stateful services to be ready..."
# Simple wait. If using managed DBs, they are already ready.
sleep 15

echo "-> Applying internal gRPC microservices..."
for service in dataset-sharder model-registry task-orchestrator parameter-server metrics-service; do
    if [ -f "k8s/base/$service.yaml" ]; then
      kubectl apply -f "k8s/base/$service.yaml"
    else
      echo "Warning: k8s/base/$service.yaml missing, skipping."
    fi
done

echo "-> Applying External REST Entrypoint (API Gateway)..."
# Inject $GATEWAY_IP into the api-gateway.yaml file dynamically
envsubst < k8s/base/api-gateway.yaml | kubectl apply -f -

# Step 5: Verification
echo "[5/5] Waiting for API Gateway LoadBalancer to bind External IP ($GATEWAY_IP)..."

# Ensure the exact static IP was propagated
while [ "$(kubectl get svc api-gateway-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || true)" != "$GATEWAY_IP" ]; do
    echo "Wait pending..."
    sleep 5
done

FINAL_IP=$(kubectl get svc api-gateway-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
echo "=========================================================="
echo "DEPLOYMENT COMPLETE!"
echo "API Gateway successfully bound to: $FINAL_IP"
echo "You can test the endpoints immediately: http://$FINAL_IP/api"
echo "=========================================================="
