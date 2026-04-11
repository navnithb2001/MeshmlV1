# MeshML Google Kubernetes Engine (GKE) Deployment Guide

This document details the configuration requirements and scripts necessary to deploy the MeshML platform directly to Google Kubernetes Engine (GKE) with zero local-to-prod code changes.

## Prerequisites

Before running the deployment script, ensure you have completed the following Pre-Flight changes:

### 1. Internal Services (gRPC Headless)
Your internal service manifests (like `k8s/base/parameter-server.yaml`, `task-orchestrator.yaml`, `dataset-sharder.yaml`, etc.) are configured as headless services to allow round-robin gRPC load balancing:
```yaml
spec:
  clusterIP: None
```

### 2. Configure Local vs. Production Storage Parity 

**Development (`docker-compose.yml`)**
Specify the explicit MinIO emulator URL for local testing:
```yaml
environment:
  - STORAGE_EMULATOR_URL=http://minio:9000
```

**Production (`k8s/base/configmap.yaml`)**
Omit the emulator. Provide the real Google Cloud Storage bucket name:
```yaml
data:
  GCS_BUCKET_NAME: "your-prod-bucket"
  # STORAGE_EMULATOR_URL is intentionally omitted
```

**Python Storage Client Update (`gcs_client.py`)**
The internal storage client is updated to default dynamically:
```python
import os
from google.cloud import storage

def get_storage_client():
    emulator = os.getenv("STORAGE_EMULATOR_URL")
    if emulator:
        return storage.Client(client_options={"api_endpoint": emulator}, project="local")
    return storage.Client()
```

### 3. Environment Secrets
The `k8s/base/secrets.yaml` file contains placeholders for your actual managed database credentials (Cloud SQL / MemoryStore) which you must fill in before deploying:
```yaml
stringData:
  DATABASE_URL: "postgresql+asyncpg://user:password@<CLOUD_SQL_IP>/meshml"
  REDIS_URL: "redis://<MEMORYSTORE_IP>:6379/0"
```

### 4. API Gateway IP Template
The `k8s/base/api-gateway.yaml` file is securely configured to request a `LoadBalancer` using an injected placeholder variable (`${GATEWAY_IP}`) instead of a hardcoded IP:
```yaml
spec:
  type: LoadBalancer
  loadBalancerIP: ${GATEWAY_IP}  
```

---

## Deployment Process

The deployment process is entirely automated by the `scripts/deploy-gke.sh` script.

To deploy your infrastructure:

1. Allow execution mode:
```bash
chmod +x scripts/deploy-gke.sh
```

2. Run the deployment script:
```bash
./scripts/deploy-gke.sh
```

### What the script does:
1. **Authentication:** Authenticates to the GKE cluster using `gcloud`.
2. **Static IP Provisioning**: Discovers or creates a static IP address (`meshml-gateway-ip`) in your region for the production frontend.
3. **Build & Push**: Loops through all internal microservices to build and push container images to Google Artifact Registry.
4. **Manifest Application**: Applies resources in rigid dependency order preventing crash-loops (ConfigMaps/Secrets -> DB/Redis -> Auth/Compute -> Gateway).
5. **Gateway Injection**: Injects the reserved static IP into `api-gateway.yaml` dynamically via `envsubst`.
6. **Load Balancer Verification**: Halts completion pending actual external IP binding on the cluster, returning your newly assigned connection string.
