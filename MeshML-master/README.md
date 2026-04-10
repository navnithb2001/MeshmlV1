# MeshML

Distributed ML training platform using microservices, gRPC streaming for control/math, and HTTP for user APIs + object storage transfers.

## Full System Design: Who Calls What

### Phase A: Ingestion (The Setup)
1. **User → API Gateway (REST)**: Uploads the `model.py` definition and the `dataset.tar.gz` archive via standard HTTP POST requests.
2. **API Gateway → Model Registry (gRPC)**: Calls `RegisterNewModel()`. The Registry saves the Python file to Object Storage (MinIO/GCS) and creates a record in the `models` table with `version_1`.
3. **API Gateway → Dataset Sharder (gRPC)**: Triggers the sharding process. The Sharder physically splits the uploaded data, saves the chunks to Object Storage, and crucially, populates the `data_batches` table in the database so the system knows what work is available.

### Phase B: Orchestration (Starting the Job)
1. **User → API Gateway (REST)**: Clicks "Start Job" on a specific dataset and model version via the UI.
2. **API Gateway → Task Orchestrator (gRPC)**: Calls `InitiateTraining(job_id, model_version)`.
3. **Task Orchestrator → Model Registry (gRPC)**: Calls `GetModelArtifact()` to generate a secure, short-lived Signed URL for the PyTorch code. The Orchestrator embeds this URL into the tasks it is preparing for the workers.

### Phase C: Training Loop (The Work)
1. **Worker ↔ Task Orchestrator (gRPC)**: The worker connects via a bidirectional stream (`StreamTasks`). The Orchestrator pushes task assignments (containing the Signed URLs) to the worker, and the worker streams back its heartbeat and task completion status (`TaskResult`).
2. **Worker → Object Storage (HTTP)**: The worker performs a standard HTTP GET using the Signed URLs to download the `model.py` and its assigned `data_batch`, bypassing internal microservice bottlenecks.
3. **Worker ↔ Parameter Server (gRPC)**: The worker pulls the latest global weights (`PullWeights`), computes gradients locally using its data shard, and pushes the gradients back (`PushGradients`) using binary tensor serialization.
4. **Parameter Server → Model Registry (Internal)**: The Parameter Server runs a background persistence loop, periodically saving checkpoints to the Registry based on a defined interval.
5. **Worker ↔ Metrics Service (gRPC)**: Loss and accuracy metrics are streamed synchronously to allow high-frequency sub-millisecond dashboard updates.

### Phase D: Completion (The Result)
1. **Parameter Server → Model Registry (gRPC)**: Once the global model reaches the `final_version` threshold (e.g., 50 or 100 global updates), the Parameter Server terminates the training loop and uploads the final `state_dict`.
2. **Model Registry**: Marks the model as COMPLETED and saves the final `.pt` file.
3. **API Gateway → User**: The database reflects the completed status, allowing the user to download the final trained weights via the REST API.

## Architectural Choice: The Protocol Split

MeshML utilizes a strict protocol split. Internal cloud-to-cloud and worker-to-cloud math operations use gRPC for maximum performance, while user-facing interactions and large blob transfers use HTTP for compatibility and throughput.

### 1. The gRPC Plane (High-Performance Control & Math)
gRPC is used for the "Live" training loop. Because it utilizes binary serialization (Protobuf) and HTTP/2 multiplexing, it eliminates JSON parsing overhead and drastically reduces latency for the thousands of micro-messages sent during training.
- **Worker ↔ Task Orchestrator:** Uses Bidirectional gRPC Streaming. Instead of workers constantly polling for work, they open a single persistent connection. The orchestrator pushes task assignments down the pipe instantly, and workers stream their status back up.
- **Worker ↔ Parameter Server:** This is the most computationally sensitive path. Gradients (the math) are serialized into flat numpy byte buffers and sent as raw Protobuf bytes. This avoids the massive CPU and memory overhead of JSON text conversion.
- **Worker ↔ Metrics Service:** Loss and accuracy metrics are streamed via gRPC, allowing for high-frequency, sub-millisecond updates to the observability stack.
- **Internal Service-to-Service:** The API Gateway uses gRPC to trigger internal commands (like `RegisterNewModel` or `InitiateTraining`) across the cluster, ensuring fast, strongly typed contracts between microservices.

### 2. The HTTP Plane (Universal Access & Blob Transfer)
HTTP is strictly reserved for "Heavy Lifting" of static files and human-to-machine interactions.
- **Worker → Object Storage (Signed URLs):** The actual download of the `dataset.tar.gz` shards and the `model.py` code happens over direct HTTP/HTTPS.
  - *Why not gRPC?* While gRPC can stream files, standard HTTP servers (like Nginx, MinIO, or GCS) are fundamentally optimized at the OS level for serving large static blobs. Using HTTP allows workers to download massive datasets with maximum network throughput and without burning CPU cycles on Protobuf deserialization.
- **User ↔ API Gateway:** All standard user actions—authentication, creating groups, uploading initial files, and triggering jobs—use REST/HTTP. This ensures the API is easily consumable by standard web browsers, React dashboards, and CLI tools.
- **Observability (WebSockets):** The API Gateway upgrades standard HTTP connections to WebSockets to push live training metrics to the user's browser, providing a real-time dashboard experience.

### 3. Hyper-Concurrency (100% Non-Blocking Architecture)
To ensure the backend effortlessly scales to thousands of concurrent users (e.g. university deployments), MeshML utilizes a completely lock-free ASGI architecture:
- **Async Relational Layer:** Native `SQLAlchemy 2.0` via `AsyncSession` prevents simultaneous user traffic from generating query database bottlenecks.
- **Zero Python Event Loop Blocks:** All third-party REST queries, massive CPU-bound serialization blocks (`torch.save`), and blob-transfer SDKs (`boto3`) operate detached efficiently inside separate CPython thread-pools (`asyncio.to_thread`) preserving hyper-fast ping/response capacity.

## What is currently supported

### Model upload contract (`/api/models/upload`)

Model file validation currently enforces:
- UTF-8 Python source
- valid Python syntax
- `create_model()` function exists
- `MODEL_METADATA` exists and is a dict literal

`MODEL_METADATA` required fields:
- `name`
- `version`
- `framework`
- `input_shape`
- `output_shape`
- `task_type` (`classification`, `regression`, `binary`)
- `loss` (`cross_entropy`, `mse`, `mae`, `bce_with_logits`, `bce`)
- `metrics` (non-empty list of strings)

### Dataset upload formats (`/api/datasets/upload`)

Supported dataset formats:
- `imagefolder`
- `csv`
- `coco`

Format can be auto-detected from uploaded content. Unknown/unsupported formats are rejected with `400`.

### Worker trainer modes

Current Python worker trainer supports task configs from model metadata:
- `classification`
- `regression`
- `binary`

Loss and metric behavior are selected from metadata at runtime.

## Local run

### 1) Start stack

```bash
docker compose -f docker/docker-compose.yml up -d
```

### 2) Initialize database

```bash
psql -h localhost -p 5432 -U meshml -d meshml -f scripts/init-db.sql
```

### 3) Install and run Python worker

```bash
pip install -e workers/python-worker
meshml-worker init --api-url http://localhost:8000
meshml-worker login --email you@example.com
meshml-worker join --invitation-code <code> --worker-id my-laptop1
meshml-worker run
```

Useful worker env vars:
- `MESHML_DISABLE_RESOURCE_THROTTLE=true` disables CPU/RAM pause monitor.
- `MESHML_EXIT_ON_JOB_COMPLETE=true` exits worker after a job fully completes.
- `MESHML_CPU_PAUSE_THRESHOLD` / `MESHML_RAM_PAUSE_THRESHOLD` override pause thresholds.

## Production Deployment

For instructions on deploying the MeshML backend platform to Google Kubernetes Engine (GKE) with zero local-to-prod code changes, please refer to the [GKE Deployment Guide](GKE-DEPLOYMENT.md).

For instructions on deploying the React Dashboard UI to Firebase Hosting, please refer to the [Firebase Deployment Guide](FIREBASE-DEPLOYMENT.md).

## Dashboard

Dashboard lives under `dashboard/`.

Run locally:

```bash
cd dashboard
npm install
npm run dev
```

Current UI includes:
- Group dashboard tabs for jobs, workers, datasets, settings
- New Training Run modal with:
  - upload new or reuse existing dataset
  - model code upload
  - convergence target (`final_version > 0`)
- Toast notifications + error boundary for frontend failures

## E2E validation

Run:

```bash
E2E_USER_EMAIL=you1@example.com \
E2E_USER_PASSWORD='StrongPass123!' \
E2E_GROUP_ID='<group-id>' \
E2E_WORKER_ID='my-laptop1' \
python tests/e2e_validation.py
```

The script validates:
- auth/login
- model upload
- dataset upload + availability
- job creation
- worker heartbeat path
- job progress / parameter-server signal

## Service endpoints (docker compose defaults)

- API Gateway: `http://localhost:8000`
- Dataset Sharder: `http://localhost:8001` and gRPC `localhost:50053`
- Task Orchestrator: `http://localhost:8002` and gRPC `localhost:50051`
- Parameter Server: `http://localhost:8003` and gRPC `localhost:50052`
- Model Registry: `http://localhost:8004` and gRPC `localhost:50054` (host-mapped)
- Metrics Service: `http://localhost:8005` and gRPC `localhost:50055`
- MinIO: `http://localhost:9000` (console `http://localhost:9001`)

## Repository layout

- `services/` microservices
- `workers/python-worker/` worker runtime + CLI
- `dashboard/` React/Vite UI
- `tests/` integration + E2E scripts
- `docker/` local compose
- `k8s/` Kubernetes manifests
- `scripts/init-db.sql` DB bootstrap
