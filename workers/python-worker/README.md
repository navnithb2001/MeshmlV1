# MeshML Python Worker

Minimal usage guide for the Python worker.

## Requirements

- Python 3.9+
- MeshML API Gateway URL
- Invitation code for a group

## Install

```bash
pip install -i https://test.pypi.org/simple/ meshml-worker==0.2.6
```

## Configure

The worker stores config at `.meshml/config.yaml` in the directory where you run `init`.
You can also set `MESHML_API_URL` to override the API Gateway URL.

## CLI Workflow

```bash
# Initialize once (writes .meshml/config.yaml in current directory)
meshml-worker init --api-url http://localhost:8000

# Login (saves token to ~/.meshml/auth.json)
meshml-worker login --email you@example.com

# Join group with invitation code
meshml-worker join --invitation-code inv_abc123 --worker-id my-laptop

# Start training loop
meshml-worker run
```

## Common Commands

```bash
meshml-worker status
meshml-worker config --show
```

## Notes

- The worker uses gRPC streams for orchestration, parameters, and metrics.
- Models and data batches are downloaded via signed URLs.
