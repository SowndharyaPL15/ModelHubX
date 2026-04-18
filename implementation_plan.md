# ModelHubX – Implementation Plan

Transitioning the project from a distributed job scheduler to **ModelHubX: A Distributed AI Model Deployment & Versioning Platform**.

## 🎯 Goal
Build a platform where users can upload machine learning models (`.pkl`, `.joblib`) and dynamically deploy them as version-controlled inference API endpoints.

## User Review Required

> [!WARNING]
> This is a complete architectural shift. The old "Worker Nodes" will be deleted. We are replacing the job queue paradigm with a **Registry & Deployment** paradigm.
> Please review the architecture below and let me know if you approve. If you approve, I will begin completely rewriting `main.py`, the `frontend`, and creating the new `inference` server.

## Proposed Architecture

I will structure the project into the following distinct micro-components:

1. **Model Storage**: A shared volume where uploaded models are physically saved.
2. **Metadata DB (Redis)**: Tracks model registries (e.g., `model:fraud-detect:v1`, `v2`) and active deployments.
3. **Core Delivery API (`api/main.py`)**: The central hub that accepts uploads and triggers deployments.
4. **Inference Engine (`inference/`)**: A generic FastAPI wrapper image. When a model is "deployed," the system binds a specific version of a model to this inference engine and serves it.
5. **Deployment Engine**: A subsystem inside the Core API that generates the **Kubernetes Deployment & Service YAMLs** automatically for the uploaded model.

---

## Proposed Changes

### [DELETE] Old Components
We will remove the job queue worker system since this is an entirely new architecture.
- `[DELETE]` worker/worker.py
- `[DELETE]` docker/Dockerfile.worker
- `[DELETE]` k8s/03-worker-cpu.yaml
- `[DELETE]` k8s/04-worker-gpu.yaml

### [NEW] API Core (ModelHubX Controller)
- `[MODIFY]` api/main.py
  - **Endpoints:**
    - `POST /api/models`: Upload a `.pkl` file. Automatically creates v1, v2, v3 metadata in Redis.
    - `GET /api/models`: List all models and their available versions.
    - `POST /api/deployments`: Request to deploy a specific model version.
    - `PUT /api/deployments/{id}/rollback`: Automatically shift an active deployment to a previous version.
  - **Kubernetes Synthesizer:** When a deployment is requested, the Python API will dynamically generate scalable `deployment.yaml` and `service.yaml` files for that specific model, mimicking enterprise behavior.

### [NEW] Inference Server Image
A lightweight, dynamic inference server that is used as the base image for every deployed model.
- `[NEW]` inference/server.py
  - A simple API containing a `/predict` endpoint that loads the injected `.pkl` model path.
- `[NEW]` docker/Dockerfile.inference
  - The universal Docker image tag `modelhubx-inference:latest` that our dynamic K8s pods will use.

### [NEW] Frontend Interface
- `[MODIFY]` frontend/index.html
  - **Model Registry View**: Drag & Drop interface to upload models. Displays versions for each model.
  - **Deployments View**: View active REST API Inference endpoints. One-click **Rollback** and **Scale** controls.

### [NEW] Docker-Compose Local Setup
- `[MODIFY]` docker-compose.yml
  - Will consist of `redis`, `modelhubx-api`, and a shared volume `model-data` to simulate a cloud bucket. Active deployments will be spawned dynamically by the API, or generated as manual tests.

---

## Open Questions

1. **Model execution simulation**: Since users and graders might not have actual working `.pkl` models on hand to test with, do you want me to include a "generate_test_model.py" script that spits out a dummy scikit-learn model so you have something to upload during the demo?
2. **Kubernetes integration**: For local testing without a real Kubernetes cluster, do you want the API to just output the beautifully formatted Kubernetes YAMLs to an `out/` folder so you can show the recruiter "Look, it generates the K8s manifest files dynamically based on model version"?

## Verification Plan

### Automated Tests
- Run `docker-compose up --build`
- Verify the backend handles model uploads and correctly increments versions (v1 -> v2).

### Manual Verification
- You will upload a dummy `.pkl` locally via the UI.
- Verify the system generates K8s infrastructure configs.
- Verify that deploying different versions reflects successfully on the Dashboard.
