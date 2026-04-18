# ⎔ ModelHubX — Premium MLOps Registry & Deployment Deck

> **"High-end MLOps platform for automated AI model versioning and Kubernetes deployment. Features a premium glassmorphic SaaS dashboard, real-time cluster metrics, and dynamic K8s manifest synthesis."**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Redis](https://img.shields.io/badge/Redis-7-DC382D?style=flat-square)](https://redis.io)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-1.28-326CE5?style=flat-square)](https://kubernetes.io)
[![Docker](https://img.shields.io/badge/Docker-24-2496ED?style=flat-square)](https://docker.com)

---

## 💎 Signature Features

### 🚀 Mission Control Dashboard
- **Hyper-Blur Glassmorphism**: A premium, aerospace-inspired UI deck with deep obsidian and clean SaaS light modes.
- **Real-time Diagnostics**: Live metrics tracking Total Model Versions, Active Replicas, and simulated p99 Gateway Latency.
- **Dual-Theme Engine**: Seamlessly switch between **Obsidian Dark** and **Clean SaaS Light** with persistent user memory.

### 🏗️ K8s Infrastructure Synthesis
- **Automated Manifesting**: Generates production-ready Kubernetes YAMLs (Deployment, Service, HPA) on the fly for every uploaded model.
- **HPA Auto-scaling**: Dynamic resource limit calculation and Pod horizontal scaling (1-10 replicas) built into the synthesis engine.

### 📦 AI Model Registry
- **Versioned Storage**: Immutable model versioning (v1, v2, v3...) with automated latest-release tracking.
- **Inference Gateway**: Simulated ClusterIP gateway that mimics production inference routing for local testing.

---

## ⚡ Quick Start

### 1. Launch the Hub
```bash
# Start the API and Dashboard
docker compose up --build -d
```

### 2. Access Mission Control
- **Dashboard**: `open http://localhost:8000` (Directly served from API)
- **Inference Docs**: `http://localhost:8000/docs` (Swagger UI)

### 3. Usage Lifecycle
1. **Upload**: Use the UI to push `.pkl` models to the **Registry**.
2. **Synthesize**: Scale your pod replicas and click **Deploy to K8s** to generate production manifests.
3. **Test**: Use the **Inference Gateway** panel to send real JSON payloads to your deployed models.

---

## 🛠️ Internal Tech Stack

- **Backend**: Python 3.11 / FastAPI (Non-blocking I/O)
- **Storage**: Redis 7 (Metadata & Versioning)
- **Frontend**: Vanilla JS / HTML5 / Premium CSS (Glassmorphism Tier)
- **DevOps**: Docker Compose / Kubernetes Manifest Synth Engine

---

## 📝 License
MIT — Build anything with it.
