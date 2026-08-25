import os
import uuid
import json
import time
import pickle
from datetime import datetime
from typing import Optional, List

import redis
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ── App Setup ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="ModelHubX",
    description="Distributed AI Model Deployment & Versioning Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Env & Storage ───────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
MODELS_DIR = os.getenv("MODELS_DIR", "./models")
OUT_DIR = os.getenv("OUT_DIR", "./out")

# Resolve FRONTEND_DIR dynamically by scanning likely paths
FRONTEND_DIR = os.getenv("FRONTEND_DIR")
if not FRONTEND_DIR:
    possible_dirs = [
        "/frontend",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend"),
        "./frontend",
        "../frontend"
    ]
    for d in possible_dirs:
        if os.path.exists(os.path.join(d, "index.html")):
            FRONTEND_DIR = d
            break
    if not FRONTEND_DIR:
        FRONTEND_DIR = "../frontend"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

def get_redis():
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# ── Keys ────────────────────────────────────────────────────────────────────
MODELS_HSET = "modelhub:models"           # Map of name -> list of versions
DEPLOYMENTS_HSET = "modelhub:deployments" # Map of deployment_id -> deployment_metadata

# ── Serving ─────────────────────────────────────────────────────────────────
@app.get("/")
def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "ModelHubX API is running. Dashboard file not found."}

# Mount other static assets if any
if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

# ── Models ──────────────────────────────────────────────────────────────────
class DeploymentRequest(BaseModel):
    model_name: str
    version: str
    replicas: int = 1

class PredictRequest(BaseModel):
    input_data: list

# ── Kubernetes Synthesizer ──────────────────────────────────────────────────
def generate_k8s_manifests(model_name: str, version: str, replicas: int, deployment_id: str):
    """
    Simulates enterprise MLOps platform generating K8s manifests for model deployment.
    """
    safe_name = model_name.lower().replace("_", "-").replace(" ", "-")
    timestamp = datetime.utcnow().isoformat()
    
    # Generate Deployment YAML
    deployment_yaml = f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {safe_name}-{version}-deployment
  namespace: modelhubx
  labels:
    app: {safe_name}
    version: {version}
    deployment_id: {deployment_id}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {safe_name}
      version: {version}
  template:
    metadata:
      labels:
        app: {safe_name}
        version: {version}
    spec:
      containers:
      - name: model-inference
        image: modelhubx-inference:latest
        env:
        - name: MODEL_PATH
          value: "/models/{model_name}/{version}.pkl"
        - name: DEPLOYMENT_ID
          value: "{deployment_id}"
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        resources:
          limits:
            memory: "1Gi"
            cpu: "500m"
          requests:
            memory: "512Mi"
            cpu: "250m"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {safe_name}-{version}-hpa
  namespace: modelhubx
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {safe_name}-{version}-deployment
  minReplicas: {replicas}
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
"""
    # Generate Service YAML
    service_yaml = f"""apiVersion: v1
kind: Service
metadata:
  name: {safe_name}-{version}-service
  namespace: modelhubx
spec:
  selector:
    app: {safe_name}
    version: {version}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8080
  type: ClusterIP
"""
    
    # Save the manifests locally
    deploy_file = os.path.join(OUT_DIR, f"{safe_name}-{version}-deployment.yaml")
    svc_file = os.path.join(OUT_DIR, f"{safe_name}-{version}-service.yaml")
    
    with open(deploy_file, "w") as f:
        f.write(deployment_yaml)
    with open(svc_file, "w") as f:
        f.write(service_yaml)
        
    return deployment_yaml, service_yaml


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    try:
        r = get_redis()
        r.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Redis unavailable: {e}")

@app.post("/api/models", tags=["Model Registry"])
async def upload_model(name: str = Form(...), file: UploadFile = File(...)):
    """Upload a model file, auto-assigning it a sequential version (v1, v2)."""
    r = get_redis()
    
    # Get existing history
    history_raw = r.hget(MODELS_HSET, name)
    history = json.loads(history_raw) if history_raw else []
    
    # Determine new version
    version = f"v{len(history) + 1}"
    
    # Save file physically
    model_dir = os.path.join(MODELS_DIR, name.replace(" ", "_"))
    os.makedirs(model_dir, exist_ok=True)
    file_path = os.path.join(model_dir, f"{version}.pkl")
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # Append to history
    model_record = {
        "version": version,
        "filename": file.filename,
        "size_bytes": len(content),
        "uploaded_at": datetime.utcnow().isoformat(),
        "path": file_path
    }
    history.append(model_record)
    
    r.hset(MODELS_HSET, name, json.dumps(history))
    
    return {
        "message": "Model uploaded successfully",
        "name": name,
        "version": version,
        "path": file_path
    }

@app.get("/api/models", tags=["Model Registry"])
def list_models():
    """List all registered models and their versions."""
    r = get_redis()
    all_models = r.hgetall(MODELS_HSET)
    response = []
    
    for name, history_raw in all_models.items():
        history = json.loads(history_raw)
        response.append({
            "name": name,
            "versions": history,
            "latest_version": history[-1]["version"] if history else None
        })
    return response

@app.post("/api/deployments", tags=["Deployments"])
def deploy_model(req: DeploymentRequest):
    """Deploy a model version, triggering K8s manifest auto-generation."""
    r = get_redis()
    
    # Validate model exists
    history_raw = r.hget(MODELS_HSET, req.model_name)
    if not history_raw:
        raise HTTPException(status_code=404, detail="Model not found")
        
    history = json.loads(history_raw)
    target_env = next((v for v in history if v["version"] == req.version), None)
    if not target_env:
        raise HTTPException(status_code=404, detail=f"Version {req.version} not found")
        
    deployment_id = str(uuid.uuid4())[:8]
    
    # 1. Synthesize Kubernetes manifests
    deploy_yaml, svc_yaml = generate_k8s_manifests(
        req.model_name, req.version, req.replicas, deployment_id
    )
    
    # 2. Record deployment
    endpoint = f"/predict/{deployment_id}"
    deployment = {
        "id": deployment_id,
        "model_name": req.model_name,
        "version": req.version,
        "replicas": req.replicas,
        "status": "Active",
        "endpoint": endpoint,
        "deployed_at": datetime.utcnow().isoformat(),
        "manifests": {
            "deployment": deploy_yaml,
            "service": svc_yaml
        },
        "model_path": target_env["path"]
    }
    
    r.hset(DEPLOYMENTS_HSET, deployment_id, json.dumps(deployment))
    
    return {
        "message": "Deployment successful",
        "deployment": deployment
    }

@app.get("/api/deployments", tags=["Deployments"])
def list_deployments():
    """List all active model API endpoints."""
    r = get_redis()
    deps_raw = r.hgetall(DEPLOYMENTS_HSET)
    return [json.loads(v) for v in deps_raw.values()]

@app.get("/api/deployments/{deployment_id}/manifests", tags=["Deployments"])
def get_deployment_manifests(deployment_id: str):
    """Retrieve the generated K8s manifests for a deployment."""
    r = get_redis()
    dep_raw = r.hget(DEPLOYMENTS_HSET, deployment_id)
    if not dep_raw:
        raise HTTPException(status_code=404, detail="Deployment not found")
    deployment = json.loads(dep_raw)
    return deployment["manifests"]

@app.put("/api/deployments/{deployment_id}/rollback", tags=["Deployments"])
def rollback_deployment(deployment_id: str, version: str):
    """Update an existing deployment to a different version (Rolling Update)."""
    r = get_redis()
    dep_raw = r.hget(DEPLOYMENTS_HSET, deployment_id)
    if not dep_raw:
        raise HTTPException(status_code=404, detail="Deployment not found")
        
    deployment = json.loads(dep_raw)
    
    # Validate the new version exists
    history_raw = r.hget(MODELS_HSET, deployment["model_name"])
    history = json.loads(history_raw)
    target_env = next((v for v in history if v["version"] == version), None)
    if not target_env:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")
        
    # Update deployment
    deployment["version"] = version
    deployment["model_path"] = target_env["path"]
    deployment["rolled_back_at"] = datetime.utcnow().isoformat()
    
    # Regenerate K8s manifests
    deploy_yaml, svc_yaml = generate_k8s_manifests(
        deployment["model_name"], version, deployment["replicas"], deployment_id
    )
    deployment["manifests"]["deployment"] = deploy_yaml
    deployment["manifests"]["service"] = svc_yaml
    
    r.hset(DEPLOYMENTS_HSET, deployment_id, json.dumps(deployment))
    return {"message": "Rollback successful", "deployment": deployment}

@app.delete("/api/deployments/{deployment_id}", tags=["Deployments"])
def remove_deployment(deployment_id: str):
    """Tear down a deployment."""
    r = get_redis()
    success = r.hdel(DEPLOYMENTS_HSET, deployment_id)
    if not success:
        raise HTTPException(status_code=404, detail="Deployment not found")
    return {"message": "Deployment terminated"}


# ── Internal Gateway Predict Simulation ──────────────────────────────────────
# Instead of doing Docker-in-Docker to run real isolated inference pods for Local testing,
# We simply dynamically load the selected model's .pkl file exactly as the deployed K8s Pod would.
# This simulates hitting the ClusterIP service of the deployed model!
@app.post("/predict/{deployment_id}", tags=["Inference Gateway"])
async def predict_gateway(deployment_id: str, req: PredictRequest):
    """
    Hit the exposed model endpoint with input data (e.g. {"input_data": [[1,2,3,4]]}).
    """
    r = get_redis()
    dep_raw = r.hget(DEPLOYMENTS_HSET, deployment_id)
    if not dep_raw:
        raise HTTPException(status_code=404, detail="Inference endpoint not found. Deployment may be terminated.")
        
    deployment = json.loads(dep_raw)
    model_path = deployment["model_path"]
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model file missing from disk!")
        
    try:
        print(f"DEBUG: [K8s Simulated Ingress] Routing request for {deployment_id} to Pod {model_path}")
        with open(model_path, "rb") as f:
            bundled = pickle.load(f)
            
        model = bundled["model"]
        version = bundled["metadata"].get("version", "unknown")
        
        # Make real prediction using the loaded model
        # Normally input_data might be passed as json nested arrays.
        predictions = model.predict(req.input_data).tolist()
        
        print(f"DEBUG: [K8s Simulated Ingress] Inference completed for version {version}")
        return {
            "deployment_id": deployment_id,
            "model_name": deployment["model_name"],
            "model_version": version,
            "predictions": predictions
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}. Make sure input matches model features!")
