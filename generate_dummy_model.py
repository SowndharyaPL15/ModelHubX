import os
import pickle
import argparse
from datetime import datetime
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.datasets import make_classification
except ImportError:
    print("scikit-learn is required to generate the dummy model.")
    print("Install it with: pip install scikit-learn")
    exit(1)

def generate_dummy_model(name="fraud_detection", version=None):
    if version:
        versions = [version]
    else:
        versions = ["v1", "v2"]
        
    print(f"Generating dummy models for '{name}'...")
    
    for v in versions:
        filename = f"{name}_{v}.pkl"
        print(f"  -> Creating {filename}...")
        
        # Create a simple dataset and train a dummy model
        # Use different random states to make v1 and v2 slightly different
        rs = 42 if v == "v1" else 99
        X, y = make_classification(n_samples=100, n_features=4, random_state=rs)
        model = LogisticRegression()
        model.fit(X, y)
        
        # Bundle model and metadata
        bundled_model = {
            "model": model,
            "metadata": {
                "name": name,
                "version": v,
                "features": 4,
                "generated_at": datetime.now().isoformat()
            }
        }
        
        with open(filename, "wb") as f:
            pickle.dump(bundled_model, f)
        
        print(f"  [SUCCESS] Generated {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy ML models for ModelHubX demo")
    parser.add_argument("--name", default="fraud_detection", help="Base model name")
    parser.add_argument("--version", default=None, help="Specific version (default: v1 and v2)")
    args = parser.parse_args()
    
    generate_dummy_model(args.name, args.version)
    print("\nNext Steps:")
    print("1. Run 'docker-compose up --build'")
    print("2. Open http://localhost:8000 (or your frontend port)")
    print(f"3. Upload the generated .pkl files to see ModelHubX versioning in action!")
